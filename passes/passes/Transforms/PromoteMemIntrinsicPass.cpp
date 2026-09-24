/*
 * GenMC -- Generic Model Checking.
 *
 * This project is dual-licensed under the Apache License 2.0 and the MIT License.
 * You may choose to use, distribute, or modify this software under either license.
 *
 * Apache License 2.0:
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * MIT License:
 *     https://opensource.org/licenses/MIT
 */

#include <llvm/IR/Operator.h>
#include "PromoteMemIntrinsicPass.hpp"
#include "genmc/Support/Error.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ranges>
#include <utility>
#include <vector>

/* With opaque pointers, no ptr->int->ptr cast is needed */
#define CAST_PTR_TO_TYPE_THROUGH_INT(builder, ptr, dstTy, dataLayout) ptr

#define CONSTEXPR_GET_FIELD_TYPE(MI, Field) MI->get##Field()->getType()

/* Helper macro that moves constant expressions into their own instructions */
#define LOWER_CONSTEXPR(builder, MI, Field)                                                        \
	if (auto *constExpr = dyn_cast<ConstantExpr>(MI->get##Field())) {                          \
		Value *newGEP = builder.Insert(constExpr->getAsInstruction());                     \
		[[maybe_unused]] Type *destType = CONSTEXPR_GET_FIELD_TYPE(MI, Field);             \
		Value *castedGEP = CAST_PTR_TO_TYPE_THROUGH_INT(                                   \
			builder, newGEP, destType,                                                 \
			MI->getParent()->getParent()->getParent()->getDataLayout());               \
		MI->set##Field(castedGEP);                                                         \
	}

using namespace llvm;

/**
 * Lower a call to: __memcpy_chk(void * dest, const void * src, size_t len, size_t destlen);
 *
 * Reference implementation (see __memcpy_chk source):
 * 	if dstlen < len
 * 		__chk_fail ();
 * 	return memcpy(dest, src, len);
 *
 * __memcpy_chk spec:
 * http://refspecs.linux-foundation.org/LSB_4.0.0/LSB-Core-generic/LSB-Core-generic/libc---memcpy-chk-1.html
 * __memcpy_chk source:
 * https://sourceware.org/git/?p=glibc.git;a=blob;f=debug/memcpy_chk.c;h=6f8628ab945c5e8d2738f7382238c1e1006afe41;hb=HEAD
 * memcpy spec: https://pubs.opengroup.org/onlinepubs/9699919799/functions/memcpy.html
 */
static void lowerFortifiedMemCpy(CallInst *CI, Function &F)
{
	/* Get arguments */
	auto *dst_ptr = CI->getArgOperand(0);
	auto *src_ptr = CI->getArgOperand(1);
	auto *len = CI->getArgOperand(2);
	auto *dst_len = CI->getArgOperand(3);

	/* Split the basic block at the call to __memcpy_chk in bb (before call-instr.) and contBB
	 * (rest of the block) */
	auto *bb = CI->getParent();
	auto *contBB =
		bb->splitBasicBlock(CI); /* Creates a dummy terminator for bb -> remove later */

	/* Insert the llvm.memcpy intrinsic at the start of contBB */
	IRBuilder<> contBuilder(contBB, contBB->begin());
	contBuilder.CreateMemCpy(dst_ptr, Align(1), src_ptr, Align(1), len);
	CI->replaceAllUsesWith(
		dst_ptr); /* memcpy(...) returns the dst_ptr back, see: memcpy Spec */

	/* Create a new basic block "memcpy__chk_fail" for inside the IF => aborts using an
	 * unreachable-instruction */
	auto *failBB = BasicBlock::Create(F.getContext(), "memcpy__chk_fail", &F);
	auto *assertFailFun = F.getParent()->getFunction("__VERIFIER_assert_fail");
	CallInst::Create(assertFailFun, {}, "", failBB);
	new UnreachableInst(F.getContext(), failBB);

	/* Compare arguments: dstlen < len */
	auto *dummyTerminator = bb->getTerminator();
	IRBuilder<> termBuilder(dummyTerminator);
	auto *cmp = termBuilder.CreateICmpULT(dst_len, len);

	/* Replace the dummy terminator: We jump to failBB if the condition holds */
	dummyTerminator->eraseFromParent();
	termBuilder.SetInsertPoint(bb);
	termBuilder.CreateCondBr(cmp, /* True */ failBB, /* False */ contBB);
}

/**
 * We collect and lower all calls to the fortified version of memcpy: __memcpy_chk
 * The fortified version checks for buffer overflows before performing the memcpy.
 * @see lowerFortifiedMemCpy for implementation details
 */
static auto lowerFortifiedCalls(Function &F) -> bool
{
	auto modified = false;
	SmallVector<CallInst *, 8> fortifiedCalls;

	/* Collect call-instructions to __memcpy_chk */
	for (auto &I : instructions(F)) {
		auto *ci = dyn_cast<CallInst>(&I);
		if (!ci)
			continue;

		auto *calledFun = dyn_cast<Function>(ci->getCalledOperand());
		if (!calledFun || calledFun->getName() != "__memcpy_chk")
			continue;

		fortifiedCalls.push_back(ci);
		modified = true;
	}

	for (auto *ci : fortifiedCalls)
		lowerFortifiedMemCpy(ci, F);
	for (auto *ci : fortifiedCalls)
		ci->eraseFromParent();
	return modified;
}

/* The type an operand names, or null when the pass cannot type it: a global, an alloca, or a
 * GEP -- instruction or constant expression -- whose all-zero indices name the object it
 * indexes into (`getelementptr [2 x i32], ptr %a, i64 0, i64 0` is the array; promoting by its
 * result element type would copy one element of several) and whose other indices name the
 * member. A null, an undef, a cast expression, a load, a call result or a parameter has no
 * type here, and an intrinsic over it is lowered to a loop instead. */
static auto getPromotionGEPType(Value *op) -> Type *
{
	if (auto *v = dyn_cast<GlobalVariable>(op))
		return v->getValueType();
	if (auto *ai = dyn_cast<AllocaInst>(op))
		return ai->getAllocatedType();
	if (auto *gep = dyn_cast<GEPOperator>(op)) {
		if (gep->hasAllZeroIndices())
			return gep->getSourceElementType();
		return gep->getResultElementType();
	}
	return nullptr;
}

static auto isPromotableMemIntrinsicOperand(Value *op) -> bool
{
	return getPromotionGEPType(op) != nullptr;
}

/* The type a mem intrinsic of `len` bytes is promoted over, given the type its destination
 * names. After SROA a copy is often byte-addressed -- `getelementptr i8, ptr %p, i64 5` for a
 * three-byte tail of a struct -- and the named type (i8) is then smaller than the copy. Such a
 * copy is promoted over `[len/size x T]`: the same accesses the intrinsic performs, at the width
 * the destination names, which is the width the program reads them back at. */
static auto getPromotionTypeForLen(Type *named, uint64_t len, const DataLayout &DL) -> Type *
{
	auto size = DL.getTypeStoreSize(named);
	if (size >= len || size == 0)
		return named;
	if ((named->isIntegerTy() || named->isPointerTy()) && len % size == 0)
		return ArrayType::get(named, len / size);
	return named;
}

static void promoteMemCpy(IRBuilder<> &builder, Value *dst, Value *src,
			  const std::vector<Value *> &args, Type *typ, uint64_t &remainingLen,
			  Type *gepTy)
{
	if (remainingLen == 0)
		return;

	auto *srcGEP = builder.CreateInBoundsGEP(gepTy, src, args, "memcpy.src.gep");
	auto *dstGEP = builder.CreateInBoundsGEP(gepTy, dst, args, "memcpy.dst.gep");

	auto len = builder.GetInsertBlock()->getModule()->getDataLayout().getTypeStoreSize(typ);
	if (len > remainingLen) {
		/* The copy ends inside this leaf: its length is not a multiple of the type it is
		 * promoted over, which happens when a compiler splits a struct copy and the tail
		 * is padding. The bytes that are copied are copied one at a time, at the width
		 * a byte-addressed intrinsic already implies for them. */
		auto *i8Ty = IntegerType::getInt8Ty(typ->getContext());
		auto *i64Ty = IntegerType::getInt64Ty(typ->getContext());
		for (uint64_t b = 0; b < remainingLen; ++b) {
			auto *off = Constant::getIntegerValue(i64Ty, APInt(64, b));
			auto *sg = builder.CreateInBoundsGEP(i8Ty, srcGEP, {off}, "memcpy.src.tail");
			auto *dg = builder.CreateInBoundsGEP(i8Ty, dstGEP, {off}, "memcpy.dst.tail");
			builder.CreateStore(builder.CreateLoad(i8Ty, sg, "memcpy.src.tail.load"), dg);
		}
		remainingLen = 0;
		return;
	}

	remainingLen -= len;
	auto *srcLoad = builder.CreateLoad(typ, srcGEP, "memcpy.src.load");
	builder.CreateStore(srcLoad, dstGEP);
}

static void promoteMemSet(IRBuilder<> &builder, Value *dst, Value *argVal,
			  const std::vector<Value *> &args, Type *typ, Type *gepTy)
{
	VERIFY(typ->isIntegerTy() || typ->isPointerTy());
	VERIFY(isa<ConstantInt>(argVal));

	const auto &DL = builder.GetInsertBlock()->getParent()->getParent()->getDataLayout();
	auto sizeInBits = typ->isIntegerTy() ? typ->getIntegerBitWidth()
					     : DL.getPointerTypeSizeInBits(typ);
	const long int ival = cast<ConstantInt>(argVal)->getSExtValue();
	Value *val = Constant::getIntegerValue(typ, APInt(sizeInBits, ival));

	Value *dstGEP = builder.CreateInBoundsGEP(gepTy, dst, args, "memset.dst.gep");
	builder.CreateStore(val, dstGEP);
}

template <typename F>
static void promoteMemIntrinsic(Type *typ, std::vector<Value *> &args, F &&promoteFun)
{
	auto *i32Ty = IntegerType::getInt32Ty(typ->getContext());

	if (!isa<StructType>(typ) && !isa<ArrayType>(typ) && !isa<VectorType>(typ)) {
		std::forward<F>(promoteFun)(typ, args);
		return;
	}

	if (auto *arrayType = dyn_cast<ArrayType>(typ)) {
#ifdef LLVM_HAS_GLOBALOBJECT_GET_METADATA
		auto numElems = arrayType->getNumElements();
#else
		auto numElems = arrayType->getArrayNumElements();
#endif
		for (auto i = 0U; i < numElems; i++) {
			args.push_back(Constant::getIntegerValue(i32Ty, APInt(32, i)));
			promoteMemIntrinsic(arrayType->getElementType(), args, promoteFun);
			args.pop_back();
		}
	} else if (auto *structType = dyn_cast<StructType>(typ)) {
		for (auto i = 0U; i < structType->getNumElements(); ++i) {
			args.push_back(Constant::getIntegerValue(i32Ty, APInt(32, i)));
			promoteMemIntrinsic(structType->getElementType(i), args, promoteFun);
			args.pop_back();
		}
	} else {
		UNREACHABLE();
	}
}

static auto canPromoteMemIntrinsic(MemIntrinsic *MI) -> bool
{
	/* Skip if length is not a constant */
	auto *length = dyn_cast<ConstantInt>(MI->getLength());
	if (!length) {
		WARN_ONCE("memintr-length",
			  "Cannot promote non-constant-length mem intrinsic! Skipping...\n");
		return false;
	}

	/*
	 * For memcpy(), make sure source and dest live in the same address space
	 * (This also makes sure we are not copying from genmc's space to userspace)
	 */
	auto *MCI = dyn_cast<MemCpyInst>(MI);
	VERIFY(!MCI || MCI->getSourceAddressSpace() == MCI->getDestAddressSpace());
	if (MCI && !(isPromotableMemIntrinsicOperand(MCI->getDest()) ||
		     isPromotableMemIntrinsicOperand(MCI->getSource()))) {
		WARN_ONCE("memintr-opaque", "Cannot promote memcpy() due to both src and dst being "
					    "opaque! Skipping...\n");
		return false;
	}

	auto *MSI = dyn_cast<MemSetInst>(MI);
	if (MSI && !isPromotableMemIntrinsicOperand(MSI->getDest())) {
		WARN_ONCE("memintr-dst",
			  "Cannot promote memset() due to dst being opaque! Skipping...\n");
		return false;
	}

	/*
	 * Finally, this is one of the cases we can currently handle.
	 * We produce a warning anyway because, e.g., if a small struct has no atomic
	 * fields, clang might initialize it with memcpy(), and then read it with a
	 * 64bit access, and mixed-size accesses are __bad__ news
	 */
	WARN_ONCE("promote-memintrinsic", "Memory intrinsic found! Attempting to promote it...\n");
	return true;
}

/**
 * Due to LLVM's use of opaque pointers, we infer the types based on the instruction defining the
 * value (Alloca, GEP, GlobalVal => @see getPromotionGEPType()). To allow copying between different
 * (compatible) data-types, we perform a cast either src->dst or dst->src.
 *
 * Note: We assume that the compiler only generates memcpy()s between "compatible" types:
 *  - compatible: struct {i32, i32, i32} and [3 x i32]
 *  - non-compatible:  struct {i32, i64, i32} and [4 x i32]
 * Promotion in the second case would lead to "mixed-sized accesses".
 */
static auto getRecastedOperands(MemCpyInst *MI, IRBuilder<> &builder) -> std::pair<Value *, Value *>
{
	auto *dst = MI->getDest();
	auto *src = MI->getSource();

	auto dataLayout = MI->getParent()->getParent()->getParent()->getDataLayout();
	auto indexSize = dataLayout.getIndexSizeInBits(dst->getType()->getPointerAddressSpace());
	const std::vector<Value *> gepArgs = {Constant::getIntegerValue(
		IntegerType::get(builder.getContext(), indexSize),
		APInt(indexSize, 0))}; // Ex: getelementptr %type, %ptr, i32 0

	/* Case src->dst */
	if (isPromotableMemIntrinsicOperand(dst)) {
		src = CAST_PTR_TO_TYPE_THROUGH_INT(builder, src, dst->getType(), dataLayout);
		return {builder.CreateGEP(getPromotionGEPType(dst), src, ArrayRef(gepArgs)), dst};
	}

	/* Case dst->src */
	VERIFY(isPromotableMemIntrinsicOperand(src));
	dst = CAST_PTR_TO_TYPE_THROUGH_INT(builder, dst, src->getType(), dataLayout);
	return {src, builder.CreateGEP(getPromotionGEPType(src), dst, ArrayRef(gepArgs))};
}

/* Tries to promote a memcpy() instruction.
 * We also allow the promotion of memcpy()s w/ constant arguments:
 *
 * call void @llvm.memcpy.p0.p0.i64(
 *      ptr align 8 getelementptr inbounds (%struct.queue_t, ptr @queue, i64 0, i32 0, i32 1),
 *      ptr align 8 %_3,
 *      i64 8,
 *      i1 false)
 *
 * by moving constant expressions into their own instruction and proceeding as normal. */
static auto tryPromoteMemCpy(MemCpyInst *MI, SmallVector<llvm::MemIntrinsic *, 8> &promoted) -> bool
{
	if (!canPromoteMemIntrinsic(MI))
		return false;

	/* We only copy "len" bytes (3rd arg in llvm.memcpy) */
	auto len = cast<ConstantInt>(MI->getLength())->getZExtValue();

	/* Remove memcpy's with len=0 completely as no-ops (generated by rustc) */
	if (len == 0) {
		WARN_ONCE("memintr-zero-length",
			  "Cannot promote zero-length mem intrinsic! Removing instruction...\n");
		promoted.push_back(MI);
		return true;
	}

	IRBuilder<> builder(MI);
	auto *i64Ty = IntegerType::getInt64Ty(MI->getContext());
	auto *nullInt = Constant::getNullValue(i64Ty);

	/* Check for constexpr arguments */
	LOWER_CONSTEXPR(builder, MI, Source); // NOLINT(misc-const-correctness)
	LOWER_CONSTEXPR(builder, MI, Dest);   // NOLINT(misc-const-correctness)

	/* Recast args to same types for GEP indexing later */
	auto [src, dst] = getRecastedOperands(MI, builder);
	const auto &DL = MI->getParent()->getModule()->getDataLayout();
	auto *dstTyp = getPromotionTypeForLen(getPromotionGEPType(dst), len, DL);
	VERIFY(dstTyp);

	/* To ensure we only copy "len" bytes from total */
	auto typeSizeDst = DL.getTypeStoreSize(dstTyp);
	VERIFY(typeSizeDst >= len);

	std::vector<Value *> args = {nullInt};
	promoteMemIntrinsic(dstTyp, args, [&](Type *typ, const std::vector<Value *> &args) {
		promoteMemCpy(builder, dst, src, args, typ, len, dstTyp);
	});
	promoted.push_back(MI);
	return true;
}

/* A memset whose destination the pass cannot see (a phi, a call result, a parameter) is still
 * promotable when its length is a constant: it is `len / w` stores of the pattern at width w,
 * where w is the alignment the intrinsic itself asserts for the destination, capped at a word.
 * clang emits exactly this shape for a constructor loop zeroing one field per iteration
 * (`memset(phi-ptr, 0, 1)` for each std::atomic<bool> of an array), which instcombine used to
 * fold into a plain store; a pipeline that does not run instcombine hands it here. */
static auto tryPromoteOpaqueMemSet(MemSetInst *MS, SmallVector<MemIntrinsic *, 8> &promoted) -> bool
{
	auto *lenC = dyn_cast<ConstantInt>(MS->getLength());
	auto *valC = dyn_cast<ConstantInt>(MS->getValue());
	if (!lenC || !valC)
		return false;
	auto len = lenC->getZExtValue();
	if (len == 0) {
		promoted.push_back(MS);
		return true;
	}
	uint64_t w = MS->getDestAlign().valueOrOne().value();
	if (w > 8)
		w = 8;
	while (w > 1 && len % w != 0)
		w /= 2;
	IRBuilder<> builder(MS);
	auto *elemTy = IntegerType::get(MS->getContext(), 8 * w);
	auto *i64Ty = IntegerType::getInt64Ty(MS->getContext());
	/* The pattern byte replicated to the store width */
	uint64_t byte = valC->getZExtValue() & 0xff, pattern = 0;
	for (uint64_t i = 0; i < w; ++i)
		pattern |= byte << (8 * i);
	auto *val = ConstantInt::get(elemTy, pattern);
	for (uint64_t i = 0; i < len / w; ++i) {
		auto *gep = builder.CreateInBoundsGEP(elemTy, MS->getDest(),
						      {ConstantInt::get(i64Ty, i)}, "memset.opaque.gep");
		builder.CreateStore(val, gep);
	}
	promoted.push_back(MS);
	return true;
}

static auto tryPromoteMemSet(MemSetInst *MS, SmallVector<MemIntrinsic *, 8> &promoted) -> bool
{
	if (!isPromotableMemIntrinsicOperand(MS->getDest()) && tryPromoteOpaqueMemSet(MS, promoted))
		return true;
	if (!canPromoteMemIntrinsic(MS))
		return false;

	auto *dst = MS->getDest();
	auto *val = MS->getValue();

	auto *i64Ty = IntegerType::getInt64Ty(MS->getContext());
	auto *nullInt = Constant::getNullValue(i64Ty);
	auto *dstTyp = getPromotionGEPType(dst);
	VERIFY(dstTyp);
	if (auto *lenC = dyn_cast<ConstantInt>(MS->getLength()))
		dstTyp = getPromotionTypeForLen(dstTyp, lenC->getZExtValue(),
						MS->getParent()->getModule()->getDataLayout());

	IRBuilder<> builder(MS);
	std::vector<Value *> args = {nullInt};

	promoteMemIntrinsic(dstTyp, args, [&](Type *typ, const std::vector<Value *> &args) {
		promoteMemSet(builder, dst, val, args, typ, dstTyp);
	});
	promoted.push_back(MS);
	return true;
}

static void removePromoted(std::ranges::input_range auto &&promoted)
{
	for (auto *MI : promoted) {
		/* Are MI's operands used anywhere else? */
		auto *dst = dyn_cast<BitCastInst>(MI->getRawDest());
		CastInst *src = nullptr;
		if (auto *MC = dyn_cast<MemCpyInst>(MI))
			src = dyn_cast<BitCastInst>(MC->getRawSource());

		MI->eraseFromParent();
		if (dst && dst->hasNUses(0))
			dst->eraseFromParent();
		if (src && src->hasNUses(0))
			src->eraseFromParent();
	}
}

/* A mem intrinsic whose length is not a constant, and every memmove, becomes a byte loop:
 *
 *     i = 0
 *   hdr: if (i < len) goto body else goto post
 *   body: dst[k] = (memset ? val : src[k]), where k = i, or len-1-i when a memmove's
 *         destination starts inside its source; i = i + 1; goto hdr
 *
 * Each iteration is one load and one store the checker models like any other access, and the
 * loop is bounded like every other loop. Without this the intrinsic reached the interpreter,
 * whose memset()/memcpy() handlers are an unconditional error (they cannot know the width of
 * the accesses that follow), and a program with a single runtime-length memset -- a vector
 * assign, a bitset clear -- could not be explored past it. Measured on a hypergraph rewriting
 * engine: the composed program stopped at "Invalid call to memset()" inside its first rewrite.
 */
static auto promoteRuntimeLength(MemIntrinsic *MI, SmallVector<MemIntrinsic *, 8> &promoted,
				 std::optional<unsigned> bound) -> bool
{
	auto &ctx = MI->getContext();
	auto *F = MI->getFunction();
	auto *len = MI->getLength();
	auto *lenTy = len->getType();
	auto *i8Ty = IntegerType::getInt8Ty(ctx);
	auto *dst = MI->getRawDest();
	Value *src = nullptr;
	if (auto *MT = dyn_cast<MemTransferInst>(MI))
		src = MT->getRawSource();
	const bool backwardIfOverlapping = isa<MemMoveInst>(MI);

	/* The width of the loop's accesses: the scalar the destination (and source) names, so a
	 * store here is the width of the load that reads it back -- the same rule the constant
	 * case follows -- and a copy of n words is n iterations rather than 8n. A byte loop over
	 * the remainder covers a length that is not a multiple of the width. */
	const auto &DL = F->getParent()->getDataLayout();
	auto scalarWidth = [&](Value *op) -> uint64_t {
		if (!op || !isPromotableMemIntrinsicOperand(op))
			return 0;
		Type *t = getPromotionGEPType(op);
		if (!t)
			return 0;
		while (auto *at = dyn_cast<ArrayType>(t))
			t = at->getElementType();
		if (!t->isIntegerTy() && !t->isPointerTy())
			return 0;
		const auto w = DL.getTypeStoreSize(t).getFixedValue();
		return (w == 1 || w == 2 || w == 4 || w == 8) ? w : 0;
	};
	uint64_t w = scalarWidth(dst);
	if (src) {
		const auto ws = scalarWidth(src);
		if (w == 0) w = ws;
		else if (ws != 0 && ws != w) w = 1;
	}
	if (w <= 1) {
		/* Operands the pass cannot see through (parameters, phis, call results), and a
		 * byte-addressed GEP after SROA (`getelementptr i8, ptr %p, i64 k`) that names i8
		 * for memory written at word width: the alignment the intrinsic asserts is the
		 * element width clang gives a copy of T[n] -- alignof(T) -- capped at a word and
		 * at both operands' alignments. A byte loop over word-written memory reads bytes
		 * no write label covers, which the checker reports as an uninitialised read. */
		uint64_t a = MI->getDestAlign().valueOrOne().value();
		if (auto *MT = dyn_cast<MemTransferInst>(MI))
			a = std::min<uint64_t>(a, MT->getSourceAlign().valueOrOne().value());
		w = std::max<uint64_t>(w, std::min<uint64_t>(a, 8));
		while (w > 1 && (w & (w - 1)) != 0) --w;
		if (w == 0) w = 1;
	}
	auto *elemTy = IntegerType::get(ctx, 8 * static_cast<unsigned>(w));

	BasicBlock *pre = MI->getParent();
	BasicBlock *post = pre->splitBasicBlock(MI, "memintr.post");
	BasicBlock *hdr = BasicBlock::Create(ctx, "memintr.hdr", F, post);
	BasicBlock *body = BasicBlock::Create(ctx, "memintr.body", F, post);

	pre->getTerminator()->eraseFromParent();
	IRBuilder<> b(pre);
	Value *backward = nullptr;
	if (backwardIfOverlapping) {
		/* dst inside [src, src+len): copy from the top down so no byte is overwritten
		 * before it is read */
		auto *dstI = b.CreatePtrToInt(dst, lenTy);
		auto *srcI = b.CreatePtrToInt(src, lenTy);
		auto *srcEnd = b.CreateAdd(srcI, len);
		backward = b.CreateAnd(b.CreateICmpUGT(dstI, srcI), b.CreateICmpULT(dstI, srcEnd));
	}

	/* Two loops: `words` elements of width w, then the byte remainder. Each is the same
	 * shape: hdr tests i < count, body moves element k and advances. */
	auto *wC = ConstantInt::get(lenTy, w);
	Value *words = (w == 1) ? len : b.CreateUDiv(len, wC, "memintr.words");
	Value *tail = (w == 1) ? ConstantInt::get(lenTy, 0)
			       : b.CreateSub(len, b.CreateMul(words, wC), "memintr.tail");
	/* The pattern of a memset at width w: the byte replicated */
	Value *setVal = nullptr, *setByte = nullptr;
	if (!src) {
		setByte = b.CreateTrunc(cast<MemSetInst>(MI)->getValue(), i8Ty);
		Value *wide = b.CreateZExt(setByte, elemTy);
		setVal = wide;
		for (uint64_t k = 1; k < w; ++k)
			setVal = b.CreateOr(setVal, b.CreateShl(wide, ConstantInt::get(elemTy, 8 * k)));
	}
	BasicBlock *tailHdr = BasicBlock::Create(ctx, "memintr.tail.hdr", F, post);
	BasicBlock *tailBody = BasicBlock::Create(ctx, "memintr.tail.body", F, post);
	/* Everything the loops read from `pre` is computed before its branch. */
	Value *tailOff = (w == 1) ? nullptr : b.CreateMul(words, wC);

	/* LoopUnrollPass (--unroll=N) kills a thread when a loop header is entered for the
	 * (N+1)th time, which a loop of N or more iterations does. A copy that long fails an
	 * assertion here, before its first store. */
	BasicBlock *start = pre;
	if (bound) {
		auto *nC = ConstantInt::get(lenTy, *bound);
		Value *tooLong = b.CreateICmpUGE(words, nC);
		if (w != 1)
			tooLong = b.CreateOr(tooLong, b.CreateICmpUGE(tail, nC));
		BasicBlock *fail = BasicBlock::Create(ctx, "memintr.too.long", F, post);
		start = BasicBlock::Create(ctx, "memintr.start", F, post);
		b.CreateCondBr(tooLong, fail, start);

		IRBuilder<> fb(fail);
		fb.SetCurrentDebugLocation(MI->getDebugLoc());
		auto *ptrTy = PointerType::getUnqual(ctx);
		auto assertFail = F->getParent()->getOrInsertFunction(
			"__VERIFIER_assert_fail",
			FunctionType::get(Type::getVoidTy(ctx),
					  {ptrTy, ptrTy, Type::getInt32Ty(ctx)}, false));
		/* Named: the execution engine maps globals to memory by name, and two unnamed
		 * globals share one slot. */
		auto *msg = fb.CreateGlobalString(
			"a memory intrinsic lowered to a loop is longer than the --unroll bound",
			"memintr.too.long.msg");
		fb.CreateCall(assertFail, {msg, msg, ConstantInt::get(Type::getInt32Ty(ctx), 0)});
		fb.CreateUnreachable();
		b.SetInsertPoint(start);
	}

	auto emitLoop = [&](BasicBlock *h, BasicBlock *bd, BasicBlock *from, BasicBlock *to, Value *count,
			    Type *ty, Value *val, Value *byteOffset, bool down) {
		IRBuilder<> lb(h);
		auto *i = lb.CreatePHI(lenTy, 2, "memintr.i");
		i->addIncoming(ConstantInt::get(lenTy, 0), from);
		lb.CreateCondBr(lb.CreateICmpULT(i, count), bd, to);
		lb.SetInsertPoint(bd);
		Value *k = i;
		if (down)
			k = lb.CreateSub(lb.CreateSub(count, ConstantInt::get(lenTy, 1)), i);
		/* byte address = base + byteOffset + k * width, indexed in units of the type */
		Value *dstBase = byteOffset ? lb.CreateInBoundsGEP(i8Ty, dst, {byteOffset}) : dst;
		auto *dstElem = lb.CreateInBoundsGEP(ty, dstBase, {k}, "memintr.dst");
		Value *v = val;
		if (src) {
			Value *srcBase = byteOffset ? lb.CreateInBoundsGEP(i8Ty, src, {byteOffset}) : src;
			auto *srcElem = lb.CreateInBoundsGEP(ty, srcBase, {k}, "memintr.src");
			v = lb.CreateLoad(ty, srcElem);
		}
		lb.CreateStore(v, dstElem);
		i->addIncoming(lb.CreateAdd(i, ConstantInt::get(lenTy, 1)), bd);
		lb.CreateBr(h);
	};
	/* Ascending copy: the words from the bottom, then the tail bytes at offset words*w. */
	if (backward) {
		BasicBlock *downTailHdr = BasicBlock::Create(ctx, "memintr.down.tail.hdr", F, post);
		BasicBlock *downTailBody = BasicBlock::Create(ctx, "memintr.down.tail.body", F, post);
		BasicBlock *downHdr = BasicBlock::Create(ctx, "memintr.down.hdr", F, post);
		BasicBlock *downBody = BasicBlock::Create(ctx, "memintr.down.body", F, post);
		b.CreateCondBr(backward, downTailHdr, hdr);
		/* Descending copy, for a destination inside the source: the tail bytes first, then
		 * the words, each from the top. The word loop's stores cover the tail's source
		 * bytes, so the tail is read before the words are written. */
		emitLoop(downTailHdr, downTailBody, start, downHdr, tail, i8Ty, setByte, tailOff, true);
		emitLoop(downHdr, downBody, downTailHdr, post, words, elemTy, setVal, nullptr, true);
	} else {
		b.CreateBr(hdr);
	}
	emitLoop(hdr, body, start, tailHdr, words, elemTy, setVal, nullptr, false);
	if (w == 1) {
		/* No tail: the word loop was the whole copy */
		IRBuilder<> tb(tailHdr);
		tb.CreateBr(post);
		IRBuilder<> tbb(tailBody);
		tbb.CreateUnreachable();
	} else {
		emitLoop(tailHdr, tailBody, hdr, post, tail, i8Ty, setByte, tailOff, false);
	}

	promoted.push_back(MI);
	return true;
}

auto PromoteMemIntrinsicPass::run(Function &F, FunctionAnalysisManager & /*FAM*/)
	-> PreservedAnalyses
{
	/* Locate mem intrinsics of interest */
	SmallVector<llvm::MemIntrinsic *, 8> promoted;
	auto modified = false;

	modified |= lowerFortifiedCalls(F);
	/* Gathered first: the runtime-length promotion splits blocks, which an instruction
	 * iterator over the function must not see mid-walk. */
	SmallVector<MemIntrinsic *, 8> found;
	for (auto &I : instructions(F))
		if (auto *MI = dyn_cast<MemIntrinsic>(&I))
			found.push_back(MI);
	const auto bound = (unroll_ && !noUnroll_.count(F.getName().str()))
				   ? unroll_
				   : std::optional<unsigned>();
	for (auto *MI : found) {
		/* The loop form covers what the typed promotion cannot: a runtime length, a memmove,
		 * a memcpy whose operands are both opaque, a memset whose destination is opaque and
		 * whose value or length is not a constant. Left alone, each reaches the
		 * interpreter's handler, which is an unconditional error. */
		auto *MC = dyn_cast<MemCpyInst>(MI);
		auto *MS = dyn_cast<MemSetInst>(MI);
		const bool constLen = isa<ConstantInt>(MI->getLength());
		const bool opaqueCpy = MC && !isPromotableMemIntrinsicOperand(MC->getDest()) &&
				       !isPromotableMemIntrinsicOperand(MC->getSource());
		const bool opaqueSet = MS && !isPromotableMemIntrinsicOperand(MS->getDest()) &&
				       !(constLen && isa<ConstantInt>(MS->getValue()));
		if (isa<MemMoveInst>(MI) || !constLen || opaqueCpy || opaqueSet) {
			if (!lowerOpaque_)
				continue; /* the second instance, after SROA and mem2reg */
			if (MC && MC->getSourceAddressSpace() != MC->getDestAddressSpace())
				continue;
			modified |= promoteRuntimeLength(MI, promoted, bound);
			continue;
		}
		if (auto *MC = dyn_cast<MemCpyInst>(MI))
			modified |= tryPromoteMemCpy(MC, promoted);
		if (auto *MS = dyn_cast<MemSetInst>(MI))
			modified |= tryPromoteMemSet(MS, promoted);
	}

	/* Erase promoted intrinsics from the code */
	removePromoted(promoted);
	if (std::getenv("HG_GENMC_VERIFY_IR") && llvm::verifyFunction(F, &llvm::errs())) {
		llvm::errs() << "PromoteMemIntrinsicPass: malformed IR in " << F.getName() << "\n";
		std::error_code ec;
		llvm::raw_fd_ostream os("/tmp/malformed_fn.ll", ec);
		F.print(os);
	}
	return modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
