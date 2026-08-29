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

#include "PromoteMemIntrinsicPass.hpp"
#include "genmc/Support/Error.hpp"
#include <llvm/ADT/Twine.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <ranges>
#include <utility>

/* We have to do ptr->int->ptr cast for older LLVMs (no opaque ptrs)
 * Note: Bitcasts wouldn't work due to casting of differently sized pointees */
#if LLVM_VERSION_MAJOR < 15
#define CAST_PTR_TO_TYPE_THROUGH_INT(builder, ptr, dstTy, dataLayout)                              \
	builder.CreateIntToPtr(                                                                    \
		builder.CreatePtrToInt(                                                            \
			ptr, dataLayout.getIntPtrType(                                             \
				     ptr->getContext()) /* Int type of same size as a ptr */),     \
		dstTy)
#else
#define CAST_PTR_TO_TYPE_THROUGH_INT(builder, ptr, dstTy, dataLayout) ptr
#endif

/* Before LLVM-15, Constant Expression GEP's are implicitly casted to *i8 */
#if LLVM_VERSION_MAJOR < 15
#define CONSTEXPR_GET_FIELD_TYPE(MI, Field) Type::getInt8PtrTy(MI->getContext())
#else
#define CONSTEXPR_GET_FIELD_TYPE(MI, Field) MI->get##Field()->getType()
#endif

/* Helper macro that moves constant expressions into their own instructions */
#define LOWER_CONSTEXPR(builder, MI, Field)                                                        \
	if (auto *CE = dyn_cast<ConstantExpr>(MI->get##Field())) {                                 \
		Value *newGEP = builder.Insert(CE->getAsInstruction());                            \
		Type *destType = CONSTEXPR_GET_FIELD_TYPE(MI, Field);                              \
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

	/* Split the basic block at the call to __memcpy_chk in BB (before call-instr.) and contBB
	 * (rest of the block) */
	auto *BB = CI->getParent();
	auto *contBB =
		BB->splitBasicBlock(CI); /* Creates a dummy terminator for BB -> remove later */

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
	auto *dummyTerminator = BB->getTerminator();
	IRBuilder<> termBuilder(dummyTerminator);
	auto *cmp = termBuilder.CreateICmpULT(dst_len, len);

	/* Replace the dummy terminator: We jump to failBB if the condition holds */
	dummyTerminator->eraseFromParent();
	termBuilder.SetInsertPoint(BB);
	auto *cnd_br = termBuilder.CreateCondBr(cmp, /* True */ failBB, /* False */ contBB);
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

static auto isPromotableMemIntrinsicOperand(Value *op) -> bool
{
	/* Constant to capture MemSet too */
	return isa<Constant>(op) || isa<AllocaInst>(op) || isa<GetElementPtrInst>(op);
}

static auto getPromotionGEPType(Value *op) -> Type *
{
	VERIFY(isPromotableMemIntrinsicOperand(op));
	if (auto *v = dyn_cast<GlobalVariable>(op))
		return v->getValueType();
	if (auto *ai = dyn_cast<AllocaInst>(op))
		return ai->getAllocatedType();
	if (auto *gepi = dyn_cast<GetElementPtrInst>(op)) {
		/* A GEP whose indices are all zero names the object it indexes into, not a
		 * member of it: `getelementptr [2 x i32], ptr %a, i64 0, i64 0` is the array,
		 * and a memcpy of the whole array through it must be promoted field by field
		 * over the array's type. Its result element type (i32 here) is only the first
		 * element, and promoting by that either trips typeSizeDst >= len or copies one
		 * element of several. */
		if (gepi->hasAllZeroIndices())
			return gepi->getSourceElementType();
		return gepi->getResultElementType();
	}
	UNREACHABLE();
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
	auto *dstStore = builder.CreateStore(srcLoad, dstGEP);
}

static void promoteMemSet(IRBuilder<> &builder, Value *dst, Value *argVal,
			  const std::vector<Value *> &args, Type *typ, Type *gepTy)
{
	VERIFY(typ->isIntegerTy() || typ->isPointerTy());
	VERIFY(isa<ConstantInt>(argVal));

	const auto &DL = builder.GetInsertBlock()->getParent()->getParent()->getDataLayout();
	auto sizeInBits = typ->isIntegerTy() ? typ->getIntegerBitWidth()
					     : DL.getPointerTypeSizeInBits(typ);
	long int ival = dyn_cast<ConstantInt>(argVal)->getSExtValue();
	Value *val = Constant::getIntegerValue(typ, APInt(sizeInBits, ival));

	Value *dstGEP = builder.CreateInBoundsGEP(gepTy, dst, args, "memset.dst.gep");
	Value *dstStore = builder.CreateStore(val, dstGEP);
}

template <typename F>
static void promoteMemIntrinsic(Type *typ, std::vector<Value *> &args, F &&promoteFun)
{
	auto *i32Ty = IntegerType::getInt32Ty(typ->getContext());

	if (!isa<StructType>(typ) && !isa<ArrayType>(typ) && !isa<VectorType>(typ)) {
		promoteFun(typ, args);
		return;
	}

	if (auto *AT = dyn_cast<ArrayType>(typ)) {
#ifdef LLVM_HAS_GLOBALOBJECT_GET_METADATA
		auto n = AT->getNumElements();
#else
		auto n = AT->getArrayNumElements();
#endif
		for (auto i = 0U; i < n; i++) {
			args.push_back(Constant::getIntegerValue(i32Ty, APInt(32, i)));
			promoteMemIntrinsic(AT->getElementType(), args, promoteFun);
			args.pop_back();
		}
	} else if (auto *ST = dyn_cast<StructType>(typ)) {
		auto i = 0U;
		for (auto it = ST->element_begin(); i < ST->getNumElements(); ++it, ++i) {
			args.push_back(Constant::getIntegerValue(i32Ty, APInt(32, i)));
			promoteMemIntrinsic(*it, args, promoteFun);
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
	if (MCI && !isPromotableMemIntrinsicOperand(MCI->getDest()) &&
	    !isPromotableMemIntrinsicOperand(MCI->getSource())) {
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
	std::vector<Value *> gepArgs = {Constant::getIntegerValue(
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
	auto len = dyn_cast<ConstantInt>(MI->getLength())->getZExtValue();

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
	LOWER_CONSTEXPR(builder, MI, Source);
	LOWER_CONSTEXPR(builder, MI, Dest);

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
static auto promoteRuntimeLength(MemIntrinsic *MI, SmallVector<MemIntrinsic *, 8> &promoted) -> bool
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
	b.CreateBr(hdr);

	b.SetInsertPoint(hdr);
	auto *i = b.CreatePHI(lenTy, 2, "memintr.i");
	i->addIncoming(ConstantInt::get(lenTy, 0), pre);
	b.CreateCondBr(b.CreateICmpULT(i, len), body, post);

	b.SetInsertPoint(body);
	Value *k = i;
	if (backward) {
		auto *fromTop = b.CreateSub(b.CreateSub(len, ConstantInt::get(lenTy, 1)), i);
		k = b.CreateSelect(backward, fromTop, i);
	}
	auto *dstElem = b.CreateInBoundsGEP(i8Ty, dst, {k}, "memintr.dst");
	Value *byte = nullptr;
	if (src) {
		auto *srcElem = b.CreateInBoundsGEP(i8Ty, src, {k}, "memintr.src");
		byte = b.CreateLoad(i8Ty, srcElem);
	} else {
		byte = b.CreateTrunc(cast<MemSetInst>(MI)->getValue(), i8Ty);
	}
	b.CreateStore(byte, dstElem);
	auto *next = b.CreateAdd(i, ConstantInt::get(lenTy, 1));
	i->addIncoming(next, body);
	b.CreateBr(hdr);

	promoted.push_back(MI);
	return true;
}

auto PromoteMemIntrinsicPass::run(Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses
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
	for (auto *MI : found) {
		if (isa<MemMoveInst>(MI) || !isa<ConstantInt>(MI->getLength())) {
			if (auto *MC = dyn_cast<MemCpyInst>(MI); MC && MC->getSourceAddressSpace() != MC->getDestAddressSpace())
				continue;
			modified |= promoteRuntimeLength(MI, promoted);
			continue;
		}
		if (auto *MC = dyn_cast<MemCpyInst>(MI))
			modified |= tryPromoteMemCpy(MC, promoted);
		if (auto *MS = dyn_cast<MemSetInst>(MI))
			modified |= tryPromoteMemSet(MS, promoted);
	}

	/* Erase promoted intrinsics from the code */
	removePromoted(promoted);
	return modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
