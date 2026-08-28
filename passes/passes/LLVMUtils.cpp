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

#include "LLVMUtils.hpp"
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/ValueHandle.h>

#include <unordered_set>

using namespace llvm;

bool areSameLoadOrdering(AtomicOrdering o1, AtomicOrdering o2)
{
	return o1 == o2 ||
	       (o1 == AtomicOrdering::Acquire && o2 == AtomicOrdering::AcquireRelease) ||
	       (o1 == AtomicOrdering::AcquireRelease && o2 == AtomicOrdering::Acquire) ||
	       (o1 == AtomicOrdering::Monotonic && o2 == AtomicOrdering::Release) ||
	       (o1 == AtomicOrdering::Release && o2 == AtomicOrdering::Monotonic);
}

Value *stripCasts(Value *val)
{
	while (isa<CastInst>(val))
		val = dyn_cast<CastInst>(val)->getOperand(0);
	return val;
}

Value *stripCastsGEPs(Value *val)
{
	while (true) {
		if (auto *ci = dyn_cast<CastInst>(val))
			val = ci->getOperand(0);
		else if (auto *gepi = dyn_cast<GetElementPtrInst>(val))
			val = gepi->getPointerOperand();
		else
			break;
	}
	return val;
}

Value *getNonConstantOp(const Instruction *i)
{
	if (isa<Constant>(i->getOperand(1)))
		return i->getOperand(0);
	if (isa<Constant>(i->getOperand(0)))
		return i->getOperand(1);
	return nullptr;
}

/*
 * If V is a binop/cmpop, and one of the operators of V is a constant,
 * returns the other operator of V.
 * If both operators of V are non-const, returns nullptr.
 */
Value *getNonConstOpFromBinopOrCmp(const Value *v)
{
	if (auto *bop = dyn_cast<BinaryOperator>(v)) {
		return getNonConstantOp(bop);
	} else if (auto *cop = dyn_cast<CmpInst>(v)) {
		return getNonConstantOp(cop);
	}
	return nullptr;
}

Value *stripCastsConstOps(Value *val)
{
	while (true) {
		if (auto *ci = dyn_cast<CastInst>(val)) {
			val = ci->getOperand(0);
		} else if (auto *v = getNonConstOpFromBinopOrCmp(val)) {
			val = v;
		} else {
			break;
		}
	}
	return val;
}

std::string getCalledFunOrStripValName(const CallInst &ci)
{
	if (auto *fun = ci.getCalledFunction())
		return fun->getName().str();
	return CallInstWrapper(const_cast<CallInst *>(&ci))
		.getCalledOperand()
		->stripPointerCasts()
		->getName()
		.str();
}

bool isIntrinsicCallNoSideEffects(const Instruction &i)
{
	auto *ci = dyn_cast<CallInst>(&i);
	if (!ci)
		return false;

	return isCleanInternalFunction(getCalledFunOrStripValName(*ci));
}

AtomicCmpXchgInst *extractsFromCAS(ExtractValueInst *extract)
{
	if (!extract->getType()->isIntegerTy() || extract->getNumIndices() > 1)
		return nullptr;
	return dyn_cast<AtomicCmpXchgInst>(extract->getAggregateOperand());
}

/* i1 depends on i2 iff i2 is reachable from i1 through operand edges. Reachability is a
 * visited-set search: an instruction that has been explored once does not reach i2 through
 * any path, so it is never explored again. Iterative, so a long dependence chain cannot
 * exhaust the stack. */
bool isDependentOn(const Instruction *i1, const Instruction *i2)
{
	if (!i1 || !i2)
		return false;

	llvm::DenseSet<const Instruction *> visited;
	llvm::SmallVector<const Instruction *, 64> work;
	work.push_back(i1);
	while (!work.empty()) {
		const auto *cur = work.pop_back_val();
		if (!visited.insert(cur).second)
			continue;
		for (auto &u : cur->operands()) {
			if (auto *i = dyn_cast<Instruction>(u.get())) {
				if (i == i2)
					return true;
				if (!visited.count(i))
					work.push_back(i);
			}
		}
	}
	return false;
}

bool hasSideEffects(const Instruction *i, const VSet<Function *> *cleanFuns /* = nullptr */)
{
	if (isa<AllocaInst>(i))
		return true;
	if (i->mayHaveSideEffects()) {
		if (auto *ci = dyn_cast<CallInst>(i)) {
			auto name = getCalledFunOrStripValName(*ci);
			if (isInternalFunction(name))
				return !isCleanInternalFunction(name);
			if (!cleanFuns)
				return true;
			CallInstWrapper CW(const_cast<CallInst *>(ci));
			const auto *fun =
				dyn_cast<Function>(CW.getCalledOperand()->stripPointerCasts());
			if (!fun || !cleanFuns->count(const_cast<Function *>(fun)))
				return true;
		} else if (!isa<LoadInst>(i) && !isa<FenceInst>(i)) {
			return true;
		}
	}
	return false;
}

bool isAlloc(const Instruction *i, const VSet<Function *> *allocFuns /* = nullptr */)
{
	auto *si = i->stripPointerCasts();
	if (isa<AllocaInst>(si))
		return true;

	auto *ci = dyn_cast<CallInst>(si);
	if (!ci)
		return false;

	if (isAllocFunction(getCalledFunOrStripValName(*ci)))
		return true;

	CallInstWrapper CW(const_cast<CallInst *>(ci));
	const auto *fun = dyn_cast<Function>(CW.getCalledOperand()->stripPointerCasts());
	return allocFuns && allocFuns->count(const_cast<Function *>(fun));
}

auto hasLoadSemantics(llvm::Instruction *I) -> bool
{
	/* Overapproximate with function calls some of which might be modeled as loads */
	auto *ci = llvm::dyn_cast<llvm::CallInst>(I);
	return llvm::isa<llvm::LoadInst>(I) || llvm::isa<llvm::AtomicCmpXchgInst>(I) ||
	       llvm::isa<llvm::AtomicRMWInst>(I) ||
	       (ci && ci->getCalledFunction() &&
		hasGlobalLoadSemantics(ci->getCalledFunction()->getName().str()));
}

auto getInstKind(llvm::Instruction *I) -> ActionKind
{
	return hasLoadSemantics(I) ? ActionKind::Load : ActionKind::NonLoad;
}

void annotateInstruction(llvm::Instruction *i, const std::string &type, uint64_t value)
{
	auto &ctx = i->getContext();
	auto *md = i->getMetadata(type);

	/* If there are already metadata, accumulate */
	uint64_t mValue = value;
	if (md) {
		auto old = dyn_cast<ConstantInt>(
				   dyn_cast<ConstantAsMetadata>(md->getOperand(0))->getValue())
				   ->getZExtValue();
		mValue |= old;
	}

	auto *node =
		MDNode::get(ctx, ConstantAsMetadata::get(ConstantInt::get(ctx, APInt(64, mValue))));
	i->setMetadata(type, node);
	return;
}

BasicBlock *tryThreadSuccessor(BranchInst *term, BasicBlock *succ)
{
	auto *succTerm = dyn_cast<BranchInst>(succ->getTerminator());
	if (!succTerm || succTerm != &*succ->begin() || succTerm->isConditional())
		return nullptr;

	/* If there are PHIs that depend on SUCC be conservative and
	 * do not transform, as B might jump to DESTBB too */
	auto *destBB = succTerm->getSuccessor(0);
	if (isa<PHINode>(&*destBB->begin()))
		return nullptr;

	for (auto i = 0u; i < term->getNumSuccessors(); i++) {
		if (term->getSuccessor(i) == succ) {
			term->setSuccessor(i, destBB);
			return destBB;
		}
	}
	return nullptr;
}

void replaceUsesWithIf(Value *Old, Value *New, llvm::function_ref<bool(Use &U)> ShouldReplace)
{
	// assert(New && "Value::replaceUsesWithIf(<null>) is invalid!");
	// assert(New->getType() == old->getType() &&
	//        "replaceUses of value with new value of different type!");

	SmallVector<TrackingVH<Constant>, 8> Consts;
	SmallPtrSet<Constant *, 8> Visited;

	for (auto UI = Old->use_begin(), E = Old->use_end(); UI != E;) {
		Use &U = *UI;
		++UI;
		if (!ShouldReplace(U))
			continue;
		// Must handle Constants specially, we cannot call replaceUsesOfWith on a
		// constant because they are uniqued.
		if (auto *C = dyn_cast<Constant>(U.getUser())) {
			if (!isa<GlobalValue>(C)) {
				if (Visited.insert(C).second)
					Consts.push_back(TrackingVH<Constant>(C));
				continue;
			}
		}
		U.set(New);
	}

	while (!Consts.empty()) {
		// FIXME: handleOperandChange() updates all the uses in a given Constant,
		//        not just the one passed to ShouldReplace
		Consts.pop_back_val()->handleOperandChange(Old, New);
	}
}
