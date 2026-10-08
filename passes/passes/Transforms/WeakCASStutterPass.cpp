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

#include "WeakCASStutterPass.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>

using namespace llvm;

namespace {

/* Regions larger than this are not a retry loop's failure path. */
constexpr unsigned kMaxRegionBlocks = 32;

class StutterCheck {
public:
	StutterCheck(AtomicCmpXchgInst *cas, const DataLayout &dl) : cas_(cas), dl_(dl) {}

	auto run() -> bool
	{
		casBB_ = cas_->getParent();
		auto *br = dyn_cast<BranchInst>(casBB_->getTerminator());
		if (!br || !br->isConditional())
			return false;

		/* The branch is on the success flag, possibly negated. */
		auto *cond = br->getCondition();
		auto negated = false;
		if (auto *bo = dyn_cast<BinaryOperator>(cond)) {
			if (bo->getOpcode() == Instruction::Xor && isa<ConstantInt>(bo->getOperand(1)) &&
			    cast<ConstantInt>(bo->getOperand(1))->isOne()) {
				cond = bo->getOperand(0);
				negated = true;
			}
		}
		if (!isFlag(cond))
			return false;
		auto *succBB = br->getSuccessor(negated ? 1 : 0);
		auto *failBB = br->getSuccessor(negated ? 0 : 1);

		/* The failure path: every block reachable from the failure edge before the CAS
		 * block is reached again. It must close back onto the CAS block. */
		region_.insert(casBB_);
		SmallVector<BasicBlock *, 16> work;
		if (failBB != casBB_) {
			region_.insert(failBB);
			work.push_back(failBB);
		}
		auto reachesCas = (failBB == casBB_);
		while (!work.empty()) {
			auto *bb = work.pop_back_val();
			auto *term = bb->getTerminator();
			if (!isa<BranchInst>(term) && !isa<SwitchInst>(term))
				return false;
			for (auto *s : successors(bb)) {
				if (s == casBB_) {
					reachesCas = true;
					continue;
				}
				if (region_.insert(s).second) {
					if (region_.size() > kMaxRegionBlocks)
						return false;
					work.push_back(s);
				}
			}
		}
		if (!reachesCas || region_.contains(succBB))
			return false;

		/* The CAS is the same operation on the next arrival. */
		if (!invariant(cas_->getPointerOperand()) || !invariant(cas_->getCompareOperand()) ||
		    !invariant(cas_->getNewValOperand()))
			return false;

		for (auto *bb : region_) {
			auto beforeCas = (bb == casBB_);
			for (auto &i : *bb) {
				if (&i == cas_) {
					beforeCas = false;
					continue;
				}
				if (auto *phi = dyn_cast<PHINode>(&i)) {
					if (!phiKeepsValue(phi))
						return false;
					continue;
				}
				if (i.isTerminator() || isa<DbgInfoIntrinsic>(&i))
					continue;
				if (auto *si = dyn_cast<StoreInst>(&i)) {
					/* A store before the CAS in the CAS block runs on every
					 * arrival, the first included, so the stutter's copy repeats
					 * one that is kept. */
					if (!beforeCas || si->isVolatile() ||
					    !invariant(si->getPointerOperand()) ||
					    !invariant(si->getValueOperand()))
						return false;
					continue;
				}
				if (i.mayReadOrWriteMemory() || i.mayHaveSideEffects())
					return false;
			}
		}
		return true;
	}

private:
	auto isFlag(Value *v) const -> bool
	{
		auto *ev = dyn_cast<ExtractValueInst>(v);
		return ev && ev->getAggregateOperand() == cas_ && ev->getNumIndices() == 1 &&
		       ev->getIndices()[0] == 1;
	}

	auto isLoaded(Value *v) const -> bool
	{
		auto *ev = dyn_cast<ExtractValueInst>(v);
		return ev && ev->getAggregateOperand() == cas_ && ev->getNumIndices() == 1 &&
		       ev->getIndices()[0] == 0;
	}

	/* Bit-for-bit conversions only: the value is the same bits on both sides. */
	auto stripCasts(Value *v) const -> Value *
	{
		while (auto *c = dyn_cast<CastInst>(v)) {
			if (!isa<PtrToIntInst>(c) && !isa<IntToPtrInst>(c) && !isa<BitCastInst>(c))
				break;
			if (dl_.getTypeSizeInBits(c->getSrcTy()) != dl_.getTypeSizeInBits(c->getDestTy()))
				break;
			v = c->getOperand(0);
		}
		return v;
	}

	auto inRegion(Value *v) const -> bool
	{
		auto *i = dyn_cast<Instruction>(v);
		return i && region_.contains(i->getParent());
	}

	/* Under a spurious failure the CAS returns its compare operand, so a value carried
	 * back to `phi` keeps the phi's value when it is the phi itself or the CAS's loaded
	 * value with the phi as the compare operand. */
	auto equalUnderStutter(Value *v, PHINode *phi) const -> bool
	{
		if (v == phi)
			return true;
		return isLoaded(stripCasts(v)) && stripCasts(cas_->getCompareOperand()) == phi;
	}

	auto phiKeepsValue(PHINode *phi) const -> bool
	{
		for (unsigned k = 0; k < phi->getNumIncomingValues(); ++k)
			if (region_.contains(phi->getIncomingBlock(k)) &&
			    !equalUnderStutter(phi->getIncomingValue(k), phi))
				return false;
		return true;
	}

	/* The value on the next arrival at the CAS equals the value on this one. Phis are
	 * checked separately (phiKeepsValue), so a phi of the region counts as invariant
	 * here and the whole CAS is rejected if one does not keep its value. */
	auto invariant(Value *v) -> bool
	{
		if (!inRegion(v) || isa<PHINode>(v))
			return true;
		if (v == cas_ || isFlag(v))
			return false;
		if (isLoaded(v))
			return invariant(cas_->getCompareOperand());
		auto *i = cast<Instruction>(v);
		if (i->mayReadOrWriteMemory() || i->mayHaveSideEffects())
			return false;
		auto [it, fresh] = memo_.try_emplace(i, true);
		if (!fresh)
			return it->second;
		auto ok = true;
		for (auto &op : i->operands())
			ok = ok && invariant(op.get());
		memo_[i] = ok;
		return ok;
	}

	AtomicCmpXchgInst *cas_;
	const DataLayout &dl_;
	BasicBlock *casBB_ = nullptr;
	SmallPtrSet<BasicBlock *, 16> region_;
	DenseMap<Instruction *, bool> memo_;
};

} // namespace

auto WeakCASStutterPass::run(Function &F, FunctionAnalysisManager & /*FAM*/) -> PreservedAnalyses
{
	const auto &dl = F.getParent()->getDataLayout();
	const auto report = std::getenv("HG_GENMC_STUTTER_REPORT") != nullptr;
	auto modified = false;

	for (auto &inst : instructions(F)) {
		auto *casi = dyn_cast<AtomicCmpXchgInst>(&inst);
		if (!casi || !casi->isWeak())
			continue;
		const auto stutter = StutterCheck(casi, dl).run();
		if (report) {
			errs() << "HG-STUTTER " << (stutter ? "strong " : "weak   ") << F.getName();
			if (const auto &loc = casi->getDebugLoc())
				errs() << " " << loc->getFilename() << ":" << loc.getLine();
			errs() << "\n";
		}
		if (stutter) {
			casi->setWeak(false);
			modified = true;
		}
	}
	return modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
