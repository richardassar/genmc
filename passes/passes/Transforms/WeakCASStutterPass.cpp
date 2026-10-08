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
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <string>

using namespace llvm;

namespace {

/* Regions larger than this are not a retry loop's failure path. */
constexpr unsigned kMaxRegionBlocks = 32;

/* Applies the two tests described in WeakCASStutterPass.hpp to one weak CAS. */
class StutterCheck {
public:
	StutterCheck(AtomicCmpXchgInst *cas, const DataLayout &dl, const LoopInfo &li,
		     const DominatorTree &dt)
		: cas_(cas), dl_(dl), li_(li), dt_(dt)
	{}

	auto run() -> bool
	{
		casBB_ = cas_->getParent();
		auto *loop = li_.getLoopFor(casBB_);
		if (!loop)
			return fail("the CAS is not in a loop");
		/* The decision on the flag can sit a few blocks after the CAS, behind unconditional
		 * branches or a conditional one whose targets coincide. Those blocks run on
		 * success and on failure alike. */
		decBB_ = casBB_;
		for (auto k = 0; k < 4; ++k) {
			auto *t = dyn_cast<BranchInst>(decBB_->getTerminator());
			if (!t || (t->isConditional() && t->getSuccessor(0) != t->getSuccessor(1)))
				break;
			auto *next = t->getSuccessor(0);
			if (next->getUniquePredecessor() != decBB_ || !loop->contains(next))
				break;
			chain_.insert(next);
			decBB_ = next;
		}
		auto *br = dyn_cast<BranchInst>(decBB_->getTerminator());
		if (!br || !br->isConditional())
			return fail("no conditional branch follows the CAS");

		/* The branch is on the success flag, possibly negated. */
		auto *cond = br->getCondition();
		auto negated = false;
		if (auto *flag = negatedFlag(cond)) {
			cond = flag;
			negated = true;
		}
		if (!isFlag(cond))
			return fail("the CAS block's branch is not on the success flag");
		auto *succBB = br->getSuccessor(negated ? 1 : 0);
		auto *failBB = br->getSuccessor(negated ? 0 : 1);
		if (loop->contains(succBB))
			return fail("the success edge stays in the loop");
		if (!loop->contains(failBB))
			return fail("the failure edge leaves the loop");

		loop_ = loop;
		failBB_ = failBB;
		if (dropIteration())
			return true;
		const std::string dropWhy = reason_;
		region_.clear();
		memo_.clear();
		if (repeatAttempt())
			return true;
		why_ = "drop: " + dropWhy + "; repeat: " + reason_;
		reason_ = why_.c_str();
		return false;
	}

	[[nodiscard]] auto reason() const -> const char * { return reason_; }

private:
	/*
	 * The iteration that ends in the spurious failure, from the loop header to the header
	 * again, contains reads and fences only (the CAS's failed read among them), and every
	 * header phi has the same value after it as before. Deleting the iteration's events from
	 * the execution leaves a consistent execution -- deleting reads and fences only removes
	 * constraints -- in which the thread starts the next iteration in the same state and
	 * every remaining read reads the same write. That execution has no spurious failure at
	 * this point and is explored.
	 */
	auto dropIteration() -> bool
	{
		auto *header = loop_->getHeader();

		/* The failure path back to the header must stay in the loop: an exit taken after a
		 * spurious failure is behaviour of its own. */
		SmallPtrSet<BasicBlock *, 16> post;
		SmallVector<BasicBlock *, 16> work;
		if (failBB_ != header) {
			post.insert(failBB_);
			work.push_back(failBB_);
		}
		while (!work.empty()) {
			auto *bb = work.pop_back_val();
			for (auto *s : successors(bb)) {
				if (s == header)
					continue;
				if (!loop_->contains(s))
					return fail("the failure path can leave the loop");
				if (s == casBB_)
					return fail("the failure path reaches the CAS without the header");
				if (post.insert(s).second) {
					if (post.size() > kMaxRegionBlocks)
						return fail("the failure path is larger than a retry loop");
					work.push_back(s);
				}
			}
		}
		post.insert(chain_.begin(), chain_.end());

		/* The blocks an iteration passes on its way from the header to the CAS. */
		SmallPtrSet<BasicBlock *, 16> fromHeader;
		work.push_back(header);
		fromHeader.insert(header);
		while (!work.empty()) {
			auto *bb = work.pop_back_val();
			if (bb == casBB_)
				continue;
			for (auto *s : successors(bb))
				if (s != header && loop_->contains(s) && fromHeader.insert(s).second)
					work.push_back(s);
		}
		SmallPtrSet<BasicBlock *, 16> toCas;
		work.push_back(casBB_);
		toCas.insert(casBB_);
		while (!work.empty()) {
			auto *bb = work.pop_back_val();
			if (bb == header)
				continue;
			for (auto *p : predecessors(bb))
				if (loop_->contains(p) && toCas.insert(p).second)
					work.push_back(p);
		}

		region_.insert(post.begin(), post.end());
		for (auto *bb : fromHeader)
			if (toCas.contains(bb))
				region_.insert(bb);
		if (region_.size() > kMaxRegionBlocks)
			return fail("the iteration is larger than a retry loop");

		for (auto *bb : region_) {
			/* A load before the CAS is also in the execution where the CAS succeeds,
			 * which is explored, so an error it meets is reported there. A load after
			 * the CAS runs only on a failure, with the spurious failure's value. */
			auto beforeCas = !post.contains(bb);
			for (auto &i : *bb) {
				if (&i == cas_) {
					beforeCas = false;
					continue;
				}
				if (isa<DbgInfoIntrinsic>(&i) || i.isTerminator())
					continue;
				if (auto *phi = dyn_cast<PHINode>(&i)) {
					if (bb == header && !phiKeepsValue(phi))
						return fail("a loop-carried value changes in the iteration");
					continue;
				}
				if (isa<FenceInst>(&i) || (beforeCas && isa<LoadInst>(&i)))
					continue;
				if (i.mayReadOrWriteMemory() || i.mayHaveSideEffects())
					return fail("the iteration writes, calls or has another side "
						    "effect");
			}
		}
		return true;
	}

	/*
	 * The spurious failure is followed by the same attempt from the same state: the path
	 * back to the CAS repeats the previous arrival's, loop-carried values are unchanged and
	 * the only memory accesses besides the CAS are stores of unchanged values to unchanged
	 * addresses that the previous arrival also executed.
	 */
	auto repeatAttempt() -> bool
	{
		auto *loop = loop_;
		auto *failBB = failBB_;
		region_.insert(casBB_);
		region_.insert(chain_.begin(), chain_.end());
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
				return fail("the failure path ends in something other than a branch");
			for (auto *s : successors(bb)) {
				if (s == casBB_) {
					reachesCas = true;
					continue;
				}
				if (!loop->contains(s)) {
					if (!dt_.dominates(bb, casBB_))
						return fail("the failure path can leave the loop");
					continue;
				}
				if (region_.insert(s).second) {
					if (region_.size() > kMaxRegionBlocks)
						return fail("the failure path is larger than a retry loop");
					work.push_back(s);
				}
			}
		}
		if (!reachesCas)
			return fail("the failure path does not return to the CAS");

		/* The CAS is the same operation on the next arrival. */
		if (!invariant(cas_->getPointerOperand()) || !invariant(cas_->getCompareOperand()) ||
		    !invariant(cas_->getNewValOperand()))
			return fail("the CAS's operands change on the failure path");

		for (auto *bb : region_) {
			auto ranBefore = (bb == casBB_) || dt_.dominates(bb, casBB_);
			for (auto &i : *bb) {
				if (&i == cas_) {
					ranBefore = false;
					continue;
				}
				if (auto *phi = dyn_cast<PHINode>(&i)) {
					if (!phiKeepsValue(phi))
						return fail("a loop-carried value changes on the failure "
							    "path");
					continue;
				}
				if (isa<DbgInfoIntrinsic>(&i))
					continue;
				if (i.isTerminator()) {
					if (bb == casBB_ || chain_.contains(bb))
						continue;
					Value *c = nullptr;
					if (auto *b = dyn_cast<BranchInst>(&i); b && b->isConditional())
						c = b->getCondition();
					else if (auto *sw = dyn_cast<SwitchInst>(&i))
						c = sw->getCondition();
					if (c && !invariant(c))
						return fail("a branch on the failure path depends on a "
							    "changed value");
					continue;
				}
				if (auto *si = dyn_cast<StoreInst>(&i)) {
					/* The previous arrival executed this store with the same
					 * operands, so the stutter's copy repeats one that is kept. */
					if (!ranBefore || si->isVolatile() ||
					    !invariant(si->getPointerOperand()) ||
					    !invariant(si->getValueOperand()))
						return fail("a store on the failure path is not a repeat");
					continue;
				}
				if (i.mayReadOrWriteMemory() || i.mayHaveSideEffects())
					return fail("the failure path reads memory, calls or has "
						    "another side effect");
			}
		}
		return true;
	}

	auto fail(const char *why) -> bool
	{
		reason_ = why;
		return false;
	}

	/* The success flag, possibly widened and narrowed again (the inlined
	 * compare_exchange_weak passes it through an i8). */
	auto isFlag(Value *v) const -> bool
	{
		while (true) {
			if (auto *c = dyn_cast<ZExtInst>(v))
				v = c->getOperand(0);
			else if (auto *t = dyn_cast<TruncInst>(v); t && t->getType()->isIntegerTy(1))
				v = t->getOperand(0);
			else
				break;
		}
		auto *ev = dyn_cast<ExtractValueInst>(v);
		return ev && ev->getAggregateOperand() == cas_ && ev->getNumIndices() == 1 &&
		       ev->getIndices()[0] == 1;
	}

	/* The flag when `v` is `xor flag, true`. */
	auto negatedFlag(Value *v) const -> Value *
	{
		auto *bo = dyn_cast<BinaryOperator>(v);
		if (!bo || bo->getOpcode() != Instruction::Xor)
			return nullptr;
		auto *c = dyn_cast<ConstantInt>(bo->getOperand(1));
		if (!c || !c->isOne() || !isFlag(bo->getOperand(0)))
			return nullptr;
		return bo->getOperand(0);
	}

	auto isLoaded(Value *v) const -> bool
	{
		auto *ev = dyn_cast<ExtractValueInst>(v);
		return ev && ev->getAggregateOperand() == cas_ && ev->getNumIndices() == 1 &&
		       ev->getIndices()[0] == 0;
	}

	/* The arm a select on the flag takes when the CAS fails; null if `v` is not one. */
	auto failureArm(Value *v) const -> Value *
	{
		auto *sel = dyn_cast<SelectInst>(v);
		if (!sel)
			return nullptr;
		if (isFlag(sel->getCondition()))
			return sel->getFalseValue();
		if (negatedFlag(sel->getCondition()))
			return sel->getTrueValue();
		return nullptr;
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
	 * back to `phi` keeps the phi's value when it is the phi itself, or the CAS's loaded
	 * value with the phi as the compare operand, or a select on the flag whose failure
	 * arm is one of those. */
	auto equalUnderStutter(Value *v, PHINode *phi) const -> bool
	{
		v = stripCasts(v);
		if (auto *arm = failureArm(v))
			v = stripCasts(arm);
		if (v == phi)
			return true;
		return isLoaded(v) && stripCasts(cas_->getCompareOperand()) == phi;
	}

	auto phiKeepsValue(PHINode *phi) const -> bool
	{
		for (unsigned k = 0; k < phi->getNumIncomingValues(); ++k)
			if (region_.contains(phi->getIncomingBlock(k)) &&
			    !equalUnderStutter(phi->getIncomingValue(k), phi))
				return false;
		return true;
	}

	/* The value on the next arrival at the CAS equals the value on the previous one. Phis
	 * are checked separately (phiKeepsValue) and the whole CAS is rejected if one does not
	 * keep its value, so a phi of the region counts as invariant here. */
	auto invariant(Value *v) -> bool
	{
		if (!inRegion(v) || isa<PHINode>(v))
			return true;
		if (auto *arm = failureArm(v))
			return invariant(arm);
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
	const LoopInfo &li_;
	const DominatorTree &dt_;
	BasicBlock *casBB_ = nullptr;
	SmallPtrSet<BasicBlock *, 16> region_;
	DenseMap<Instruction *, bool> memo_;
	const char *reason_ = "";
	std::string why_;
	Loop *loop_ = nullptr;
	BasicBlock *failBB_ = nullptr;
	BasicBlock *decBB_ = nullptr;
	SmallPtrSet<BasicBlock *, 4> chain_;
};

} // namespace

auto WeakCASStutterPass::run(Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses
{
	const auto report = std::getenv("HG_GENMC_STUTTER_REPORT") != nullptr;
	SmallVector<AtomicCmpXchgInst *, 8> weak;
	for (auto &inst : instructions(F))
		if (auto *casi = dyn_cast<AtomicCmpXchgInst>(&inst); casi && casi->isWeak())
			weak.push_back(casi);
	if (weak.empty())
		return PreservedAnalyses::all();

	const auto &dl = F.getParent()->getDataLayout();
	const auto &li = FAM.getResult<LoopAnalysis>(F);
	const auto &dt = FAM.getResult<DominatorTreeAnalysis>(F);
	auto modified = false;
	for (auto *casi : weak) {
		StutterCheck check(casi, dl, li, dt);
		const auto stutter = check.run();
		if (report) {
			errs() << "HG-STUTTER " << (stutter ? "strong " : "weak   ") << F.getName();
			if (const auto &loc = casi->getDebugLoc())
				errs() << " " << loc->getFilename() << ":" << loc.getLine();
			if (!stutter)
				errs() << " -- " << check.reason();
			errs() << "\n";
			if (!stutter && std::getenv("HG_GENMC_STUTTER_REPORT")[0] == '2')
				casi->getParent()->getParent()->print(errs());
		}
		if (stutter) {
			casi->setWeak(false);
			modified = true;
		}
	}
	if (!modified)
		return PreservedAnalyses::all();
	PreservedAnalyses pa;
	pa.preserveSet<CFGAnalyses>();
	return pa;
}
