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

#include "FunctionInlinerPass.hpp"
#include "passes/InternalFunctions.hpp"
#include <llvm/ADT/SCCIterator.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace llvm;

/**
 * Do not inline any functions that are (mutually) recursive.
 *
 * Example transforms that are permitted (below only f1 will be inlined):
 * f->f1->f2->f3->f4->f2->f3->f4... => f->f2->f3->f4->f2->f3->f4
 * f->f1->f2->f2->f2->... => main->f2->f2->f2...
 */
static auto inlineCall(CallBase *cb) -> bool
{
	llvm::InlineFunctionInfo ifi;
#if LLVM_VERSION_MAJOR >= 11
	return InlineFunction(*cb, ifi).isSuccess();
#else
	return InlineFunction(*cb, ifi);
#endif
}

/* Inline every call to `toInline`. The call sites come from the function's own use list, which
 * LLVM maintains, rather than from a scan of every instruction in the module: on a module of a
 * few million instructions the scan was the whole cost of this pass, once per function. */
static auto inlineFunction(Function *toInline) -> bool
{
	std::vector<CallBase *> calls;
	for (auto *u : toInline->users()) {
		auto *cb = dyn_cast<CallBase>(u);
		if (!cb || cb->getCalledFunction() != toInline)
			continue;
		if (cb->getFunction() == toInline) /* No need to inline calls to itself */
			continue;
		calls.push_back(cb);
	}
	auto changed = false;
	for (auto *ci : calls)
		changed |= inlineCall(ci);
	return changed;
}

auto FunctionInlinerPass::run(Module &M, ModuleAnalysisManager &AM) -> PreservedAnalyses
{
	CallGraph CG(M);

	/* The SCC iteration is post-order -- a function's callees come before it -- so by the
	 * time a function is inlined into its callers its own inlinable calls are already gone,
	 * and nothing is inlined twice. One walk of the SCC forest decides recursion for every
	 * function at once; deciding it per function re-walked the forest that many times. */
	auto changed = false;
	for (auto sccIt = scc_begin(&CG); !sccIt.isAtEnd(); ++sccIt) {
		if (sccIt.hasCycle())
			continue; /* (mutually) recursive: never inlined */
		for (auto *node : *sccIt) {
			auto *F = node->getFunction();
			/* Skip functions with empty bodies, external declarations and GenMC's own */
			if (!F || F->isDeclaration() || isInternalFunction(F->getName().str()))
				continue;
			changed |= inlineFunction(F);
		}
	}
	return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
