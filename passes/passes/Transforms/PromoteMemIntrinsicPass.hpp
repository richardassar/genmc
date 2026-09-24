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

#ifndef GENMC_PROMOTE_MEMINTRINSIC_PASS_HPP
#define GENMC_PROMOTE_MEMINTRINSIC_PASS_HPP

#include "genmc/ADT/VSet.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

#include <llvm/Passes/PassBuilder.h>

using namespace llvm;

class PromoteMemIntrinsicPass : public PassInfoMixin<PromoteMemIntrinsicPass> {
public:
	/* The pass runs twice. The first instance promotes the intrinsics whose operands name a
	 * type and leaves the rest in place; the second, after SROA and mem2reg have folded
	 * closure fields and pointer slots into the allocas they held, promotes what became
	 * typed and lowers whatever is still opaque to a loop at the intrinsic's alignment. A
	 * copy lowered at alignment width before that folding is read back at its fields'
	 * widths, and a read inside a wider write is one the checker cannot resolve. */
	explicit PromoteMemIntrinsicPass(bool lowerOpaque = true) : lowerOpaque_(lowerOpaque) {}
	/* UNROLL and NOUNROLLFUNS are --unroll and --no-unroll: a lowered copy whose loop
	 * reaches the bound fails an assertion before its first store. */
	PromoteMemIntrinsicPass(bool lowerOpaque, std::optional<unsigned> unroll,
				const VSet<std::string> &noUnrollFuns)
		: lowerOpaque_(lowerOpaque), unroll_(unroll), noUnroll_(noUnrollFuns)
	{}
	auto run(Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses;

private:
	bool lowerOpaque_;
	std::optional<unsigned> unroll_;
	VSet<std::string> noUnroll_;
};

#endif /* GENMC_PROMOTE_MEMINTRINSIC_PASS_HPP */
