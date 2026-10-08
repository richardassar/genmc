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

#ifndef GENMC_WEAK_CAS_STUTTER_PASS_HPP
#define GENMC_WEAK_CAS_STUTTER_PASS_HPP

#include <llvm/Passes/PassBuilder.h>

using namespace llvm;

/**
 * Makes a weak CAS strong when its spurious failure adds no behaviour. The checker models a
 * spurious failure as a failed read of the write the CAS would have succeeded on, so the CAS
 * returns its expected value. Two shapes of retry loop qualify:
 *
 *  - Drop the iteration. From the loop header to the header again the iteration holds reads
 *    and fences only (loads before the CAS; nothing but pure code after it), the failure path
 *    stays in the loop, and every header phi has the same value after the iteration as before.
 *    Deleting the iteration's events leaves a consistent execution -- deleting reads and fences
 *    only removes constraints -- in which the thread starts the next iteration in the same state
 *    and every remaining read reads the same write.
 *
 *  - Repeat the attempt. The failure path returns to the CAS with every loop-carried value and
 *    branch condition unchanged, and the only memory accesses on the cycle are stores of
 *    unchanged values to unchanged addresses that the previous arrival also executed (a
 *    Treiber push storing the link before the CAS). The execution with the spurious failure
 *    differs from the one without it by one read and the repeated stores. Remaining
 *    assumption: no other thread writes a stored address between the two copies; for a
 *    non-atomic store such a writer is a data race, which the checker reports through the copy
 *    that is kept.
 *
 * Every other weak CAS stays weak and its spurious failures are explored: one not in a loop,
 * one whose failure leaves the loop, loads after the CAS, calls, writes a changed value, or
 * changes a loop-carried value such as an attempt counter. Runs before SpinAssume and unrolling,
 * which rewrite the cycle. HG_GENMC_STUTTER_REPORT=1 prints the decision per CAS (=2 also
 * prints the function of a CAS kept weak); HG_GENMC_NO_STUTTER=1 disables the pass.
 */
class WeakCASStutterPass : public PassInfoMixin<WeakCASStutterPass> {
public:
	auto run(Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses;
};

#endif /* GENMC_WEAK_CAS_STUTTER_PASS_HPP */
