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
 * Makes a weak CAS strong when its spurious failure is a stutter: the failure edge
 * returns to the CAS with every loop-carried value unchanged and with no memory access
 * on the way other than stores that write the same value to the same address again.
 *
 * The checker models a spurious failure as a failed read of the write the CAS would
 * have succeeded on, so the value it returns equals the expected value. On such a cycle
 * the thread then re-executes the same CAS from the same local state, and the
 * execution with the spurious failure differs from the one without it only by one read
 * (and the repeated stores). Dropping a read removes happens-before edges and never
 * makes a consistent execution inconsistent, so every outcome reachable through the
 * spurious failure is reachable without it. Run before unrolling, which breaks the
 * cycle into copies.
 *
 * A weak CAS whose failure leads anywhere else -- out of the loop, into a load, a call,
 * an RMW, a store of a changed value, or a loop-carried value that changes (an attempt
 * counter, a "contended" flag) -- stays weak and its spurious failures are explored.
 *
 * Remaining assumption for the repeated stores: no other thread writes the stored
 * address between the two copies. For a non-atomic store such a writer is a data race,
 * which the checker reports through the copy that is kept.
 */
class WeakCASStutterPass : public PassInfoMixin<WeakCASStutterPass> {
public:
	auto run(Function &F, FunctionAnalysisManager &FAM) -> PreservedAnalyses;
};

#endif /* GENMC_WEAK_CAS_STUTTER_PASS_HPP */
