/*
 * Progress reporting for long runs (HG_GENMC_PROGRESS). A heartbeat thread prints, every
 * HG_GENMC_HEARTBEAT seconds (30 when unset), what the checker is doing RIGHT NOW: during the
 * transformation the pass and the function it is on, that function's size and how long the
 * pass has been on it; during exploration the executions completed, the revisit stack, the
 * instructions interpreted in the current execution and in total. The point is a run inside
 * one long pass or one long execution, where no per-pass or per-execution line is coming.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace hgprog {

struct State {
	std::mutex mtx;                 // guards the strings
	std::string phase{"startup"};
	std::string pass;
	std::string function;
	std::atomic<uint64_t> functionInsts{0};
	std::atomic<int64_t> passStartNs{0};
	std::atomic<uint64_t> explored{0};
	std::atomic<uint64_t> blocked{0};
	std::atomic<uint64_t> execDepth{0};
	std::atomic<uint64_t> pendingRevisits{0};
	std::atomic<uint64_t> instsThisExec{0};
	std::atomic<uint64_t> instsTotal{0};
	std::atomic<int64_t> exploreStartNs{0};
};

inline State &state()
{
	static State s;
	return s;
}

inline bool enabled() { return std::getenv("HG_GENMC_PROGRESS") != nullptr; }

inline int64_t nowNs()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		       std::chrono::steady_clock::now().time_since_epoch())
		.count();
}

inline unsigned long rssMb()
{
	std::ifstream f("/proc/self/statm");
	unsigned long size = 0, resident = 0;
	f >> size >> resident;
	return resident * 4096UL / (1024UL * 1024UL);
}

inline void setPass(const std::string &pass, const std::string &function, uint64_t functionInsts)
{
	auto &s = state();
	std::lock_guard<std::mutex> lk(s.mtx);
	s.phase = "transform";
	s.pass = pass;
	s.function = function;
	s.functionInsts.store(functionInsts, std::memory_order_relaxed);
	s.passStartNs.store(nowNs(), std::memory_order_relaxed);
}

inline void setPhase(const char *phase)
{
	auto &s = state();
	std::lock_guard<std::mutex> lk(s.mtx);
	s.phase = phase;
	if (s.phase == "explore" && s.exploreStartNs.load(std::memory_order_relaxed) == 0)
		s.exploreStartNs.store(nowNs(), std::memory_order_relaxed);
}

inline void heartbeat()
{
	auto &s = state();
	std::string phase, pass, function;
	{
		std::lock_guard<std::mutex> lk(s.mtx);
		phase = s.phase;
		pass = s.pass;
		function = s.function;
	}
	const auto now = nowNs();
	std::cerr << "HG-HEARTBEAT phase=" << phase << " rss_mb=" << rssMb();
	if (phase == "transform") {
		const double inPass = (now - s.passStartNs.load(std::memory_order_relaxed)) / 1e9;
		std::cerr << " pass=" << pass << " fn=" << function
			  << " fn_insts=" << s.functionInsts.load(std::memory_order_relaxed)
			  << " in_pass_s=" << inPass;
	} else if (phase == "explore") {
		const double secs = (now - s.exploreStartNs.load(std::memory_order_relaxed)) / 1e9;
		const auto n = s.explored.load(std::memory_order_relaxed);
		std::cerr << " explored=" << n << " blocked=" << s.blocked.load(std::memory_order_relaxed)
			  << " exec_depth=" << s.execDepth.load(std::memory_order_relaxed)
			  << " pending_revisits=" << s.pendingRevisits.load(std::memory_order_relaxed)
			  << " insts_this_exec=" << s.instsThisExec.load(std::memory_order_relaxed)
			  << " insts_total=" << s.instsTotal.load(std::memory_order_relaxed)
			  << " elapsed_s=" << secs << " rate_per_s=" << (secs > 0 ? n / secs : 0.0);
	}
	std::cerr << "\n";
}

/* Idempotent; the thread is detached and dies with the process. */
inline void start()
{
	static std::once_flag once;
	if (!enabled())
		return;
	std::call_once(once, [] {
		const char *hb = std::getenv("HG_GENMC_HEARTBEAT");
		const int secs = hb ? std::max(1, std::atoi(hb)) : 30;
		std::thread([secs] {
			for (;;) {
				std::this_thread::sleep_for(std::chrono::seconds(secs));
				heartbeat();
			}
		}).detach();
	});
}

}  // namespace hgprog
