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
#include <map>
#include <vector>
#include <algorithm>
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
	// The current pass group: how much of the module it has been through, and what each
	// pass has cost so far. A function is counted when its first pass of the group starts.
	std::atomic<uint64_t> groupTotalFns{0};
	std::atomic<uint64_t> groupTotalInsts{0};
	std::atomic<uint64_t> groupDoneFns{0};
	std::atomic<uint64_t> groupDoneInsts{0};
	std::map<std::string, double> passSeconds;      // guarded by mtx
	std::map<std::string, uint64_t> passInsts;      // instructions the pass has been over
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

/* A module-level pass (or a pass group) starts: the whole module is its problem size. */
inline void beginGroup(uint64_t totalFns, uint64_t totalInsts)
{
	auto &s = state();
	std::lock_guard<std::mutex> lk(s.mtx);
	s.groupTotalFns.store(totalFns, std::memory_order_relaxed);
	s.groupTotalInsts.store(totalInsts, std::memory_order_relaxed);
	s.groupDoneFns.store(0, std::memory_order_relaxed);
	s.groupDoneInsts.store(0, std::memory_order_relaxed);
	s.function.clear();
}

inline void setPass(const std::string &pass, const std::string &function, uint64_t functionInsts)
{
	auto &s = state();
	std::lock_guard<std::mutex> lk(s.mtx);
	s.phase = "transform";
	if (function != s.function) {
		s.groupDoneFns.fetch_add(1, std::memory_order_relaxed);
		s.groupDoneInsts.fetch_add(functionInsts, std::memory_order_relaxed);
	}
	s.pass = pass;
	s.function = function;
	s.functionInsts.store(functionInsts, std::memory_order_relaxed);
	s.passStartNs.store(nowNs(), std::memory_order_relaxed);
}

/* A pass finished on a function: charge its time and the function's size to that pass. */
inline void endPass(const std::string &pass, double secs, uint64_t functionInsts)
{
	auto &s = state();
	std::lock_guard<std::mutex> lk(s.mtx);
	s.passSeconds[pass] += secs;
	s.passInsts[pass] += functionInsts;
}

/* The costliest passes so far, "name=secs/Minsts" -- seconds per million instructions is the
 * number that says whether a pass is growing faster than the module. */
inline std::string passTable(size_t top)
{
	auto &s = state();
	std::vector<std::pair<std::string, double>> v;
	{
		std::lock_guard<std::mutex> lk(s.mtx);
		for (auto &kv : s.passSeconds)
			v.emplace_back(kv.first, kv.second);
	}
	std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.second > b.second; });
	std::string out;
	for (size_t i = 0; i < v.size() && i < top; ++i) {
		uint64_t insts = 0;
		{
			std::lock_guard<std::mutex> lk(s.mtx);
			insts = s.passInsts[v[i].first];
		}
		const double perM = insts ? v[i].second / (insts / 1e6) : 0.0;
		out += (i ? " " : "") + v[i].first + "=" + std::to_string(v[i].second).substr(0, 7) + "s/" +
		       std::to_string(perM).substr(0, 6) + "s.per.Minst";
	}
	return out;
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
			  << " in_pass_s=" << inPass
			  << " group_fns=" << s.groupDoneFns.load(std::memory_order_relaxed) << "/"
			  << s.groupTotalFns.load(std::memory_order_relaxed)
			  << " group_insts=" << s.groupDoneInsts.load(std::memory_order_relaxed) << "/"
			  << s.groupTotalInsts.load(std::memory_order_relaxed)
			  << " top=[" << passTable(3) << "]";
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
