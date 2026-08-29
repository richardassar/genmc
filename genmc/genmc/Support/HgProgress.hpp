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
#include <functional>
#include <cmath>
#include <cstdio>
#include <string>
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

namespace llvm {
class Instruction;
}

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
	// Exploration profile (HG_GENMC_PROFILE): every 4096th interpreted instruction is
	// attributed to its function and source line, so the heartbeat names where an execution
	// spends its instructions -- the loop that iterates to the bound, not a guess about it.
	std::map<std::string, uint64_t> profile;        // guarded by mtx
	uint64_t profileSamples{0};                     // guarded by mtx
	// Executions started (each replay of a revisit is one), and the instructions each thread
	// has interpreted since the last completed execution: a first execution that never
	// completes is then attributed to the thread that runs it.
	std::atomic<uint64_t> starts{0};
	static constexpr size_t kThreads = 64;
	std::atomic<uint64_t> threadInsts[kThreads] = {};
	// The instruction the interpreter is executing, and the interpreter's own account of the
	// executing thread's call stack (set by the interpreter; empty until it runs).
	std::atomic<const llvm::Instruction *> curInst{nullptr};
	std::function<std::string()> stackDump;
	// Choice attribution (estimation mode). The state-space estimate of an execution is the
	// product, over its reads and writes, of the number of alternatives each one has; every
	// alternative set larger than one is booked here against the site of the instruction
	// being interpreted, weighted by log2 of its size, so the table at the end names where
	// the estimate comes from. A store that becomes a new alternative for earlier loads is
	// booked at the store's site with the log2 growth it causes.
	std::map<std::string, std::pair<uint64_t, double>> choices; // site -> (count, sum log2); mtx
	uint64_t execChoicePoints{0};                                // this execution; mtx
	double execChoiceLog2{0};                                    // this execution; mtx
	uint64_t log2Samples{0};                                     // Welford over executions; mtx
	long double log2Mean{0}, log2M2{0};                          // of log2(sample); mtx
	long double residualSum{0};                                  // |log2 sample - booked|; mtx
	uint64_t choicePointsSum{0};                                 // over executions; mtx
};

/* The function and source line of an instruction, "fn[:line]"; defined by the interpreter. */
std::string siteName(const llvm::Instruction *I);

inline State &state()
{
	static State s;
	return s;
}

inline bool enabled() { return std::getenv("HG_GENMC_PROGRESS") != nullptr; }
inline std::string profileTable(size_t n);

inline bool profiling()
{
	static const bool on = [] {
		const bool v = std::getenv("HG_GENMC_PROFILE") != nullptr;
		if (v)
			std::atexit([] { std::cerr << "HG-PROFILE at exit: [" << profileTable(12) << "]\n"; });
		return v;
	}();
	return on;
}

inline void sample(const std::string &site)
{
	auto &s = state();
	std::lock_guard<std::mutex> g(s.mtx);
	++s.profile[site];
	++s.profileSamples;
}

inline void choice(size_t alternatives)
{
	auto &s = state();
	const auto *I = s.curInst.load(std::memory_order_relaxed);
	const double w = std::log2(static_cast<double>(alternatives));
	std::lock_guard<std::mutex> g(s.mtx);
	auto &e = s.choices[siteName(I)];
	++e.first;
	e.second += w;
	++s.execChoicePoints;
	s.execChoiceLog2 += w;
}

inline void choiceRevisit(double log2Growth)
{
	if (log2Growth <= 0)
		return;
	auto &s = state();
	const auto *I = s.curInst.load(std::memory_order_relaxed);
	std::lock_guard<std::mutex> g(s.mtx);
	auto &e = s.choices[siteName(I) + " (store revisits earlier loads)"];
	++e.first;
	e.second += log2Growth;
	++s.execChoicePoints;
	s.execChoiceLog2 += log2Growth;
}

/* One estimation sample ended: SAMPLE is the product of the alternative counts. */
inline void endEstimationSample(long double sample)
{
	auto &s = state();
	std::lock_guard<std::mutex> g(s.mtx);
	const long double l = std::log2(sample);
	++s.log2Samples;
	const long double d = l - s.log2Mean;
	s.log2Mean += d / s.log2Samples;
	s.log2M2 += d * (l - s.log2Mean);
	s.residualSum += std::fabs(l - (long double)s.execChoiceLog2);
	s.choicePointsSum += s.execChoicePoints;
	s.execChoicePoints = 0;
	s.execChoiceLog2 = 0;
}

inline std::string choiceTable(size_t n)
{
	auto &s = state();
	std::vector<std::pair<std::string, std::pair<uint64_t, double>>> v;
	long double mean = 0, sd = 0, residual = 0;
	uint64_t samples = 0, points = 0;
	{
		std::lock_guard<std::mutex> g(s.mtx);
		v.assign(s.choices.begin(), s.choices.end());
		samples = s.log2Samples;
		mean = s.log2Mean;
		sd = samples > 1 ? std::sqrt(s.log2M2 / (samples - 1)) : 0;
		residual = samples ? s.residualSum / samples : 0;
		points = s.choicePointsSum;
	}
	std::sort(v.begin(), v.end(),
		  [](auto &a, auto &b) { return a.second.second > b.second.second; });
	double total = 0;
	for (auto &e : v)
		total += e.second.second;
	char buf[256];
	std::snprintf(buf, sizeof(buf),
		      "HG-CHOICES samples=%llu log2_mean=%.2Lf log2_sd=%.2Lf booked_residual=%.3Lf "
		      "choice_points_per_exec=%.1f booked_log2_per_exec=%.2f\n",
		      (unsigned long long)samples, mean, sd, residual,
		      samples ? (double)points / samples : 0.0, samples ? total / samples : 0.0);
	std::string out = buf;
	for (size_t i = 0; i < v.size() && i < n; ++i) {
		std::snprintf(buf, sizeof(buf), "HG-CHOICE %6.2f%% log2/exec=%8.2f n/exec=%8.2f %s\n",
			      total > 0 ? 100.0 * v[i].second.second / total : 0.0,
			      samples ? v[i].second.second / samples : 0.0,
			      samples ? (double)v[i].second.first / samples : 0.0,
			      v[i].first.c_str());
		out += buf;
	}
	return out;
}

inline std::string profileTable(size_t n)
{
	auto &s = state();
	std::vector<std::pair<std::string, uint64_t>> v;
	uint64_t total = 0;
	{
		std::lock_guard<std::mutex> g(s.mtx);
		v.assign(s.profile.begin(), s.profile.end());
		total = s.profileSamples;
	}
	std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.second > b.second; });
	std::string out;
	for (size_t i = 0; i < v.size() && i < n; ++i) {
		if (i) out += " ";
		out += v[i].first + "=" + std::to_string(total ? 100 * v[i].second / total : 0) + "%";
	}
	return out;
}

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
		std::cerr << " starts=" << s.starts.load(std::memory_order_relaxed) << " thread_insts=[";
		for (size_t t = 0; t < State::kThreads; ++t) {
			const auto v = s.threadInsts[t].load(std::memory_order_relaxed);
			if (v) std::cerr << t << ":" << v << " ";
		}
		std::cerr << "]";
		if (profiling())
			std::cerr << " profile=[" << profileTable(8) << "]";
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
