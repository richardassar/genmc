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

#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include "LLVMModule.hpp"
#include "genmc/Support/Error.hpp"
#include "genmc/Support/SExprVisitor.hpp"
#include "genmc/Support/HgProgress.hpp"
#include "passes/LLIConfig.hpp"
#include "passes/ModuleInfo.hpp"
#include "passes/Transforms/BarrierResultCheckerPass.hpp"
#include "passes/Transforms/BisimilarityCheckerPass.hpp"
#include "passes/Transforms/CallInfoCollectionPass.hpp"
#include "passes/Transforms/CodeCondenserPass.hpp"
#include "passes/Transforms/ConfirmationAnnotationPass.hpp"
#include "passes/Transforms/DeclareInternalsPass.hpp"
#include "passes/Transforms/DefineLibcFunsPass.hpp"
#include "passes/Transforms/EliminateAnnotationsPass.hpp"
#include "passes/Transforms/EliminateCASPHIsPass.hpp"
#include "passes/Transforms/EliminateCastsPass.hpp"
#include "passes/Transforms/EliminateRedundantInstPass.hpp"
#include "passes/Transforms/EliminateUnusedCodePass.hpp"
#include "passes/Transforms/EscapeCheckerPass.hpp"
#include "passes/Transforms/FunctionInlinerPass.hpp"
#include "passes/Transforms/IntrinsicLoweringPass.hpp"
#include "passes/Transforms/LoadAnnotationPass.hpp"
#include "passes/Transforms/LocalSimplifyCFGPass.hpp"
#include "passes/Transforms/LoopJumpThreadingPass.hpp"
#include "passes/Transforms/LoopUnrollPass.hpp"
#include "passes/Transforms/MDataCollectionPass.hpp"
#include "passes/Transforms/MMDetectorPass.hpp"
#include "passes/Transforms/PromoteMemIntrinsicPass.hpp"
#include "passes/Transforms/PropagateAssumesPass.hpp"
#include "passes/Transforms/RustPrepPass.hpp"
#include "passes/Transforms/SpinAssumePass.hpp"
#include "passes/Transforms/StrengthenCASPass.hpp"

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#if __has_include(<llvm/Plugins/PassPlugin.h>)
#include <llvm/Plugins/PassPlugin.h>
#else
#include <llvm/Passes/PassPlugin.h>
#endif
#include <llvm/Support/Debug.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/DeadArgumentElimination.h>
#include <llvm/Transforms/Scalar/JumpThreading.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
namespace fs = std::filesystem;

namespace LLVMModule {

auto parseLLVMModule(const std::string &filename, const std::unique_ptr<llvm::LLVMContext> &ctx)
	-> std::unique_ptr<llvm::Module>
{
	llvm::SMDiagnostic err;

	/* A saved module of a large program carries block names that repeated splitting has
	 * grown past the reader's 1024-byte default (-non-global-value-max-name-size), and the
	 * reader refuses the file. It is told the size to accept before it opens one. */
	auto &opts = llvm::cl::getRegisteredOptions();
	if (auto it = opts.find("non-global-value-max-name-size"); it != opts.end())
		it->second->addOccurrence(0, "non-global-value-max-name-size", "1048576");

	auto mod = llvm::parseIRFile(filename, err, *ctx);
	if (!mod) {
		err.print(filename.c_str(), llvm::dbgs());
		ERROR("Could not parse LLVM IR!");
	}
	return mod;
}

/**
 * Used to link a collection of modules into a single module
 */
auto linkAllModules(std::vector<std::unique_ptr<llvm::Module>> modules) -> std::unique_ptr<Module>
{
	Linker linker(*modules[0]);
	for (size_t i = 1; i < modules.size(); ++i) {
		if (linker.linkInModule(std::move(modules[i]))) {
			ERROR("Could not link the LLVM IR!");
		}
	}

	return std::move(modules[0]);
}

/**
 * Used for Rust-builds. We link all produced llvm-ir files in the target-directory together.
 */
auto parseLinkAllLLVMModules(const fs::path &dirname, const std::unique_ptr<llvm::LLVMContext> &ctx)
	-> std::unique_ptr<llvm::Module>
{
	llvm::SMDiagnostic err;

	/* Search for all *.bc files under "dir / target / * / deps /" */
	std::vector<fs::path> bc_files;
	for (const auto &dir : fs::directory_iterator(dirname / "target")) {
		if (!fs::is_directory(dir)) {
			continue;
		}

		auto deps_dir = dir.path() / "deps";
		if (!fs::exists(deps_dir)) {
			continue;
		}

		for (const auto &file : fs::directory_iterator(deps_dir)) {
			if (file.path().extension() != ".bc") {
				continue;
			}

			bc_files.push_back(file.path());
		}
	}

	/* Parse all modules */
	std::vector<std::unique_ptr<Module>> modules;
	for (const auto &bc_file : bc_files) {
		std::unique_ptr<Module> module = parseIRFile(bc_file.string(), err, *ctx);
		if (!module) {
			err.print(bc_file.c_str(), llvm::dbgs());
			ERROR("Could not parse LLVM IR!");
		}
		modules.push_back(std::move(module));
	}

	/* Link them all together */
	return linkAllModules(std::move(modules));
}

auto cloneModule(const std::unique_ptr<llvm::Module> &mod,
		 const std::unique_ptr<llvm::LLVMContext> &ctx) -> std::unique_ptr<llvm::Module>
{
	/* Roundtrip the module to a stream and then back into the new context */
	std::string str;
	llvm::raw_string_ostream stream(str);

	llvm::WriteBitcodeToFile(*mod, stream);

	const llvm::StringRef ref(stream.str());
	std::unique_ptr<llvm::MemoryBuffer> buf(llvm::MemoryBuffer::getMemBuffer(ref));

	return std::move(llvm::parseBitcodeFile(buf->getMemBufferRef(), *ctx).get());
}

static void initializeVariableInfo(ModuleInfo &MI, PassModuleInfo &PI)
{
	for (auto &kv : PI.varInfo.globalInfo)
		MI.varInfo.globalInfo[MI.idInfo.VID.at(kv.first)] = kv.second;
	for (auto &kv : PI.varInfo.localInfo) {
		if (MI.idInfo.VID.contains(kv.first))
			MI.varInfo.localInfo[MI.idInfo.VID.at(kv.first)] = kv.second;
	}
	MI.varInfo.internalInfo = PI.varInfo.internalInfo;
}

static void initializeAnnotationInfo(ModuleInfo &MI, PassModuleInfo &PI)
{
	using Transformer = SExprTransformer<llvm::Value *>;
	Transformer transformer;

	for (auto &kv : PI.annotInfo.annotMap) {
		MI.annotInfo.annotMap[MI.idInfo.VID.at(kv.first)] = std::make_pair(
			kv.second.first,
			transformer.transform(&*kv.second.second,
					      [&](llvm::Value *v) { return MI.idInfo.VID.at(v); }));
	}
}

static void initializeModuleInfo(ModuleInfo &MI, PassModuleInfo &PI, const llvm::Module &mod)
{
	MI.collectIDs(mod);
	initializeVariableInfo(MI, PI);
	initializeAnnotationInfo(MI, PI);
	MI.determinedMM = PI.determinedMM;
	MI.barrierResultsUsed = PI.barrierResultsUsed;
}

auto transformLLVMModule(llvm::Module &mod, ModuleInfo &MI, const LLIConfig *conf) -> bool
{
	PassModuleInfo PI;

	/* NOTE: The order between the analyses, the builder and the managers matters */

	/* First, register the analyses that we are about to use.
	 * We also use an (unused) pass builder to load default analyses */
	llvm::LoopAnalysisManager lam;
	llvm::CGSCCAnalysisManager cgam;
	llvm::FunctionAnalysisManager fam;
	llvm::ModuleAnalysisManager mam;

	mam.registerPass([&] { return MDataInfo(); });
	mam.registerPass([&] { return BarrierResultAnalysis(); });
	mam.registerPass([&] { return MMAnalysis(); });
	mam.registerPass([&] { return CallAnalysis(); });
	mam.registerPass([&] { return EscapeAnalysis(); });
	fam.registerPass([&] { return BisimilarityAnalysis(); });
	fam.registerPass([&] { return LoadAnnotationAnalysis(); });

	/* Progress through the transformation, when asked for: which pass is running, how long it
	 * took, the resident set after it and the module's instruction count -- the numbers that
	 * say where the time and memory of a large module go, and whether to keep waiting. */
	llvm::PassInstrumentationCallbacks pic;
	const bool hgProgress = std::getenv("HG_GENMC_PROGRESS") != nullptr;
	static std::chrono::steady_clock::time_point hgPassStart;
	auto hgRss = [] {
		std::ifstream f("/proc/self/statm");
		unsigned long size = 0, resident = 0;
		f >> size >> resident;
		return resident * 4096UL / (1024UL * 1024UL);
	};
	auto hgInsts = [&mod] {
		unsigned long n = 0;
		for (auto &F : mod)
			for (auto &BB : F)
				n += BB.size();
		return n;
	};
	if (hgProgress) {
		hgprog::start();
		pic.registerBeforeNonSkippedPassCallback([&](llvm::StringRef name, llvm::Any ir) {
			/* A function pass is reported to the heartbeat only: which function it is on
			 * and how large, so a pass that stays on one function for an hour is
			 * visible from outside. */
			if (const auto *fp = llvm::any_cast<const llvm::Function *>(&ir)) {
				const auto *F = *fp;
				unsigned long n = 0;
				for (auto &BB : *F)
					n += BB.size();
				hgprog::setPass(name.str(), F->getName().str(), n);
				return;
			}
			if (llvm::any_cast<const llvm::Module *>(&ir) == nullptr)
				return;
			{
				unsigned long fns = 0;
				for (auto &F : mod)
					if (!F.isDeclaration())
						++fns;
				hgprog::beginGroup(fns, hgInsts());
			}
			hgprog::setPass(name.str(), "<module>", 0);
			hgPassStart = std::chrono::steady_clock::now();
			std::cerr << "HG-PASS begin " << name.str() << " rss_mb=" << hgRss() << "\n";
		});
		pic.registerAfterPassCallback([&](llvm::StringRef name, llvm::Any ir, const llvm::PreservedAnalyses &) {
			/* Function passes run once per function; counting the whole module after each
			 * of them is quadratic and was itself the slowest part of the transform. */
			const bool modulePass = llvm::any_cast<const llvm::Module *>(&ir) != nullptr;
			if (!modulePass) {
				if (const auto *fp = llvm::any_cast<const llvm::Function *>(&ir)) {
					auto &st = hgprog::state();
					const double secs = (hgprog::nowNs() - st.passStartNs.load(std::memory_order_relaxed)) / 1e9;
					hgprog::endPass(name.str(), secs, st.functionInsts.load(std::memory_order_relaxed));
				}
				return;
			}
			const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - hgPassStart).count();
			std::cerr << "HG-PASS end   " << name.str() << " secs=" << secs << " rss_mb=" << hgRss()
				  << " insts=" << hgInsts() << " passes=[" << hgprog::passTable(6) << "]\n";
		});
		mam.registerPass([&] { return llvm::PassInstrumentationAnalysis(&pic); });
		fam.registerPass([&] { return llvm::PassInstrumentationAnalysis(&pic); });
	}

	llvm::PassBuilder PB(nullptr, llvm::PipelineTuningOptions(), std::nullopt, &pic);
	PB.registerModuleAnalyses(mam);
	PB.registerCGSCCAnalyses(cgam);
	PB.registerFunctionAnalyses(fam);
	PB.registerLoopAnalyses(lam);
	PB.crossRegisterProxies(lam, fam, cgam, mam);

	/* Then create two pass managers: a basic one and one that
	runs some loop passes */
	llvm::ModulePassManager basicOptsMGR;

	basicOptsMGR.addPass(DeclareInternalsPass());
	basicOptsMGR.addPass(DefineLibcFunsPass());
	basicOptsMGR.addPass(MDataCollectionPass(PI));
	if (conf->rust)
		basicOptsMGR.addPass(RustPrepPass());
	if (conf->inlineFunctions)
		basicOptsMGR.addPass(FunctionInlinerPass());
	{
		llvm::FunctionPassManager fpm;
		/* Run after the inliner because it might generate new memcpys. Typed operands only
		 * here; the opaque ones wait for SROA and mem2reg below. */
		fpm.addPass(PromoteMemIntrinsicPass(/*lowerOpaque=*/false));
		if (conf->castElimination)
			fpm.addPass(EliminateCastsPass());
		fpm.addPass(SROAPass(SROAOptions::PreserveCFG));
		fpm.addPass(PromotePass()); // Mem2Reg
		/* Now that closure fields and pointer slots are the allocas they held, the copies
		 * that were opaque above are typed; what is still opaque is lowered to a loop. */
		fpm.addPass(PromoteMemIntrinsicPass(/*lowerOpaque=*/true, conf->unroll,
						    conf->noUnrollFuns));
		fpm.addPass(IntrinsicLoweringPass());
		basicOptsMGR.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
	}
	basicOptsMGR.addPass(DeadArgumentEliminationPass());
	{
		llvm::FunctionPassManager fpm;
		fpm.addPass(LocalSimplifyCFGPass());
		fpm.addPass(EliminateAnnotationsPass({.annotHelper = conf->helper,
						      .annotConf = conf->confirmation,
						      .annotFinal = conf->finalWrite}));
		fpm.addPass(EliminateRedundantInstPass());
		basicOptsMGR.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
	}

	if (!conf->bam)
		basicOptsMGR.addPass(BarrierResultCheckerPass(PI));
	if (conf->mmDetector)
		basicOptsMGR.addPass(MMDetectorPass(PI));

	auto preserved = basicOptsMGR.run(mod, mam);

	llvm::ModulePassManager loopOptsMGR;

	{
		llvm::FunctionPassManager fpm;
		fpm.addPass(EliminateCASPHIsPass());
		fpm.addPass(llvm::JumpThreadingPass());
		fpm.addPass(EliminateUnusedCodePass());
		if (conf->codeCondenser && !conf->liveness)
			fpm.addPass(CodeCondenserPass());
		if (conf->loopJumpThreading)
			fpm.addPass(createFunctionToLoopPassAdaptor(LoopJumpThreadingPass()));
		loopOptsMGR.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
	}
	if (conf->spinAssume)
		loopOptsMGR.addPass(SpinAssumePass(conf->liveness));
	if (conf->unroll.has_value())
		loopOptsMGR.addPass(
			createModuleToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(
				LoopUnrollPass(*conf->unroll, conf->noUnrollFuns))));
	preserved.intersect(loopOptsMGR.run(mod, mam));

	/* Run annotation passes last so that the module is stable */
	{
		llvm::FunctionPassManager fpm;
		if (conf->assumePropagation)
			fpm.addPass(PropagateAssumesPass());
		if (conf->confirmation)
			fpm.addPass(ConfirmationAnnotationPass());
		if (conf->loadAnnot)
			fpm.addPass(LoadAnnotationPass(PI.annotInfo));
		fpm.addPass(StrengthenCASPass());
		basicOptsMGR.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
	}

	preserved.intersect(basicOptsMGR.run(mod, mam));

	initializeModuleInfo(MI, PI, mod);

	assert(!llvm::verifyModule(mod, &llvm::dbgs()));
	return true;
}

void printLLVMModule(llvm::Module &mod, const std::string &filename)
{
	auto flags = llvm::sys::fs::OF_None;
	std::error_code errs;
	auto out = std::make_unique<llvm::raw_fd_ostream>(filename.c_str(), errs, flags);
	/* A ".bc" name writes bitcode: it round-trips exactly and parses in seconds, where the
	 * textual form of a multi-gigabyte module emits block labels its own reader rejects. */
	if (filename.size() > 3 && filename.compare(filename.size() - 3, 3, ".bc") == 0) {
		/* Block and value names that the transformation has grown by repeated splitting
		 * (".loopexit.split-lp.loopexit.split-lp...", kilobytes long on a large module)
		 * are rejected by LLVM's textual reader past 1024 bytes
		 * (-non-global-value-max-name-size), and a re-fed module passes through that
		 * reader. Exploration needs no local names, so every one of them is dropped. */
		for (auto &F : mod)
			for (auto &BB : F) {
				BB.setName("");
				for (auto &I : BB)
					if (I.hasName())
						I.setName("");
			}
		llvm::WriteBitcodeToFile(mod, *out);
		return;
	}

	/* TODO: Do we need an exception? If yes, properly handle it */
	if (errs) {
		WARN("Failed to write transformed module to file {}: {}\n", filename,
		     errs.message());
		return;
	}
	mod.print(*out, nullptr);
}

} // namespace LLVMModule
