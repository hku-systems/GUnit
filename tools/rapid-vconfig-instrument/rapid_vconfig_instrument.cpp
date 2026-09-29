#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "virtual_dim_pass.h"

#include <string>
#include <system_error>

using namespace llvm;

namespace {

cl::opt<std::string> InputPath("input", cl::Required,
                               cl::desc("Input LLVM bitcode"));
cl::opt<std::string> OutputPath("output", cl::Required,
                                cl::desc("Transformed LLVM bitcode"));
cl::opt<std::string> EntrySymbol("entry-symbol", cl::Required,
                                 cl::desc("Rewritten RAPID entry symbol"));

constexpr StringLiteral RapidContextAttr = "rapid.vconfig.context";
constexpr StringLiteral RapidProcessedAttr = "rapid.vconfig.processed";

int fail(StringRef Reason) {
  errs() << Reason << '\n';
  return 1;
}

bool validEntryAbi(const Function &Entry) {
  return !Entry.isDeclaration() && !Entry.isVarArg() &&
         Entry.getReturnType()->isVoidTy() && !Entry.arg_empty() &&
         Entry.getArg(Entry.arg_size() - 1)->getType()->isPointerTy();
}

int writeModuleAtomically(const Module &Module, StringRef Path) {
  int FileDescriptor = -1;
  SmallString<256> TemporaryPath;
  std::error_code EC = sys::fs::createUniqueFile(
      (Path + ".tmp-%%%%%%").str(), FileDescriptor, TemporaryPath);
  if (EC)
    return fail("vconfig_output_write_failed");

  {
    raw_fd_ostream Output(FileDescriptor, true);
    WriteBitcodeToFile(Module, Output);
    Output.flush();
    if (Output.has_error()) {
      sys::fs::remove(TemporaryPath);
      return fail("vconfig_output_write_failed");
    }
  }

  EC = sys::fs::rename(TemporaryPath, Path);
  if (EC) {
    sys::fs::remove(TemporaryPath);
    return fail("vconfig_output_write_failed");
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "RAPID VConfig instrumenter\n");

  LLVMContext Context;
  SMDiagnostic Diagnostic;
  std::unique_ptr<Module> Module = parseIRFile(InputPath, Diagnostic, Context);
  if (!Module) {
    Diagnostic.print(argv[0], errs());
    return fail("vconfig_input_invalid");
  }

  Function *Entry = Module->getFunction(EntrySymbol);
  if (!Entry)
    return fail("vconfig_entry_missing");
  if (!validEntryAbi(*Entry))
    return fail("vconfig_entry_abi_invalid");

  Entry->addFnAttr(RapidContextAttr);
  ModuleAnalysisManager AnalysisManager;
  VirtualDimPass Pass;
  Pass.run(*Module, AnalysisManager);
  if (!Entry->hasFnAttribute(RapidProcessedAttr))
    return fail("vconfig_transform_failed");

  if (verifyModule(*Module, &errs()))
    return fail("vconfig_output_invalid");

  return writeModuleAtomically(*Module, OutputPath);
}
