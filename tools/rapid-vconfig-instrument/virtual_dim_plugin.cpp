//===- VirtualDimPlugin.cpp - VirtualDim pass plugin ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "virtual_dim_pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

static void addVirtualDimPass(ModulePassManager &MPM) {
  MPM.addPass(VirtualDimPass());
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "RapidVConfigPassPlugin",
          LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "rapid-vconfig" || Name == "virtual-dim") {
                    addVirtualDimPass(MPM);
                    return true;
                  }
                  return false;
                });

            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  addVirtualDimPass(MPM);
                });
          }};
}
