//===- VirtualDim.h - Virtual grid/block mapping pass ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_VIRTUALDIM_H
#define LLVM_TRANSFORMS_SCALAR_VIRTUALDIM_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class VirtualDimPass : public PassInfoMixin<VirtualDimPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_VIRTUALDIM_H
