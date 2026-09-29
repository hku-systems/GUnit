//===- VirtualDim.cpp - Virtual grid/block mapping pass --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "virtual_dim_pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

using namespace llvm;

namespace {

struct VDim {
  uint32_t Gx = 1;
  uint32_t Gy = 1;
  uint32_t Gz = 1;
  uint32_t Bx = 1;
  uint32_t By = 1;
  uint32_t Bz = 1;
};

struct VDimValues {
  Value *Gx = nullptr;
  Value *Gy = nullptr;
  Value *Gz = nullptr;
  Value *Bx = nullptr;
  Value *By = nullptr;
  Value *Bz = nullptr;
  Value *BlockIdxX = nullptr;
  Value *BlockIdxY = nullptr;
  Value *BlockIdxZ = nullptr;
  Value *ThreadIdxX = nullptr;
  Value *ThreadIdxY = nullptr;
  Value *ThreadIdxZ = nullptr;
};

enum class VDimMode { Const, Args, RapidContext };

static constexpr StringLiteral RapidContextAttr = "rapid.vconfig.context";
static constexpr StringLiteral RapidProcessedAttr = "rapid.vconfig.processed";
static constexpr StringLiteral RapidWarpAlignedAttr =
    "rapid.vconfig.warp_aligned";

struct VDimAnnotation {
  VDimMode Mode = VDimMode::Const;
  VDim Const;
};

enum class RapidContextSyncActionKind { Barrier, FixedLoop, DynamicLoop };
enum class DynamicLoopUpdateKind { Add, Shl, Mul };

struct RapidContextSyncAction {
  RapidContextSyncActionKind Kind = RapidContextSyncActionKind::Barrier;
  unsigned Count = 0;
  unsigned TripCount = 0;
  Value *Start = nullptr;
  ConstantInt *Step = nullptr;
  Value *Bound = nullptr;
  Value *EnterCondition = nullptr;
  ICmpInst::Predicate Predicate = ICmpInst::BAD_ICMP_PREDICATE;
  DynamicLoopUpdateKind UpdateKind = DynamicLoopUpdateKind::Add;
  bool EnterOnTrue = true;
  bool ContinueOnTrue = true;
};

struct DynamicLoopSyncInfo {
  Loop *Target = nullptr;
  Value *Start = nullptr;
  ConstantInt *Step = nullptr;
  Value *Bound = nullptr;
  Value *EnterCondition = nullptr;
  ICmpInst::Predicate Predicate = ICmpInst::BAD_ICMP_PREDICATE;
  DynamicLoopUpdateKind UpdateKind = DynamicLoopUpdateKind::Add;
  bool EnterOnTrue = true;
  bool ContinueOnTrue = true;
};

struct RapidContextSyncPlan {
  SmallVector<RapidContextSyncAction, 4> InactiveSyncActions;
  BasicBlock *GuardBlock = nullptr;
  bool RequiresWarpAlignedBlock = false;
  StringRef UnsupportedReason;
  StringRef UnsupportedDetail;
};

static bool parseVDimAnnotation(StringRef S, VDim &Out) {
  if (!S.starts_with("vdim="))
    return false;
  S = S.drop_front(5);
  SmallVector<uint32_t, 6> Values;
  while (!S.empty()) {
    StringRef Tok;
    std::tie(Tok, S) = S.split(',');
    Tok = Tok.trim();
    if (Tok.empty())
      return false;
    uint64_t V = 0;
    if (Tok.getAsInteger(10, V))
      return false;
    Values.push_back(static_cast<uint32_t>(V));
  }

  if (Values.size() == 4) {
    Out.Gx = Values[0];
    Out.Gy = Values[1];
    Out.Gz = 1;
    Out.Bx = Values[2];
    Out.By = Values[3];
    Out.Bz = 1;
    return true;
  }
  if (Values.size() == 6) {
    Out.Gx = Values[0];
    Out.Gy = Values[1];
    Out.Gz = Values[2];
    Out.Bx = Values[3];
    Out.By = Values[4];
    Out.Bz = Values[5];
    return true;
  }
  return false;
}

static bool getVDimAnnotation(Module &M, Function &F, VDimAnnotation &Out) {
  GlobalVariable *GA = M.getNamedGlobal("llvm.global.annotations");
  if (!GA)
    return false;
  auto *Arr = dyn_cast<ConstantArray>(GA->getOperand(0));
  if (!Arr)
    return false;

  for (unsigned I = 0; I < Arr->getNumOperands(); ++I) {
    auto *CS = dyn_cast<ConstantStruct>(Arr->getOperand(I));
    if (!CS || CS->getNumOperands() < 2)
      continue;
    Value *FnVal = CS->getOperand(0)->stripPointerCasts();
    if (FnVal != &F)
      continue;
    auto *AnnGV = dyn_cast<GlobalVariable>(
        CS->getOperand(1)->stripPointerCasts());
    if (!AnnGV || !AnnGV->hasInitializer())
      continue;
    auto *AnnData = dyn_cast<ConstantDataArray>(AnnGV->getInitializer());
    if (!AnnData)
      continue;
    StringRef AnnStr = AnnData->getAsCString();
    if (AnnStr == "vdim" || AnnStr == "vdim-args") {
      Out.Mode = VDimMode::Args;
      return true;
    }
    if (parseVDimAnnotation(AnnStr, Out.Const)) {
      Out.Mode = VDimMode::Const;
      return true;
    }
  }
  return false;
}

static ConstantInt *getConstI32(LLVMContext &Ctx, uint32_t V) {
  return ConstantInt::get(Type::getInt32Ty(Ctx), V);
}

static StructType *getOrCreateDim3Type(Module &M) {
  LLVMContext &Ctx = M.getContext();
  if (StructType *Ty = StructType::getTypeByName(Ctx, "struct.dim3"))
    return Ty;
  auto *I32 = Type::getInt32Ty(Ctx);
  return StructType::create(Ctx, {I32, I32, I32}, "struct.dim3");
}

static bool isNvvmKernel(Module &M, Function &F) {
  if (F.getCallingConv() == CallingConv::PTX_Kernel)
    return true;
  NamedMDNode *NMD = M.getNamedMetadata("nvvm.annotations");
  if (!NMD)
    return false;
  for (MDNode *MD : NMD->operands()) {
    if (!MD || MD->getNumOperands() < 2)
      continue;
    auto *VMD = dyn_cast<ValueAsMetadata>(MD->getOperand(0));
    if (!VMD || VMD->getValue() != &F)
      continue;
    auto *Tag = dyn_cast<MDString>(MD->getOperand(1));
    if (Tag && Tag->getString() == "kernel")
      return true;
  }
  return false;
}

static void replaceNvvmAnnotations(Module &M, Function *OldF,
                                   Function *NewF) {
  NamedMDNode *NMD = M.getNamedMetadata("nvvm.annotations");
  if (!NMD)
    return;
  for (MDNode *MD : NMD->operands()) {
    if (!MD || MD->getNumOperands() < 1)
      continue;
    auto *VMD = dyn_cast<ValueAsMetadata>(MD->getOperand(0));
    if (!VMD || VMD->getValue() != OldF)
      continue;
    MD->replaceOperandWith(0, ValueAsMetadata::get(NewF));
  }
}

static bool getDim3Args(Function &F, Argument *&GridArg, Argument *&BlockArg,
                        StructType *&DimTy) {
  if (F.arg_size() < 2)
    return false;
  GridArg = F.getArg(F.arg_size() - 2);
  BlockArg = F.getArg(F.arg_size() - 1);
  if (!GridArg->getType()->isPointerTy() ||
      !BlockArg->getType()->isPointerTy())
    return false;
  if (GridArg->hasByValAttr() || BlockArg->hasByValAttr())
    return false;
  auto hasMarker = [](Argument *Arg, StringRef Name) {
    return Arg->getName() == Name || Arg->hasAttribute("vdim.arg");
  };
  if (!hasMarker(GridArg, "vdim.grid") || !hasMarker(BlockArg, "vdim.block"))
    return false;
  DimTy = getOrCreateDim3Type(*F.getParent());
  return true;
}

static Function *appendVDimArgs(Function &F) {
  Module &M = *F.getParent();
  LLVMContext &Ctx = F.getContext();
  FunctionType *OldTy = F.getFunctionType();
  SmallVector<Type *, 8> Params(OldTy->params());
  auto *PtrTy = PointerType::get(Ctx, 0);
  Params.push_back(PtrTy);
  Params.push_back(PtrTy);

  FunctionType *NewTy = FunctionType::get(OldTy->getReturnType(), Params,
                                          OldTy->isVarArg());
  Function *NewF = Function::Create(NewTy, F.getLinkage(), "", &M);
  NewF->copyAttributesFrom(&F);
  NewF->setCallingConv(F.getCallingConv());
  NewF->setSubprogram(F.getSubprogram());
  NewF->takeName(&F);

  auto NewArgIt = NewF->arg_begin();
  for (Argument &OldArg : F.args()) {
    Argument *NewArg = &*NewArgIt++;
    NewArg->takeName(&OldArg);
    OldArg.replaceAllUsesWith(NewArg);
  }

  unsigned GridIdx = OldTy->getNumParams();
  unsigned BlockIdx = GridIdx + 1;
  NewF->getArg(GridIdx)->setName("vdim.grid");
  NewF->getArg(BlockIdx)->setName("vdim.block");
  Attribute VDimAttr = Attribute::get(Ctx, "vdim.arg");
  NewF->addParamAttr(GridIdx, VDimAttr);
  NewF->addParamAttr(BlockIdx, VDimAttr);

  NewF->splice(NewF->begin(), &F);
  F.replaceAllUsesWith(NewF);
  replaceNvvmAnnotations(M, &F, NewF);
  F.eraseFromParent();
  return NewF;
}

static bool replaceDimBuiltinUses(Function &F, const VDimValues &VD) {
  bool Changed = false;
  SmallVector<Instruction *, 8> ToErase;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      if (CI->getMetadata("vdim.physical"))
        continue;
      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      StringRef Name = Callee->getName();
      Value *Replacement = nullptr;
      if (Name.starts_with("llvm.nvvm.read.ptx.sreg.nctaid.")) {
        if (Name.ends_with(".x"))
          Replacement = VD.Gx;
        else if (Name.ends_with(".y"))
          Replacement = VD.Gy;
        else if (Name.ends_with(".z"))
          Replacement = VD.Gz;
      } else if (Name.starts_with("llvm.nvvm.read.ptx.sreg.ntid.")) {
        if (Name.ends_with(".x"))
          Replacement = VD.Bx;
        else if (Name.ends_with(".y"))
          Replacement = VD.By;
        else if (Name.ends_with(".z"))
          Replacement = VD.Bz;
      } else if (Name.starts_with("llvm.nvvm.read.ptx.sreg.ctaid.")) {
        if (Name.ends_with(".x"))
          Replacement = VD.BlockIdxX;
        else if (Name.ends_with(".y"))
          Replacement = VD.BlockIdxY;
        else if (Name.ends_with(".z"))
          Replacement = VD.BlockIdxZ;
      } else if (Name.starts_with("llvm.nvvm.read.ptx.sreg.tid.")) {
        if (Name.ends_with(".x"))
          Replacement = VD.ThreadIdxX;
        else if (Name.ends_with(".y"))
          Replacement = VD.ThreadIdxY;
        else if (Name.ends_with(".z"))
          Replacement = VD.ThreadIdxZ;
      }

      if (!Replacement)
        continue;
      CI->replaceAllUsesWith(Replacement);
      ToErase.push_back(CI);
      Changed = true;
    }
  }

  for (Instruction *I : ToErase)
    I->eraseFromParent();

  return Changed;
}

static Function *getOrDeclareIntrinsic(Module &M, StringRef Name) {
  if (Function *F = M.getFunction(Name))
    return F;
  auto *I32 = Type::getInt32Ty(M.getContext());
  return Function::Create(FunctionType::get(I32, {}, false),
                          Function::ExternalLinkage, Name, &M);
}

static Value *readSReg(IRBuilder<> &B, Module &M, StringRef Name) {
  Function *F = getOrDeclareIntrinsic(M, Name);
  return B.CreateCall(F, {});
}

static Function *getOrDeclareBarrier0(Module &M) {
  if (Function *F = M.getFunction("llvm.nvvm.barrier.cta.sync.aligned.all"))
    return F;
  auto *I32 = Type::getInt32Ty(M.getContext());
  auto *FnTy = FunctionType::get(Type::getVoidTy(M.getContext()), {I32}, false);
  return Function::Create(FnTy, Function::ExternalLinkage,
                          "llvm.nvvm.barrier.cta.sync.aligned.all", &M);
}

static void markPhysical(Value *V) {
  if (auto *CI = dyn_cast<CallInst>(V))
    CI->setMetadata("vdim.physical", MDNode::get(CI->getContext(), {}));
}

static Value *readPhysicalSReg(IRBuilder<> &B, Module &M, StringRef Name) {
  Value *V = readSReg(B, M, Name);
  markPhysical(V);
  return V;
}

static Value *linearizePhysical3D(IRBuilder<> &B, Value *X, Value *Y, Value *Z,
                                  Value *DimX, Value *DimY, StringRef Prefix) {
  Value *YStride = B.CreateMul(Y, DimX, Twine(Prefix) + ".y_stride");
  Value *XY = B.CreateMul(DimX, DimY, Twine(Prefix) + ".xy");
  Value *ZStride = B.CreateMul(Z, XY, Twine(Prefix) + ".z_stride");
  return B.CreateAdd(B.CreateAdd(X, YStride, Twine(Prefix) + ".xy_linear"),
                     ZStride, Twine(Prefix) + ".linear");
}

static void decomposeLinear3D(IRBuilder<> &B, Value *Linear, Value *DimX,
                              Value *DimY, StringRef Prefix, Value *&OutX,
                              Value *&OutY, Value *&OutZ) {
  OutX = B.CreateURem(Linear, DimX, Twine(Prefix) + ".x");
  Value *DivX = B.CreateUDiv(Linear, DimX, Twine(Prefix) + ".div_x");
  OutY = B.CreateURem(DivX, DimY, Twine(Prefix) + ".y");
  OutZ = B.CreateUDiv(DivX, DimY, Twine(Prefix) + ".z");
}

static bool isBarrier0Call(const CallBase &CB) {
  Function *Callee = CB.getCalledFunction();
  if (!Callee)
    return false;
  StringRef Name = Callee->getName();
  if (Name == "llvm.nvvm.barrier0")
    return CB.arg_empty();
  if (Name != "llvm.nvvm.barrier.cta.sync.aligned.all" ||
      CB.arg_size() != 1)
    return false;
  auto *BarrierId = dyn_cast<ConstantInt>(CB.getArgOperand(0));
  return BarrierId && BarrierId->isZero();
}

static bool isWarpSensitiveCallName(StringRef Name) {
  return Name.starts_with("llvm.nvvm.shfl") ||
         Name.starts_with("llvm.nvvm.vote") ||
         Name.starts_with("llvm.nvvm.match") ||
         Name.starts_with("llvm.nvvm.redux") ||
         Name.starts_with("llvm.nvvm.activemask") ||
         Name.starts_with("llvm.nvvm.bar.warp");
}

static bool isBlockUniformIntrinsicName(StringRef Name) {
  return Name.starts_with("llvm.smin.") || Name.starts_with("llvm.smax.") ||
         Name.starts_with("llvm.umin.") || Name.starts_with("llvm.umax.");
}

static bool isBlockUniformValue(Value *V,
                                SmallPtrSetImpl<Value *> &Visited) {
  if (isa<Constant>(V) || isa<Argument>(V))
    return true;
  if (!Visited.insert(V).second)
    return false;

  if (auto *CB = dyn_cast<CallBase>(V)) {
    Function *Callee = CB->getCalledFunction();
    if (!Callee)
      return false;
    StringRef Name = Callee->getName();
    if (Name.starts_with("llvm.nvvm.read.ptx.sreg.ctaid.") ||
        Name.starts_with("llvm.nvvm.read.ptx.sreg.nctaid.") ||
        Name.starts_with("llvm.nvvm.read.ptx.sreg.ntid."))
      return true;
    if (!isBlockUniformIntrinsicName(Name))
      return false;
    return llvm::all_of(CB->args(), [&](Use &Operand) {
      return isBlockUniformValue(Operand.get(), Visited);
    });
  }

  auto *I = dyn_cast<Instruction>(V);
  if (!I || isa<LoadInst>(I) || isa<PHINode>(I) || I->mayHaveSideEffects())
    return false;
  return llvm::all_of(I->operands(), [&](Use &Operand) {
    return isBlockUniformValue(Operand.get(), Visited);
  });
}

static BasicBlock *findUniformEarlyReturnGuard(
    Function &Entry, PostDominatorTree &PostDominators, LoopInfo &Loops,
    unsigned &BarrierCount) {
  BasicBlock &EntryBlock = Entry.getEntryBlock();
  auto *Branch = dyn_cast<BranchInst>(EntryBlock.getTerminator());
  if (!Branch || !Branch->isConditional())
    return nullptr;

  SmallPtrSet<Value *, 16> Visited;
  if (!isBlockUniformValue(Branch->getCondition(), Visited))
    return nullptr;

  BasicBlock *Body = nullptr;
  for (unsigned I = 0; I < 2; ++I) {
    BasicBlock *Exit = Branch->getSuccessor(I);
    BasicBlock *Candidate = Branch->getSuccessor(1 - I);
    if (!isa<ReturnInst>(Exit->getTerminator()) ||
        !Candidate->hasNPredecessors(1))
      continue;
    Body = Candidate;
    break;
  }
  if (!Body)
    return nullptr;

  for (Instruction &I : EntryBlock) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (CB && isBarrier0Call(*CB))
      return nullptr;
  }

  BarrierCount = 0;
  for (Instruction &I : instructions(Entry)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || !isBarrier0Call(*CB))
      continue;
    BasicBlock *BarrierBlock = CB->getParent();
    if (Loops.getLoopFor(BarrierBlock) ||
        !PostDominators.dominates(BarrierBlock, Body))
      return nullptr;
    ++BarrierCount;
  }
  return BarrierCount > 0 ? Body : nullptr;
}

class SyncPathSummaryAnalyzer {
public:
  explicit SyncPathSummaryAnalyzer(bool &RequiresWarpAlignedBlock)
      : RequiresWarpAlignedBlock(RequiresWarpAlignedBlock) {}

  bool summarizeBlockCalls(BasicBlock &BB, unsigned &BarrierCount) {
    uint64_t Total = 0;
    for (Instruction &I : BB) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;

      if (auto *Asm = dyn_cast<InlineAsm>(CB->getCalledOperand())) {
        StringRef Source = Asm->getAsmString();
        if (Source.contains("bar.") || Source.contains("shfl.") ||
            Source.contains("vote.") || Source.contains("match.") ||
            Source.contains("redux.") || Source.contains("activemask"))
          return fail("vconfig_inline_asm_unsupported",
                      "synchronization inline assembly");
        continue;
      }

      Function *Callee = CB->getCalledFunction();
      if (!Callee) {
        Callee = dyn_cast<Function>(
            CB->getCalledOperand()->stripPointerCasts());
      }
      if (!Callee)
        return fail("vconfig_barrier_unsupported", "indirect helper call");

      StringRef Name = Callee->getName();
      if (Name.starts_with("llvm.nvvm.barrier")) {
        if (!isBarrier0Call(*CB))
          return fail("vconfig_barrier_unsupported",
                      "unsupported block barrier");
        ++Total;
        RequiresWarpAlignedBlock = true;
      } else if (isWarpSensitiveCallName(Name)) {
        RequiresWarpAlignedBlock = true;
      } else if (!Callee->isDeclaration()) {
        unsigned HelperCount = 0;
        if (!summarizeFunction(*Callee, HelperCount))
          return false;
        Total += HelperCount;
      }

      if (Total > std::numeric_limits<unsigned>::max())
        return fail("vconfig_barrier_unsupported",
                    "barrier summary overflow");
    }

    BarrierCount = static_cast<unsigned>(Total);
    return true;
  }

  bool summarizeFunction(Function &F, unsigned &BarrierCount) {
    auto Cached = FunctionSummaries.find(&F);
    if (Cached != FunctionSummaries.end()) {
      BarrierCount = Cached->second;
      return true;
    }
    if (!ActiveFunctions.insert(&F).second)
      return fail("vconfig_barrier_unsupported", "recursive helper call");

    SmallPtrSet<BasicBlock *, 32> Region;
    for (BasicBlock &BB : F)
      Region.insert(&BB);
    bool Success = summarizeRegion(&F.getEntryBlock(), Region, nullptr,
                                   nullptr, BarrierCount);
    ActiveFunctions.erase(&F);
    if (!Success)
      return false;
    FunctionSummaries[&F] = BarrierCount;
    return true;
  }

  bool summarizeRegion(BasicBlock *Entry,
                       const SmallPtrSetImpl<BasicBlock *> &Region,
                       BasicBlock *BackedgeSource,
                       BasicBlock *BackedgeTarget,
                       unsigned &BarrierCount) {
    DenseMap<BasicBlock *, unsigned> MemoizedBlocks;
    SmallPtrSet<BasicBlock *, 32> ActiveBlocks;
    return summarizeBlockPaths(Entry, Region, BackedgeSource, BackedgeTarget,
                               MemoizedBlocks, ActiveBlocks, BarrierCount);
  }

  StringRef unsupportedReason() const { return UnsupportedReason; }
  StringRef unsupportedDetail() const { return UnsupportedDetail; }

private:
  bool fail(StringRef Reason, StringRef Detail) {
    if (UnsupportedReason.empty()) {
      UnsupportedReason = Reason;
      UnsupportedDetail = Detail;
    }
    return false;
  }

  bool summarizeBlockPaths(
      BasicBlock *BB, const SmallPtrSetImpl<BasicBlock *> &Region,
      BasicBlock *BackedgeSource, BasicBlock *BackedgeTarget,
      DenseMap<BasicBlock *, unsigned> &MemoizedBlocks,
      SmallPtrSetImpl<BasicBlock *> &ActiveBlocks, unsigned &BarrierCount) {
    auto Cached = MemoizedBlocks.find(BB);
    if (Cached != MemoizedBlocks.end()) {
      BarrierCount = Cached->second;
      return true;
    }
    if (!ActiveBlocks.insert(BB).second)
      return fail("vconfig_barrier_unsupported",
                  "helper control-flow cycle");

    unsigned LocalCount = 0;
    if (!summarizeBlockCalls(*BB, LocalCount))
      return false;

    bool HasSuffix = false;
    unsigned CommonSuffix = 0;
    for (BasicBlock *Succ : successors(BB)) {
      unsigned Suffix = 0;
      bool IsBackedge = BB == BackedgeSource && Succ == BackedgeTarget;
      if (!IsBackedge && Region.contains(Succ) &&
          !summarizeBlockPaths(Succ, Region, BackedgeSource, BackedgeTarget,
                               MemoizedBlocks, ActiveBlocks, Suffix))
        return false;
      if (!HasSuffix) {
        CommonSuffix = Suffix;
        HasSuffix = true;
      } else if (CommonSuffix != Suffix) {
        return fail("vconfig_barrier_unsupported",
                    "barrier path summary mismatch");
      }
    }

    uint64_t Total = static_cast<uint64_t>(LocalCount) + CommonSuffix;
    if (Total > std::numeric_limits<unsigned>::max())
      return fail("vconfig_barrier_unsupported", "barrier summary overflow");
    ActiveBlocks.erase(BB);
    BarrierCount = static_cast<unsigned>(Total);
    MemoizedBlocks[BB] = BarrierCount;
    return true;
  }

  bool &RequiresWarpAlignedBlock;
  DenseMap<Function *, unsigned> FunctionSummaries;
  SmallPtrSet<Function *, 16> ActiveFunctions;
  StringRef UnsupportedReason;
  StringRef UnsupportedDetail;
};

static RapidContextSyncAction makeBarrierSyncAction(unsigned Count) {
  RapidContextSyncAction Action;
  Action.Kind = RapidContextSyncActionKind::Barrier;
  Action.Count = Count;
  return Action;
}

static RapidContextSyncAction makeFixedLoopSyncAction(unsigned TripCount,
                                                      unsigned BarrierCount) {
  RapidContextSyncAction Action;
  Action.Kind = RapidContextSyncActionKind::FixedLoop;
  Action.Count = BarrierCount;
  Action.TripCount = TripCount;
  return Action;
}

static RapidContextSyncAction
makeDynamicLoopSyncAction(const DynamicLoopSyncInfo &Info,
                          unsigned BarrierCount) {
  RapidContextSyncAction Action;
  Action.Kind = RapidContextSyncActionKind::DynamicLoop;
  Action.Count = BarrierCount;
  Action.Start = Info.Start;
  Action.Step = Info.Step;
  Action.Bound = Info.Bound;
  Action.EnterCondition = Info.EnterCondition;
  Action.Predicate = Info.Predicate;
  Action.UpdateKind = Info.UpdateKind;
  Action.EnterOnTrue = Info.EnterOnTrue;
  Action.ContinueOnTrue = Info.ContinueOnTrue;
  return Action;
}

static bool analyzeFixedTripSelfLoop(BasicBlock &BB, BasicBlock *&ExitBlock,
                                     unsigned &TripCount) {
  auto *Term = dyn_cast<BranchInst>(BB.getTerminator());
  if (!Term || !Term->isConditional())
    return false;
  if (Term->getSuccessor(0) != &BB)
    return false;
  ExitBlock = Term->getSuccessor(1);

  auto *Cmp = dyn_cast<ICmpInst>(Term->getCondition());
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_ULT)
    return false;
  auto *Bound = dyn_cast<ConstantInt>(Cmp->getOperand(1));
  if (!Bound || Bound->getBitWidth() > 32)
    return false;

  for (PHINode &Phi : BB.phis()) {
    ConstantInt *Start = nullptr;
    Value *LoopIncoming = nullptr;
    for (unsigned I = 0; I < Phi.getNumIncomingValues(); ++I) {
      BasicBlock *IncomingBlock = Phi.getIncomingBlock(I);
      Value *IncomingValue = Phi.getIncomingValue(I);
      if (IncomingBlock == &BB) {
        LoopIncoming = IncomingValue;
      } else {
        Start = dyn_cast<ConstantInt>(IncomingValue);
      }
    }
    if (!Start || !Start->isZero() || !LoopIncoming)
      continue;

    auto *Next = dyn_cast<BinaryOperator>(LoopIncoming);
    if (!Next || Next->getOpcode() != Instruction::Add)
      continue;
    auto *Step = dyn_cast<ConstantInt>(Next->getOperand(1));
    if (Next->getOperand(0) != &Phi || !Step || !Step->isOne())
      continue;
    if (Cmp->getOperand(0) != Next)
      continue;

    TripCount = static_cast<unsigned>(Bound->getZExtValue());
    return TripCount > 0;
  }

  return false;
}

static bool getSimpleFixedTripCount(Loop &L, unsigned &TripCount) {
  BasicBlock *Header = L.getHeader();
  BasicBlock *Latch = L.getLoopLatch();
  BasicBlock *Preheader = L.getLoopPreheader();
  if (!Header || !Latch || !Preheader)
    return false;

  auto *Branch = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!Branch || !Branch->isConditional())
    return false;
  bool ContinueOnTrue = Branch->getSuccessor(0) == Header;
  if (!ContinueOnTrue && Branch->getSuccessor(1) != Header)
    return false;

  if (auto *ConditionPhi = dyn_cast<PHINode>(Branch->getCondition())) {
    if (ConditionPhi->getParent() == Header) {
      auto *Start = dyn_cast<ConstantInt>(
          ConditionPhi->getIncomingValueForBlock(Preheader));
      auto *Next =
          dyn_cast<ConstantInt>(ConditionPhi->getIncomingValueForBlock(Latch));
      if (Start && Next && Start->getType()->isIntegerTy(1) &&
          Start->getZExtValue() == static_cast<uint64_t>(ContinueOnTrue) &&
          Next->getZExtValue() != static_cast<uint64_t>(ContinueOnTrue)) {
        TripCount = 2;
        return true;
      }
    }
  }

  auto *Cmp = dyn_cast<ICmpInst>(Branch->getCondition());
  if (!Cmp)
    return false;

  for (PHINode &Phi : Header->phis()) {
    auto *Start = dyn_cast<ConstantInt>(
        Phi.getIncomingValueForBlock(Preheader));
    Value *LatchIncoming = Phi.getIncomingValueForBlock(Latch);
    auto *Next = dyn_cast_or_null<BinaryOperator>(LatchIncoming);
    if (!Start || !Next || Next->getOpcode() != Instruction::Add)
      continue;

    ConstantInt *Step = nullptr;
    if (Next->getOperand(0) == &Phi)
      Step = dyn_cast<ConstantInt>(Next->getOperand(1));
    else if (Next->getOperand(1) == &Phi)
      Step = dyn_cast<ConstantInt>(Next->getOperand(0));
    if (!Step || Step->isZero() || Step->isNegative())
      continue;

    CmpInst::Predicate Predicate = Cmp->getPredicate();
    Value *Compared = Cmp->getOperand(0);
    auto *Bound = dyn_cast<ConstantInt>(Cmp->getOperand(1));
    if (Compared != Next && Compared != &Phi) {
      Compared = Cmp->getOperand(1);
      Bound = dyn_cast<ConstantInt>(Cmp->getOperand(0));
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    bool ComparesNext = Compared == Next;
    bool ComparesCurrent = Compared == &Phi;
    if ((!ComparesNext && !ComparesCurrent) || !Bound || Start->isNegative() ||
        Bound->isNegative())
      continue;

    uint64_t StartValue = Start->getZExtValue();
    uint64_t StepValue = Step->getZExtValue();
    uint64_t BoundValue = Bound->getZExtValue();
    if (BoundValue <= StartValue)
      continue;
    uint64_t Distance = BoundValue - StartValue;
    uint64_t Trips = 0;

    if ((Predicate == ICmpInst::ICMP_ULT ||
         Predicate == ICmpInst::ICMP_SLT) &&
        ContinueOnTrue) {
      Trips = (Distance + StepValue - 1) / StepValue;
      if (ComparesCurrent)
        ++Trips;
    } else if (((Predicate == ICmpInst::ICMP_EQ && !ContinueOnTrue) ||
                (Predicate == ICmpInst::ICMP_NE && ContinueOnTrue)) &&
               ComparesNext && Distance % StepValue == 0) {
      Trips = Distance / StepValue;
    }

    if (Trips == 0 || Trips > std::numeric_limits<unsigned>::max())
      continue;
    TripCount = static_cast<unsigned>(Trips);
    return true;
  }
  return false;
}

static bool getSimpleDynamicLoopSyncInfo(Loop &L,
                                         DynamicLoopSyncInfo &Info) {
  BasicBlock *Header = L.getHeader();
  BasicBlock *Latch = L.getLoopLatch();
  BasicBlock *Preheader = L.getLoopPreheader();
  BasicBlock *Exit = L.getExitBlock();
  if (!Header || !Latch || !Exit || L.getParentLoop())
    return false;

  auto *LatchBranch = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBranch || !LatchBranch->isConditional())
    return false;
  bool ContinueOnTrue = LatchBranch->getSuccessor(0) == Header;
  if (!ContinueOnTrue && LatchBranch->getSuccessor(1) != Header)
    return false;

  auto *Cmp = dyn_cast<ICmpInst>(LatchBranch->getCondition());
  if (!Cmp)
    return false;

  BasicBlock *EntryGuard = nullptr;
  BranchInst *EntryBranch = nullptr;
  if (Preheader) {
    EntryGuard = Preheader->getSinglePredecessor();
    EntryBranch =
        EntryGuard ? dyn_cast<BranchInst>(EntryGuard->getTerminator())
                   : nullptr;
  } else {
    Preheader = L.getLoopPredecessor();
    EntryGuard = Preheader;
    EntryBranch =
        EntryGuard ? dyn_cast<BranchInst>(EntryGuard->getTerminator())
                   : nullptr;
  }
  if (!EntryBranch || !EntryBranch->isConditional())
    return false;
  bool EnterOnTrue = EntryBranch->getSuccessor(0) == Header ||
                     EntryBranch->getSuccessor(0) == Preheader;
  if (!EnterOnTrue && EntryBranch->getSuccessor(1) != Header &&
      EntryBranch->getSuccessor(1) != Preheader)
    return false;
  if (EntryBranch->getSuccessor(EnterOnTrue ? 1 : 0) != Exit)
    return false;
  SmallPtrSet<Value *, 16> EntryVisited;
  if (!isBlockUniformValue(EntryBranch->getCondition(), EntryVisited))
    return false;

  for (PHINode &Phi : Header->phis()) {
    Value *Start = Phi.getIncomingValueForBlock(Preheader);
    Value *LatchIncoming = Phi.getIncomingValueForBlock(Latch);
    auto *Next = dyn_cast_or_null<BinaryOperator>(LatchIncoming);
    if (!Start || !Next)
      continue;

    SmallPtrSet<Value *, 16> StartVisited;
    if (!isBlockUniformValue(Start, StartVisited))
      continue;

    ConstantInt *Step = nullptr;
    DynamicLoopUpdateKind UpdateKind = DynamicLoopUpdateKind::Add;
    if (Next->getOpcode() == Instruction::Add) {
      if (Next->getOperand(0) == &Phi)
        Step = dyn_cast<ConstantInt>(Next->getOperand(1));
      else if (Next->getOperand(1) == &Phi)
        Step = dyn_cast<ConstantInt>(Next->getOperand(0));
      UpdateKind = DynamicLoopUpdateKind::Add;
    } else if (Next->getOpcode() == Instruction::Shl) {
      if (Next->getOperand(0) != &Phi)
        continue;
      Step = dyn_cast<ConstantInt>(Next->getOperand(1));
      UpdateKind = DynamicLoopUpdateKind::Shl;
    } else if (Next->getOpcode() == Instruction::Mul) {
      if (Next->getOperand(0) == &Phi)
        Step = dyn_cast<ConstantInt>(Next->getOperand(1));
      else if (Next->getOperand(1) == &Phi)
        Step = dyn_cast<ConstantInt>(Next->getOperand(0));
      UpdateKind = DynamicLoopUpdateKind::Mul;
    }
    if (!Step || Step->isZero() || Step->isNegative())
      continue;
    if (UpdateKind == DynamicLoopUpdateKind::Mul && Step->isOne())
      continue;

    CmpInst::Predicate Predicate = Cmp->getPredicate();
    Value *Compared = Cmp->getOperand(0);
    Value *Bound = Cmp->getOperand(1);
    if (Compared != Next) {
      Compared = Cmp->getOperand(1);
      Bound = Cmp->getOperand(0);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    bool IsLessThan = Predicate == ICmpInst::ICMP_ULT ||
                      Predicate == ICmpInst::ICMP_SLT;
    bool IsEqualityExit =
        (Predicate == ICmpInst::ICMP_EQ && !ContinueOnTrue) ||
        (Predicate == ICmpInst::ICMP_NE && ContinueOnTrue);
    if (Compared != Next || (!IsLessThan && !IsEqualityExit))
      continue;

    SmallPtrSet<Value *, 16> BoundVisited;
    if (!isBlockUniformValue(Bound, BoundVisited))
      continue;

    Info.Target = &L;
    Info.Start = Start;
    Info.Step = Step;
    Info.Bound = Bound;
    Info.EnterCondition = EntryBranch->getCondition();
    Info.Predicate = Predicate;
    Info.UpdateKind = UpdateKind;
    Info.EnterOnTrue = EnterOnTrue;
    Info.ContinueOnTrue = ContinueOnTrue;
    return true;
  }
  return false;
}

static bool buildDynamicLoopSyncPlan(
    Function &Entry, LoopInfo &Loops, DominatorTree &Dominators,
    PostDominatorTree &PostDominators,
    SmallVectorImpl<RapidContextSyncAction> &Actions) {
  DynamicLoopSyncInfo DynamicInfo;
  uint64_t DynamicBarrierCount = 0;
  unsigned PreBarrierCount = 0;
  unsigned PostBarrierCount = 0;

  for (Instruction &I : instructions(Entry)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || !isBarrier0Call(*CB))
      continue;

    BasicBlock *BarrierBlock = CB->getParent();
    Loop *ContainingLoop = Loops.getLoopFor(BarrierBlock);
    if (!ContainingLoop)
      continue;

    uint64_t ExecutionsPerDynamicIteration = 1;
    Loop *Current = ContainingLoop;
    for (; Current; Current = Current->getParentLoop()) {
      BasicBlock *Latch = Current->getLoopLatch();
      if (!Latch || !Dominators.dominates(BarrierBlock, Latch))
        return false;

      unsigned FixedTripCount = 0;
      if (getSimpleFixedTripCount(*Current, FixedTripCount)) {
        if (ExecutionsPerDynamicIteration >
            std::numeric_limits<unsigned>::max() / FixedTripCount)
          return false;
        ExecutionsPerDynamicIteration *= FixedTripCount;
        continue;
      }

      DynamicLoopSyncInfo Candidate;
      if (!getSimpleDynamicLoopSyncInfo(*Current, Candidate))
        return false;
      if (DynamicInfo.Target && DynamicInfo.Target != Candidate.Target)
        return false;
      DynamicInfo = Candidate;
      break;
    }
    if (!DynamicInfo.Target || !Current || Current != DynamicInfo.Target)
      return false;

    DynamicBarrierCount += ExecutionsPerDynamicIteration;
    if (DynamicBarrierCount > std::numeric_limits<unsigned>::max())
      return false;
  }

  if (!DynamicInfo.Target || DynamicBarrierCount == 0)
    return false;

  BasicBlock *Header = DynamicInfo.Target->getHeader();
  BasicBlock *Exit = DynamicInfo.Target->getExitBlock();
  if (!Header || !Exit)
    return false;
  for (Instruction &I : instructions(Entry)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || !isBarrier0Call(*CB) || Loops.getLoopFor(CB->getParent()))
      continue;
    BasicBlock *BarrierBlock = CB->getParent();
    if (Dominators.dominates(BarrierBlock, Header)) {
      ++PreBarrierCount;
    } else if (Dominators.dominates(Exit, BarrierBlock) &&
               PostDominators.dominates(BarrierBlock, Exit)) {
      ++PostBarrierCount;
    } else {
      return false;
    }
  }

  if (PreBarrierCount)
    Actions.push_back(makeBarrierSyncAction(PreBarrierCount));
  Actions.push_back(makeDynamicLoopSyncAction(
      DynamicInfo, static_cast<unsigned>(DynamicBarrierCount)));
  if (PostBarrierCount)
    Actions.push_back(makeBarrierSyncAction(PostBarrierCount));
  return true;
}

static bool countFixedLoopBarrierExecutions(Function &Entry, LoopInfo &Loops,
                                            DominatorTree &Dominators,
                                            unsigned AcyclicBarrierCount,
                                            unsigned &TotalBarrierCount) {
  uint64_t Total = AcyclicBarrierCount;
  for (Instruction &I : instructions(Entry)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || !isBarrier0Call(*CB))
      continue;

    BasicBlock *BarrierBlock = CB->getParent();
    Loop *ContainingLoop = Loops.getLoopFor(BarrierBlock);
    if (!ContainingLoop)
      continue;

    uint64_t Executions = 1;
    for (Loop *Current = ContainingLoop; Current;
         Current = Current->getParentLoop()) {
      unsigned TripCount = 0;
      BasicBlock *Latch = Current->getLoopLatch();
      if (!Latch || !Dominators.dominates(BarrierBlock, Latch) ||
          !getSimpleFixedTripCount(*Current, TripCount))
        return false;
      if (Executions > std::numeric_limits<unsigned>::max() / TripCount)
        return false;
      Executions *= TripCount;
    }

    Total += Executions;
    if (Total > std::numeric_limits<unsigned>::max())
      return false;
  }

  TotalBarrierCount = static_cast<unsigned>(Total);
  return TotalBarrierCount > 0;
}

static AllocaInst *getLoadedAlloca(Value *V) {
  auto *Load = dyn_cast<LoadInst>(V);
  if (!Load)
    return nullptr;
  return dyn_cast<AllocaInst>(Load->getPointerOperand()->stripPointerCasts());
}

static bool isStoreToAlloca(const StoreInst &Store, const AllocaInst *Alloca) {
  return Store.getPointerOperand()->stripPointerCasts() == Alloca;
}

static bool isZeroStoreToAlloca(const StoreInst &Store,
                                const AllocaInst *Alloca) {
  if (!isStoreToAlloca(Store, Alloca))
    return false;
  auto *Stored = dyn_cast<ConstantInt>(Store.getValueOperand());
  return Stored && Stored->isZero();
}

static bool isOneIncrementStoreToAlloca(const StoreInst &Store,
                                        const AllocaInst *Alloca) {
  if (!isStoreToAlloca(Store, Alloca))
    return false;

  auto *Add = dyn_cast<BinaryOperator>(Store.getValueOperand());
  if (!Add || Add->getOpcode() != Instruction::Add)
    return false;

  auto isLoadFromAlloca = [&](Value *V) {
    auto *Load = dyn_cast<LoadInst>(V);
    return Load && Load->getPointerOperand()->stripPointerCasts() == Alloca;
  };
  auto isOne = [](Value *V) {
    auto *Const = dyn_cast<ConstantInt>(V);
    return Const && Const->isOne();
  };

  return (isLoadFromAlloca(Add->getOperand(0)) && isOne(Add->getOperand(1))) ||
         (isLoadFromAlloca(Add->getOperand(1)) && isOne(Add->getOperand(0)));
}

static bool blockHasZeroStoreToAlloca(BasicBlock &BB,
                                      const AllocaInst *Alloca) {
  for (Instruction &I : BB) {
    auto *Store = dyn_cast<StoreInst>(&I);
    if (Store && isZeroStoreToAlloca(*Store, Alloca))
      return true;
  }
  return false;
}

static bool blockHasOneIncrementStoreToAlloca(BasicBlock &BB,
                                              const AllocaInst *Alloca) {
  for (Instruction &I : BB) {
    auto *Store = dyn_cast<StoreInst>(&I);
    if (Store && isOneIncrementStoreToAlloca(*Store, Alloca))
      return true;
  }
  return false;
}

static bool allIndexStoresAreExpected(Function &F, const AllocaInst *Alloca,
                                      BasicBlock &Preheader,
                                      BasicBlock &Latch) {
  for (Instruction &I : instructions(F)) {
    auto *Store = dyn_cast<StoreInst>(&I);
    if (!Store || !isStoreToAlloca(*Store, Alloca))
      continue;

    if (Store->getParent() == &Preheader &&
        isZeroStoreToAlloca(*Store, Alloca))
      continue;
    if (Store->getParent() == &Latch &&
        isOneIncrementStoreToAlloca(*Store, Alloca))
      continue;
    return false;
  }
  return true;
}

static bool collectLoopBlocks(BasicBlock *BodyEntry, BasicBlock *Header,
                              BasicBlock *ExitBlock,
                              SmallVectorImpl<BasicBlock *> &LoopBlocks) {
  SmallVector<BasicBlock *, 8> Worklist;
  SmallPtrSet<BasicBlock *, 16> Visited;
  Worklist.push_back(BodyEntry);

  while (!Worklist.empty()) {
    BasicBlock *Current = Worklist.pop_back_val();
    if (Current == Header || Current == ExitBlock)
      return false;
    if (!Visited.insert(Current).second)
      continue;
    LoopBlocks.push_back(Current);

    for (BasicBlock *Succ : successors(Current)) {
      if (Succ == Header)
        continue;
      if (Succ == ExitBlock)
        return false;
      Worklist.push_back(Succ);
    }
  }

  return !LoopBlocks.empty();
}

static bool analyzeFixedTripAllocaLoop(BasicBlock &Header,
                                       BasicBlock *&BodyEntry,
                                       BasicBlock *&ExitBlock,
                                       unsigned &TripCount,
                                       SmallVectorImpl<BasicBlock *> &LoopBlocks) {
  auto *Term = dyn_cast<BranchInst>(Header.getTerminator());
  if (!Term || !Term->isConditional())
    return false;

  auto *Cmp = dyn_cast<ICmpInst>(Term->getCondition());
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_ULT)
    return false;

  AllocaInst *IndexAlloca = getLoadedAlloca(Cmp->getOperand(0));
  auto *Bound = dyn_cast<ConstantInt>(Cmp->getOperand(1));
  if (!IndexAlloca || !Bound || Bound->getBitWidth() > 32)
    return false;

  BodyEntry = Term->getSuccessor(0);
  ExitBlock = Term->getSuccessor(1);
  if (BodyEntry == &Header || ExitBlock == &Header || BodyEntry == ExitBlock)
    return false;

  LoopBlocks.clear();
  if (!collectLoopBlocks(BodyEntry, &Header, ExitBlock, LoopBlocks))
    return false;

  BasicBlock *Latch = nullptr;
  for (BasicBlock *LoopBlock : LoopBlocks) {
    if (!is_contained(successors(LoopBlock), &Header))
      continue;
    if (Latch)
      return false;
    Latch = LoopBlock;
  }
  if (!Latch)
    return false;

  BasicBlock *Preheader = nullptr;
  for (BasicBlock *Pred : predecessors(&Header)) {
    if (Pred == Latch)
      continue;
    if (Preheader)
      return false;
    Preheader = Pred;
  }
  if (!Preheader)
    return false;

  if (!blockHasZeroStoreToAlloca(*Preheader, IndexAlloca))
    return false;
  if (!blockHasOneIncrementStoreToAlloca(*Latch, IndexAlloca))
    return false;
  if (!allIndexStoresAreExpected(*Header.getParent(), IndexAlloca, *Preheader,
                                 *Latch))
    return false;

  TripCount = static_cast<unsigned>(Bound->getZExtValue());
  return TripCount > 0;
}

static bool reachableTailHasBarrier(
    BasicBlock *Start,
    const DenseMap<BasicBlock *, unsigned> &BlockBarrierCounts,
    SmallPtrSetImpl<BasicBlock *> &VisitedBlocks) {
  SmallVector<BasicBlock *, 8> Worklist;
  Worklist.push_back(Start);

  while (!Worklist.empty()) {
    BasicBlock *Current = Worklist.pop_back_val();
    if (!VisitedBlocks.insert(Current).second)
      continue;
    if (BlockBarrierCounts.lookup(Current) > 0)
      return true;

    for (BasicBlock *Succ : successors(Current))
      Worklist.push_back(Succ);
  }

  return false;
}

static bool scanLoopBodySyncCalls(ArrayRef<BasicBlock *> LoopBlocks,
                                  BasicBlock *BodyEntry,
                                  const DenseMap<BasicBlock *, unsigned>
                                      &BlockBarrierCounts,
                                  unsigned &BarrierCount) {
  for (BasicBlock *BB : LoopBlocks) {
    unsigned Count = BlockBarrierCounts.lookup(BB);
    if (Count == 0)
      continue;
    if (BB != BodyEntry ||
        BarrierCount > std::numeric_limits<unsigned>::max() - Count)
      return false;
    BarrierCount += Count;
  }
  return true;
}

static RapidContextSyncPlan analyzeRapidContextSync(Function &Entry) {
  RapidContextSyncPlan Plan;
  bool HasBarrier = false;
  DenseMap<BasicBlock *, unsigned> BlockBarrierCounts;
  SyncPathSummaryAnalyzer SummaryAnalyzer(Plan.RequiresWarpAlignedBlock);
  for (BasicBlock &BB : Entry) {
    unsigned Count = 0;
    if (!SummaryAnalyzer.summarizeBlockCalls(BB, Count)) {
      Plan.UnsupportedReason = SummaryAnalyzer.unsupportedReason();
      Plan.UnsupportedDetail = SummaryAnalyzer.unsupportedDetail();
      return Plan;
    }
    BlockBarrierCounts[&BB] = Count;
    HasBarrier |= Count > 0;
  }

  if (!HasBarrier)
    return Plan;

  DominatorTree Dominators(Entry);
  LoopInfo Loops(Dominators);
  PostDominatorTree PostDominators(Entry);
  unsigned AcyclicBarrierCount = 0;
  bool HasLoopBarrier = false;
  bool AllBarriersPostDominateEntry = true;
  for (BasicBlock &BB : Entry) {
    unsigned Count = BlockBarrierCounts.lookup(&BB);
    if (Count == 0)
      continue;

    if (Loops.getLoopFor(&BB)) {
      HasLoopBarrier = true;
      continue;
    }
    if (!PostDominators.dominates(&BB, &Entry.getEntryBlock()))
      AllBarriersPostDominateEntry = false;
    if (AcyclicBarrierCount > std::numeric_limits<unsigned>::max() - Count) {
      Plan.UnsupportedReason = "vconfig_barrier_unsupported";
      Plan.UnsupportedDetail = "barrier summary overflow";
      return Plan;
    }
    AcyclicBarrierCount += Count;
  }

  if (!HasLoopBarrier) {
    if (AllBarriersPostDominateEntry) {
      Plan.InactiveSyncActions.push_back(
          makeBarrierSyncAction(AcyclicBarrierCount));
      return Plan;
    }

    unsigned RegionBarrierCount = 0;
    Plan.GuardBlock = findUniformEarlyReturnGuard(
        Entry, PostDominators, Loops, RegionBarrierCount);
    if (Plan.GuardBlock) {
      Plan.InactiveSyncActions.push_back(
          makeBarrierSyncAction(RegionBarrierCount));
      return Plan;
    }

    Plan.UnsupportedReason = "vconfig_barrier_unsupported";
    return Plan;
  }

  if (AllBarriersPostDominateEntry) {
    Loop *OnlyBarrierLoop = nullptr;
    bool OnlySimpleSelfLoopBarriers = true;
    for (Instruction &I : instructions(Entry)) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || !isBarrier0Call(*CB))
        continue;
      Loop *ContainingLoop = Loops.getLoopFor(CB->getParent());
      if (!ContainingLoop)
        continue;
      if (ContainingLoop->getNumBlocks() != 1 ||
          (OnlyBarrierLoop && OnlyBarrierLoop != ContainingLoop)) {
        OnlySimpleSelfLoopBarriers = false;
        break;
      }
      OnlyBarrierLoop = ContainingLoop;
    }

    if (!OnlySimpleSelfLoopBarriers) {
      unsigned TotalBarrierCount = 0;
      if (countFixedLoopBarrierExecutions(Entry, Loops, Dominators,
                                          AcyclicBarrierCount,
                                          TotalBarrierCount)) {
        Plan.InactiveSyncActions.push_back(
            makeBarrierSyncAction(TotalBarrierCount));
        return Plan;
      }
    }
  }

  if (buildDynamicLoopSyncPlan(Entry, Loops, Dominators, PostDominators,
                               Plan.InactiveSyncActions))
    return Plan;

  SmallPtrSet<BasicBlock *, 16> VisitedBlocks;
  BasicBlock *Current = &Entry.getEntryBlock();
  while (Current) {
    if (!VisitedBlocks.insert(Current).second) {
      Plan.UnsupportedReason = "vconfig_barrier_unsupported";
      return Plan;
    }

    BasicBlock *LoopExit = nullptr;
    unsigned LoopTripCount = 0;
    if (analyzeFixedTripSelfLoop(*Current, LoopExit, LoopTripCount)) {
      unsigned LoopBarrierCount = BlockBarrierCounts.lookup(Current);
      if (LoopBarrierCount > 0) {
        Plan.InactiveSyncActions.push_back(
            makeFixedLoopSyncAction(LoopTripCount, LoopBarrierCount));
      }
      Current = LoopExit;
      continue;
    }

    BasicBlock *LoopBodyEntry = nullptr;
    SmallVector<BasicBlock *, 16> LoopBlocks;
    if (analyzeFixedTripAllocaLoop(*Current, LoopBodyEntry, LoopExit,
                                   LoopTripCount, LoopBlocks)) {
      unsigned LoopBarrierCount = 0;
      if (!scanLoopBodySyncCalls(LoopBlocks, LoopBodyEntry,
                                 BlockBarrierCounts, LoopBarrierCount)) {
        Plan.UnsupportedReason = "vconfig_barrier_unsupported";
        return Plan;
      }
      if (LoopBarrierCount > 0) {
        Plan.InactiveSyncActions.push_back(
            makeFixedLoopSyncAction(LoopTripCount, LoopBarrierCount));
      }
      Current = LoopExit;
      continue;
    }

    unsigned BarrierCount = BlockBarrierCounts.lookup(Current);
    if (BarrierCount > 0)
      Plan.InactiveSyncActions.push_back(makeBarrierSyncAction(BarrierCount));

    Instruction *Term = Current->getTerminator();
    if (isa<ReturnInst>(Term))
      return Plan;
    auto *Branch = dyn_cast<BranchInst>(Term);
    if (Branch && Branch->isUnconditional()) {
      Current = Branch->getSuccessor(0);
      continue;
    }
    if (Branch && Branch->isConditional()) {
      SmallPtrSet<BasicBlock *, 16> TailBlocks;
      for (BasicBlock *Succ : successors(Current)) {
        if (reachableTailHasBarrier(Succ, BlockBarrierCounts, TailBlocks)) {
          Plan.UnsupportedReason = "vconfig_barrier_unsupported";
          return Plan;
        }
      }
      return Plan;
    }

    Plan.UnsupportedReason = "vconfig_barrier_unsupported";
    return Plan;
  }

  return Plan;
}

static Value *cloneUniformValue(Value *Original, IRBuilder<> &B,
                                DenseMap<Value *, Value *> &Cloned) {
  if (isa<Constant>(Original) || isa<Argument>(Original))
    return Original;
  if (auto It = Cloned.find(Original); It != Cloned.end())
    return It->second;

  auto *InstructionValue = dyn_cast<Instruction>(Original);
  if (!InstructionValue || isa<PHINode>(InstructionValue) ||
      isa<LoadInst>(InstructionValue) || InstructionValue->mayHaveSideEffects())
    return nullptr;
  if (auto *CB = dyn_cast<CallBase>(InstructionValue)) {
    Function *Callee = CB->getCalledFunction();
    if (!Callee)
      return nullptr;
    StringRef Name = Callee->getName();
    if (!isBlockUniformIntrinsicName(Name) &&
        !Name.starts_with("llvm.nvvm.read.ptx.sreg.ctaid.") &&
        !Name.starts_with("llvm.nvvm.read.ptx.sreg.nctaid.") &&
        !Name.starts_with("llvm.nvvm.read.ptx.sreg.ntid."))
      return nullptr;
  }

  Instruction *Copy = InstructionValue->clone();
  for (unsigned Index = 0; Index < Copy->getNumOperands(); ++Index) {
    Value *Operand = cloneUniformValue(Copy->getOperand(Index), B, Cloned);
    if (!Operand) {
      Copy->deleteValue();
      return nullptr;
    }
    Copy->setOperand(Index, Operand);
  }
  B.Insert(Copy,
           InstructionValue->hasName()
               ? Twine(InstructionValue->getName()) + ".vdim.inactive"
               : Twine());
  Cloned[Original] = Copy;
  return Copy;
}

static void emitInactiveSyncActions(Function &F, BasicBlock *InactiveBB,
                                    BasicBlock *InsertBefore,
                                    ArrayRef<RapidContextSyncAction> Actions) {
  IRBuilder<> B(InactiveBB);
  Function *Barrier0 = nullptr;
  auto emitBarrierCalls = [&](IRBuilder<> &Builder, unsigned Count) {
    if (!Barrier0)
      Barrier0 = getOrDeclareBarrier0(*F.getParent());
    for (unsigned I = 0; I < Count; ++I)
      Builder.CreateCall(Barrier0, {Builder.getInt32(0)});
  };

  for (const RapidContextSyncAction &Action : Actions) {
    if (Action.Kind == RapidContextSyncActionKind::Barrier) {
      emitBarrierCalls(B, Action.Count);
      continue;
    }

    if (Action.Kind == RapidContextSyncActionKind::FixedLoop) {
      BasicBlock *LoopBB =
        BasicBlock::Create(F.getContext(), "vdim.inactive.sync.loop", &F,
                           InsertBefore);
      BasicBlock *AfterBB =
        BasicBlock::Create(F.getContext(), "vdim.inactive.sync.after", &F,
                           InsertBefore);
      B.CreateBr(LoopBB);

      IRBuilder<> LoopBuilder(LoopBB);
      PHINode *Index = LoopBuilder.CreatePHI(
          Type::getInt32Ty(F.getContext()), 2, "vdim.inactive.sync.i");
      Index->addIncoming(B.getInt32(0), B.GetInsertBlock());
      emitBarrierCalls(LoopBuilder, Action.Count);
      Value *Next = LoopBuilder.CreateAdd(Index, LoopBuilder.getInt32(1),
                                          "vdim.inactive.sync.next");
      Index->addIncoming(Next, LoopBB);
      Value *KeepGoing = LoopBuilder.CreateICmpULT(
          Next, LoopBuilder.getInt32(Action.TripCount),
          "vdim.inactive.sync.keep_going");
      LoopBuilder.CreateCondBr(KeepGoing, LoopBB, AfterBB);

      B.SetInsertPoint(AfterBB);
      continue;
    }

    BasicBlock *HeaderBB = BasicBlock::Create(
        F.getContext(), "vdim.inactive.sync.dynamic.header", &F, InsertBefore);
    BasicBlock *LoopBB = BasicBlock::Create(
        F.getContext(), "vdim.inactive.sync.dynamic.loop", &F, InsertBefore);
    BasicBlock *AfterBB = BasicBlock::Create(
        F.getContext(), "vdim.inactive.sync.dynamic.after", &F, InsertBefore);
    B.CreateBr(HeaderBB);

    IRBuilder<> HeaderBuilder(HeaderBB);
    DenseMap<Value *, Value *> Cloned;
    Value *EnterCondition =
        cloneUniformValue(Action.EnterCondition, HeaderBuilder, Cloned);
    Value *Start = cloneUniformValue(Action.Start, HeaderBuilder, Cloned);
    Value *Step = cloneUniformValue(Action.Step, HeaderBuilder, Cloned);
    Value *Bound = cloneUniformValue(Action.Bound, HeaderBuilder, Cloned);
    if (!EnterCondition || !Start || !Step || !Bound) {
      HeaderBuilder.CreateBr(AfterBB);
    } else {
      HeaderBuilder.CreateCondBr(
          EnterCondition, Action.EnterOnTrue ? LoopBB : AfterBB,
          Action.EnterOnTrue ? AfterBB : LoopBB);

      IRBuilder<> LoopBuilder(LoopBB);
      PHINode *Index = LoopBuilder.CreatePHI(
          Start->getType(), 2, "vdim.inactive.sync.dynamic.i");
      Index->addIncoming(Start, HeaderBB);
      emitBarrierCalls(LoopBuilder, Action.Count);
      Value *Next = nullptr;
      switch (Action.UpdateKind) {
      case DynamicLoopUpdateKind::Add:
        Next = LoopBuilder.CreateAdd(
            Index, Step, "vdim.inactive.sync.dynamic.next");
        break;
      case DynamicLoopUpdateKind::Shl:
        Next = LoopBuilder.CreateShl(
            Index, Step, "vdim.inactive.sync.dynamic.next");
        break;
      case DynamicLoopUpdateKind::Mul:
        Next = LoopBuilder.CreateMul(
            Index, Step, "vdim.inactive.sync.dynamic.next");
        break;
      }
      Index->addIncoming(Next, LoopBB);
      Value *KeepGoing = LoopBuilder.CreateICmp(
          Action.Predicate, Next, Bound,
          "vdim.inactive.sync.dynamic.keep_going");
      LoopBuilder.CreateCondBr(
          KeepGoing, Action.ContinueOnTrue ? LoopBB : AfterBB,
          Action.ContinueOnTrue ? AfterBB : LoopBB);
    }

    B.SetInsertPoint(AfterBB);
  }

  B.CreateRetVoid();
}

static bool insertActiveGuard(Function &F, const VDimAnnotation &Ann,
                              VDimValues &VD,
                              ArrayRef<RapidContextSyncAction> InactiveActions =
                                  {},
                              bool RequiresWarpAlignedBlock = false,
                              BasicBlock *DelayedGuardBlock = nullptr) {
  if (F.isDeclaration())
    return false;
  if (!F.getReturnType()->isVoidTy())
    return false;

  LLVMContext &Ctx = F.getContext();
  Argument *GridArg = nullptr;
  Argument *BlockArg = nullptr;
  Argument *ContextArg = nullptr;
  StructType *DimTy = nullptr;
  if (Ann.Mode == VDimMode::Args) {
    if (!getDim3Args(F, GridArg, BlockArg, DimTy))
      return false;
  } else if (Ann.Mode == VDimMode::RapidContext) {
    if (F.arg_empty() || !F.getArg(F.arg_size() - 1)->getType()->isPointerTy())
      return false;
    ContextArg = F.getArg(F.arg_size() - 1);
  }

  BasicBlock *Entry = &F.getEntryBlock();
  Instruction *SplitPt = &*Entry->getFirstInsertionPt();
  BasicBlock *PrefixBB = Entry->splitBasicBlock(
      SplitPt, DelayedGuardBlock ? "vdim.prefix" : "vdim.active");
  BasicBlock *GuardBB = Entry;
  BasicBlock *ActiveBB = PrefixBB;
  if (DelayedGuardBlock) {
    GuardBB = DelayedGuardBlock;
    Instruction *GuardSplitPt = &*GuardBB->getFirstInsertionPt();
    ActiveBB = GuardBB->splitBasicBlock(GuardSplitPt, "vdim.active");
  }
  BasicBlock *InactiveBB = BasicBlock::Create(Ctx, "vdim.inactive", &F, ActiveBB);

  Instruction *OldTerm = Entry->getTerminator();
  Instruction *GuardOldTerm = GuardBB->getTerminator();
  IRBuilder<> B(OldTerm);

  if (Ann.Mode == VDimMode::Const) {
    VD.Gx = getConstI32(Ctx, Ann.Const.Gx);
    VD.Gy = getConstI32(Ctx, Ann.Const.Gy);
    VD.Gz = getConstI32(Ctx, Ann.Const.Gz);
    VD.Bx = getConstI32(Ctx, Ann.Const.Bx);
    VD.By = getConstI32(Ctx, Ann.Const.By);
    VD.Bz = getConstI32(Ctx, Ann.Const.Bz);
  } else if (Ann.Mode == VDimMode::Args) {
    auto *I32 = Type::getInt32Ty(Ctx);
    IRBuilder<> BE(&*Entry->getFirstInsertionPt());
    AllocaInst *GridFallback = BE.CreateAlloca(DimTy);
    GridFallback->setAlignment(Align(4));
    AllocaInst *BlockFallback = BE.CreateAlloca(DimTy);
    BlockFallback->setAlignment(Align(4));

    auto storeFallback = [&](AllocaInst *Dst, StringRef XName, StringRef YName,
                             StringRef ZName) {
      Value *X = readSReg(B, *F.getParent(), XName);
      Value *Y = readSReg(B, *F.getParent(), YName);
      Value *Z = readSReg(B, *F.getParent(), ZName);
      markPhysical(X);
      markPhysical(Y);
      markPhysical(Z);
      Value *Dst0 = B.CreateStructGEP(DimTy, Dst, 0);
      Value *Dst1 = B.CreateStructGEP(DimTy, Dst, 1);
      Value *Dst2 = B.CreateStructGEP(DimTy, Dst, 2);
      B.CreateStore(X, Dst0);
      B.CreateStore(Y, Dst1);
      B.CreateStore(Z, Dst2);
    };

    storeFallback(GridFallback, "llvm.nvvm.read.ptx.sreg.nctaid.x",
                  "llvm.nvvm.read.ptx.sreg.nctaid.y",
                  "llvm.nvvm.read.ptx.sreg.nctaid.z");
    storeFallback(BlockFallback, "llvm.nvvm.read.ptx.sreg.ntid.x",
                  "llvm.nvvm.read.ptx.sreg.ntid.y",
                  "llvm.nvvm.read.ptx.sreg.ntid.z");

    auto *GridPtrTy = cast<PointerType>(GridArg->getType());
    auto *BlockPtrTy = cast<PointerType>(BlockArg->getType());
    Value *GridIsNull =
        B.CreateICmpEQ(GridArg, ConstantPointerNull::get(GridPtrTy));
    Value *BlockIsNull =
        B.CreateICmpEQ(BlockArg, ConstantPointerNull::get(BlockPtrTy));
    Value *GridPtr = B.CreateSelect(GridIsNull, (Value *)GridFallback, GridArg);
    Value *BlockPtr =
        B.CreateSelect(BlockIsNull, (Value *)BlockFallback, BlockArg);

    auto loadDim = [&](Value *Ptr, unsigned Idx) -> Value * {
      Value *Field = B.CreateStructGEP(DimTy, Ptr, Idx);
      return B.CreateLoad(I32, Field);
    };
    VD.Gx = loadDim(GridPtr, 0);
    VD.Gy = loadDim(GridPtr, 1);
    VD.Gz = loadDim(GridPtr, 2);
    VD.Bx = loadDim(BlockPtr, 0);
    VD.By = loadDim(BlockPtr, 1);
    VD.Bz = loadDim(BlockPtr, 2);
  } else {
    auto *I8 = Type::getInt8Ty(Ctx);
    auto *I32 = Type::getInt32Ty(Ctx);
    auto *ArrayTy = ArrayType::get(I32, 6);
    IRBuilder<> BE(&*Entry->getFirstInsertionPt());
    AllocaInst *Fallback = BE.CreateAlloca(ArrayTy, nullptr, "vdim.fallback");
    Fallback->setAlignment(Align(4));

    auto storePhysical = [&](unsigned Index, StringRef Name) {
      Value *Physical = readSReg(B, *F.getParent(), Name);
      markPhysical(Physical);
      Value *Field = B.CreateGEP(
          ArrayTy, Fallback, {B.getInt32(0), B.getInt32(Index)});
      B.CreateStore(Physical, Field);
    };
    storePhysical(0, "llvm.nvvm.read.ptx.sreg.nctaid.x");
    storePhysical(1, "llvm.nvvm.read.ptx.sreg.nctaid.y");
    storePhysical(2, "llvm.nvvm.read.ptx.sreg.nctaid.z");
    storePhysical(3, "llvm.nvvm.read.ptx.sreg.ntid.x");
    storePhysical(4, "llvm.nvvm.read.ptx.sreg.ntid.y");
    storePhysical(5, "llvm.nvvm.read.ptx.sreg.ntid.z");

    auto *ContextPtrTy = cast<PointerType>(ContextArg->getType());
    Value *ContextIsNull =
        B.CreateICmpEQ(ContextArg, ConstantPointerNull::get(ContextPtrTy));
    Value *ConfigPtr = B.CreateSelect(ContextIsNull, (Value *)Fallback,
                                      ContextArg, "vdim.config");
    auto loadField = [&](uint64_t Offset, StringRef Name) -> Value * {
      Value *Field = B.CreateGEP(I8, ConfigPtr, B.getInt64(Offset));
      return B.CreateLoad(I32, Field, Name);
    };
    VD.Gx = loadField(0, "vdim.grid.x");
    VD.Gy = loadField(4, "vdim.grid.y");
    VD.Gz = loadField(8, "vdim.grid.z");
    VD.Bx = loadField(12, "vdim.block.x");
    VD.By = loadField(16, "vdim.block.y");
    VD.Bz = loadField(20, "vdim.block.z");
  }

  Value *PhysicalGridX =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.nctaid.x");
  Value *PhysicalGridY =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.nctaid.y");
  Value *PhysicalBlockX =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.ntid.x");
  Value *PhysicalBlockY =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.ntid.y");
  Value *PhysicalBidx =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.ctaid.x");
  Value *PhysicalBidy =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.ctaid.y");
  Value *PhysicalBidz =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.ctaid.z");
  Value *PhysicalTidx =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.tid.x");
  Value *PhysicalTidy =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.tid.y");
  Value *PhysicalTidz =
      readPhysicalSReg(B, *F.getParent(), "llvm.nvvm.read.ptx.sreg.tid.z");

  Value *BlockLinear =
      linearizePhysical3D(B, PhysicalBidx, PhysicalBidy, PhysicalBidz,
                          PhysicalGridX, PhysicalGridY, "vdim.block");
  Value *ThreadLinear =
      linearizePhysical3D(B, PhysicalTidx, PhysicalTidy, PhysicalTidz,
                          PhysicalBlockX, PhysicalBlockY, "vdim.thread");

  decomposeLinear3D(B, BlockLinear, VD.Gx, VD.Gy, "vdim.block", VD.BlockIdxX,
                    VD.BlockIdxY, VD.BlockIdxZ);
  decomposeLinear3D(B, ThreadLinear, VD.Bx, VD.By, "vdim.thread",
                    VD.ThreadIdxX, VD.ThreadIdxY, VD.ThreadIdxZ);

  Value *LogicalGridXY = B.CreateMul(VD.Gx, VD.Gy, "vdim.grid.xy");
  Value *LogicalGridThreads =
      B.CreateMul(LogicalGridXY, VD.Gz, "vdim.grid.blocks");
  Value *LogicalBlockXY = B.CreateMul(VD.Bx, VD.By, "vdim.block.xy");
  Value *LogicalBlockThreads =
      B.CreateMul(LogicalBlockXY, VD.Bz, "vdim.block.threads");

  Value *InGrid = B.CreateICmpULT(BlockLinear, LogicalGridThreads,
                                  "vdim.block.active");
  Value *InBlock = B.CreateICmpULT(ThreadLinear, LogicalBlockThreads,
                                   "vdim.thread.active");
  Value *Active = B.CreateAnd(InGrid, InBlock, "vdim.active.thread");

  if (DelayedGuardBlock) {
    IRBuilder<> GuardBuilder(GuardOldTerm);
    GuardBuilder.CreateCondBr(Active, ActiveBB, InactiveBB);
    GuardOldTerm->eraseFromParent();
  } else {
    B.CreateCondBr(Active, ActiveBB, InactiveBB);
    OldTerm->eraseFromParent();
  }

  emitInactiveSyncActions(F, InactiveBB, ActiveBB, InactiveActions);
  return true;
}

static CallInst *findCallByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *Callee = CI->getCalledFunction();
      if (Callee && Callee->getName() == Name)
        return CI;
    }
  }
  return nullptr;
}

static AllocaInst *findArgsAlloca(CallInst *LaunchCall) {
  if (!LaunchCall || LaunchCall->arg_size() < 6)
    return nullptr;
  Value *ArgsPtr = LaunchCall->getArgOperand(5)->stripPointerCasts();
  if (auto *GEP = dyn_cast<GetElementPtrInst>(ArgsPtr))
    ArgsPtr = GEP->getPointerOperand()->stripPointerCasts();
  return dyn_cast<AllocaInst>(ArgsPtr);
}

static GlobalVariable *getOrCreateVDimGlobal(Module &M, StringRef Name) {
  if (GlobalVariable *GV = M.getNamedGlobal(Name))
    return GV;
  auto *PtrTy = PointerType::get(M.getContext(), 0);
  auto *Init = ConstantPointerNull::get(PtrTy);
  auto *GV = new GlobalVariable(M, PtrTy, false, GlobalValue::InternalLinkage,
                                Init, Name);
  GV->setAlignment(M.getDataLayout().getPointerABIAlignment(0));
  return GV;
}

static Function *getOrCreateVDimSet(Module &M, GlobalVariable *GridGV,
                                    GlobalVariable *BlockGV) {
  Function *F = M.getFunction("vdim_set");
  if (F && !F->isDeclaration())
    return F;

  LLVMContext &Ctx = M.getContext();
  if (!F) {
    auto *PtrTy = PointerType::get(Ctx, 0);
    auto *FnTy = FunctionType::get(Type::getVoidTy(Ctx), {PtrTy, PtrTy}, false);
    F = Function::Create(FnTy, GlobalValue::ExternalLinkage, "vdim_set", &M);
    F->setCallingConv(CallingConv::C);
  }

  FunctionType *FTy = F->getFunctionType();
  if (!F->isDeclaration())
    return F;

  if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy() ||
      !FTy->getParamType(1)->isPointerTy()) {
    return nullptr;
  }
  F->removeParamAttr(0, Attribute::ByVal);
  F->removeParamAttr(1, Attribute::ByVal);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> B(Entry);
  B.CreateStore(F->getArg(0), GridGV);
  B.CreateStore(F->getArg(1), BlockGV);
  B.CreateRetVoid();
  return F;
}

static bool instrumentHostStub(Function &F, const VDimAnnotation &Ann,
                               Module &M) {
  if (Ann.Mode != VDimMode::Args)
    return false;
  CallInst *Pop = findCallByName(F, "__cudaPopCallConfiguration");
  CallInst *Launch = findCallByName(F, "cudaLaunchKernel");
  if (!Pop || !Launch)
    return false;

  Value *GridPtr = Pop->getArgOperand(0);
  auto *GridAlloca = dyn_cast<AllocaInst>(GridPtr->stripPointerCasts());
  if (!GridAlloca)
    return false;
  auto *DimTy = dyn_cast<StructType>(GridAlloca->getAllocatedType());
  if (!DimTy || DimTy->getNumElements() != 3)
    return false;

  GlobalVariable *GridGV = getOrCreateVDimGlobal(M, "__vdim_grid");
  GlobalVariable *BlockGV = getOrCreateVDimGlobal(M, "__vdim_block");
  getOrCreateVDimSet(M, GridGV, BlockGV);

  AllocaInst *ArgsAlloca = findArgsAlloca(Launch);
  if (!ArgsAlloca)
    return false;
  auto *ArrSize = dyn_cast<ConstantInt>(ArgsAlloca->getArraySize());
  if (!ArrSize)
    return false;
  uint64_t N = ArrSize->getZExtValue();
  Type *ArgElemTy = ArgsAlloca->getAllocatedType();
  Align ArgsAlign = ArgsAlloca->getAlign();

  IRBuilder<> BE(&*F.getEntryBlock().getFirstInsertionPt());
  AllocaInst *NewArgs = BE.CreateAlloca(ArgElemTy, BE.getInt64(N + 2));
  NewArgs->setAlignment(ArgsAlign);
  ArgsAlloca->replaceAllUsesWith(NewArgs);
  ArgsAlloca->eraseFromParent();

  IRBuilder<> B(Launch);
  Value *IdxN = B.getInt64(N);
  Value *IdxN1 = B.getInt64(N + 1);
  Value *GepN = B.CreateGEP(ArgElemTy, NewArgs, IdxN);
  Value *GepN1 = B.CreateGEP(ArgElemTy, NewArgs, IdxN1);
  Value *GridArgPtr = B.CreateBitCast(GridGV, ArgElemTy);
  Value *BlockArgPtr = B.CreateBitCast(BlockGV, ArgElemTy);
  B.CreateStore(GridArgPtr, GepN);
  B.CreateStore(BlockArgPtr, GepN1);
  return true;
}

static bool strengthenSharedGlobalAlignments(Module &M) {
  DenseMap<GlobalVariable *, Align> RequiredAlignments;
  auto RecordAlignment = [&](Value *Pointer, Align Alignment) {
    auto *Global = dyn_cast<GlobalVariable>(getUnderlyingObject(Pointer));
    if (!Global || Global->getAddressSpace() != 3)
      return;
    auto It = RequiredAlignments.find(Global);
    if (It == RequiredAlignments.end() ||
        It->second.value() < Alignment.value())
      RequiredAlignments[Global] = Alignment;
  };

  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        RecordAlignment(Load->getPointerOperand(), Load->getAlign());
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        RecordAlignment(Store->getPointerOperand(), Store->getAlign());
      } else if (auto *Atomic = dyn_cast<AtomicRMWInst>(&I)) {
        RecordAlignment(Atomic->getPointerOperand(), Atomic->getAlign());
      } else if (auto *Compare = dyn_cast<AtomicCmpXchgInst>(&I)) {
        RecordAlignment(Compare->getPointerOperand(), Compare->getAlign());
      } else if (auto *Transfer = dyn_cast<MemTransferInst>(&I)) {
        RecordAlignment(Transfer->getRawDest(),
                        Transfer->getDestAlign().valueOrOne());
        RecordAlignment(Transfer->getRawSource(),
                        Transfer->getSourceAlign().valueOrOne());
      } else if (auto *Set = dyn_cast<MemSetInst>(&I)) {
        RecordAlignment(Set->getRawDest(), Set->getDestAlign().valueOrOne());
      }
    }
  }

  bool Changed = false;
  for (auto &[Global, Required] : RequiredAlignments) {
    if (Global->getAlign().valueOrOne().value() >= Required.value())
      continue;
    Global->setAlignment(Required);
    Changed = true;
  }
  return Changed;
}

} // namespace

PreservedAnalyses VirtualDimPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool Changed = strengthenSharedGlobalAlignments(M);

  for (Function &F : make_early_inc_range(M)) {
    if (F.isDeclaration() || F.hasFnAttribute("vdim.processed") ||
        F.hasFnAttribute(RapidProcessedAttr))
      continue;

    if (F.hasFnAttribute(RapidContextAttr)) {
      RapidContextSyncPlan SyncPlan = analyzeRapidContextSync(F);
      if (!SyncPlan.UnsupportedReason.empty()) {
        errs() << SyncPlan.UnsupportedReason << '\n';
        if (!SyncPlan.UnsupportedDetail.empty())
          errs() << SyncPlan.UnsupportedDetail << '\n';
        continue;
      }

      VDimAnnotation Ann;
      Ann.Mode = VDimMode::RapidContext;
      VDimValues VD;
      bool LocalChanged = insertActiveGuard(
          F, Ann, VD, SyncPlan.InactiveSyncActions,
          SyncPlan.RequiresWarpAlignedBlock, SyncPlan.GuardBlock);
      if (LocalChanged)
        LocalChanged |= replaceDimBuiltinUses(F, VD);
      if (LocalChanged) {
        F.addFnAttr(RapidProcessedAttr);
        if (SyncPlan.RequiresWarpAlignedBlock)
          F.addFnAttr(RapidWarpAlignedAttr);
        Changed = true;
      }
      continue;
    }

    VDimAnnotation Ann;
    if (!getVDimAnnotation(M, F, Ann))
      continue;
    if (isNvvmKernel(M, F)) {
      Function *Cur = &F;
      if (Ann.Mode == VDimMode::Args) {
        Argument *GridArg = nullptr;
        Argument *BlockArg = nullptr;
        StructType *DimTy = nullptr;
        if (!getDim3Args(*Cur, GridArg, BlockArg, DimTy)) {
          Cur = appendVDimArgs(*Cur);
          Changed = true;
        }
      }

      VDimValues VD;
      bool LocalChanged = false;
      LocalChanged |= insertActiveGuard(*Cur, Ann, VD);
      if (LocalChanged)
        LocalChanged |= replaceDimBuiltinUses(*Cur, VD);
      if (LocalChanged) {
        Cur->addFnAttr("vdim.processed");
        Changed = true;
      }
    } else {
      bool LocalChanged = instrumentHostStub(F, Ann, M);
      if (LocalChanged) {
        F.addFnAttr("vdim.processed");
        Changed = true;
      }
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
