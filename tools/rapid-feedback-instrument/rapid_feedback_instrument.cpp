#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using namespace llvm;

namespace {

cl::opt<std::string> InputPath("input", cl::Required,
                               cl::desc("Linked LLVM bitcode input"));
cl::opt<std::string> OutputPath("output", cl::Required,
                                cl::desc("Instrumented LLVM bitcode output"));
cl::opt<std::string> BuildSpecPath("build-spec", cl::Required,
                                   cl::desc("Phase 2 build_spec.json"));
cl::opt<std::string> MetadataPath("metadata-out", cl::Required,
                                  cl::desc("Feedback metadata JSON output"));
cl::opt<bool> InstrumentSelects(
    "instrument-selects", cl::init(false),
    cl::desc("Emit single-sided evaluated probes for LLVM select instructions"));

constexpr uint32_t EdgeMapSize = 65536u;
constexpr uint32_t MemoryIndexBuckets = 65536u;
constexpr uint32_t MemoryIndexDataBuckets = 61440u;

struct PayloadSlot {
  uint16_t Slot;
  uint32_t ArgIndex;
  bool IsTopLevel;
  std::optional<int64_t> ByteOffset;
};

struct BuildSpec {
  std::string KernelId;
  std::string EntrySymbol;
  bool MemoryFeedbackEnabled = false;
  std::vector<PayloadSlot> PayloadSlots;
};

using TopLevelSlotMap = std::map<const Argument *, uint16_t>;
using NestedSlotMap =
    std::map<std::pair<const Argument *, int64_t>, uint16_t>;
using SpilledArgumentMap = std::map<const AllocaInst *, const Argument *>;

uint32_t stableHash(StringRef Text) {
  uint32_t Hash = 2166136261u;
  for (char Byte : Text) {
    Hash ^= static_cast<uint8_t>(Byte);
    Hash *= 16777619u;
  }
  return Hash;
}

Error buildSpecError(const Twine &Message) {
  return createStringError(inconvertibleErrorCode(), Message);
}

Expected<BuildSpec> readBuildSpec(StringRef Path) {
  auto Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return buildSpecError(Twine("cannot read build spec: ") + Path);

  Expected<json::Value> Parsed = json::parse((*Buffer)->getBuffer());
  if (!Parsed)
    return Parsed.takeError();
  json::Object *Root = Parsed->getAsObject();
  if (!Root)
    return buildSpecError("build spec root must be an object");

  std::optional<StringRef> KernelId = Root->getString("kernel_id");
  std::optional<StringRef> EntrySymbol = Root->getString("entry_symbol");
  std::optional<int64_t> AbiVersion = Root->getInteger("entry_abi_version");
  std::optional<StringRef> EntryContext = Root->getString("entry_context");
  if (!KernelId || !EntrySymbol || AbiVersion != 1 || !EntryContext ||
      *EntryContext != "rapid_kernel_context_v1") {
    return buildSpecError("build spec does not use rapid_kernel_context_v1");
  }

  BuildSpec Spec;
  Spec.KernelId = KernelId->str();
  Spec.EntrySymbol = EntrySymbol->str();
  Spec.MemoryFeedbackEnabled =
      Root->getBoolean("feedback_memory_enabled").value_or(false);

  const json::Array *Slots = Root->getArray("feedback_payload_slots");
  if (!Slots)
    return buildSpecError("feedback_payload_slots must be an array");
  for (const json::Value &Value : *Slots) {
    const json::Object *Object = Value.getAsObject();
    if (!Object)
      return buildSpecError("payload slot must be an object");
    std::optional<int64_t> Slot = Object->getInteger("slot");
    std::optional<int64_t> ArgIndex = Object->getInteger("arg_index");
    const json::Array *FieldPath = Object->getArray("field_path");
    std::optional<int64_t> ByteOffset = Object->getInteger("byte_offset");
    if (!Slot || !ArgIndex || !FieldPath || *Slot < 0 || *Slot > UINT16_MAX ||
        *ArgIndex < 0 || *ArgIndex > UINT32_MAX) {
      return buildSpecError("payload slot fields are invalid");
    }
    const bool IsTopLevel = FieldPath->empty();
    if (!IsTopLevel && !ByteOffset)
      return buildSpecError("nested payload slot requires byte_offset");
    Spec.PayloadSlots.push_back({static_cast<uint16_t>(*Slot),
                                 static_cast<uint32_t>(*ArgIndex), IsTopLevel,
                                 ByteOffset});
  }
  return Spec;
}

uint8_t accessWidth(const DataLayout &Layout, Type *AccessType) {
  TypeSize Size = Layout.getTypeStoreSize(AccessType);
  if (Size.isScalable() || Size.getFixedValue() > UINT8_MAX)
    return 0u;
  return static_cast<uint8_t>(Size.getFixedValue());
}

Value *castToGenericPointer(IRBuilder<> &Builder, Value *Pointer) {
  auto *GenericPointerType = PointerType::get(Builder.getContext(), 0u);
  if (Pointer->getType() == GenericPointerType)
    return Pointer;
  return Builder.CreateAddrSpaceCast(Pointer, GenericPointerType,
                                     "rapid.feedback.ptr");
}

std::optional<uint16_t>
findPayloadSlot(Value *AccessPointer, const DataLayout &Layout,
                const TopLevelSlotMap &TopLevelSlots,
                const NestedSlotMap &NestedSlots,
                const SpilledArgumentMap &SpilledArguments) {
  const Value *Underlying = getUnderlyingObject(AccessPointer);
  if (const auto *UnderlyingArg = dyn_cast<Argument>(Underlying)) {
    auto SlotIt = TopLevelSlots.find(UnderlyingArg);
    if (SlotIt != TopLevelSlots.end())
      return SlotIt->second;
    return std::nullopt;
  }

  const auto *PointerLoad = dyn_cast<LoadInst>(Underlying);
  if (!PointerLoad)
    return std::nullopt;
  int64_t FieldOffset = 0;
  const Value *FieldBase = GetPointerBaseWithConstantOffset(
      PointerLoad->getPointerOperand(), FieldOffset, Layout);
  const Value *FieldUnderlying = getUnderlyingObject(FieldBase);
  const Argument *AggregateArg = dyn_cast<Argument>(FieldUnderlying);
  if (!AggregateArg) {
    if (const auto *Spill = dyn_cast<AllocaInst>(FieldUnderlying)) {
      auto SpillIt = SpilledArguments.find(Spill);
      if (SpillIt != SpilledArguments.end())
        AggregateArg = SpillIt->second;
    }
  }
  if (!AggregateArg)
    return std::nullopt;
  if (FieldOffset == 0) {
    auto TopLevelIt = TopLevelSlots.find(AggregateArg);
    if (TopLevelIt != TopLevelSlots.end())
      return TopLevelIt->second;
  }
  auto SlotIt = NestedSlots.find({AggregateArg, FieldOffset});
  if (SlotIt == NestedSlots.end())
    return std::nullopt;
  return SlotIt->second;
}

Error writeJson(StringRef Path, json::Object Root) {
  std::error_code ErrorCode;
  raw_fd_ostream Output(Path, ErrorCode, sys::fs::OF_Text);
  if (ErrorCode)
    return createStringError(ErrorCode, "cannot open feedback metadata");
  Output << formatv("{0:2}\n", json::Value(std::move(Root)));
  return Error::success();
}

Error instrumentModule(Module &Module, const BuildSpec &Spec,
                       json::Object &Metadata) {
  Function *Entry = Module.getFunction(Spec.EntrySymbol);
  if (!Entry || Entry->isDeclaration())
    return buildSpecError("entry symbol is missing or has no body");
  if (Entry->arg_empty() ||
      !Entry->getArg(Entry->arg_size() - 1)->getType()->isPointerTy())
    return buildSpecError("entry context parameter is missing");

  LLVMContext &Context = Module.getContext();
  Type *VoidType = Type::getVoidTy(Context);
  Type *I8Type = Type::getInt8Ty(Context);
  Type *I16Type = Type::getInt16Ty(Context);
  Type *I32Type = Type::getInt32Ty(Context);
  Type *GenericPointerType = PointerType::get(Context, 0u);
  FunctionCallee BasicBlockProbe = Module.getOrInsertFunction(
      "__rapid_feedback_bb", FunctionType::get(VoidType, {I32Type}, false));
  FunctionCallee MemoryProbe = Module.getOrInsertFunction(
      "__rapid_feedback_mem",
      FunctionType::get(VoidType,
                        {GenericPointerType, I32Type, I16Type, I8Type, I8Type,
                         GenericPointerType},
                        false));

  json::Array CfgSites;
  uint32_t BlockIndex = 0u;
  for (BasicBlock &Block : *Entry) {
    BasicBlock::iterator InsertPoint = Block.getFirstInsertionPt();
    if (InsertPoint == Block.end())
      continue;
    const std::string Key =
        (Twine(Spec.KernelId) + ":" + Entry->getName() + ":bb:" +
         Twine(BlockIndex))
            .str();
    const uint32_t SiteId = stableHash(Key) % EdgeMapSize;
    IRBuilder<> Builder(&*InsertPoint);
    Builder.CreateCall(BasicBlockProbe, ConstantInt::get(I32Type, SiteId));

    json::Object Site;
    Site["site_id"] = SiteId;
    if (InstrumentSelects)
      Site["site_kind"] = "basic_block";
    Site["function"] = Entry->getName();
    Site["block_index"] = BlockIndex;
    Site["block_name"] = Block.hasName() ? Block.getName() : "";
    CfgSites.push_back(std::move(Site));
    ++BlockIndex;
  }

  if (InstrumentSelects) {
    std::vector<SelectInst *> SelectInstructions;
    for (BasicBlock &Block : *Entry) {
      for (Instruction &Instruction : Block) {
        if (auto *Select = dyn_cast<SelectInst>(&Instruction))
          SelectInstructions.push_back(Select);
      }
    }

    json::Array SelectSites;
    uint32_t SelectOrdinal = 0u;
    for (SelectInst *Select : SelectInstructions) {
      const std::string Key =
          (Twine(Spec.KernelId) + ":" + Entry->getName() + ":select:" +
           Twine(SelectOrdinal))
              .str();
      const uint32_t SiteId = stableHash(Key) % EdgeMapSize;
      IRBuilder<> Builder(Select);
      // This pseudo-site is deliberately single-sided: it records only that
      // the select was evaluated, not whether its true or false value won.
      Builder.CreateCall(BasicBlockProbe, ConstantInt::get(I32Type, SiteId));

      json::Object Site;
      Site["site_id"] = SiteId;
      Site["site_kind"] = "select_evaluated";
      Site["function"] = Entry->getName();
      Site["instruction_ordinal"] = SelectOrdinal;
      SelectSites.push_back(std::move(Site));

      json::Object CfgSite;
      CfgSite["site_id"] = SiteId;
      CfgSite["site_kind"] = "select_evaluated";
      CfgSite["function"] = Entry->getName();
      CfgSite["instruction_ordinal"] = SelectOrdinal;
      CfgSites.push_back(std::move(CfgSite));
      ++SelectOrdinal;
    }
    Metadata["instrumented_basic_block_sites"] = BlockIndex;
    Metadata["instrumented_select_sites"] = SelectSites.size();
    Metadata["select_sites"] = std::move(SelectSites);
  }

  TopLevelSlotMap TopLevelSlots;
  NestedSlotMap NestedSlots;
  if (Spec.MemoryFeedbackEnabled) {
    for (const PayloadSlot &Slot : Spec.PayloadSlots) {
      if (Slot.ArgIndex >= Entry->arg_size() - 1)
        return buildSpecError("payload arg index is outside the entry signature");
      Argument *PayloadArg = Entry->getArg(Slot.ArgIndex);
      if (Slot.IsTopLevel) {
        TopLevelSlots[PayloadArg] = Slot.Slot;
      } else {
        NestedSlots[{PayloadArg, *Slot.ByteOffset}] = Slot.Slot;
      }
    }
  }

  SpilledArgumentMap SpilledArguments;
  for (BasicBlock &Block : *Entry) {
    for (Instruction &Instruction : Block) {
      auto *Spill = dyn_cast<AllocaInst>(&Instruction);
      if (!Spill)
        continue;

      const Argument *StoredArgument = nullptr;
      bool Ambiguous = false;
      for (User *User : Spill->users()) {
        auto *Store = dyn_cast<StoreInst>(User);
        if (!Store || getUnderlyingObject(Store->getPointerOperand()) != Spill)
          continue;
        const Value *StoredValue = Store->getValueOperand();
        if (StoredValue->getType()->isPointerTy())
          StoredValue = StoredValue->stripPointerCasts();
        const auto *ArgumentValue = dyn_cast<Argument>(StoredValue);
        if (!ArgumentValue ||
            (StoredArgument && StoredArgument != ArgumentValue)) {
          Ambiguous = true;
          break;
        }
        StoredArgument = ArgumentValue;
      }
      if (!Ambiguous && StoredArgument)
        SpilledArguments[Spill] = StoredArgument;
    }
  }

  std::vector<Instruction *> MemoryInstructions;
  for (BasicBlock &Block : *Entry) {
    for (Instruction &Instruction : Block) {
      if (isa<LoadInst>(Instruction) || isa<StoreInst>(Instruction))
        MemoryInstructions.push_back(&Instruction);
    }
  }

  json::Array MemorySites;
  uint32_t UnknownMemorySites = 0u;
  uint32_t MemoryOrdinal = 0u;
  Value *KernelContext = Entry->getArg(Entry->arg_size() - 1);
  const DataLayout &Layout = Module.getDataLayout();
  for (Instruction *Instruction : MemoryInstructions) {
    Value *AccessPointer = nullptr;
    Type *AccessType = nullptr;
    StringRef AccessKind;
    uint8_t AccessKindValue = 0u;
    if (auto *Load = dyn_cast<LoadInst>(Instruction)) {
      AccessPointer = Load->getPointerOperand();
      AccessType = Load->getType();
      AccessKind = "read";
    } else {
      auto *Store = cast<StoreInst>(Instruction);
      AccessPointer = Store->getPointerOperand();
      AccessType = Store->getValueOperand()->getType();
      AccessKind = "write";
      AccessKindValue = 1u;
    }

    std::optional<uint16_t> ArgSlot = findPayloadSlot(
        AccessPointer, Layout, TopLevelSlots, NestedSlots, SpilledArguments);
    if (!ArgSlot) {
      ++UnknownMemorySites;
      ++MemoryOrdinal;
      continue;
    }

    const std::string Key =
        (Twine(Spec.KernelId) + ":" + Entry->getName() + ":mem:" +
         Twine(MemoryOrdinal))
            .str();
    const uint32_t SiteId = stableHash(Key);
    const uint8_t Width = accessWidth(Layout, AccessType);
    IRBuilder<> Builder(Instruction);
    Builder.CreateCall(
        MemoryProbe,
        {KernelContext, ConstantInt::get(I32Type, SiteId),
         ConstantInt::get(I16Type, *ArgSlot),
         ConstantInt::get(I8Type, AccessKindValue),
         ConstantInt::get(I8Type, Width),
         castToGenericPointer(Builder, AccessPointer)});

    json::Object Site;
    Site["mem_site_id"] = SiteId;
    Site["function"] = Entry->getName();
    Site["instruction_ordinal"] = MemoryOrdinal;
    Site["arg_slot"] = *ArgSlot;
    Site["access_kind"] = AccessKind;
    Site["width"] = Width;
    MemorySites.push_back(std::move(Site));
    ++MemoryOrdinal;
  }

  Entry->setMemoryEffects(MemoryEffects::unknown());

  json::Object PatternEncodings;
  PatternEncodings["single"] = 0;
  PatternEncodings["full_broadcast"] = 1;
  PatternEncodings["full_contiguous"] = 2;
  PatternEncodings["full_other"] = 3;
  PatternEncodings["partial_broadcast"] = 4;
  PatternEncodings["partial_contiguous"] = 5;
  PatternEncodings["partial_other"] = 6;

  Metadata["schema_version"] = 2;
  Metadata["feedback_abi_version"] = 1;
  Metadata["edge_map_size"] = EdgeMapSize;
  Metadata["simt_memcov_buckets"] = MemoryIndexBuckets;
  Metadata["simt_memcov_data_buckets"] = MemoryIndexDataBuckets;
  Metadata["memory_metric_version"] = "rapid-simt-memcov-v1";
  Metadata["memory_map_bits"] = MemoryIndexDataBuckets;
  Metadata["memory_sector_bytes"] = 32;
  Metadata["thread_activity_map_bits"] =
      MemoryIndexBuckets - MemoryIndexDataBuckets;
  Metadata["memory_pattern_encodings"] = std::move(PatternEncodings);
  Metadata["memory_hash_contract_version"] =
      "rapid-simt-memcov-hash-v1";
  Metadata["kernel_id"] = Spec.KernelId;
  Metadata["entry_symbol"] = Spec.EntrySymbol;
  Metadata["instrumented_cfg_sites"] = CfgSites.size();
  Metadata["instrumented_memory_sites"] = MemorySites.size();
  Metadata["unknown_memory_sites"] = UnknownMemorySites;
  Metadata["cfg_sites"] = std::move(CfgSites);
  Metadata["memory_sites"] = std::move(MemorySites);
  return Error::success();
}

} // namespace

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "RAPID LLVM feedback instrumentation\n");

  Expected<BuildSpec> Spec = readBuildSpec(BuildSpecPath);
  if (!Spec) {
    logAllUnhandledErrors(Spec.takeError(), errs(), "rapid-feedback-instrument: ");
    return 1;
  }

  LLVMContext Context;
  SMDiagnostic Diagnostic;
  std::unique_ptr<Module> Module = parseIRFile(InputPath, Diagnostic, Context);
  if (!Module) {
    Diagnostic.print(Argv[0], errs());
    return 1;
  }

  json::Object Metadata;
  if (Error ErrorValue = instrumentModule(*Module, *Spec, Metadata)) {
    logAllUnhandledErrors(std::move(ErrorValue), errs(),
                          "rapid-feedback-instrument: ");
    return 1;
  }
  if (verifyModule(*Module, &errs())) {
    errs() << "rapid-feedback-instrument: instrumented module verification failed\n";
    return 1;
  }

  std::error_code ErrorCode;
  raw_fd_ostream Output(OutputPath, ErrorCode, sys::fs::OF_None);
  if (ErrorCode) {
    errs() << "rapid-feedback-instrument: cannot open output: "
           << ErrorCode.message() << "\n";
    return 1;
  }
  WriteBitcodeToFile(*Module, Output);
  Output.flush();

  if (Error ErrorValue = writeJson(MetadataPath, std::move(Metadata))) {
    logAllUnhandledErrors(std::move(ErrorValue), errs(),
                          "rapid-feedback-instrument: ");
    return 1;
  }
  return 0;
}
