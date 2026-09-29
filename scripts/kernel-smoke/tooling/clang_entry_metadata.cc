#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::tooling;

namespace {

struct ParamInfo {
  int index = 0;
  std::string name;
  std::string type;
  std::string kind;
  std::string pointer_role;
  std::uint64_t size_bytes = 0;
  std::uint64_t align_bytes = 0;
  llvm::json::Value domain = nullptr;
  llvm::json::Value type_layout = nullptr;
  llvm::json::Value pointee_layout = nullptr;
  llvm::json::Value type_info = nullptr;
  std::string materialization_status = "ok";
  std::vector<std::string> materialization_reason_codes;
  std::vector<std::string> materialization_blockers;
};

struct SourceLocInfo {
  std::string file;
  unsigned line = 0;
  unsigned column = 0;
};

struct SymbolInfo {
  std::string qualified_name;
  std::string source_file;
  unsigned line = 0;
  std::vector<ParamInfo> args;
  std::string type_shim;
  std::string type_shim_status = "ok";
  std::vector<std::string> type_shim_reason_codes;
  std::vector<std::string> type_shim_missing_dependencies;
  std::vector<std::string> type_shim_system_headers;
  std::vector<std::string> type_shim_system_includes;
  std::string materialization_status = "ok";
  std::vector<std::string> materialization_reason_codes;
  std::vector<std::string> materialization_blockers;
};

struct MaterializationFacts {
  std::set<std::string> reason_codes;
  std::set<std::string> blockers;
};

struct ShimState {
  ASTContext &ctx;
  const SourceManager &sm;
  std::vector<const Decl *> ordered;
  llvm::SmallPtrSet<const Decl *, 32> seen;
  std::set<std::string> reason_codes;
  std::set<std::string> missing_dependencies;
  std::set<std::string> system_header_paths;
  std::set<std::string> system_include_stmts;
};

struct Request {
  std::string source_file;
  std::string cwd;
  std::vector<std::string> args;
  std::set<std::string> ptx_symbols;
};

struct RunState {
  std::set<std::string> wanted_symbols;
  std::unordered_map<std::string, SymbolInfo> matched;
  std::vector<std::string> diagnostics;
};

struct IncludeRoot {
  std::string path;
  bool is_system = false;
};

std::optional<Request> loadRequest(const std::string &path, std::string &error) {
  auto buf_or_err = llvm::MemoryBuffer::getFile(path);
  if (!buf_or_err) {
    error = "failed to read request file";
    return std::nullopt;
  }

  auto parsed = llvm::json::parse(buf_or_err->get()->getBuffer());
  if (!parsed) {
    error = "failed to parse request JSON";
    return std::nullopt;
  }
  auto *obj = parsed->getAsObject();
  if (!obj) {
    error = "request JSON root must be an object";
    return std::nullopt;
  }

  Request req;
  auto source = obj->getString("source_file");
  auto cwd = obj->getString("cwd");
  auto *args = obj->getArray("args");
  auto *symbols = obj->getArray("ptx_symbols");
  if (!source || !cwd || !args || !symbols) {
    error = "request JSON missing source_file/cwd/args/ptx_symbols";
    return std::nullopt;
  }

  req.source_file = std::string(*source);
  req.cwd = std::string(*cwd);
  for (auto &value : *args) {
    auto s = value.getAsString();
    if (!s) {
      error = "request args must be strings";
      return std::nullopt;
    }
    req.args.emplace_back(*s);
  }
  for (auto &value : *symbols) {
    auto s = value.getAsString();
    if (!s) {
      error = "request ptx_symbols must be strings";
      return std::nullopt;
    }
    req.ptx_symbols.insert(std::string(*s));
  }

  return req;
}

std::string normalizePathWithCwd(llvm::StringRef raw, llvm::StringRef cwd) {
  llvm::SmallString<256> path(raw);
  if (!llvm::sys::path::is_absolute(path)) {
    llvm::SmallString<256> base(cwd);
    llvm::sys::fs::make_absolute(base, path);
  }
  llvm::sys::path::remove_dots(path, true);
  return std::string(path.str());
}

std::vector<IncludeRoot> parseIncludeRoots(const Request &req) {
  std::vector<IncludeRoot> roots;
  std::set<std::pair<std::string, bool>> seen;
  auto add_root = [&](llvm::StringRef raw, bool is_system) {
    if (raw.empty()) {
      return;
    }
    std::string normalized = normalizePathWithCwd(raw, req.cwd);
    auto key = std::make_pair(normalized, is_system);
    if (seen.insert(key).second) {
      roots.push_back(IncludeRoot{normalized, is_system});
    }
  };
  auto add_cuda_roots = [&](llvm::StringRef cuda_root) {
    if (cuda_root.empty()) {
      return;
    }
    std::string normalized = normalizePathWithCwd(cuda_root, req.cwd);
    llvm::SmallString<256> include_dir(normalized);
    llvm::sys::path::append(include_dir, "include");
    add_root(include_dir, true);
    llvm::SmallString<256> target_include(normalized);
    llvm::sys::path::append(target_include, "targets", "x86_64-linux", "include");
    add_root(target_include, true);
  };

  std::string source_path = normalizePathWithCwd(req.source_file, req.cwd);
  llvm::StringRef source_parent = llvm::sys::path::parent_path(source_path);
  if (!source_parent.empty()) {
    add_root(source_parent, false);
  }

  for (size_t i = 0; i < req.args.size(); ++i) {
    llvm::StringRef item(req.args[i]);
    if (item == "-I" && i + 1 < req.args.size()) {
      add_root(req.args[++i], false);
      continue;
    }
    if (item.starts_with("-I") && item.size() > 2) {
      add_root(item.drop_front(2), false);
      continue;
    }
    if (item == "-isystem" && i + 1 < req.args.size()) {
      add_root(req.args[++i], true);
      continue;
    }
    if (item == "--cuda-path" && i + 1 < req.args.size()) {
      add_cuda_roots(req.args[++i]);
      continue;
    }
    if (item.starts_with("--cuda-path=") && item.size() > strlen("--cuda-path=")) {
      add_cuda_roots(item.drop_front(strlen("--cuda-path=")));
      continue;
    }
  }
  return roots;
}

std::optional<std::string> bestIncludeHint(llvm::StringRef definition_file, const std::vector<IncludeRoot> &roots) {
  if (definition_file.empty()) {
    return std::nullopt;
  }
  if (definition_file.ends_with("/cuda_fp16.hpp") || definition_file == "cuda_fp16.hpp") {
    return std::string("#include <cuda_fp16.h>");
  }
  std::string normalized = normalizePathWithCwd(definition_file, "");
  std::string best_stmt;
  std::tuple<int, size_t, std::string> best_key{99, static_cast<size_t>(-1), {}};
  for (const auto &root : roots) {
    llvm::StringRef header(normalized);
    llvm::StringRef root_ref(root.path);
    if (!header.starts_with(root_ref)) {
      continue;
    }
    llvm::StringRef suffix = header.drop_front(root_ref.size());
    while (!suffix.empty() && (suffix.front() == '/' || suffix.front() == '\\')) {
      suffix = suffix.drop_front();
    }
    if (suffix.empty()) {
      continue;
    }
    std::string include_suffix = std::string(suffix);
    std::replace(include_suffix.begin(), include_suffix.end(), '\\', '/');
    std::string stmt = root.is_system ? (std::string("#include <") + include_suffix + ">")
                                      : (std::string("#include \"") + include_suffix + "\"");
    auto key = std::make_tuple(root.is_system ? 0 : 1, include_suffix.size(), root.path);
    if (best_stmt.empty() || key < best_key) {
      best_key = key;
      best_stmt = std::move(stmt);
    }
  }
  if (best_stmt.empty()) {
    return std::nullopt;
  }
  return best_stmt;
}

std::string mangleDecl(ASTContext &ctx, const NamedDecl *decl) {
  std::string out;
  llvm::raw_string_ostream os(out);
  auto mctx = ItaniumMangleContext::create(ctx, ctx.getDiagnostics());
  if (!mctx->shouldMangleDeclName(decl)) {
    os << decl->getNameAsString();
    return os.str();
  }
  if (const auto *fd = llvm::dyn_cast<FunctionDecl>(decl)) {
    mctx->mangleName(GlobalDecl(fd), os);
    return os.str();
  }
  os << decl->getNameAsString();
  return os.str();
}

bool hasCudaGlobalAttr(const FunctionDecl *fd) {
  return fd && fd->hasAttr<CUDAGlobalAttr>();
}

bool isInterestingKernel(const FunctionDecl *fd) {
  if (!fd) {
    return false;
  }
  if (!hasCudaGlobalAttr(fd)) {
    return false;
  }
  auto name = fd->getNameAsString();
  if (name.find("__device_stub__") != std::string::npos) {
    return false;
  }
  return true;
}

unsigned sourceLine(const FunctionDecl *fd, ASTContext &ctx) {
  const auto &sm = ctx.getSourceManager();
  auto loc = sm.getExpansionLoc(fd->getLocation());
  auto presumed = sm.getPresumedLoc(loc);
  if (presumed.isInvalid()) {
    return 0;
  }
  return presumed.getLine();
}

std::string sourceFile(const FunctionDecl *fd, ASTContext &ctx) {
  const auto &sm = ctx.getSourceManager();
  auto loc = sm.getExpansionLoc(fd->getLocation());
  auto presumed = sm.getPresumedLoc(loc);
  if (presumed.isInvalid()) {
    return {};
  }
  return std::string(presumed.getFilename());
}

std::optional<SourceLocInfo> sourceLoc(SourceLocation loc, const SourceManager &sm) {
  auto spelling = sm.getSpellingLoc(loc);
  auto presumed = sm.getPresumedLoc(spelling);
  if (presumed.isInvalid()) {
    return std::nullopt;
  }

  SourceLocInfo info;
  info.file = std::string(presumed.getFilename());
  info.line = presumed.getLine();
  info.column = presumed.getColumn();
  return info;
}

llvm::json::Object toJson(const SourceLocInfo &loc) {
  return llvm::json::Object{{"file", llvm::json::fixUTF8(loc.file)},
                            {"line", static_cast<std::int64_t>(loc.line)},
                            {"column", static_cast<std::int64_t>(loc.column)}};
}

std::string namedDeclName(const NamedDecl *decl) {
  if (!decl) {
    return {};
  }
  if (const auto *tag = llvm::dyn_cast<TagDecl>(decl)) {
    return tag->getQualifiedNameAsString();
  }
  return decl->getQualifiedNameAsString();
}

PrintingPolicy manifestTypePrintingPolicy(ASTContext &ctx) {
  PrintingPolicy policy(ctx.getLangOpts());
  policy.FullyQualifiedName = true;
  return policy;
}

std::string manifestTypeName(QualType type, ASTContext &ctx) {
  PrintingPolicy policy = manifestTypePrintingPolicy(ctx);
  return TypeName::getFullyQualifiedName(type, ctx, policy, false);
}

std::string tagKindString(TagTypeKind kind) {
  switch (kind) {
  case TagTypeKind::Struct:
    return "struct";
  case TagTypeKind::Interface:
    return "interface";
  case TagTypeKind::Union:
    return "union";
  case TagTypeKind::Class:
    return "class";
  case TagTypeKind::Enum:
    return "enum";
  }
  return {};
}

std::string declKindString(const NamedDecl *decl) {
  if (!decl) {
    return {};
  }
  if (const auto *record = llvm::dyn_cast<RecordDecl>(decl)) {
    return tagKindString(record->getTagKind());
  }
  if (llvm::isa<EnumDecl>(decl)) {
    return "enum";
  }
  if (llvm::isa<TypedefNameDecl>(decl)) {
    return "typedef";
  }
  return decl->getDeclKindName();
}

std::optional<std::string> declUsr(const NamedDecl *decl) {
  if (!decl) {
    return std::nullopt;
  }
  llvm::SmallString<128> usr;
  if (index::generateUSRForDecl(decl, usr)) {
    return std::nullopt;
  }
  return std::string(usr.str());
}

const NamedDecl *namedDeclForType(QualType qt, ASTContext &ctx) {
  qt = qt.getNonReferenceType();
  while (!qt.isNull()) {
    if (const auto *ptr = qt->getAs<PointerType>()) {
      qt = ptr->getPointeeType();
      continue;
    }
    if (const auto *member_ptr = qt->getAs<MemberPointerType>()) {
      qt = member_ptr->getPointeeType();
      continue;
    }
    if (const auto *array = ctx.getAsArrayType(qt)) {
      qt = array->getElementType();
      continue;
    }
    if (const auto *paren = llvm::dyn_cast<ParenType>(qt.getTypePtr())) {
      qt = paren->getInnerType();
      continue;
    }
    if (const auto *adjusted = llvm::dyn_cast<AdjustedType>(qt.getTypePtr())) {
      qt = adjusted->getAdjustedType();
      continue;
    }
    if (const auto *macro = llvm::dyn_cast<MacroQualifiedType>(qt.getTypePtr())) {
      qt = macro->getUnderlyingType();
      continue;
    }
    if (const auto *elaborated = llvm::dyn_cast<ElaboratedType>(qt.getTypePtr())) {
      qt = elaborated->getNamedType();
      continue;
    }
    if (const auto *typedef_ty = llvm::dyn_cast<TypedefType>(qt.getTypePtr())) {
      return typedef_ty->getDecl()->getCanonicalDecl();
    }
    if (const auto *using_ty = llvm::dyn_cast<UsingType>(qt.getTypePtr())) {
      return using_ty->getFoundDecl()->getCanonicalDecl();
    }
    if (const auto *tag = qt->getAsTagDecl()) {
      if (tag->getDeclName().isEmpty()) {
        if (const auto *typedef_decl = tag->getTypedefNameForAnonDecl()) {
          return typedef_decl->getCanonicalDecl();
        }
      }
      return llvm::dyn_cast<NamedDecl>(tag->getCanonicalDecl());
    }
    QualType desugared = qt.getSingleStepDesugaredType(ctx);
    if (desugared == qt) {
      break;
    }
    qt = desugared;
  }
  return nullptr;
}

const Decl *printableDecl(const NamedDecl *decl) {
  if (!decl) {
    return nullptr;
  }
  if (const auto *spec = llvm::dyn_cast<ClassTemplateSpecializationDecl>(decl)) {
    if (const auto *tmpl = spec->getSpecializedTemplate()) {
      return tmpl;
    }
  }
  if (const auto *record = llvm::dyn_cast<CXXRecordDecl>(decl)) {
    if (const auto *tmpl = record->getDescribedClassTemplate()) {
      return tmpl;
    }
  }
  if (const auto *record = llvm::dyn_cast<RecordDecl>(decl)) {
    if (const auto *def = record->getDefinition()) {
      return def;
    }
    return record;
  }
  if (const auto *enum_decl = llvm::dyn_cast<EnumDecl>(decl)) {
    return enum_decl;
  }
  if (const auto *typedef_decl = llvm::dyn_cast<TypedefNameDecl>(decl)) {
    return typedef_decl;
  }
  if (const auto *var_decl = llvm::dyn_cast<VarDecl>(decl)) {
    return var_decl;
  }
  return decl;
}

const NamedDecl *owningRecordForNestedDecl(const NamedDecl *decl) {
  if (!decl) {
    return nullptr;
  }
  const DeclContext *ctx = decl->getDeclContext();
  while (ctx && !llvm::isa<TranslationUnitDecl>(ctx)) {
    if (const auto *record = llvm::dyn_cast<RecordDecl>(ctx)) {
      if (const auto *cxx_record = llvm::dyn_cast<CXXRecordDecl>(record)) {
        if (const auto *tmpl = cxx_record->getDescribedClassTemplate()) {
          return tmpl;
        }
      }
      if (const RecordDecl *def = record->getDefinition()) {
        return llvm::dyn_cast<NamedDecl>(def);
      }
      return llvm::dyn_cast<NamedDecl>(record);
    }
    ctx = ctx->getParent();
  }
  return nullptr;
}

void addShimReason(ShimState &state, llvm::StringRef code, llvm::StringRef detail = "") {
  state.reason_codes.insert(code.str());
  if (!detail.empty()) {
    state.missing_dependencies.insert(detail.str());
  }
}

bool recordHeaderBackedShimDependency(SourceLocation loc, ShimState &state) {
  if (loc.isInvalid()) {
    return false;
  }
  SourceLocation expansion_loc = state.sm.getExpansionLoc(loc);
  if (state.sm.isWrittenInMainFile(expansion_loc)) {
    return false;
  }
  llvm::StringRef file = state.sm.getFilename(expansion_loc);
  if (file.empty()) {
    return false;
  }
  state.system_header_paths.insert(file.str());
  return true;
}

bool isHeaderBackedShimDecl(SourceLocation loc, ShimState &state) {
  if (loc.isInvalid()) {
    return false;
  }
  SourceLocation expansion_loc = state.sm.getExpansionLoc(loc);
  return !state.sm.isWrittenInMainFile(expansion_loc);
}

bool isSupportedShimDeclContext(const DeclContext *ctx) {
  while (ctx && !llvm::isa<TranslationUnitDecl>(ctx)) {
    if (const auto *ns = llvm::dyn_cast<NamespaceDecl>(ctx)) {
      if (ns->isAnonymousNamespace()) {
        return false;
      }
      ctx = ns->getParent();
      continue;
    }
    if (const auto *linkage = llvm::dyn_cast<LinkageSpecDecl>(ctx)) {
      ctx = linkage->getParent();
      continue;
    }
    return false;
  }
  return ctx != nullptr;
}

std::vector<const NamespaceDecl *> namespaceDeclContextChain(const DeclContext *ctx) {
  std::vector<const NamespaceDecl *> namespaces;
  while (ctx && !llvm::isa<TranslationUnitDecl>(ctx)) {
    if (const auto *ns = llvm::dyn_cast<NamespaceDecl>(ctx)) {
      namespaces.push_back(ns);
      ctx = ns->getParent();
      continue;
    }
    if (const auto *linkage = llvm::dyn_cast<LinkageSpecDecl>(ctx)) {
      ctx = linkage->getParent();
      continue;
    }
    break;
  }
  std::reverse(namespaces.begin(), namespaces.end());
  return namespaces;
}

std::string renderDeclWithNamespaceContext(const Decl *decl, llvm::StringRef decl_text) {
  std::vector<const NamespaceDecl *> namespaces = namespaceDeclContextChain(decl->getDeclContext());
  if (namespaces.empty()) {
    return decl_text.str();
  }

  std::string wrapped;
  llvm::raw_string_ostream os(wrapped);
  for (const NamespaceDecl *ns : namespaces) {
    if (ns->isInline()) {
      os << "inline ";
    }
    os << "namespace " << ns->getNameAsString() << " {\n";
  }
  os << decl_text << "\n";
  for (auto it = namespaces.rbegin(); it != namespaces.rend(); ++it) {
    os << "} // namespace " << (*it)->getNameAsString() << "\n";
  }
  os.flush();
  return wrapped;
}

void collectShimDeclsForType(QualType qt, ShimState &state);
void collectShimDeclsForTypeLoc(TypeLoc tl, ShimState &state);
void collectShimDecl(const NamedDecl *decl, ShimState &state);
void collectTemplateArgumentDependencies(const TemplateArgument &arg, ShimState &state);
void collectTemplateArgumentLocDependencies(const TemplateArgumentLoc &arg_loc, ShimState &state);

void collectExprDependencies(const Expr *expr, ShimState &state);

bool isSupportedConstantVar(const VarDecl *var_decl) {
  return var_decl && (var_decl->isConstexpr() || var_decl->getType().isConstQualified());
}

const NamedDecl *owningDeclForValue(const ValueDecl *value_decl) {
  if (!value_decl) {
    return nullptr;
  }
  if (const auto *enum_const = llvm::dyn_cast<EnumConstantDecl>(value_decl)) {
    return llvm::dyn_cast<NamedDecl>(enum_const->getDeclContext());
  }
  if (const auto *var_decl = llvm::dyn_cast<VarDecl>(value_decl)) {
    if (var_decl->isStaticDataMember()) {
      return llvm::dyn_cast<NamedDecl>(var_decl->getDeclContext());
    }
  }
  return nullptr;
}

void collectValueDependency(const ValueDecl *value_decl, ShimState &state) {
  if (!value_decl) {
    return;
  }
  if (const auto *var_decl = llvm::dyn_cast<VarDecl>(value_decl)) {
    if (!isSupportedConstantVar(var_decl)) {
      addShimReason(state, "constexpr_dep_missing", var_decl->getQualifiedNameAsString());
      return;
    }
  }
  if (const auto *owner = owningDeclForValue(value_decl)) {
    collectShimDecl(owner, state);
    return;
  }
  if (const auto *named_decl = llvm::dyn_cast<NamedDecl>(value_decl)) {
    collectShimDecl(named_decl, state);
  }
}

class ExprDependencyVisitor : public RecursiveASTVisitor<ExprDependencyVisitor> {
public:
  explicit ExprDependencyVisitor(ShimState &state) : State(state) {}

  bool VisitDeclRefExpr(DeclRefExpr *expr) {
    collectValueDependency(expr ? expr->getDecl() : nullptr, State);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *expr) {
    const ValueDecl *member_decl = expr ? expr->getMemberDecl() : nullptr;
    if (llvm::isa<EnumConstantDecl>(member_decl)) {
      collectValueDependency(member_decl, State);
      return true;
    }
    if (const auto *var_decl = llvm::dyn_cast_or_null<VarDecl>(member_decl)) {
      if (var_decl->isStaticDataMember()) {
        collectValueDependency(var_decl, State);
      }
    }
    return true;
  }

  bool VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *expr) {
    if (!expr) {
      return true;
    }
    for (const auto &arg_loc : expr->template_arguments()) {
      collectTemplateArgumentLocDependencies(arg_loc, State);
    }
    return true;
  }

  bool VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *expr) {
    if (!expr) {
      return true;
    }
    for (const auto &arg_loc : expr->template_arguments()) {
      collectTemplateArgumentLocDependencies(arg_loc, State);
    }
    return true;
  }

private:
  ShimState &State;
};

void collectExprDependencies(const Expr *expr, ShimState &state) {
  if (!expr) {
    return;
  }
  ExprDependencyVisitor visitor(state);
  visitor.TraverseStmt(const_cast<Expr *>(expr));
}

void collectTemplateArgumentDependencies(const TemplateArgument &arg, ShimState &state) {
  switch (arg.getKind()) {
  case TemplateArgument::Null:
    return;
  case TemplateArgument::Type:
    collectShimDeclsForType(arg.getAsType(), state);
    return;
  case TemplateArgument::Declaration:
    collectValueDependency(arg.getAsDecl(), state);
    return;
  case TemplateArgument::NullPtr:
    collectShimDeclsForType(arg.getNullPtrType(), state);
    return;
  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    if (const auto *tmpl = arg.getAsTemplateOrTemplatePattern().getAsTemplateDecl()) {
      collectShimDecl(tmpl, state);
    }
    return;
  case TemplateArgument::Integral:
    collectShimDeclsForType(arg.getIntegralType(), state);
    return;
  case TemplateArgument::StructuralValue:
    collectShimDeclsForType(arg.getStructuralValueType(), state);
    return;
  case TemplateArgument::Expression:
    collectExprDependencies(arg.getAsExpr(), state);
    return;
  case TemplateArgument::Pack:
    for (const auto &pack_arg : arg.pack_elements()) {
      collectTemplateArgumentDependencies(pack_arg, state);
    }
    return;
  }
}

void collectTemplateArgumentLocDependencies(const TemplateArgumentLoc &arg_loc, ShimState &state) {
  const TemplateArgument &arg = arg_loc.getArgument();
  switch (arg.getKind()) {
  case TemplateArgument::Null:
    return;
  case TemplateArgument::Type:
    if (TypeSourceInfo *tsi = arg_loc.getTypeSourceInfo()) {
      collectShimDeclsForTypeLoc(tsi->getTypeLoc(), state);
    } else {
      collectShimDeclsForType(arg.getAsType(), state);
    }
    return;
  case TemplateArgument::Declaration:
    if (Expr *expr = arg_loc.getSourceDeclExpression()) {
      collectExprDependencies(expr, state);
    }
    collectValueDependency(arg.getAsDecl(), state);
    return;
  case TemplateArgument::NullPtr:
    if (Expr *expr = arg_loc.getSourceNullPtrExpression()) {
      collectExprDependencies(expr, state);
    }
    collectShimDeclsForType(arg.getNullPtrType(), state);
    return;
  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    if (const auto *tmpl = arg.getAsTemplateOrTemplatePattern().getAsTemplateDecl()) {
      collectShimDecl(tmpl, state);
    }
    return;
  case TemplateArgument::Integral:
    if (Expr *expr = arg_loc.getSourceIntegralExpression()) {
      collectExprDependencies(expr, state);
    }
    collectShimDeclsForType(arg.getIntegralType(), state);
    return;
  case TemplateArgument::StructuralValue:
    if (Expr *expr = arg_loc.getSourceStructuralValueExpression()) {
      collectExprDependencies(expr, state);
    }
    collectShimDeclsForType(arg.getStructuralValueType(), state);
    return;
  case TemplateArgument::Expression:
    collectExprDependencies(arg_loc.getSourceExpression(), state);
    return;
  case TemplateArgument::Pack:
    for (const auto &pack_arg : arg.pack_elements()) {
      collectTemplateArgumentDependencies(pack_arg, state);
    }
    return;
  }
}

void collectShimDeclsForTypeLoc(TypeLoc tl, ShimState &state) {
  if (tl.isNull()) {
    return;
  }
  if (auto qualified = tl.getAs<QualifiedTypeLoc>()) {
    collectShimDeclsForTypeLoc(qualified.getUnqualifiedLoc(), state);
    return;
  }
  if (auto elaborated = tl.getAs<ElaboratedTypeLoc>()) {
    collectShimDeclsForTypeLoc(elaborated.getNamedTypeLoc(), state);
    return;
  }
  if (auto paren = tl.getAs<ParenTypeLoc>()) {
    collectShimDeclsForTypeLoc(paren.getInnerLoc(), state);
    return;
  }
  if (auto macro = tl.getAs<MacroQualifiedTypeLoc>()) {
    collectShimDeclsForTypeLoc(macro.getInnerLoc(), state);
    return;
  }
  if (auto ptr = tl.getAs<PointerTypeLoc>()) {
    collectShimDeclsForTypeLoc(ptr.getPointeeLoc(), state);
    return;
  }
  if (auto member_ptr = tl.getAs<MemberPointerTypeLoc>()) {
    collectShimDeclsForTypeLoc(member_ptr.getPointeeLoc(), state);
    return;
  }
  if (auto constant_array = tl.getAs<ConstantArrayTypeLoc>()) {
    collectExprDependencies(constant_array.getSizeExpr(), state);
    collectShimDeclsForTypeLoc(constant_array.getElementLoc(), state);
    return;
  }
  if (auto dependent_array = tl.getAs<DependentSizedArrayTypeLoc>()) {
    collectExprDependencies(dependent_array.getSizeExpr(), state);
    collectShimDeclsForTypeLoc(dependent_array.getElementLoc(), state);
    return;
  }
  if (auto template_spec = tl.getAs<TemplateSpecializationTypeLoc>()) {
    for (unsigned i = 0; i < template_spec.getNumArgs(); ++i) {
      collectTemplateArgumentLocDependencies(template_spec.getArgLoc(i), state);
    }
    collectShimDeclsForType(template_spec.getType(), state);
    return;
  }
  if (auto dependent_template_spec = tl.getAs<DependentTemplateSpecializationTypeLoc>()) {
    for (unsigned i = 0; i < dependent_template_spec.getNumArgs(); ++i) {
      collectTemplateArgumentLocDependencies(dependent_template_spec.getArgLoc(i), state);
    }
    collectShimDeclsForType(dependent_template_spec.getType(), state);
    return;
  }
  if (auto typedef_loc = tl.getAs<TypedefTypeLoc>()) {
    collectShimDecl(typedef_loc.getTypedefNameDecl(), state);
    return;
  }
  if (auto using_loc = tl.getAs<UsingTypeLoc>()) {
    collectShimDecl(using_loc.getFoundDecl(), state);
    return;
  }
  if (auto tag_loc = tl.getAs<TagTypeLoc>()) {
    collectShimDecl(tag_loc.getDecl(), state);
    return;
  }
  collectShimDeclsForType(tl.getType(), state);
}

void collectAnonymousTagDependencies(const TagDecl *tag, ShimState &state) {
  if (!tag || !tag->getDeclName().isEmpty()) {
    return;
  }
  if (const auto *record = llvm::dyn_cast<RecordDecl>(tag)) {
    if (const auto *def = record->getDefinition()) {
      for (const auto *field : def->fields()) {
        if (const auto *tsi = field->getTypeSourceInfo()) {
          collectShimDeclsForTypeLoc(tsi->getTypeLoc(), state);
        } else {
          collectShimDeclsForType(field->getType(), state);
        }
      }
    }
    return;
  }
  if (const auto *enum_decl = llvm::dyn_cast<EnumDecl>(tag)) {
    if (enum_decl->isCompleteDefinition()) {
      for (const auto *enumerator : enum_decl->enumerators()) {
        collectExprDependencies(enumerator->getInitExpr(), state);
      }
    }
  }
}

void collectRecordMemberDeclDependencies(const RecordDecl *record, ShimState &state);

void collectRecordDependencyFields(const RecordDecl *record, ShimState &state) {
  if (!record) {
    return;
  }
  const RecordDecl *def = record->getDefinition();
  if (!def) {
    return;
  }
  if (const auto *cxx_def = llvm::dyn_cast<CXXRecordDecl>(def)) {
    for (const CXXBaseSpecifier &base : cxx_def->bases()) {
      QualType base_type = base.getType();
      if (const auto *base_record = base_type->getAsCXXRecordDecl()) {
        collectRecordDependencyFields(base_record, state);
      } else {
        collectShimDeclsForType(base_type, state);
      }
    }
  }
  for (const auto *field : def->fields()) {
    if (const auto *tsi = field->getTypeSourceInfo()) {
      collectShimDeclsForTypeLoc(tsi->getTypeLoc(), state);
    } else {
      collectShimDeclsForType(field->getType(), state);
    }
  }
  collectRecordMemberDeclDependencies(def, state);
}

void collectRecordMemberDeclDependencies(const RecordDecl *record, ShimState &state) {
  if (!record) {
    return;
  }
  const RecordDecl *def = record->getDefinition();
  if (!def) {
    return;
  }
  for (const Decl *member : def->decls()) {
    if (!member || llvm::isa<FieldDecl>(member)) {
      continue;
    }
    if (const auto *nested_record = llvm::dyn_cast<RecordDecl>(member)) {
      if (nested_record != def) {
        collectRecordDependencyFields(nested_record, state);
      }
      continue;
    }
    if (const auto *class_template = llvm::dyn_cast<ClassTemplateDecl>(member)) {
      if (const auto *templated = class_template->getTemplatedDecl()) {
        collectRecordDependencyFields(templated, state);
      }
      continue;
    }
    if (const auto *enum_decl = llvm::dyn_cast<EnumDecl>(member)) {
      if (enum_decl->isCompleteDefinition()) {
        for (const auto *enumerator : enum_decl->enumerators()) {
          collectExprDependencies(enumerator->getInitExpr(), state);
        }
      }
      continue;
    }
    if (const auto *typedef_decl = llvm::dyn_cast<TypedefNameDecl>(member)) {
      collectShimDeclsForType(typedef_decl->getUnderlyingType(), state);
      continue;
    }
    if (const auto *var_decl = llvm::dyn_cast<VarDecl>(member)) {
      if (var_decl->isStaticDataMember()) {
        collectShimDeclsForType(var_decl->getType(), state);
        if (const Expr *init = var_decl->getInit()) {
          collectExprDependencies(init, state);
        }
      }
    }
  }
}

void collectShimDecl(const NamedDecl *decl, ShimState &state) {
  const Decl *printable = printableDecl(decl);
  if (!printable || state.seen.contains(printable)) {
    return;
  }
  SourceLocation loc = state.sm.getExpansionLoc(printable->getLocation());
  if (!state.sm.isWrittenInBuiltinFile(loc) && isHeaderBackedShimDecl(loc, state)) {
    recordHeaderBackedShimDependency(loc, state);
    return;
  }
  if (const auto *tag = llvm::dyn_cast<TagDecl>(printable)) {
    if (tag->getTagKind() == TagTypeKind::Union) {
      return;
    }
    if (tag->getDeclName().isEmpty()) {
      collectAnonymousTagDependencies(tag, state);
      if (llvm::isa<RecordDecl>(tag) || llvm::isa<EnumDecl>(tag)) {
        return;
      }
      addShimReason(state, "anonymous_type_unsupported", "anonymous decl");
      return;
    }
    if (const auto *record = llvm::dyn_cast<CXXRecordDecl>(tag)) {
      if (record->isLocalClass()) {
        addShimReason(state, "local_type_unsupported", namedDeclName(tag));
        return;
      }
    }
  }
  if (!isSupportedShimDeclContext(printable->getDeclContext())) {
    if (recordHeaderBackedShimDependency(loc, state)) {
      return;
    }
    if (const auto *owner = owningRecordForNestedDecl(decl)) {
      if (owner != decl) {
        collectShimDecl(owner, state);
        return;
      }
    }
    addShimReason(state, "scoped_type_shim_unsupported", namedDeclName(decl));
    return;
  }
  state.seen.insert(printable);

  if (const auto *class_template = llvm::dyn_cast<ClassTemplateDecl>(printable)) {
    if (const auto *templated = class_template->getTemplatedDecl()) {
      collectRecordDependencyFields(templated, state);
    }
  } else if (const auto *record = llvm::dyn_cast<RecordDecl>(printable)) {
    collectRecordDependencyFields(record, state);
  } else if (const auto *typedef_decl = llvm::dyn_cast<TypedefNameDecl>(printable)) {
    QualType underlying = typedef_decl->getUnderlyingType();
    const TagDecl *anon_tag = underlying.isNull() ? nullptr : underlying->getAsTagDecl();
    if (anon_tag && anon_tag->getDeclName().isEmpty()) {
      collectAnonymousTagDependencies(anon_tag, state);
    } else {
      collectShimDeclsForType(underlying, state);
    }
  } else if (const auto *var_decl = llvm::dyn_cast<VarDecl>(printable)) {
    if (!(var_decl->isConstexpr() || var_decl->getType().isConstQualified())) {
      addShimReason(state, "constexpr_dep_missing", var_decl->getQualifiedNameAsString());
      return;
    }
  }

  state.ordered.push_back(printable);
}

void collectShimDeclsForType(QualType qt, ShimState &state) {
  if (const auto *constant_array = state.ctx.getAsConstantArrayType(qt)) {
    collectExprDependencies(constant_array->getSizeExpr(), state);
    collectShimDeclsForType(constant_array->getElementType(), state);
    return;
  }
  if (const auto *dependent_array = llvm::dyn_cast<DependentSizedArrayType>(qt.getTypePtr())) {
    collectExprDependencies(dependent_array->getSizeExpr(), state);
    collectShimDeclsForType(dependent_array->getElementType(), state);
    return;
  }
  if (const auto *template_spec = llvm::dyn_cast<TemplateSpecializationType>(qt.getTypePtr())) {
    for (const auto &arg : template_spec->template_arguments()) {
      collectTemplateArgumentDependencies(arg, state);
    }
  } else if (const auto *dependent_template_spec = llvm::dyn_cast<DependentTemplateSpecializationType>(qt.getTypePtr())) {
    for (const auto &arg : dependent_template_spec->template_arguments()) {
      collectTemplateArgumentDependencies(arg, state);
    }
  }
  const NamedDecl *decl = namedDeclForType(qt, state.ctx);
  if (!decl) {
    return;
  }
  collectShimDecl(decl, state);
}

void renderTypeShim(SymbolInfo &info, const FunctionDecl *fd, ASTContext &ctx) {
  ShimState state{ctx, ctx.getSourceManager(), {}, {}, {}, {}, {}};
  for (const ParmVarDecl *param : fd->parameters()) {
    if (const auto *tsi = param->getTypeSourceInfo()) {
      collectShimDeclsForTypeLoc(tsi->getTypeLoc(), state);
    } else {
      QualType shim_type = param->getOriginalType();
      if (shim_type.isNull()) {
        shim_type = param->getType();
      }
      collectShimDeclsForType(shim_type, state);
    }
  }
  if (!state.reason_codes.empty()) {
    info.type_shim_status = "unsupported";
    info.type_shim_reason_codes.assign(state.reason_codes.begin(), state.reason_codes.end());
    info.type_shim_missing_dependencies.assign(state.missing_dependencies.begin(), state.missing_dependencies.end());
    info.type_shim.clear();
    return;
  }

  std::string body;
  llvm::raw_string_ostream os(body);
  PrintingPolicy policy = manifestTypePrintingPolicy(ctx);
  os << "#ifndef __KSMOKE_TYPE_SHIM_V1_CUH__\n";
  os << "#define __KSMOKE_TYPE_SHIM_V1_CUH__\n\n";
  for (const Decl *decl : state.ordered) {
    if (!decl) {
      continue;
    }
    std::string decl_text;
    if (const auto *typedef_decl = llvm::dyn_cast<TypedefNameDecl>(decl)) {
      QualType underlying = typedef_decl->getUnderlyingType();
      const TagDecl *anon_tag = underlying.isNull() ? nullptr : underlying->getAsTagDecl();
      if (anon_tag && anon_tag->getDeclName().isEmpty()) {
        std::string tag_text;
        llvm::raw_string_ostream tag_os(tag_text);
        const TagDecl *def = anon_tag->getDefinition() ? anon_tag->getDefinition() : anon_tag;
        def->print(tag_os, policy, 0, true);
        tag_os.flush();
        llvm::StringRef trimmed = llvm::StringRef(tag_text).rtrim();
        if (trimmed.ends_with(";")) {
          trimmed = trimmed.drop_back();
        }
        decl_text = std::string("typedef ") + trimmed.str() + " " + typedef_decl->getNameAsString() + ";";
      }
    }
    if (decl_text.empty()) {
      llvm::raw_string_ostream decl_os(decl_text);
      decl->print(decl_os, 0, true);
      decl_os.flush();
    }
    if (llvm::isa<TypedefNameDecl>(decl) || llvm::isa<RecordDecl>(decl) || llvm::isa<ClassTemplateDecl>(decl) || llvm::isa<EnumDecl>(decl)) {
      llvm::StringRef trimmed = llvm::StringRef(decl_text).rtrim();
      if (!trimmed.ends_with(";")) {
        decl_text += ";";
      }
    }
    os << renderDeclWithNamespaceContext(decl, llvm::StringRef(decl_text).rtrim()) << "\n\n";
  }
  os << "#endif // __KSMOKE_TYPE_SHIM_V1_CUH__\n";
  info.type_shim = os.str();
  info.type_shim_system_headers.assign(state.system_header_paths.begin(), state.system_header_paths.end());
  info.type_shim_system_includes.assign(state.system_include_stmts.begin(), state.system_include_stmts.end());
}

QualType peelNominalType(QualType qt, ASTContext &ctx) {
  qt = qt.getNonReferenceType();
  while (!qt.isNull()) {
    if (const auto *ptr = qt->getAs<PointerType>()) {
      qt = ptr->getPointeeType();
      continue;
    }
    if (const auto *member_ptr = qt->getAs<MemberPointerType>()) {
      qt = member_ptr->getPointeeType();
      continue;
    }
    if (const auto *array = ctx.getAsArrayType(qt)) {
      qt = array->getElementType();
      continue;
    }
    if (const auto *paren = llvm::dyn_cast<ParenType>(qt.getTypePtr())) {
      qt = paren->getInnerType();
      continue;
    }
    if (const auto *adjusted = llvm::dyn_cast<AdjustedType>(qt.getTypePtr())) {
      qt = adjusted->getAdjustedType();
      continue;
    }
    if (const auto *macro = llvm::dyn_cast<MacroQualifiedType>(qt.getTypePtr())) {
      qt = macro->getUnderlyingType();
      continue;
    }
    if (const auto *elaborated = llvm::dyn_cast<ElaboratedType>(qt.getTypePtr())) {
      qt = elaborated->getNamedType();
      continue;
    }
    QualType desugared = qt.getSingleStepDesugaredType(ctx);
    if (desugared == qt) {
      break;
    }
    qt = desugared;
  }
  return qt.getUnqualifiedType();
}

llvm::json::Value typeInfoForQualType(QualType spelled_type, QualType semantic_type, ASTContext &ctx) {
  QualType nominal_type = peelNominalType(spelled_type, ctx);
  if (nominal_type.isNull()) {
    nominal_type = peelNominalType(semantic_type, ctx);
  }

  const TagDecl *tag_decl = nominal_type.isNull() ? nullptr : nominal_type->getAsTagDecl();
  const NamedDecl *named_decl = tag_decl ? llvm::dyn_cast<NamedDecl>(tag_decl->getCanonicalDecl()) : nullptr;
  if (!named_decl) {
    return llvm::json::Value(llvm::json::Object{});
  }

  llvm::json::Object decl{{"kind", declKindString(named_decl)},
                          {"qualified_name", namedDeclName(named_decl)}};
  if (auto usr = declUsr(named_decl)) {
    decl["usr"] = *usr;
  }

  const auto &sm = ctx.getSourceManager();
  if (auto loc = sourceLoc(named_decl->getLocation(), sm)) {
    decl["decl_loc"] = toJson(*loc);
  }

  llvm::json::Object definition;
  if (const auto *record = llvm::dyn_cast<RecordDecl>(named_decl)) {
    if (const auto *def = record->getDefinition()) {
      definition["status"] = "available";
      if (auto loc = sourceLoc(def->getLocation(), sm)) {
        definition["loc"] = toJson(*loc);
      }
    } else {
      definition["status"] = "declaration_only";
    }
  } else if (const auto *enum_decl = llvm::dyn_cast<EnumDecl>(named_decl)) {
    if (enum_decl->isCompleteDefinition()) {
      definition["status"] = "available";
      if (auto loc = sourceLoc(enum_decl->getLocation(), sm)) {
        definition["loc"] = toJson(*loc);
      }
    } else {
      definition["status"] = "declaration_only";
    }
  }
  if (!definition.empty()) {
    decl["definition"] = std::move(definition);
  }

  return llvm::json::Value(std::move(decl));
}

llvm::json::Value typeInfoFor(const ParmVarDecl *param, ASTContext &ctx) {
  QualType spelled_type = param->getTypeSourceInfo() ? param->getTypeSourceInfo()->getType() : param->getOriginalType();
  return typeInfoForQualType(spelled_type, param->getType(), ctx);
}

std::string apsIntToDecimalString(const llvm::APSInt &value) {
  llvm::SmallString<32> out;
  value.toString(out, 10);
  return std::string(out.str());
}

const EnumDecl *enumDeclFor(QualType type, ASTContext &ctx) {
  QualType canonical = ctx.getCanonicalType(type.getNonReferenceType());
  if (const auto *enum_type = canonical->getAs<EnumType>()) {
    return enum_type->getDecl();
  }
  return nullptr;
}

std::optional<llvm::json::Object> enumDomainFor(QualType type, ASTContext &ctx) {
  const EnumDecl *enum_decl = enumDeclFor(type, ctx);
  if (!enum_decl) {
    return std::nullopt;
  }
  const EnumDecl *def = enum_decl->getDefinition();
  if (!def || !def->isCompleteDefinition()) {
    return std::nullopt;
  }

  llvm::json::Array values;
  for (const EnumConstantDecl *enumerator : def->enumerators()) {
    values.emplace_back(llvm::json::Object{{"name", llvm::json::fixUTF8(enumerator->getNameAsString())},
                                           {"value", apsIntToDecimalString(enumerator->getInitVal())}});
  }
  return llvm::json::Object{{"kind", "enum"},
                            {"values", std::move(values)},
                            {"allow_unknown", false}};
}

std::string rawLayoutKindFor(QualType type, ASTContext &ctx) {
  QualType canonical = type.getCanonicalType();
  if (canonical->isPointerType()) {
    return "pointer";
  }
  if (canonical->isArithmeticType() || canonical->isEnumeralType()) {
    return "scalar";
  }
  if (ctx.getAsConstantArrayType(canonical)) {
    return "array";
  }
  if (const auto *record_type = canonical->getAs<RecordType>()) {
    return record_type->getDecl()->isUnion() ? "union" : "record";
  }
  return "opaque";
}

std::uint64_t typeSizeBytes(QualType type, ASTContext &ctx) {
  return static_cast<std::uint64_t>(ctx.getTypeSizeInChars(type).getQuantity());
}

std::uint64_t typeAlignBytes(QualType type, ASTContext &ctx) {
  return static_cast<std::uint64_t>(ctx.getTypeAlignInChars(type).getQuantity());
}

llvm::json::Object layoutNodeForType(const std::string &name, QualType type,
                                     ASTContext &ctx, unsigned depth,
                                     const std::string &index);

bool nodeSubtreeContainsPointer(const llvm::json::Object &node);

bool nodeValueIsPtrBearing(const llvm::json::Value &value) {
  if (const auto *obj = value.getAsObject()) {
    return nodeSubtreeContainsPointer(*obj);
  }
  return true;
}

bool nodeSubtreeContainsPointer(const llvm::json::Object &node) {
  if (auto kind = node.getString("kind")) {
    if (*kind == "pointer" || *kind == "opaque_with_ptr") {
      return true;
    }
  }
  if (auto status = node.getString("layout_status")) {
    if (*status == "partial" || *status == "opaque") {
      return true;
    }
  }
  if (const auto *fields = node.getArray("fields")) {
    for (const auto &field : *fields) {
      if (nodeValueIsPtrBearing(field)) {
        return true;
      }
    }
  }
  if (const auto *element = node.get("element")) {
    if (nodeValueIsPtrBearing(*element)) {
      return true;
    }
  }
  return false;
}

void normalizeAggregateKind(llvm::json::Object &node) {
  if (auto kind = node.getString("kind")) {
    if (*kind == "union" || *kind == "opaque") {
      node["kind"] = "opaque_with_ptr";
    } else if (*kind == "record" || *kind == "array") {
      node["kind"] = nodeSubtreeContainsPointer(node) ? "opaque_with_ptr" : "opaque_val";
    }
  }
}

std::string argKindFor(QualType type, ASTContext &ctx, const llvm::json::Value &type_layout) {
  QualType canonical = type.getCanonicalType();
  if (canonical->isPointerType()) {
    return "pointer";
  }
  if (canonical->isArithmeticType() || canonical->isEnumeralType()) {
    return "scalar";
  }
  std::string raw_kind = rawLayoutKindFor(type, ctx);
  if (raw_kind == "union" || raw_kind == "opaque") {
    return "opaque_with_ptr";
  }
  if (const auto *layout = type_layout.getAsObject()) {
    return nodeSubtreeContainsPointer(*layout) ? "opaque_with_ptr" : "opaque_val";
  }
  return "opaque_val";
}

void addMaterializationFact(MaterializationFacts &facts, llvm::StringRef code,
                            llvm::StringRef detail = "") {
  facts.reason_codes.insert(code.str());
  if (!detail.empty()) {
    facts.blockers.insert(detail.str());
  }
}

void mergeMaterializationFacts(MaterializationFacts &dst, const MaterializationFacts &src) {
  dst.reason_codes.insert(src.reason_codes.begin(), src.reason_codes.end());
  dst.blockers.insert(src.blockers.begin(), src.blockers.end());
}

std::vector<std::string> materializationSetToVector(const std::set<std::string> &values) {
  return std::vector<std::string>(values.begin(), values.end());
}

void assignMaterializationFacts(const MaterializationFacts &facts, std::string &status,
                                std::vector<std::string> &reason_codes,
                                std::vector<std::string> &blockers) {
  status = facts.reason_codes.empty() ? "ok" : "unsafe";
  reason_codes = materializationSetToVector(facts.reason_codes);
  blockers = materializationSetToVector(facts.blockers);
}

void appendMaterializationFacts(llvm::json::Object &obj, const MaterializationFacts &facts) {
  if (facts.reason_codes.empty() && facts.blockers.empty()) {
    return;
  }
  llvm::json::Array reason_codes;
  llvm::json::Array blockers;
  for (const auto &code : facts.reason_codes) {
    reason_codes.emplace_back(llvm::json::fixUTF8(code));
  }
  for (const auto &blocker : facts.blockers) {
    blockers.emplace_back(llvm::json::fixUTF8(blocker));
  }
  obj["materialization_status"] = "unsafe";
  obj["materialization_reason_codes"] = std::move(reason_codes);
  obj["materialization_blockers"] = std::move(blockers);
}

std::string fieldAccessName(AccessSpecifier access) {
  switch (access) {
  case AS_public:
    return "public";
  case AS_protected:
    return "protected";
  case AS_private:
    return "private";
  case AS_none:
    return "none";
  }
}

bool isConstStorageType(QualType type, ASTContext &ctx) {
  QualType canonical = type.getCanonicalType();
  if (const auto *array_type = ctx.getAsArrayType(canonical)) {
    return isConstStorageType(array_type->getElementType(), ctx);
  }
  return canonical.isConstQualified();
}

std::string materializationPathForChild(const std::string &parent, const std::string &child) {
  if (parent.empty()) {
    return child;
  }
  if (child.empty()) {
    return parent;
  }
  return parent + "." + child;
}

MaterializationFacts materializationFactsForType(QualType type, ASTContext &ctx,
                                                 const std::string &path, unsigned depth,
                                                 llvm::SmallPtrSetImpl<const RecordDecl *> &seen);

MaterializationFacts materializationFactsForRecord(const RecordDecl *record, ASTContext &ctx,
                                                   const std::string &path, unsigned depth,
                                                   llvm::SmallPtrSetImpl<const RecordDecl *> &seen) {
  MaterializationFacts facts;
  const RecordDecl *def = record ? record->getDefinition() : nullptr;
  if (!def || !def->isCompleteDefinition()) {
    addMaterializationFact(facts, "incomplete_record", path + ": record definition is incomplete");
    return facts;
  }
  const auto *canonical_def = llvm::cast<RecordDecl>(def->getCanonicalDecl());
  if (seen.contains(canonical_def)) {
    return facts;
  }
  seen.insert(canonical_def);

  if (const auto *cxx_def = llvm::dyn_cast<CXXRecordDecl>(def)) {
    if (!cxx_def->isTriviallyCopyable()) {
      addMaterializationFact(facts, "non_trivially_copyable",
                             path + ": record is not trivially copyable");
    }
    if (cxx_def->isPolymorphic()) {
      addMaterializationFact(facts, "virtual_method_or_vptr",
                             path + ": record is polymorphic or carries a vptr");
    }
    for (const CXXBaseSpecifier &base : cxx_def->bases()) {
      std::string base_name = base.getType().getAsString(manifestTypePrintingPolicy(ctx));
      std::string base_path = materializationPathForChild(path, base_name);
      if (base.isVirtual()) {
        addMaterializationFact(facts, "virtual_base",
                               base_path + ": virtual base cannot be field-wise materialized");
      }
      if (base.getAccessSpecifier() == AS_private || base.getAccessSpecifier() == AS_protected) {
        addMaterializationFact(
            facts, "non_public_base",
            base_path + ": " + fieldAccessName(base.getAccessSpecifier()) +
                " base cannot be directly written by generated decode");
      }
      if (!base.isVirtual() && base.getAccessSpecifier() == AS_public) {
        if (const auto *base_record = base.getType()->getAsCXXRecordDecl()) {
          mergeMaterializationFacts(
              facts, materializationFactsForRecord(base_record, ctx, base_path, depth + 1, seen));
        }
      }
    }
  }

  unsigned field_ordinal = 0;
  for (const FieldDecl *field : def->fields()) {
    std::string field_name = field->getNameAsString();
    if (field_name.empty()) {
      field_name = "anon" + std::to_string(field_ordinal);
    }
    std::string field_path = materializationPathForChild(path, field_name);
    AccessSpecifier access = field->getAccess();
    if (access == AS_private) {
      addMaterializationFact(facts, "private_data_field",
                             field_path + ": private data field cannot be directly written");
    } else if (access == AS_protected) {
      addMaterializationFact(facts, "protected_data_field",
                             field_path + ": protected data field cannot be directly written");
    }
    if (field->getType()->isReferenceType()) {
      addMaterializationFact(facts, "reference_field",
                             field_path + ": reference data field cannot be rebound by decode");
    }
    if (isConstStorageType(field->getType(), ctx)) {
      addMaterializationFact(facts, "const_assignment_blocker",
                             field_path + ": const data field cannot be assigned after construction");
    }
    mergeMaterializationFacts(
        facts, materializationFactsForType(field->getType(), ctx, field_path, depth + 1, seen));
    ++field_ordinal;
  }

  seen.erase(canonical_def);
  return facts;
}

MaterializationFacts materializationFactsForType(QualType type, ASTContext &ctx,
                                                 const std::string &path, unsigned depth,
                                                 llvm::SmallPtrSetImpl<const RecordDecl *> &seen) {
  constexpr unsigned kMaxMaterializationDepth = 8;
  MaterializationFacts facts;
  if (type.isNull() || depth >= kMaxMaterializationDepth) {
    return facts;
  }
  QualType canonical = type.getCanonicalType();
  if (canonical->isReferenceType()) {
    addMaterializationFact(facts, "reference_type", path + ": reference type cannot be materialized");
    return facts;
  }
  if (canonical->isPointerType() || canonical->isArithmeticType() || canonical->isEnumeralType()) {
    return facts;
  }
  if (const auto *array_type = ctx.getAsConstantArrayType(canonical)) {
    return materializationFactsForType(array_type->getElementType(), ctx, path + "[]", depth + 1, seen);
  }
  if (const auto *record_type = canonical->getAs<RecordType>()) {
    return materializationFactsForRecord(record_type->getDecl(), ctx, path, depth + 1, seen);
  }
  return facts;
}

MaterializationFacts materializationFactsForType(QualType type, ASTContext &ctx,
                                                 const std::string &path) {
  llvm::SmallPtrSet<const RecordDecl *, 16> seen;
  return materializationFactsForType(type, ctx, path, 0, seen);
}

MaterializationFacts materializationFactsForField(const FieldDecl *field, ASTContext &ctx,
                                                  const std::string &path) {
  MaterializationFacts facts;
  if (!field) {
    return facts;
  }
  AccessSpecifier access = field->getAccess();
  if (access == AS_private) {
    addMaterializationFact(facts, "private_data_field",
                           path + ": private data field cannot be directly written");
  } else if (access == AS_protected) {
    addMaterializationFact(facts, "protected_data_field",
                           path + ": protected data field cannot be directly written");
  }
  if (field->getType()->isReferenceType()) {
    addMaterializationFact(facts, "reference_field",
                           path + ": reference data field cannot be rebound by decode");
  }
  if (isConstStorageType(field->getType(), ctx)) {
    addMaterializationFact(facts, "const_assignment_blocker",
                           path + ": const data field cannot be assigned after construction");
  }
  mergeMaterializationFacts(facts, materializationFactsForType(field->getType(), ctx, path));
  return facts;
}

llvm::json::Object layoutNodeForField(const FieldDecl *field, std::uint64_t offset_bits,
                                      const std::string &field_name, const std::string &index,
                                      ASTContext &ctx, unsigned depth) {
  llvm::json::Object node = layoutNodeForType(field_name, field->getType(), ctx, depth, index);
  appendMaterializationFacts(node, materializationFactsForField(field, ctx, index));
  if (field->isBitField()) {
    node["bit_offset"] = static_cast<std::int64_t>(offset_bits);
    node["bit_width"] = static_cast<std::int64_t>(field->getBitWidthValue());
  }
  return node;
}

void appendRecordFieldsFor(const RecordDecl *def, const std::string &parent_index,
                           ASTContext &ctx, unsigned depth, bool &complete,
                           llvm::json::Array &fields) {
  if (!def || !def->isCompleteDefinition()) {
    complete = false;
    return;
  }

  const ASTRecordLayout &layout = ctx.getASTRecordLayout(def);
  if (const auto *cxx_def = llvm::dyn_cast<CXXRecordDecl>(def)) {
    for (const CXXBaseSpecifier &base : cxx_def->bases()) {
      if (base.isVirtual() || base.getAccessSpecifier() != AS_public) {
        complete = false;
        continue;
      }
      QualType base_type = base.getType();
      const CXXRecordDecl *base_record = base_type->getAsCXXRecordDecl();
      const RecordDecl *base_def = base_record ? base_record->getDefinition() : nullptr;
      if (!base_def || !base_def->isCompleteDefinition()) {
        complete = false;
        continue;
      }
      appendRecordFieldsFor(base_def, parent_index, ctx, depth, complete, fields);
    }
  }

  unsigned field_ordinal = 0;
  for (const FieldDecl *field : def->fields()) {
    std::uint64_t offset_bits = static_cast<std::uint64_t>(layout.getFieldOffset(field_ordinal));
    std::string field_name = field->getNameAsString();
    if (field_name.empty()) {
      field_name = "anon" + std::to_string(field_ordinal);
    }
    std::string child_index = parent_index.empty() ? field_name : parent_index + "." + field_name;
    llvm::json::Object field_node = layoutNodeForField(field, offset_bits, field_name, child_index, ctx, depth + 1);
    if (auto status = field_node.getString("layout_status")) {
      if (*status != "complete") {
        complete = false;
      }
    }
    fields.emplace_back(std::move(field_node));
    ++field_ordinal;
  }
}

llvm::json::Array recordFieldsFor(const RecordDecl *def, const std::string &parent_index,
                                  ASTContext &ctx, unsigned depth,
                                  bool &complete) {
  llvm::json::Array fields;
  complete = true;
  appendRecordFieldsFor(def, parent_index, ctx, depth, complete, fields);
  return fields;
}

llvm::json::Object layoutNodeForType(const std::string &name, QualType type,
                                     ASTContext &ctx, unsigned depth,
                                     const std::string &index) {
  constexpr unsigned kMaxLayoutDepth = 8;
  PrintingPolicy policy(ctx.getLangOpts());
  std::string raw_kind = rawLayoutKindFor(type, ctx);
  llvm::json::Object node{{"name", llvm::json::fixUTF8(name)},
                          {"index", llvm::json::fixUTF8(index)},
                          {"type", llvm::json::fixUTF8(manifestTypeName(type, ctx))},
                          {"kind", raw_kind},
                          {"size_bytes", static_cast<std::int64_t>(typeSizeBytes(type, ctx))},
                          {"align_bytes", static_cast<std::int64_t>(typeAlignBytes(type, ctx))}};
  llvm::json::Value type_info = typeInfoForQualType(type, type, ctx);
  if (const auto *type_info_obj = type_info.getAsObject()) {
    if (!type_info_obj->empty()) {
      node["type_info"] = std::move(type_info);
    }
  }
  appendMaterializationFacts(node, materializationFactsForType(type, ctx, index));

  if (auto domain = enumDomainFor(type, ctx)) {
    node["domain"] = std::move(*domain);
  }

  if (depth >= kMaxLayoutDepth) {
    node["kind"] = "opaque_with_ptr";
    node["layout_status"] = "partial";
    return node;
  }

  if (raw_kind == "pointer") {
    node["pointer_role"] = "payload_buffer";
    node["pointee_layout"] = layoutNodeForType("$pointee", type->getPointeeType(), ctx, depth + 1,
                                               index + ".*");
  } else if (raw_kind == "record" || raw_kind == "union") {
    const auto *record_type = type.getCanonicalType()->getAs<RecordType>();
    const RecordDecl *def = record_type ? record_type->getDecl()->getDefinition() : nullptr;
    bool complete = false;
    llvm::json::Array fields = recordFieldsFor(def, index, ctx, depth, complete);
    node["layout_status"] = complete ? "complete" : "partial";
    node["fields"] = std::move(fields);
    normalizeAggregateKind(node);
  } else if (raw_kind == "array") {
    const auto *array_type = ctx.getAsConstantArrayType(type.getCanonicalType());
    if (array_type) {
      node["layout_status"] = "complete";
      node["element_count"] = static_cast<std::int64_t>(array_type->getSize().getLimitedValue());
      node["element"] = layoutNodeForType("$element", array_type->getElementType(), ctx, depth + 1,
                                          index + "[]");
      normalizeAggregateKind(node);
    } else {
      node["layout_status"] = "opaque";
      node["kind"] = "opaque_with_ptr";
    }
  } else if (raw_kind == "opaque") {
    node["kind"] = "opaque_with_ptr";
    node["layout_status"] = "opaque";
  }
  return node;
}

llvm::json::Value typeLayoutFor(QualType type, const std::string &arg_name, ASTContext &ctx) {
  std::string raw_kind = rawLayoutKindFor(type, ctx);
  if (raw_kind != "record" && raw_kind != "union" && raw_kind != "array" && raw_kind != "opaque") {
    return nullptr;
  }
  if (raw_kind == "opaque") {
    return llvm::json::Object{{"layout_status", "opaque"}};
  }

  llvm::json::Object root;
  if (raw_kind == "record" || raw_kind == "union") {
    const auto *record_type = type.getCanonicalType()->getAs<RecordType>();
    const RecordDecl *def = record_type ? record_type->getDecl()->getDefinition() : nullptr;
    bool complete = false;
    llvm::json::Array fields = recordFieldsFor(def, arg_name, ctx, 0, complete);
    root["layout_status"] = complete ? "complete" : "partial";
    root["fields"] = std::move(fields);
    appendMaterializationFacts(root, materializationFactsForType(type, ctx, arg_name));
    return llvm::json::Value(std::move(root));
  }

  const auto *array_type = ctx.getAsConstantArrayType(type.getCanonicalType());
  if (!array_type) {
    return llvm::json::Object{{"layout_status", "opaque"}};
  }
  root["layout_status"] = "complete";
  root["element_count"] = static_cast<std::int64_t>(array_type->getSize().getLimitedValue());
  root["element"] = layoutNodeForType("$element", array_type->getElementType(), ctx, 1,
                                      arg_name + "[]");
  appendMaterializationFacts(root, materializationFactsForType(type, ctx, arg_name));
  return llvm::json::Value(std::move(root));
}

MaterializationFacts materializationFactsForParam(const ParmVarDecl *param, ASTContext &ctx,
                                                  const std::string &name) {
  if (!param) {
    return MaterializationFacts{};
  }
  return materializationFactsForType(param->getType(), ctx, name);
}

MaterializationFacts materializationFactsForParams(const std::vector<ParamInfo> &params) {
  MaterializationFacts facts;
  for (const auto &param : params) {
    for (const auto &code : param.materialization_reason_codes) {
      facts.reason_codes.insert(code);
    }
    for (const auto &blocker : param.materialization_blockers) {
      facts.blockers.insert(blocker);
    }
  }
  return facts;
}

std::vector<ParamInfo> paramsFor(const FunctionDecl *fd, ASTContext &ctx) {
  std::vector<ParamInfo> out;
  int index = 0;
  for (const ParmVarDecl *param : fd->parameters()) {
    ParamInfo info;
    info.index = index++;
    info.name = param->getNameAsString();
    if (info.name.empty()) {
      info.name = "unnamed";
    }
    info.type = manifestTypeName(param->getOriginalType(), ctx);
    info.type_layout = typeLayoutFor(param->getType(), info.name, ctx);
    info.kind = argKindFor(param->getType(), ctx, info.type_layout);
    if (info.kind == "pointer") {
      info.pointer_role = "payload_buffer";
      info.pointee_layout = layoutNodeForType("$pointee", param->getType()->getPointeeType(), ctx, 1,
                                              info.name + ".*");
    }
    info.size_bytes = static_cast<std::uint64_t>(ctx.getTypeSizeInChars(param->getType()).getQuantity());
    info.align_bytes = static_cast<std::uint64_t>(ctx.getTypeAlignInChars(param->getType()).getQuantity());
    if (auto domain = enumDomainFor(param->getType(), ctx)) {
      info.domain = llvm::json::Value(std::move(*domain));
    }
    info.type_info = typeInfoFor(param, ctx);
    assignMaterializationFacts(materializationFactsForParam(param, ctx, info.name),
                               info.materialization_status,
                               info.materialization_reason_codes,
                               info.materialization_blockers);
    out.push_back(std::move(info));
  }
  return out;
}

llvm::json::Value toJson(const SymbolInfo &info) {
  llvm::json::Array args;
  llvm::json::Array reason_codes;
  llvm::json::Array missing_dependencies;
  llvm::json::Array system_headers;
  llvm::json::Array system_includes;
  for (const auto &arg : info.args) {
    llvm::json::Object arg_json{{"index", arg.index},
                                {"name", llvm::json::fixUTF8(arg.name)},
                                {"type", llvm::json::fixUTF8(arg.type)},
                                {"kind", llvm::json::fixUTF8(arg.kind)},
                                {"size_bytes", static_cast<std::int64_t>(arg.size_bytes)},
                                {"align_bytes", static_cast<std::int64_t>(arg.align_bytes)}};
    if (arg.kind == "pointer" && !arg.pointer_role.empty()) {
      arg_json["pointer_role"] = llvm::json::fixUTF8(arg.pointer_role);
      arg_json["pointee_layout"] = arg.pointee_layout;
    }
    arg_json["domain"] = arg.domain;
    arg_json["type_layout"] = arg.type_layout;
    if (const auto *type_info_obj = arg.type_info.getAsObject()) {
      if (!type_info_obj->empty()) {
        arg_json["type_info"] = arg.type_info;
      }
    }
    if (arg.materialization_status == "unsafe" ||
        !arg.materialization_reason_codes.empty() ||
        !arg.materialization_blockers.empty()) {
      llvm::json::Array materialization_reason_codes;
      llvm::json::Array materialization_blockers;
      for (const auto &code : arg.materialization_reason_codes) {
        materialization_reason_codes.emplace_back(llvm::json::fixUTF8(code));
      }
      for (const auto &blocker : arg.materialization_blockers) {
        materialization_blockers.emplace_back(llvm::json::fixUTF8(blocker));
      }
      arg_json["materialization_status"] = llvm::json::fixUTF8(arg.materialization_status);
      arg_json["materialization_reason_codes"] = std::move(materialization_reason_codes);
      arg_json["materialization_blockers"] = std::move(materialization_blockers);
    }
    args.emplace_back(std::move(arg_json));
  }
  for (const auto &code : info.type_shim_reason_codes) {
    reason_codes.emplace_back(llvm::json::fixUTF8(code));
  }
  for (const auto &dep : info.type_shim_missing_dependencies) {
    missing_dependencies.emplace_back(llvm::json::fixUTF8(dep));
  }
  for (const auto &header : info.type_shim_system_headers) {
    system_headers.emplace_back(llvm::json::fixUTF8(header));
  }
  for (const auto &include_stmt : info.type_shim_system_includes) {
    system_includes.emplace_back(llvm::json::fixUTF8(include_stmt));
  }
  llvm::json::Array materialization_reason_codes;
  llvm::json::Array materialization_blockers;
  for (const auto &code : info.materialization_reason_codes) {
    materialization_reason_codes.emplace_back(llvm::json::fixUTF8(code));
  }
  for (const auto &blocker : info.materialization_blockers) {
    materialization_blockers.emplace_back(llvm::json::fixUTF8(blocker));
  }
  llvm::json::Object result{{"qualified_name", llvm::json::fixUTF8(info.qualified_name)},
                            {"source_file", llvm::json::fixUTF8(info.source_file)},
                            {"line", static_cast<std::int64_t>(info.line)},
                            {"args", std::move(args)},
                            {"type_shim", llvm::json::fixUTF8(info.type_shim)},
                            {"type_shim_status", llvm::json::fixUTF8(info.type_shim_status)},
                            {"type_shim_reason_codes", std::move(reason_codes)},
                            {"type_shim_missing_dependencies", std::move(missing_dependencies)},
                            {"type_shim_system_headers", std::move(system_headers)},
                            {"type_shim_system_includes", std::move(system_includes)}};
  if (info.materialization_status == "unsafe" ||
      !info.materialization_reason_codes.empty() ||
      !info.materialization_blockers.empty()) {
    result["materialization_status"] = llvm::json::fixUTF8(info.materialization_status);
    result["materialization_reason_codes"] = std::move(materialization_reason_codes);
    result["materialization_blockers"] = std::move(materialization_blockers);
  }
  return result;
}

class KernelVisitor : public RecursiveASTVisitor<KernelVisitor> {
public:
  KernelVisitor(ASTContext &ctx, RunState &state) : Ctx(ctx), State(state) {}

  bool VisitFunctionDecl(FunctionDecl *fd) {
    record(fd);
    return true;
  }

  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *ftd) {
    if (!ftd) {
      return true;
    }
    for (auto it = ftd->spec_begin(); it != ftd->spec_end(); ++it) {
      record(*it);
    }
    return true;
  }

private:
  void record(FunctionDecl *fd) {
    if (!isInterestingKernel(fd)) {
      return;
    }

    std::string mangled = mangleDecl(Ctx, fd);
    if (State.wanted_symbols.find(mangled) == State.wanted_symbols.end()) {
      return;
    }
    if (State.matched.find(mangled) != State.matched.end()) {
      return;
    }

    SymbolInfo info;
    info.qualified_name = fd->getQualifiedNameAsString();
    info.source_file = sourceFile(fd, Ctx);
    info.line = sourceLine(fd, Ctx);
    info.args = paramsFor(fd, Ctx);
    assignMaterializationFacts(materializationFactsForParams(info.args),
                               info.materialization_status,
                               info.materialization_reason_codes,
                               info.materialization_blockers);
    renderTypeShim(info, fd, Ctx);
    State.matched.emplace(mangled, std::move(info));
  }

  ASTContext &Ctx;
  RunState &State;
};

class KernelConsumer : public ASTConsumer {
public:
  explicit KernelConsumer(RunState &state) : State(state) {}

  void HandleTranslationUnit(ASTContext &ctx) override {
    KernelVisitor visitor(ctx, State);
    visitor.TraverseDecl(ctx.getTranslationUnitDecl());
  }

private:
  RunState &State;
};

class KernelAction : public ASTFrontendAction {
public:
  explicit KernelAction(RunState &state) : State(state) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &, llvm::StringRef) override {
    return std::make_unique<KernelConsumer>(State);
  }

private:
  RunState &State;
};

class KernelActionFactory : public FrontendActionFactory {
public:
  explicit KernelActionFactory(RunState &state) : State(state) {}

  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<KernelAction>(State);
  }

private:
  RunState &State;
};

llvm::json::Object buildResponse(const Request &req, const RunState &state, bool ok) {
  llvm::json::Object symbols;
  for (const auto &it : state.matched) {
    symbols[it.first] = toJson(it.second);
  }

  llvm::json::Array missing;
  for (const auto &sym : req.ptx_symbols) {
    if (state.matched.find(sym) == state.matched.end()) {
      missing.emplace_back(sym);
    }
  }

  llvm::json::Array diagnostics;
  for (const auto &diag : state.diagnostics) {
    diagnostics.emplace_back(llvm::json::fixUTF8(diag));
  }

  return llvm::json::Object{{"ok", ok},
                            {"diagnostics", std::move(diagnostics)},
                            {"symbols", std::move(symbols)},
                            {"missing_symbols", std::move(missing)}};
}

void finalizeShimIncludes(const Request &req, RunState &state) {
  const std::vector<IncludeRoot> roots = parseIncludeRoots(req);
  for (auto &entry : state.matched) {
    SymbolInfo &info = entry.second;
    std::set<std::string> include_set(info.type_shim_system_includes.begin(), info.type_shim_system_includes.end());
    std::set<std::string> header_set(info.type_shim_system_headers.begin(), info.type_shim_system_headers.end());
    for (const auto &arg : info.args) {
      const auto *type_info = arg.type_info.getAsObject();
      if (!type_info) {
        continue;
      }
      if (!type_info->get("kind")) {
        continue;
      }
      std::string chosen_header;
      if (const auto *definition_value = type_info->get("definition")) {
        if (const auto *definition = definition_value->getAsObject()) {
          if (const auto *loc_value = definition->get("loc")) {
            if (const auto *loc = loc_value->getAsObject()) {
              if (const auto *file_value = loc->get("file")) {
                if (auto file = file_value->getAsString()) {
                  chosen_header = std::string(*file);
                }
              }
            }
          }
        }
      }
      if (chosen_header.empty()) {
        if (const auto *loc_value = type_info->get("decl_loc")) {
          if (const auto *loc = loc_value->getAsObject()) {
            if (const auto *file_value = loc->get("file")) {
              if (auto file = file_value->getAsString()) {
                chosen_header = std::string(*file);
              }
            }
          }
        }
      }
      if (!chosen_header.empty() && chosen_header != info.source_file) {
        header_set.insert(chosen_header);
      }
    }
    info.type_shim_system_headers.assign(header_set.begin(), header_set.end());
    for (const auto &header_path : info.type_shim_system_headers) {
      if (auto include_stmt = bestIncludeHint(header_path, roots)) {
        include_set.insert(*include_stmt);
      }
    }
    info.type_shim_system_includes.assign(include_set.begin(), include_set.end());
  }
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  if (argc != 2) {
    llvm::errs() << "usage: clang-entry-metadata <request.json>\n";
    return 2;
  }

  std::string error;
  auto req = loadRequest(argv[1], error);
  if (!req) {
    llvm::errs() << error << "\n";
    return 2;
  }

  FixedCompilationDatabase compdb(req->cwd, req->args);
  ClangTool tool(compdb, {req->source_file});
  RunState state;
  state.wanted_symbols = req->ptx_symbols;

  class DiagnosticCapture : public DiagnosticConsumer {
  public:
    explicit DiagnosticCapture(RunState &state) : State(state) {}

    void HandleDiagnostic(DiagnosticsEngine::Level level, const Diagnostic &info) override {
      llvm::SmallString<256> message;
      info.FormatDiagnostic(message);
      if (level >= DiagnosticsEngine::Error) {
        State.diagnostics.push_back(std::string(message));
      }
    }

  private:
    RunState &State;
  } diagConsumer(state);

  tool.setDiagnosticConsumer(&diagConsumer);
  KernelActionFactory factory(state);
  int rc = tool.run(&factory);

  finalizeShimIncludes(*req, state);
  auto response = buildResponse(*req, state, rc == 0);
  llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(response)));
  return rc;
}
