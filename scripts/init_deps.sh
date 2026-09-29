#!/usr/bin/env bash
# Initialize build dependencies: LibAFL submodule + patches, rapid-llvm
# toolchain. Idempotent; safe to re-run.
#
# Why not plain `git submodule update --init rapid-llvm LibAFL`?
# LibAFL works that way, but rapid-llvm is pinned at the llvmorg-21.1.8 tag
# commit, and GitHub refuses a shallow fetch of that commit by SHA
# ("upload-pack: not our ref"). Fetch the tag explicitly instead.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

git submodule update --init LibAFL
if git -C LibAFL log --oneline -3 2>/dev/null | grep -q "Fix Rust compiler compatibility"; then
  echo "LibAFL: patches already applied"
else
  (cd LibAFL && git am ../patches/libafl/*.patch)
  echo "LibAFL: patches applied"
fi

expected=$(git ls-tree HEAD rapid-llvm | awk '{print $3}')
current=$(git -C rapid-llvm rev-parse HEAD 2>/dev/null || true)
if [[ $current == "$expected" ]]; then
  echo "rapid-llvm: already at llvmorg-21.1.8"
else
  mkdir -p rapid-llvm
  [[ -e rapid-llvm/.git ]] || git -C rapid-llvm init -q
  if ! git -C rapid-llvm remote get-url origin >/dev/null 2>&1; then
    git -C rapid-llvm remote add origin \
      "$(git config -f .gitmodules submodule.rapid-llvm.url)"
  fi
  if ! git -C rapid-llvm rev-parse -q --verify refs/tags/llvmorg-21.1.8 >/dev/null 2>&1; then
    git -C rapid-llvm fetch -q --depth 1 origin refs/tags/llvmorg-21.1.8
  fi
  git -C rapid-llvm checkout -q FETCH_HEAD
  echo "rapid-llvm: checked out llvmorg-21.1.8"
fi

git submodule status LibAFL rapid-llvm || true
