#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "$0")"
script_dir="$(dirname "$script_path")"
invoked_name="$(basename "$0")"
rapid_wrap="$script_dir/rapid-wrap"

if [[ ! -x "$rapid_wrap" ]]; then
  printf 'kernel-smoke wrapper missing rapid-wrap: %s\n' "$rapid_wrap" >&2
  exit 1
fi

find_in_path() {
  local name="$1"
  local dir candidate resolved
  IFS=':' read -r -a path_parts <<< "${PATH:-}"
  for dir in "${path_parts[@]}"; do
    [[ -z "$dir" ]] && dir='.'
    candidate="$dir/$name"
    [[ -x "$candidate" ]] || continue
    resolved="$(readlink -f "$candidate")"
    if [[ "$resolved" != "$script_path" && "$resolved" != "$script_dir/$name" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

pick_real_compiler() {
  local name="$1"
  local found
  if found="$(find_in_path "$name")"; then
    printf '%s\n' "$found"
    return 0
  fi

  case "$name" in
    nvcc)
      if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        printf '%s\n' /usr/local/cuda/bin/nvcc
        return 0
      fi
      ;;
    cc)
      for alt in gcc clang cc; do
        if found="$(find_in_path "$alt")"; then
          printf '%s\n' "$found"
          return 0
        fi
      done
      ;;
    c++)
      for alt in g++ clang++ c++; do
        if found="$(find_in_path "$alt")"; then
          printf '%s\n' "$found"
          return 0
        fi
      done
      ;;
  esac

  return 1
}

if ! real_compiler="$(pick_real_compiler "$invoked_name")"; then
  printf 'kernel-smoke wrapper could not resolve real compiler for %s\n' "$invoked_name" >&2
  exit 1
fi

exec "$rapid_wrap" "$real_compiler" "$@"
