"""Manifest-driven invoke header generation for Phase 2."""

from __future__ import annotations

from pathlib import Path

from contracts.kernel import KernelContract


def _guard_name(kernel_id: str) -> str:
    return "__PHASE2_INVOKE_" + "".join(c if c.isalnum() else "_" for c in kernel_id).upper() + "_CUH__"


def _param_list(contract: KernelContract) -> str:
    params = [f"{arg.codegen_type} {arg.name}" for arg in contract.args]
    params.append("RapidKernelContext *context")
    return ", ".join(params)


def _call_args(contract: KernelContract) -> str:
    args = [f"decoded.{arg.name}" for arg in contract.args]
    args.append("context")
    return ", ".join(args)


def _feedback_prepare_body(contract: KernelContract) -> str:
    payload_slots = contract.feedback_payload_slots
    lines = [
        "  if (context == nullptr) {",
        "    return;",
        "  }",
        "  if (threadIdx.x == 0 && blockIdx.x == 0) {",
        "    context->vconfig = vconfig;",
        "    context->feedback.bounds_count = RAPID_PAYLOAD_SLOT_COUNT;",
    ]
    for slot in payload_slots:
        lines.extend(
            [
                f"    context->feedback.bounds[{slot.slot}] = RapidPayloadBounds{{",
                f"        reinterpret_cast<uintptr_t>({slot.decoded_expr}),",
                f"        decoded.__rapid_payload_{slot.slot}_len_bytes}};",
            ]
        )
    lines.extend(["  }", "  __syncthreads();"])
    return "\n".join(lines)


class InvokeHeaderGenerator:
    """Generate invoke headers for currently supported arg-pack-v1 kernels."""

    def generate(self, *, contract: KernelContract, phase2_dir: Path) -> None:
        gen_dir = phase2_dir / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        out_path = gen_dir / "fuzzer_invoke.v1.cuh"
        guard = _guard_name(contract.kernel_id)
        include_block = '#include "type_shim.v1.cuh"\n' if contract.needs_shim else ""
        payload_slot_count = len(contract.feedback_payload_slots)
        content = f'''#ifndef {guard}
#define {guard}

#include "rapid_target_layout.v1.h"
#include "fuzzer_decode.v1.cuh"
#include "feedback/feedback_context.cuh"
{include_block}
static_assert(RAPID_PAYLOAD_SLOT_COUNT == {payload_slot_count}u,
              "generated target layout does not match generated invoke");

extern "C" __device__ void {contract.entry_symbol}({_param_list(contract)});

__device__ inline void fuzzer_feedback_prepare_v1(
    const DecodedKernelArgs &decoded, const RapidVConfig &vconfig,
    RapidKernelContext *context) {{
{_feedback_prepare_body(contract)}
}}

__device__ inline void fuzzer_invoke_v1(
    const DecodedKernelArgs &decoded, RapidKernelContext *context) {{
  {contract.entry_symbol}({_call_args(contract)});
}}

__device__ inline void fuzzer_invoke_v1(const DecodedKernelArgs &decoded) {{
  fuzzer_invoke_v1(decoded, nullptr);
}}

#endif // {guard}
'''
        out_path.write_text(content, encoding="utf-8")
