"""Kernel rewrite post-processing for kernel-smoke artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from codegen.decode import DecodeHeaderGenerator
from codegen.invoke import InvokeHeaderGenerator
from codegen.target_layout import TargetLayoutHeaderGenerator
from common import KernelPhase1Artifacts, now_iso, read_json, resolve_kernel_dir, write_json
from contracts.kernel import KernelContract, build_kernel_contract
from rewrite.executor import RewriteExecutor, VConfigRewriteResult
from rewrite.planner import RewritePlanner


_PHASE2_INPUT_INVALID_REASONS = {
    "arg_type_missing",
    "type_shim_missing",
    "array_extent_dep_missing",
    "constexpr_dep_missing",
    "enum_dep_missing",
    "anonymous_type_unsupported",
    "local_type_unsupported",
    "const_assignment_blocker",
    "pointer_pointee_not_supported",
    "pointer_pointee_layout_missing",
    "pointer_role_missing",
    "pointer_role_not_supported",
    "constraint_arg_invalid",
    "constraint_arg_missing",
    "constraint_buffer_arg_not_payload_buffer",
    "constraint_field_path_invalid",
    "constraint_field_path_not_found",
    "constraint_field_path_not_supported",
    "constraint_not_supported",
    "constraint_op_invalid",
    "constraint_scalar_arg_not_scalar",
    "constraint_unit_not_supported",
    "reference_type_not_supported",
    "reference_type",
    "reference_field",
    "callable_type_not_supported",
    "nonbuiltin_type_not_supported",
    "materialization_unsafe",
    "non_trivially_copyable",
    "opaque_with_ptr_layout_incomplete",
    "opaque_with_ptr_layout_missing",
    "private_data_field",
    "protected_data_field",
    "scoped_type_shim_unsupported",
    "scalar_size_unsupported",
    "union_not_supported",
    "virtual_base",
    "virtual_method_or_vptr",
    "non_public_base",
}


def _reason_code(reason: str) -> str:
    return reason.rsplit(":", 1)[-1]


def _is_phase2_input_invalid(reason: str) -> bool:
    return _reason_code(reason) in _PHASE2_INPUT_INVALID_REASONS


def _kernel_others(manifest: dict[str, Any]) -> dict[str, Any]:
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or not kernels or not isinstance(kernels[0], dict):
        return {}
    others = kernels[0].get("others")
    return others if isinstance(others, dict) else {}


def _support_failure_context(
    *,
    reason: str | None,
    contract: KernelContract | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not reason:
        return None

    context: dict[str, Any] = {}
    if ":" in reason:
        support_path, reason_code = reason.rsplit(":", 1)
        if support_path:
            context["support_path"] = support_path
        if reason_code:
            context["reason_code"] = reason_code
    else:
        context["reason_code"] = reason

    if _reason_code(reason) == "scoped_type_shim_unsupported" and contract is not None:
        context["type_shim_status"] = contract.type_shim_status
        if contract.type_shim_reason_codes:
            context["type_shim_reason_codes"] = contract.type_shim_reason_codes
        if contract.type_shim_missing_dependencies:
            context["type_shim_missing_dependencies"] = contract.type_shim_missing_dependencies

        others = _kernel_others(manifest or {})
        for source_key, output_key in (
            ("type_shim_system_headers", "type_shim_system_headers"),
            ("type_shim_system_includes", "type_shim_system_includes"),
        ):
            raw_values = others.get(source_key)
            if isinstance(raw_values, list):
                values = [str(value) for value in raw_values if isinstance(value, str) and value]
                if values:
                    context[output_key] = values

    return context or None


def _load_kernel_inputs(run_dir: Path, entry: dict[str, Any]) -> KernelPhase1Artifacts:
    kernel_id = str(entry.get("kernel_id", "")) or "unknown"
    kernel_dir = resolve_kernel_dir(run_dir, str(entry.get("dir", "")))
    phase2_dir = kernel_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    if not kernel_dir.is_dir():
        raise ValueError("phase1_artifact_missing")

    kernel_bc = kernel_dir / "kernel.bc"
    if not kernel_bc.is_file() or kernel_bc.stat().st_size == 0:
        raise ValueError("phase1_artifact_missing")

    manifest = read_json(kernel_dir / "manifest.json")
    metadata = read_json(kernel_dir / "metadata.json")
    if manifest is None or metadata is None:
        raise ValueError("phase1_metadata_missing")

    return KernelPhase1Artifacts(
        kernel_id=kernel_id,
        kernel_dir=kernel_dir,
        phase2_dir=phase2_dir,
        kernel_bc=kernel_bc,
        manifest_path=kernel_dir / "manifest.json",
        manifest=manifest,
        metadata=metadata,
    )


class Phase2Runner:
    """Structured Phase 2 runner with dedicated rewrite and codegen stages."""

    def __init__(self) -> None:
        self.planner = RewritePlanner()
        self.rewrite_executor = RewriteExecutor()
        self.target_layout_generator = TargetLayoutHeaderGenerator()
        self.decode_generator = DecodeHeaderGenerator()
        self.invoke_generator = InvokeHeaderGenerator()

    def _write_phase2_metadata(
        self,
        *,
        kernel: KernelPhase1Artifacts,
        input_symbol: str | None,
        entry_symbol: str | None,
        status: str,
        failure_reason: str | None,
        failure_detail: str | None,
        failure_context: dict[str, Any] | None,
        vconfig_result: VConfigRewriteResult | None,
    ) -> None:
        phase2_metadata = {
            "schema_version": 1,
            "kernel_id": kernel.kernel_id,
            "input_symbol": input_symbol,
            "entry_symbol": entry_symbol,
            "phase2_status": status,
            "failure_reason": failure_reason,
            "failure_detail": failure_detail,
        }
        if failure_context is not None:
            phase2_metadata["failure_context"] = failure_context
        if vconfig_result is not None:
            phase2_metadata.update(self._vconfig_status(vconfig_result))
        write_json(kernel.phase2_dir / "metadata.phase2.json", phase2_metadata)

    @staticmethod
    def _vconfig_status(result: VConfigRewriteResult) -> dict[str, Any]:
        status: dict[str, Any] = {
            "vconfig_requested": result.requested,
            "vconfig_enabled": result.enabled,
            "vconfig_warp_aligned": result.warp_aligned,
        }
        if result.reason is not None:
            status["vconfig_disabled_reason"] = result.reason
        return status

    def _write_build_spec(
        self,
        *,
        kernel: KernelPhase1Artifacts,
        contract: KernelContract,
        vconfig_result: VConfigRewriteResult,
    ) -> None:
        build_spec = {
            "schema_version": 1,
            "kernel_id": kernel.kernel_id,
            "entry_symbol": contract.entry_symbol,
            "entry_abi_version": 1,
            "entry_context": "rapid_kernel_context_v1",
            "feedback_memory_enabled": contract.feedback_memory_enabled,
            "feedback_payload_slot_count": len(contract.payload_slots),
            "feedback_payload_slots": [
                slot.to_plan_dict() for slot in contract.feedback_payload_slots
            ],
            "target_layout_header": "gen/rapid_target_layout.v1.h",
            "device_bc": "kernel.device.bc",
            "decode_header": "gen/fuzzer_decode.v1.cuh",
            "invoke_header": "gen/fuzzer_invoke.v1.cuh",
        }
        build_spec.update(self._vconfig_status(vconfig_result))
        if contract.needs_shim:
            build_spec["shim_header"] = "gen/type_shim.v1.cuh"
        if contract.type_shim_include_dirs:
            build_spec["type_shim_include_dirs"] = contract.type_shim_include_dirs
        write_json(kernel.phase2_dir / "build_spec.json", build_spec)

    def _materialize_type_shim(self, *, kernel: KernelPhase1Artifacts, contract: KernelContract) -> None:
        if not contract.type_shim_header:
            return
        src = kernel.kernel_dir / contract.type_shim_header
        if not src.is_file():
            raise ValueError("type_shim_missing")
        gen_dir = kernel.phase2_dir / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, gen_dir / "type_shim.v1.cuh")

    def _process_kernel(self, kernel: KernelPhase1Artifacts) -> dict[str, Any]:
        status = "failed"
        failure_reason: str | None = None
        failure_detail: str | None = None
        failure_context: dict[str, Any] | None = None
        input_symbol: str | None = None
        entry_symbol: str | None = None
        contract: KernelContract | None = None
        vconfig_result: VConfigRewriteResult | None = None

        try:
            if kernel.metadata.get("build_status") != "built":
                status = "skipped"
                failure_reason = "phase1_not_built"
            else:
                contract = build_kernel_contract(
                    kernel_id=kernel.kernel_id,
                    manifest=kernel.manifest,
                    manifest_path=kernel.manifest_path,
                    metadata=kernel.metadata,
                )
                rewrite_plan = self.planner.build_plan(contract)
                input_symbol = contract.input_symbol
                entry_symbol = contract.entry_symbol

                if not contract.supported:
                    reason = contract.support_reason or "phase2_input_invalid"
                    failure_context = _support_failure_context(
                        reason=reason,
                        contract=contract,
                        manifest=kernel.manifest,
                    )
                    raise ValueError(reason)

                vconfig_result = self.rewrite_executor.execute(
                    rewrite_plan=rewrite_plan,
                    kernel_bc=kernel.kernel_bc,
                    phase2_dir=kernel.phase2_dir,
                )

                self._materialize_type_shim(kernel=kernel, contract=contract)
                self.target_layout_generator.generate(
                    contract=contract,
                    phase2_dir=kernel.phase2_dir,
                )
                self.decode_generator.generate(
                    contract=contract,
                    phase2_dir=kernel.phase2_dir,
                )
                self.invoke_generator.generate(
                    contract=contract,
                    phase2_dir=kernel.phase2_dir,
                )
                self._write_build_spec(
                    kernel=kernel,
                    contract=contract,
                    vconfig_result=vconfig_result,
                )
                status = "built"
        except ValueError as e:
            reason = str(e)
            if failure_context is None:
                failure_context = _support_failure_context(
                    reason=reason,
                    contract=contract,
                    manifest=kernel.manifest,
                )
            if reason in {"phase1_artifact_missing", "phase1_metadata_missing"}:
                failure_reason = reason
            elif _is_phase2_input_invalid(reason):
                failure_reason = "phase2_input_invalid"
                failure_detail = reason
            elif reason in {
                "rewrite_ir_parse_fail",
                "rewrite_target_not_found",
                "rewrite_target_ambiguous",
                "input_signature_mismatch",
                "entry_symbol_missing",
                "entry_signature_mismatch",
                "entry_symbol_still_kernel",
            }:
                failure_reason = "rewrite_verify_fail"
                failure_detail = reason
            else:
                failure_reason = "rewrite_plan_invalid"
                failure_detail = reason
            status = "failed"
        except OSError as e:
            failure_reason = "rewrite_exec_fail"
            failure_detail = str(e)
            status = "failed"
        except Exception as e:  # defensive: keep stage moving across kernels
            failure_reason = "rewrite_exec_fail"
            failure_detail = str(e)
            status = "failed"

        self._write_phase2_metadata(
            kernel=kernel,
            input_symbol=input_symbol,
            entry_symbol=entry_symbol,
            status=status,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
            failure_context=failure_context,
            vconfig_result=vconfig_result,
        )
        result = {
            "kernel_id": kernel.kernel_id,
            "kernel_dir": str(kernel.kernel_dir),
            "phase2_dir": str(kernel.phase2_dir),
            "status": status,
            "failure_reason": failure_reason,
            "failure_detail": failure_detail,
        }
        if failure_context is not None:
            result["failure_context"] = failure_context
        if vconfig_result is not None:
            result.update(self._vconfig_status(vconfig_result))
        return result

    def run(self, run_dir: Path) -> dict[str, Any]:
        run_dir = run_dir.resolve()
        index_path = run_dir / "index.json"
        index = read_json(index_path)
        if index is None:
            raise ValueError(f"missing or invalid index.json: {index_path}")

        kernels = index.get("kernels")
        if not isinstance(kernels, list):
            raise ValueError("index.json missing kernels list")

        results: list[dict[str, Any]] = []
        counts = {"total": 0, "built": 0, "failed": 0, "skipped": 0}
        total_kernels = len(kernels)
        print(f"  [rewrite] processing {total_kernels} kernels ...", flush=True)

        for idx, entry in enumerate(kernels, 1):
            counts["total"] += 1
            if not isinstance(entry, dict):
                counts["failed"] += 1
                result = {
                    "kernel_id": "unknown",
                    "kernel_dir": "",
                    "phase2_dir": "",
                    "status": "failed",
                    "failure_reason": "phase1_artifact_missing",
                    "failure_detail": None,
                }
                print(f"  [rewrite] done {idx}/{total_kernels} unknown -> failed (phase1_artifact_missing)", flush=True)
                results.append(result)
                continue

            try:
                kernel = _load_kernel_inputs(run_dir, entry)
                result = self._process_kernel(kernel)
            except ValueError as e:
                kernel_id = str(entry.get("kernel_id", "")) or "unknown"
                kernel_dir = resolve_kernel_dir(run_dir, str(entry.get("dir", "")))
                phase2_dir = kernel_dir / "phase2"
                phase2_dir.mkdir(parents=True, exist_ok=True)
                result = {
                    "kernel_id": kernel_id,
                    "kernel_dir": str(kernel_dir),
                    "phase2_dir": str(phase2_dir),
                    "status": "failed",
                    "failure_reason": str(e),
                    "failure_detail": None,
                }
                write_json(
                    phase2_dir / "metadata.phase2.json",
                    {
                        "schema_version": 1,
                        "kernel_id": kernel_id,
                        "input_symbol": None,
                        "entry_symbol": None,
                        "phase2_status": "failed",
                        "failure_reason": str(e),
                        "failure_detail": None,
                    },
                )

            counts[result["status"]] += 1
            results.append(result)
            failure_reason = result.get("failure_reason")
            suffix = f" ({failure_reason})" if failure_reason else ""
            print(
                f"  [rewrite] done {idx}/{total_kernels} {result['kernel_id']} -> {result['status']}{suffix}",
                flush=True,
            )

        summary = {
            "schema_version": 1,
            "run_id": str(index.get("run_id", run_dir.name)),
            "generated_at": now_iso(),
            "counts": counts,
            "results": results,
        }
        write_json(run_dir / "rewrite_summary.json", summary)
        return summary


def run_phase2(run_dir: Path) -> dict[str, Any]:
    return Phase2Runner().run(run_dir)
