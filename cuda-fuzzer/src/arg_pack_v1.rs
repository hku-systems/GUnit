use std::{
    fs,
    mem::size_of,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

use serde::Deserialize;

const U64_BYTES: usize = size_of::<u64>();
const PAD_BYTE: u8 = 0;
const DEFAULT_MAX_PAYLOAD_LEN: usize = 4096;
const HAVOC_MAX_INSERT_BYTES: usize = 64;
const HAVOC_MAX_XOR_BYTES: usize = 16;
pub(crate) const ARG_PACK_HAVOC_MAX_STACKED_OPS: usize = 8;
pub const RAPID_VCONFIG_BYTES: usize = 24;
pub const RAPID_TASK_ENVELOPE_HEADER_BYTES: usize = RAPID_VCONFIG_BYTES + U64_BYTES;
const VCONFIG_WARP_SIZE: u32 = 32;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelManifest {
    pub schema_version: u64,
    pub kernels: Vec<KernelSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    pub symbol_name: String,
    pub display_name: String,
    pub args: Vec<ArgSpec>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    #[serde(default)]
    pub launch_policy: Option<LaunchPolicySpec>,
    #[serde(default)]
    pub others: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchPolicySpec {
    pub grid: Option<[u32; 3]>,
    pub block_candidates: Option<Vec<u32>>,
    pub physical_block_max: Option<u32>,
    pub logical_grid: Option<[u32; 3]>,
    pub logical_block: Option<[u32; 3]>,
    pub logical_block_candidates: Option<Vec<[u32; 3]>>,
    pub target_dynamic_shared_bytes: Option<u64>,
    pub coverage_memory: Option<String>,
    pub vconfig_reserved: Option<bool>,
    pub vconfig_mutation: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RapidVConfig {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RapidVConfigBounds {
    default: RapidVConfig,
    min: RapidVConfig,
    max: RapidVConfig,
    require_whole_warps: bool,
    logical_block_candidates: Option<Vec<[u32; 3]>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RapidInputEnvelope<'a> {
    pub vconfig: RapidVConfig,
    pub payload_size: u64,
    pub payload: &'a [u8],
}

#[derive(Clone, Debug, Deserialize)]
pub struct ArgSpec {
    pub index: usize,
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub kind: ArgKind,
    pub pointer_role: Option<PointerRole>,
    pub pointee_layout: Option<LayoutNode>,
    pub type_layout: Option<TypeLayout>,
    pub size_bytes: usize,
    pub align_bytes: usize,
    pub domain: Option<Domain>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ArgKind {
    Pointer,
    Scalar,
    OpaqueVal,
    OpaqueWithPtr,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PointerRole {
    PayloadBuffer,
    DerivedPointer,
    ExternalDevicePointer,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TypeLayout {
    pub layout_status: LayoutStatus,
    #[serde(default)]
    pub fields: Vec<LayoutNode>,
    pub element: Option<Box<LayoutNode>>,
    pub element_count: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LayoutStatus {
    Complete,
    Partial,
    Opaque,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LayoutNode {
    pub index: String,
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub kind: ArgKind,
    pub pointer_role: Option<PointerRole>,
    pub pointee_layout: Option<Box<LayoutNode>>,
    pub size_bytes: usize,
    pub align_bytes: usize,
    pub domain: Option<Domain>,
    #[serde(default)]
    pub fields: Vec<LayoutNode>,
    pub element: Option<Box<LayoutNode>>,
    pub element_count: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Domain {
    IntRange {
        min: Option<String>,
        max: Option<String>,
        signed: Option<bool>,
    },
    FloatRange {
        min: Option<f64>,
        max: Option<f64>,
        allow_nan: Option<bool>,
    },
    Enum {
        values: Vec<EnumValue>,
        allow_unknown: Option<bool>,
    },
    Bytes {
        min_len: Option<String>,
        max_len: Option<String>,
        elem_size_bytes: Option<usize>,
        nullable: Option<bool>,
        pattern_hex: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize)]
pub struct EnumValue {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Constraint {
    ScalarLeLogicalBlockDim {
        scalar_arg: usize,
        dimension: LogicalBlockDimension,
    },
    ScalarEqLogicalBlockDim {
        scalar_arg: usize,
        dimension: LogicalBlockDimension,
    },
    ScalarLeBufferLen {
        scalar_arg: usize,
        scalar_path: Option<Vec<String>>,
        buffer_arg: usize,
        buffer_path: Option<Vec<String>>,
        unit: ConstraintUnit,
    },
    ScalarCompareConst {
        scalar_arg: usize,
        scalar_path: Option<Vec<String>>,
        op: ConstraintOp,
        value: i64,
    },
    ScalarCompareScalar {
        lhs_arg: usize,
        lhs_path: Option<Vec<String>>,
        op: ConstraintOp,
        rhs_arg: usize,
        rhs_path: Option<Vec<String>>,
    },
    ScalarProductLeConst {
        lhs_arg: usize,
        lhs_path: Option<Vec<String>>,
        rhs_arg: usize,
        rhs_path: Option<Vec<String>>,
        value: u64,
        repair_arg: usize,
        repair_path: Option<Vec<String>>,
    },
    CountFitsBuffer {
        count_arg: usize,
        count_path: Option<Vec<String>>,
        buffer_arg: usize,
        buffer_path: Option<Vec<String>>,
        elem_size_bytes: usize,
    },
    BufferElementsLtScalar {
        buffer_arg: usize,
        buffer_path: Option<Vec<String>>,
        scalar_arg: usize,
        scalar_path: Option<Vec<String>>,
        elem_size_bytes: Option<usize>,
    },
    ExpressionCompare {
        lhs: ConstraintExpr,
        op: ConstraintOp,
        rhs: ConstraintExpr,
        repair: ConstraintRepair,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogicalBlockDimension {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConstraintExpr {
    ArgValue {
        arg: usize,
        path: Option<Vec<String>>,
    },
    PayloadLen {
        arg: usize,
        path: Option<Vec<String>>,
    },
    Const {
        value: u64,
    },
    Binary {
        op: ConstraintBinaryOp,
        lhs: Box<ConstraintExpr>,
        rhs: Box<ConstraintExpr>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub enum ConstraintBinaryOp {
    #[serde(rename = "+")]
    Add,
    #[serde(rename = "-")]
    Sub,
    #[serde(rename = "*")]
    Mul,
    #[serde(rename = "/")]
    Div,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConstraintRepair {
    ResizePayload {
        arg: usize,
        path: Option<Vec<String>>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub enum ConstraintOp {
    #[serde(rename = "<=")]
    Le,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "==")]
    Eq,
    #[serde(rename = "!=")]
    Ne,
    #[serde(rename = ">=")]
    Ge,
    #[serde(rename = ">")]
    Gt,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintUnit {
    Bytes,
    Elements,
}

impl ArgSpec {
    fn as_node(&self) -> ArgNode<'_> {
        ArgNode {
            kind: self.kind,
            type_layout: self.type_layout.as_ref(),
            fields: &[],
            element: None,
            element_count: None,
            size_bytes: self.size_bytes,
            align_bytes: self.align_bytes,
            domain: self.domain.as_ref(),
            pointee_layout: self.pointee_layout.as_ref(),
        }
    }
}

#[derive(Clone, Copy)]
struct ArgNode<'a> {
    kind: ArgKind,
    type_layout: Option<&'a TypeLayout>,
    fields: &'a [LayoutNode],
    element: Option<&'a LayoutNode>,
    element_count: Option<usize>,
    size_bytes: usize,
    align_bytes: usize,
    domain: Option<&'a Domain>,
    pointee_layout: Option<&'a LayoutNode>,
}

impl LayoutNode {
    fn as_node(&self) -> ArgNode<'_> {
        ArgNode {
            kind: self.kind,
            type_layout: None,
            fields: &self.fields,
            element: self.element.as_deref(),
            element_count: self.element_count,
            size_bytes: self.size_bytes,
            align_bytes: self.align_bytes,
            domain: self.domain.as_ref(),
            pointee_layout: self.pointee_layout.as_deref(),
        }
    }
}

impl<'a> ArgNode<'a> {
    fn payload_align_bytes(self) -> usize {
        self.pointee_layout
            .map(|pointee| pointee.align_bytes)
            .unwrap_or(1)
            .max(self.align_bytes)
            .max(1)
    }

    fn children(&self) -> Option<LayoutChildren<'a>> {
        match self.kind {
            ArgKind::OpaqueWithPtr => {
                if let Some(layout) = self.type_layout {
                    if layout.layout_status != LayoutStatus::Complete {
                        return None;
                    }
                    return Some(LayoutChildren {
                        fields: &layout.fields,
                        element: layout.element.as_deref(),
                        element_count: layout.element_count,
                    });
                }
                Some(LayoutChildren {
                    fields: self.fields,
                    element: self.element,
                    element_count: self.element_count,
                })
            }
            _ => None,
        }
    }
}

struct LayoutChildren<'a> {
    fields: &'a [LayoutNode],
    element: Option<&'a LayoutNode>,
    element_count: Option<usize>,
}

impl<'a> LayoutChildren<'a> {
    fn nodes(&self) -> Vec<ArgNode<'a>> {
        let mut nodes = self
            .fields
            .iter()
            .map(LayoutNode::as_node)
            .collect::<Vec<_>>();
        if let Some(element) = self.element {
            let count = self.element_count.unwrap_or(0);
            nodes.extend((0..count).map(|_| element.as_node()));
        }
        nodes
    }
}

fn validate_layout_node(
    node: &LayoutNode,
    context: &str,
    require_materializable: bool,
) -> Result<(), String> {
    normalize_alignment(node.align_bytes)?;
    validate_bytes_domain(&format!("layout node {context}"), node.domain.as_ref())?;
    match node.kind {
        ArgKind::Pointer => {
            let pointer_role = node
                .pointer_role
                .ok_or_else(|| format!("pointer layout node {context} missing pointer_role"))?;
            if node.pointee_layout.is_none() {
                return Err(format!(
                    "pointer layout node {context} missing pointee_layout"
                ));
            }
            if pointer_role != PointerRole::PayloadBuffer {
                return Err(format!(
                    "pointer layout node {context} uses unsupported pointer_role={pointer_role:?}; arg-pack-v1 only supports payload_buffer"
                ));
            }
            if let Some(pointee) = &node.pointee_layout {
                validate_layout_node(pointee, &format!("{context}.*"), false)?;
            }
        }
        ArgKind::Scalar => {
            if node.pointer_role.is_some() || node.pointee_layout.is_some() {
                return Err(format!(
                    "non-pointer layout node {context} must not set pointer metadata"
                ));
            }
            if !matches!(node.size_bytes, 1 | 2 | 4 | 8) {
                return Err(format!(
                    "scalar layout node {context} has unsupported size_bytes={}",
                    node.size_bytes
                ));
            }
            validate_scalar_domain(
                &format!("layout node {context}"),
                node.size_bytes,
                node.domain.as_ref(),
            )?;
        }
        ArgKind::OpaqueVal => {
            if node.pointer_role.is_some() || node.pointee_layout.is_some() {
                return Err(format!(
                    "non-pointer layout node {context} must not set pointer metadata"
                ));
            }
        }
        ArgKind::OpaqueWithPtr => {
            if require_materializable && node.fields.is_empty() && node.element.is_none() {
                return Err(format!(
                    "opaque_with_ptr layout node {context} missing recursive fields/element"
                ));
            }
        }
    }

    for field in &node.fields {
        validate_layout_node(field, &field.index, true)?;
    }
    if let Some(element) = &node.element {
        validate_layout_node(element, &format!("{context}[]"), true)?;
    }
    Ok(())
}

fn find_node_by_path<'a>(root: ArgNode<'a>, path: Option<&'a [String]>) -> Option<ArgNode<'a>> {
    let mut node = root;
    for segment in path.unwrap_or(&[]) {
        let children = node.children()?;
        let fields = children.fields;
        if let Some(field) = fields.iter().find(|field| field.name == *segment) {
            node = field.as_node();
            continue;
        }
        let index = segment.parse::<usize>().ok()?;
        let element = children.element?;
        if index >= children.element_count.unwrap_or(0) {
            return None;
        }
        node = element.as_node();
    }
    Some(node)
}

fn find_value_by_path<'a>(
    value: &'a ArgValue,
    node: ArgNode<'_>,
    path: Option<&[String]>,
) -> Option<&'a ArgValue> {
    let mut value = value;
    let mut node = node;
    for segment in path.unwrap_or(&[]) {
        let ArgValue::Aggregate(values) = value else {
            return None;
        };
        let children = node.children()?;
        if let Some((idx, field)) = children
            .fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.name == *segment)
        {
            value = values.get(idx)?;
            node = field.as_node();
            continue;
        }
        let index = segment.parse::<usize>().ok()?;
        let element = children.element?;
        let base = children.fields.len();
        if index >= children.element_count.unwrap_or(0) {
            return None;
        }
        value = values.get(base + index)?;
        node = element.as_node();
    }
    Some(value)
}

fn find_value_by_path_mut<'a>(
    value: &'a mut ArgValue,
    node: ArgNode<'_>,
    path: Option<&[String]>,
) -> Option<&'a mut ArgValue> {
    let mut value = value;
    let mut node = node;
    for segment in path.unwrap_or(&[]) {
        let ArgValue::Aggregate(values) = value else {
            return None;
        };
        let children = node.children()?;
        if let Some((idx, field)) = children
            .fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.name == *segment)
        {
            value = values.get_mut(idx)?;
            node = field.as_node();
            continue;
        }
        let index = segment.parse::<usize>().ok()?;
        let element = children.element?;
        let base = children.fields.len();
        if index >= children.element_count.unwrap_or(0) {
            return None;
        }
        value = values.get_mut(base + index)?;
        node = element.as_node();
    }
    Some(value)
}

#[derive(Clone, Debug)]
pub struct ArgPackSpec {
    symbol_name: String,
    display_name: String,
    args: Vec<ArgSpec>,
    constraints: Vec<Constraint>,
    vconfig_bounds: RapidVConfigBounds,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ArgValue {
    Pointer(Vec<u8>),
    Scalar(u64),
    OpaqueVal(Vec<u8>),
    Aggregate(Vec<ArgValue>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParsedArgPack {
    values: Vec<ArgValue>,
}

static ARG_PACK_SPEC: OnceLock<Result<ArgPackSpec, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArgPackStats {
    pub normalize_calls: u64,
    pub normalize_repack_count: u64,
    pub invalid_repair_count: u64,
    pub mutation_calls: u64,
    pub seed_generation_count: u64,
    pub payload_clamp_count: u64,
}

#[derive(Debug, Default)]
struct ArgPackStatsCounters {
    normalize_calls: AtomicU64,
    normalize_repack_count: AtomicU64,
    invalid_repair_count: AtomicU64,
    mutation_calls: AtomicU64,
    seed_generation_count: AtomicU64,
    payload_clamp_count: AtomicU64,
}

static ARG_PACK_STATS: ArgPackStatsCounters = ArgPackStatsCounters {
    normalize_calls: AtomicU64::new(0),
    normalize_repack_count: AtomicU64::new(0),
    invalid_repair_count: AtomicU64::new(0),
    mutation_calls: AtomicU64::new(0),
    seed_generation_count: AtomicU64::new(0),
    payload_clamp_count: AtomicU64::new(0),
};

#[cfg(test)]
thread_local! {
    static TEST_NORMALIZE_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn default_vconfig_bounds() -> RapidVConfigBounds {
    RapidVConfigBounds {
        default: RapidVConfig {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 1024,
            block_y: 1,
            block_z: 1,
        },
        min: RapidVConfig {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 1,
            block_y: 1,
            block_z: 1,
        },
        max: RapidVConfig {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 1024,
            block_y: 1,
            block_z: 1,
        },
        require_whole_warps: true,
        logical_block_candidates: None,
    }
}

fn validate_positive_dims(dims: [u32; 3], field: &str) -> Result<[u32; 3], String> {
    if dims.contains(&0) {
        return Err(format!("launch_policy.{field} dimensions must be positive"));
    }
    Ok(dims)
}

fn checked_dim_product(dims: [u32; 3], field: &str) -> Result<u64, String> {
    dims.into_iter().try_fold(1u64, |acc, dim| {
        acc.checked_mul(u64::from(dim))
            .ok_or_else(|| format!("launch_policy.{field} dimensions overflow"))
    })
}

fn launch_policy_bounds(raw: Option<&LaunchPolicySpec>) -> Result<RapidVConfigBounds, String> {
    let Some(raw) = raw else {
        return Ok(default_vconfig_bounds());
    };
    if raw.coverage_memory.as_deref().unwrap_or("global") != "global" {
        return Err("launch_policy.coverage_memory must be global".to_string());
    }

    let grid = validate_positive_dims(raw.grid.unwrap_or([1, 1, 1]), "grid")?;
    if grid != [1, 1, 1] {
        return Err(
            "launch_policy currently supports only physical grid [1, 1, 1]; \
             logical grid mutation is reserved for future VConfig Phase 2 rewriting"
                .to_string(),
        );
    }

    let mut candidates = raw
        .block_candidates
        .clone()
        .unwrap_or_else(|| vec![1024, 512, 256, 128, 64, 32, 16]);
    if candidates.is_empty() {
        return Err("launch_policy.block_candidates must be non-empty".to_string());
    }
    if candidates.contains(&0) {
        return Err("launch_policy.block_candidates must be positive".to_string());
    }
    candidates.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
    candidates.dedup();
    let physical_block_max = raw.physical_block_max.unwrap_or(candidates[0]);
    if physical_block_max == 0 {
        return Err("launch_policy.physical_block_max must be positive".to_string());
    }
    let selected_block = candidates
        .into_iter()
        .find(|candidate| *candidate <= physical_block_max)
        .ok_or_else(|| {
            "launch_policy.block_candidates are all above physical_block_max".to_string()
        })?;

    let selected_block = clamp_block_x_to_warp_multiple(selected_block, selected_block);
    let logical_grid = validate_positive_dims(raw.logical_grid.unwrap_or(grid), "logical_grid")?;
    let physical_grid_blocks = checked_dim_product(grid, "grid")?;
    let logical_grid_blocks = checked_dim_product(logical_grid, "logical_grid")?;
    if logical_grid_blocks > physical_grid_blocks {
        return Err("launch_policy.logical grid blocks exceed physical grid envelope".to_string());
    }

    let logical_block = validate_positive_dims(
        raw.logical_block.unwrap_or([selected_block, 1, 1]),
        "logical_block",
    )?;
    let logical_block_threads = checked_dim_product(logical_block, "logical_block")?;
    if logical_block_threads > u64::from(selected_block) {
        return Err(
            "launch_policy.logical block threads exceed physical block envelope".to_string(),
        );
    }
    let vconfig_mutation = raw.vconfig_mutation.unwrap_or(true);

    let logical_block_candidates = raw
        .logical_block_candidates
        .as_ref()
        .map(|raw_candidates| {
            if raw_candidates.is_empty() {
                return Err("launch_policy.logical_block_candidates must be non-empty".to_string());
            }
            let mut candidates = raw_candidates.clone();
            for candidate in &candidates {
                validate_positive_dims(*candidate, "logical_block_candidates")?;
                let threads = checked_dim_product(*candidate, "logical_block_candidates")?;
                if threads > u64::from(selected_block) {
                    return Err(
                        "launch_policy.logical_block_candidates exceed physical block envelope"
                            .to_string(),
                    );
                }
                if vconfig_mutation
                    && selected_block >= VCONFIG_WARP_SIZE
                    && threads % u64::from(VCONFIG_WARP_SIZE) != 0
                {
                    return Err(
                        "launch_policy.logical_block_candidates must use whole-warp thread counts"
                            .to_string(),
                    );
                }
            }
            candidates.sort_unstable();
            if candidates.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err("launch_policy.logical_block_candidates must be unique".to_string());
            }
            if candidates.binary_search(&logical_block).is_err() {
                return Err(
                    "launch_policy.logical_block_candidates must contain logical_block".to_string(),
                );
            }
            Ok(candidates)
        })
        .transpose()?;

    let default = RapidVConfig {
        grid_x: logical_grid[0],
        grid_y: logical_grid[1],
        grid_z: logical_grid[2],
        block_x: logical_block[0],
        block_y: logical_block[1],
        block_z: logical_block[2],
    };
    let min = if vconfig_mutation {
        RapidVConfig {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 1,
            block_y: 1,
            block_z: 1,
        }
    } else {
        default
    };
    Ok(RapidVConfigBounds {
        default,
        min,
        max: default,
        require_whole_warps: vconfig_mutation,
        logical_block_candidates,
    })
}

fn clamp_dim(value: u32, min: u32, max: u32) -> u32 {
    let max_dim = max.max(min).max(1);
    let min_dim = min.max(1).min(max_dim);
    value.clamp(min_dim, max_dim)
}

fn map_mutated_dim(value: u32, min: u32, max: u32, strategy: u8, step: u32) -> u32 {
    let step = step.max(1);
    let first = min.div_ceil(step).saturating_mul(step);
    let last = max - (max % step);
    if first > last {
        return clamp_dim(value, min, max);
    }

    match strategy % 4 {
        1 => first,
        2 => last,
        3 => last.saturating_sub(step).max(first),
        _ => first + (value % ((last - first) / step + 1)) * step,
    }
}

fn clamp_block_x_to_warp_multiple(value: u32, max: u32) -> u32 {
    clamp_block_x_to_warp_multiple_in_range(value, 1, max)
}

fn clamp_block_x_to_warp_multiple_in_range(value: u32, min: u32, max: u32) -> u32 {
    let clamped = clamp_dim(value, min, max);
    if max < VCONFIG_WARP_SIZE {
        return clamped;
    }
    let min_aligned = if min <= VCONFIG_WARP_SIZE {
        VCONFIG_WARP_SIZE
    } else {
        let rounded = ((u64::from(min) + u64::from(VCONFIG_WARP_SIZE - 1))
            / u64::from(VCONFIG_WARP_SIZE))
            * u64::from(VCONFIG_WARP_SIZE);
        u32::try_from(rounded).unwrap_or(u32::MAX)
    };
    let max_aligned = max - (max % VCONFIG_WARP_SIZE);
    if max_aligned == 0 || min_aligned > max_aligned {
        return clamped;
    }
    let rounded = ((u64::from(clamped) + u64::from(VCONFIG_WARP_SIZE - 1))
        / u64::from(VCONFIG_WARP_SIZE))
        * u64::from(VCONFIG_WARP_SIZE);
    u32::try_from(rounded)
        .unwrap_or(u32::MAX)
        .clamp(min_aligned, max_aligned)
}

fn greatest_common_divisor(mut lhs: u32, mut rhs: u32) -> u32 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn align_block_x_for_whole_warps(value: u32, min: u32, max: u32, y: u32, z: u32) -> Option<u32> {
    let other_dims = y.checked_mul(z)?;
    let quantum = VCONFIG_WARP_SIZE / greatest_common_divisor(VCONFIG_WARP_SIZE, other_dims);
    let min_aligned = min.div_ceil(quantum).checked_mul(quantum)?;
    let max_aligned = max - (max % quantum);
    if min_aligned > max_aligned {
        return None;
    }
    Some(
        clamp_dim(value, min, max)
            .div_ceil(quantum)
            .checked_mul(quantum)?
            .clamp(min_aligned, max_aligned),
    )
}

fn block_shape(vconfig: RapidVConfig) -> [u32; 3] {
    [vconfig.block_x, vconfig.block_y, vconfig.block_z]
}

fn logical_block_dimension(block: [u32; 3], dimension: LogicalBlockDimension) -> u32 {
    match dimension {
        LogicalBlockDimension::X => block[0],
        LogicalBlockDimension::Y => block[1],
        LogicalBlockDimension::Z => block[2],
    }
}

fn logical_vconfig_dimension(vconfig: RapidVConfig, dimension: LogicalBlockDimension) -> u32 {
    logical_block_dimension(block_shape(vconfig), dimension)
}

fn with_block_shape(mut vconfig: RapidVConfig, block: [u32; 3]) -> RapidVConfig {
    vconfig.block_x = block[0];
    vconfig.block_y = block[1];
    vconfig.block_z = block[2];
    vconfig
}

fn nearest_logical_block_candidate(block: [u32; 3], candidates: &[[u32; 3]]) -> [u32; 3] {
    *candidates
        .iter()
        .min_by_key(|candidate| {
            let distance = block
                .into_iter()
                .zip(candidate.iter().copied())
                .map(|(actual, legal)| u64::from(actual.abs_diff(legal)))
                .sum::<u64>();
            (distance, **candidate)
        })
        .expect("validated logical block candidates are non-empty")
}

fn clamp_vconfig(vconfig: RapidVConfig, bounds: &RapidVConfigBounds) -> RapidVConfig {
    let mut clamped = RapidVConfig {
        grid_x: clamp_dim(vconfig.grid_x, bounds.min.grid_x, bounds.max.grid_x),
        grid_y: clamp_dim(vconfig.grid_y, bounds.min.grid_y, bounds.max.grid_y),
        grid_z: clamp_dim(vconfig.grid_z, bounds.min.grid_z, bounds.max.grid_z),
        block_x: clamp_dim(vconfig.block_x, bounds.min.block_x, bounds.max.block_x),
        block_y: clamp_dim(vconfig.block_y, bounds.min.block_y, bounds.max.block_y),
        block_z: clamp_dim(vconfig.block_z, bounds.min.block_z, bounds.max.block_z),
    };
    if let Some(candidates) = bounds.logical_block_candidates.as_deref() {
        return with_block_shape(
            clamped,
            nearest_logical_block_candidate(block_shape(vconfig), candidates),
        );
    }
    if bounds.require_whole_warps {
        if let Some(block_x) = align_block_x_for_whole_warps(
            clamped.block_x,
            bounds.min.block_x,
            bounds.max.block_x,
            clamped.block_y,
            clamped.block_z,
        ) {
            clamped.block_x = block_x;
        } else {
            clamped.block_x = bounds.default.block_x;
            clamped.block_y = bounds.default.block_y;
            clamped.block_z = bounds.default.block_z;
        }
    }
    clamped
}

fn read_u32_at(raw: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        raw.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64_at(raw: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        raw.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn encode_rapid_input(vconfig: RapidVConfig, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(RAPID_TASK_ENVELOPE_HEADER_BYTES + payload.len());
    write_u32(&mut out, vconfig.grid_x);
    write_u32(&mut out, vconfig.grid_y);
    write_u32(&mut out, vconfig.grid_z);
    write_u32(&mut out, vconfig.block_x);
    write_u32(&mut out, vconfig.block_y);
    write_u32(&mut out, vconfig.block_z);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn parse_rapid_input_envelope(raw: &[u8]) -> Option<RapidInputEnvelope<'_>> {
    if raw.len() < RAPID_TASK_ENVELOPE_HEADER_BYTES {
        return None;
    }
    let payload_size = read_u64_at(raw, RAPID_VCONFIG_BYTES)?;
    let payload_len = usize::try_from(payload_size).ok()?;
    if payload_len != raw.len() - RAPID_TASK_ENVELOPE_HEADER_BYTES {
        return None;
    }
    Some(RapidInputEnvelope {
        vconfig: RapidVConfig {
            grid_x: read_u32_at(raw, 0)?,
            grid_y: read_u32_at(raw, 4)?,
            grid_z: read_u32_at(raw, 8)?,
            block_x: read_u32_at(raw, 12)?,
            block_y: read_u32_at(raw, 16)?,
            block_z: read_u32_at(raw, 20)?,
        },
        payload_size,
        payload: &raw[RAPID_TASK_ENVELOPE_HEADER_BYTES..],
    })
}

fn split_rapid_input_or_repair(raw: &[u8], default_vconfig: RapidVConfig) -> (RapidVConfig, &[u8]) {
    if raw.len() < RAPID_TASK_ENVELOPE_HEADER_BYTES {
        return (default_vconfig, &[]);
    }
    let vconfig = RapidVConfig {
        grid_x: read_u32_at(raw, 0).unwrap_or(default_vconfig.grid_x),
        grid_y: read_u32_at(raw, 4).unwrap_or(default_vconfig.grid_y),
        grid_z: read_u32_at(raw, 8).unwrap_or(default_vconfig.grid_z),
        block_x: read_u32_at(raw, 12).unwrap_or(default_vconfig.block_x),
        block_y: read_u32_at(raw, 16).unwrap_or(default_vconfig.block_y),
        block_z: read_u32_at(raw, 20).unwrap_or(default_vconfig.block_z),
    };
    (vconfig, &raw[RAPID_TASK_ENVELOPE_HEADER_BYTES..])
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    if alignment <= 1 {
        return Some(value);
    }
    let rem = value % alignment;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - rem)
    }
}

fn payload_len_offset(offset: usize, payload_align: usize) -> Option<usize> {
    let align = payload_align.max(1);
    let mut candidate = align_up(offset, U64_BYTES)?;
    for _ in 0..=align.max(U64_BYTES) / U64_BYTES + 1 {
        let data_start = candidate.checked_add(U64_BYTES)?;
        if data_start % align == 0 {
            return Some(candidate);
        }
        candidate = candidate.checked_add(U64_BYTES)?;
    }
    None
}

fn is_pow2(v: usize) -> bool {
    v != 0 && (v & (v - 1)) == 0
}

fn normalize_alignment(v: usize) -> Result<usize, String> {
    if is_pow2(v) {
        Ok(v)
    } else {
        Err(format!(
            "manifest align_bytes must be a power of two, got {v}"
        ))
    }
}

fn validate_scalar_domain(
    context: &str,
    size_bytes: usize,
    domain: Option<&Domain>,
) -> Result<(), String> {
    if matches!(domain, Some(Domain::FloatRange { .. })) && !matches!(size_bytes, 4 | 8) {
        return Err(format!(
            "{context} uses float_range with unsupported size_bytes={size_bytes}; only f32/f64 widths are supported"
        ));
    }
    if let Some(Domain::Enum { values, .. }) = domain {
        for value in values {
            parse_enum_discriminant(&value.value).ok_or_else(|| {
                format!(
                    "{context} enum value {} has invalid decimal discriminant {}",
                    value.name, value.value
                )
            })?;
        }
    }
    Ok(())
}

fn validate_scalar_compare_const_domain(
    node: ArgNode<'_>,
    op: ConstraintOp,
    constant: i64,
    context: &str,
) -> Result<(), String> {
    if matches!(node.domain, Some(Domain::FloatRange { .. })) {
        return validate_float_compare_const_domain(node, op, constant as f64, context);
    }
    let Some((lower, upper)) = int_domain_bounds(node) else {
        return Ok(());
    };
    let constant = constant as i128;
    let satisfiable = match op {
        ConstraintOp::Le => lower <= constant,
        ConstraintOp::Lt => lower < constant,
        ConstraintOp::Eq => lower <= constant && constant <= upper,
        ConstraintOp::Ne => lower != upper || lower != constant,
        ConstraintOp::Ge => upper >= constant,
        ConstraintOp::Gt => upper > constant,
    };
    if satisfiable {
        Ok(())
    } else {
        Err(format!(
            "unsatisfiable scalar_compare_const for {context}: domain [{lower}, {upper}] cannot satisfy {op:?} {constant}"
        ))
    }
}

fn validate_float_compare_const_domain(
    node: ArgNode<'_>,
    op: ConstraintOp,
    constant: f64,
    context: &str,
) -> Result<(), String> {
    let (lower, upper) = float_domain_bounds(node);
    let satisfiable = match op {
        ConstraintOp::Le => lower <= constant,
        ConstraintOp::Lt => lower < constant,
        ConstraintOp::Eq => lower <= constant && constant <= upper,
        ConstraintOp::Ne => lower != upper || lower != constant,
        ConstraintOp::Ge => upper >= constant,
        ConstraintOp::Gt => upper > constant,
    };
    if satisfiable {
        Ok(())
    } else {
        Err(format!(
            "unsatisfiable scalar_compare_const for {context}: float domain [{lower}, {upper}] cannot satisfy {op:?} {constant}"
        ))
    }
}

fn validate_scalar_compare_scalar_domains(
    lhs: ArgNode<'_>,
    op: ConstraintOp,
    rhs: ArgNode<'_>,
    lhs_context: &str,
    rhs_context: &str,
) -> Result<(), String> {
    let lhs_is_float = matches!(lhs.domain, Some(Domain::FloatRange { .. }));
    let rhs_is_float = matches!(rhs.domain, Some(Domain::FloatRange { .. }));
    if lhs_is_float && rhs_is_float {
        return validate_float_compare_scalar_domains(lhs, op, rhs, lhs_context, rhs_context);
    }
    if lhs_is_float || rhs_is_float {
        return Err(format!(
            "mixed float/integer scalar_compare_scalar is not supported for {lhs_context} {op:?} {rhs_context}"
        ));
    }
    let Some((lhs_lower, lhs_upper)) = int_domain_bounds(lhs) else {
        return Ok(());
    };
    let Some((rhs_lower, rhs_upper)) = int_domain_bounds(rhs) else {
        return Ok(());
    };
    let satisfiable = match op {
        ConstraintOp::Le => lhs_lower <= rhs_upper,
        ConstraintOp::Lt => lhs_lower < rhs_upper,
        ConstraintOp::Eq => lhs_lower <= rhs_upper && rhs_lower <= lhs_upper,
        ConstraintOp::Ne => {
            lhs_lower != lhs_upper || rhs_lower != rhs_upper || lhs_lower != rhs_lower
        }
        ConstraintOp::Ge => lhs_upper >= rhs_lower,
        ConstraintOp::Gt => lhs_upper > rhs_lower,
    };
    if satisfiable {
        Ok(())
    } else {
        Err(format!(
            "unsatisfiable scalar_compare_scalar for {lhs_context} {op:?} {rhs_context}: lhs domain [{lhs_lower}, {lhs_upper}], rhs domain [{rhs_lower}, {rhs_upper}]"
        ))
    }
}

fn non_negative_int_domain_bounds(node: ArgNode<'_>, context: &str) -> Result<(u64, u64), String> {
    if !matches!(node.domain, Some(Domain::IntRange { .. })) {
        return Err("scalar_product_le_const requires int_range domains".to_string());
    }
    let (lower, upper) = int_domain_bounds(node)
        .ok_or_else(|| "scalar_product_le_const requires int_range domains".to_string())?;
    if lower < 0 {
        return Err("scalar_product_le_const requires non-negative domains".to_string());
    }
    let lower = u64::try_from(lower).map_err(|_| {
        format!("scalar_product_le_const {context} lower bound is not representable")
    })?;
    let upper = u64::try_from(upper).map_err(|_| {
        format!("scalar_product_le_const {context} upper bound is not representable")
    })?;
    Ok((lower, upper))
}

fn validate_scalar_product_le_const_domains(
    lhs: ArgNode<'_>,
    rhs: ArgNode<'_>,
    value: u64,
    repair_lhs: bool,
) -> Result<(), String> {
    let lhs_bounds = non_negative_int_domain_bounds(lhs, "lhs")?;
    let rhs_bounds = non_negative_int_domain_bounds(rhs, "rhs")?;
    let (repair_lower, other_upper) = if repair_lhs {
        (lhs_bounds.0, rhs_bounds.1)
    } else {
        (rhs_bounds.0, lhs_bounds.1)
    };
    if u128::from(repair_lower) * u128::from(other_upper) <= u128::from(value) {
        return Ok(());
    }
    Err(format!(
        "unsatisfiable scalar_product_le_const: repair lower bound {repair_lower} times other upper bound {other_upper} exceeds {value}"
    ))
}

fn validate_float_compare_scalar_domains(
    lhs: ArgNode<'_>,
    op: ConstraintOp,
    rhs: ArgNode<'_>,
    lhs_context: &str,
    rhs_context: &str,
) -> Result<(), String> {
    let (lhs_lower, lhs_upper) = float_domain_bounds(lhs);
    let (rhs_lower, rhs_upper) = float_domain_bounds(rhs);
    let satisfiable = match op {
        ConstraintOp::Le => lhs_lower <= rhs_upper,
        ConstraintOp::Lt => lhs_lower < rhs_upper,
        ConstraintOp::Eq => lhs_lower <= rhs_upper && rhs_lower <= lhs_upper,
        ConstraintOp::Ne => {
            lhs_lower != lhs_upper || rhs_lower != rhs_upper || lhs_lower != rhs_lower
        }
        ConstraintOp::Ge => lhs_upper >= rhs_lower,
        ConstraintOp::Gt => lhs_upper > rhs_lower,
    };
    if satisfiable {
        Ok(())
    } else {
        Err(format!(
            "unsatisfiable scalar_compare_scalar for {lhs_context} {op:?} {rhs_context}: lhs float domain [{lhs_lower}, {lhs_upper}], rhs float domain [{rhs_lower}, {rhs_upper}]"
        ))
    }
}

fn validate_scalar_le_buffer_len_domain(
    scalar: ArgNode<'_>,
    buffer: ArgNode<'_>,
    unit: ConstraintUnit,
    scalar_context: &str,
    buffer_context: &str,
) -> Result<(), String> {
    let Some((scalar_lower, _)) = int_domain_bounds(scalar) else {
        return Ok(());
    };
    let Some(buffer_limit) = static_buffer_len_limit(buffer, unit) else {
        return Ok(());
    };
    if scalar_lower <= buffer_limit as i128 {
        return Ok(());
    }
    Err(format!(
        "unsatisfiable scalar_le_buffer_len for {scalar_context} <= {buffer_context}: scalar lower bound {scalar_lower} exceeds buffer limit {buffer_limit}"
    ))
}

fn validate_count_fits_buffer_domain(
    count: ArgNode<'_>,
    buffer: ArgNode<'_>,
    elem_size_bytes: usize,
    count_context: &str,
    buffer_context: &str,
) -> Result<(), String> {
    let Some((count_lower, _)) = int_domain_bounds(count) else {
        return Ok(());
    };
    let Some(Domain::Bytes {
        max_len: Some(max_len),
        ..
    }) = buffer.domain
    else {
        return Ok(());
    };
    let Some(max_len) = max_len.parse::<u64>().ok() else {
        return Ok(());
    };
    let buffer_limit = max_len / elem_size_bytes as u64;
    if count_lower <= buffer_limit as i128 {
        return Ok(());
    }
    Err(format!(
        "unsatisfiable count_fits_buffer for {count_context} * {elem_size_bytes} <= {buffer_context}: count lower bound {count_lower} exceeds buffer element limit {buffer_limit}"
    ))
}

fn buffer_element_constraint_size(
    buffer: ArgNode<'_>,
    elem_size_bytes: Option<usize>,
) -> Result<usize, String> {
    let elem_size = elem_size_bytes
        .or_else(|| payload_element_size(buffer))
        .ok_or_else(|| {
            "buffer_elements_lt_scalar requires elem_size_bytes or buffer element metadata"
                .to_string()
        })?;
    if !matches!(elem_size, 1 | 2 | 4 | 8) {
        return Err(
            "buffer_elements_lt_scalar elem_size_bytes must be one of 1, 2, 4, or 8".to_string(),
        );
    }
    Ok(elem_size)
}

fn static_buffer_len_limit(node: ArgNode<'_>, unit: ConstraintUnit) -> Option<u64> {
    let Some(Domain::Bytes {
        max_len: Some(max_len),
        ..
    }) = node.domain
    else {
        return None;
    };
    let max_len = max_len.parse::<u64>().ok()?;
    match unit {
        ConstraintUnit::Bytes => Some(max_len),
        ConstraintUnit::Elements => {
            let elem_size = payload_element_size(node)? as u64;
            max_len.checked_div(elem_size)
        }
    }
}

fn validate_bytes_domain(context: &str, domain: Option<&Domain>) -> Result<(), String> {
    let Some(Domain::Bytes { pattern_hex, .. }) = domain else {
        return Ok(());
    };
    if let Some(pattern_hex) = pattern_hex {
        decode_pattern_hex(pattern_hex)
            .map_err(|err| format!("{context} bytes.pattern_hex is invalid: {err}"))?;
    }
    Ok(())
}

fn decode_pattern_hex(raw: &str) -> Result<Vec<u8>, String> {
    if raw.is_empty() {
        return Err("pattern_hex must be non-empty".to_string());
    }
    if !raw.len().is_multiple_of(2) {
        return Err("pattern_hex must contain an even number of hex digits".to_string());
    }
    let mut bytes = Vec::with_capacity(raw.len() / 2);
    for idx in (0..raw.len()).step_by(2) {
        let chunk = &raw[idx..idx + 2];
        let byte = u8::from_str_radix(chunk, 16)
            .map_err(|_| format!("pattern_hex contains non-hex byte {chunk:?}"))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

fn byte_pattern(domain: Option<&Domain>) -> Result<Option<Vec<u8>>, String> {
    let Some(Domain::Bytes {
        pattern_hex: Some(pattern_hex),
        ..
    }) = domain
    else {
        return Ok(None);
    };
    decode_pattern_hex(pattern_hex).map(Some)
}

fn apply_byte_pattern(bytes: &mut [u8], pattern: &[u8]) {
    if pattern.is_empty() {
        return;
    }
    for (idx, byte) in bytes.iter_mut().enumerate() {
        *byte = pattern[idx % pattern.len()];
    }
}

fn int_domain_bounds(node: ArgNode<'_>) -> Option<(i128, i128)> {
    let Some(Domain::IntRange { min, max, signed }) = node.domain else {
        return None;
    };
    if signed.unwrap_or(false) {
        let lower = min
            .as_ref()
            .and_then(|v| v.parse::<i128>().ok())
            .unwrap_or(signed_min(node.size_bytes));
        let upper = max
            .as_ref()
            .and_then(|v| v.parse::<i128>().ok())
            .unwrap_or(signed_max(node.size_bytes));
        return Some((lower, upper.max(lower)));
    }
    let lower = min
        .as_ref()
        .and_then(|v| v.parse::<i128>().ok())
        .map(|v| v.max(0))
        .unwrap_or(0);
    let upper = max
        .as_ref()
        .and_then(|v| v.parse::<u64>().ok())
        .map(i128::from)
        .unwrap_or(u64::MAX as i128);
    Some((lower, upper.max(lower)))
}

pub fn load_manifest(path: &Path) -> Result<KernelManifest, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("failed to read manifest {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("failed to parse manifest {}: {e}", path.display()))
}

fn constraint_arg<'a>(
    args: &'a [ArgSpec],
    index: usize,
    label: &str,
) -> Result<&'a ArgSpec, String> {
    args.get(index)
        .ok_or_else(|| format!("constraint references missing {label} {index}"))
}

fn constraint_node<'a>(
    arg: &'a ArgSpec,
    path: Option<&'a [String]>,
    label: &str,
) -> Result<ArgNode<'a>, String> {
    find_node_by_path(arg.as_node(), path).ok_or_else(|| {
        format!(
            "constraint references missing {label} path {}",
            path_label(&arg.name, path)
        )
    })
}

fn validate_constraint_expr(args: &[ArgSpec], expr: &ConstraintExpr) -> Result<(), String> {
    match expr {
        ConstraintExpr::ArgValue { arg, path } => {
            let arg_spec = constraint_arg(args, *arg, "arg_value arg")?;
            let node = constraint_node(arg_spec, path.as_deref(), "arg_value")?;
            if node.kind != ArgKind::Scalar {
                return Err(format!(
                    "constraint arg_value {} is not scalar",
                    path_label(&arg_spec.name, path.as_deref())
                ));
            }
        }
        ConstraintExpr::PayloadLen { arg, path } => {
            let arg_spec = constraint_arg(args, *arg, "payload_len arg")?;
            let node = constraint_node(arg_spec, path.as_deref(), "payload_len")?;
            if node.kind != ArgKind::Pointer {
                return Err(format!(
                    "constraint payload_len {} is not a payload_buffer pointer",
                    path_label(&arg_spec.name, path.as_deref())
                ));
            }
        }
        ConstraintExpr::Const { .. } => {}
        ConstraintExpr::Binary { lhs, rhs, .. } => {
            validate_constraint_expr(args, lhs)?;
            validate_constraint_expr(args, rhs)?;
        }
    }
    Ok(())
}

fn validate_constraint_repair(args: &[ArgSpec], repair: &ConstraintRepair) -> Result<(), String> {
    match repair {
        ConstraintRepair::ResizePayload { arg, path } => {
            let arg_spec = constraint_arg(args, *arg, "resize_payload arg")?;
            let node = constraint_node(arg_spec, path.as_deref(), "resize_payload")?;
            if node.kind != ArgKind::Pointer {
                return Err(format!(
                    "constraint resize_payload {} is not a payload_buffer pointer",
                    path_label(&arg_spec.name, path.as_deref())
                ));
            }
        }
    }
    Ok(())
}

fn path_label(default: &str, path: Option<&[String]>) -> String {
    path.map(|path| path.join("."))
        .unwrap_or_else(|| default.to_string())
}

fn eval_constraint_expr(
    args: &[ArgSpec],
    values: &[ArgValue],
    expr: &ConstraintExpr,
) -> Option<u128> {
    match expr {
        ConstraintExpr::ArgValue { arg, path } => {
            let arg_node = args.get(*arg)?.as_node();
            match values
                .get(*arg)
                .and_then(|value| find_value_by_path(value, arg_node, path.as_deref()))?
            {
                ArgValue::Scalar(value) => Some(u128::from(*value)),
                _ => None,
            }
        }
        ConstraintExpr::PayloadLen { arg, path } => {
            let arg_node = args.get(*arg)?.as_node();
            match values
                .get(*arg)
                .and_then(|value| find_value_by_path(value, arg_node, path.as_deref()))?
            {
                ArgValue::Pointer(bytes) => Some(bytes.len() as u128),
                _ => None,
            }
        }
        ConstraintExpr::Const { value } => Some(u128::from(*value)),
        ConstraintExpr::Binary { op, lhs, rhs } => {
            let lhs = eval_constraint_expr(args, values, lhs)?;
            let rhs = eval_constraint_expr(args, values, rhs)?;
            match op {
                ConstraintBinaryOp::Add => lhs.checked_add(rhs),
                ConstraintBinaryOp::Sub => lhs.checked_sub(rhs),
                ConstraintBinaryOp::Mul => lhs.checked_mul(rhs),
                ConstraintBinaryOp::Div => lhs.checked_div(rhs),
            }
        }
    }
}

fn constraint_values_match(lhs: u128, op: ConstraintOp, rhs: u128) -> bool {
    match op {
        ConstraintOp::Le => lhs <= rhs,
        ConstraintOp::Lt => lhs < rhs,
        ConstraintOp::Eq => lhs == rhs,
        ConstraintOp::Ne => lhs != rhs,
        ConstraintOp::Ge => lhs >= rhs,
        ConstraintOp::Gt => lhs > rhs,
    }
}

impl ArgPackSpec {
    pub fn from_manifest(manifest: KernelManifest) -> Result<Self, String> {
        if manifest.schema_version != 1 {
            return Err(format!(
                "unsupported manifest schema_version {}",
                manifest.schema_version
            ));
        }

        if manifest.kernels.len() != 1 {
            return Err(format!(
                "arg-pack-v1 expects exactly one kernel per manifest, got {}",
                manifest.kernels.len()
            ));
        }
        let kernel = &manifest.kernels[0];
        let vconfig_bounds = launch_policy_bounds(kernel.launch_policy.as_ref())?;

        let mut args = kernel.args.clone();
        args.sort_by_key(|arg| arg.index);
        for (position, arg) in args.iter().enumerate() {
            if arg.index != position {
                return Err(format!(
                    "arg indexes must be contiguous from 0, expected {position}, got {}",
                    arg.index
                ));
            }
            normalize_alignment(arg.align_bytes)?;
            validate_bytes_domain(&format!("arg {}", arg.name), arg.domain.as_ref())?;
            match arg.kind {
                ArgKind::Pointer => {
                    if arg.size_bytes == 0 {
                        return Err(format!("pointer arg {} has size_bytes=0", arg.name));
                    }
                    let pointer_role = arg
                        .pointer_role
                        .ok_or_else(|| format!("pointer arg {} missing pointer_role", arg.name))?;
                    if arg.pointee_layout.is_none() {
                        return Err(format!("pointer arg {} missing pointee_layout", arg.name));
                    }
                    match pointer_role {
                        PointerRole::PayloadBuffer => {}
                        other => {
                            return Err(format!(
                                "pointer arg {} uses unsupported pointer_role={other:?}; arg-pack-v1 only supports payload_buffer",
                                arg.name
                            ));
                        }
                    }
                }
                ArgKind::Scalar => {
                    if arg.pointer_role.is_some() || arg.pointee_layout.is_some() {
                        return Err(format!(
                            "non-pointer arg {} must not set pointer metadata",
                            arg.name
                        ));
                    }
                    if !matches!(arg.size_bytes, 1 | 2 | 4 | 8) {
                        return Err(format!(
                            "scalar arg {} has unsupported size_bytes={}",
                            arg.name, arg.size_bytes
                        ));
                    }
                    validate_scalar_domain(
                        &format!("scalar arg {}", arg.name),
                        arg.size_bytes,
                        arg.domain.as_ref(),
                    )?;
                }
                ArgKind::OpaqueVal => {
                    if arg.pointer_role.is_some() || arg.pointee_layout.is_some() {
                        return Err(format!(
                            "non-pointer arg {} must not set pointer metadata",
                            arg.name
                        ));
                    }
                    if arg.size_bytes == 0 {
                        return Err(format!("opaque_val arg {} has size_bytes=0", arg.name));
                    }
                }
                ArgKind::OpaqueWithPtr => {
                    if arg.pointer_role.is_some() || arg.pointee_layout.is_some() {
                        return Err(format!(
                            "non-pointer arg {} must not set pointer metadata",
                            arg.name
                        ));
                    }
                    let type_layout = arg.type_layout.as_ref().ok_or_else(|| {
                        format!("opaque_with_ptr arg {} missing type_layout", arg.name)
                    })?;
                    if type_layout.layout_status != LayoutStatus::Complete {
                        return Err(format!(
                            "opaque_with_ptr arg {} requires complete type_layout",
                            arg.name
                        ));
                    }
                    for field in &type_layout.fields {
                        validate_layout_node(field, &field.index, true)?;
                    }
                    if let Some(element) = &type_layout.element {
                        validate_layout_node(element, &format!("{}[]", arg.name), true)?;
                    }
                }
            }
            if let Some(pointee) = &arg.pointee_layout {
                validate_layout_node(pointee, &format!("{}.*", arg.name), false)?;
            }
        }

        for constraint in &kernel.constraints {
            match constraint {
                Constraint::ScalarLeLogicalBlockDim {
                    scalar_arg,
                    dimension,
                }
                | Constraint::ScalarEqLogicalBlockDim {
                    scalar_arg,
                    dimension,
                } => {
                    let scalar_arg_spec = constraint_arg(&args, *scalar_arg, "scalar_arg")?;
                    if scalar_arg_spec.kind != ArgKind::Scalar {
                        return Err(format!(
                            "constraint scalar_arg {} is not scalar",
                            scalar_arg_spec.name
                        ));
                    }
                    let default_block = [block_shape(vconfig_bounds.default)];
                    let dimensions = vconfig_bounds
                        .logical_block_candidates
                        .as_deref()
                        .unwrap_or(&default_block)
                        .iter()
                        .map(|candidate| logical_block_dimension(*candidate, *dimension));
                    if matches!(constraint, Constraint::ScalarEqLogicalBlockDim { .. }) {
                        if let Some(unrepresentable) = dimensions.clone().find(|value| {
                            !scalar_domain_permits_value(scalar_arg_spec.as_node(), *value)
                        }) {
                            return Err(format!(
                                "constraint scalar_arg {} domain does not permit logical block dimension {unrepresentable}",
                                scalar_arg_spec.name
                            ));
                        }
                    } else {
                        let smallest_dimension =
                            dimensions.min().expect("logical block domain is non-empty");
                        if positive_scalar_lower_bound(scalar_arg_spec.as_node())
                            .is_some_and(|lower| lower > u64::from(smallest_dimension))
                        {
                            return Err(format!(
                                "constraint scalar_arg {} positive minimum exceeds smallest logical block dimension",
                                scalar_arg_spec.name
                            ));
                        }
                    }
                }
                Constraint::ScalarLeBufferLen {
                    scalar_arg,
                    scalar_path,
                    buffer_arg,
                    buffer_path,
                    unit,
                } => {
                    let scalar_arg_spec = constraint_arg(&args, *scalar_arg, "scalar_arg")?;
                    let buffer_arg_spec = constraint_arg(&args, *buffer_arg, "buffer_arg")?;
                    let scalar =
                        constraint_node(scalar_arg_spec, scalar_path.as_deref(), "scalar")?;
                    let buffer =
                        constraint_node(buffer_arg_spec, buffer_path.as_deref(), "buffer")?;
                    if scalar.kind != ArgKind::Scalar {
                        return Err(format!(
                            "constraint scalar_arg {} is not scalar",
                            scalar_arg_spec.name
                        ));
                    }
                    if buffer.kind != ArgKind::Pointer {
                        return Err(format!(
                            "constraint buffer_arg {} is not a payload_buffer pointer",
                            buffer_arg_spec.name
                        ));
                    }
                    if *unit == ConstraintUnit::Elements && payload_element_size(buffer).is_none() {
                        return Err(
                            "scalar_le_buffer_len unit=elements requires buffer elem_size_bytes or pointee_layout size_bytes"
                                .to_string(),
                        );
                    }
                    validate_scalar_le_buffer_len_domain(
                        scalar,
                        buffer,
                        *unit,
                        &path_label(&scalar_arg_spec.name, scalar_path.as_deref()),
                        &path_label(&buffer_arg_spec.name, buffer_path.as_deref()),
                    )?;
                }
                Constraint::ScalarCompareConst {
                    scalar_arg,
                    scalar_path,
                    op,
                    value,
                } => {
                    let scalar_arg_spec = constraint_arg(&args, *scalar_arg, "scalar_arg")?;
                    let scalar =
                        constraint_node(scalar_arg_spec, scalar_path.as_deref(), "scalar")?;
                    if scalar.kind != ArgKind::Scalar {
                        return Err(format!(
                            "constraint scalar_arg {} is not scalar",
                            scalar_arg_spec.name
                        ));
                    }
                    validate_scalar_compare_const_domain(
                        scalar,
                        *op,
                        *value,
                        &path_label(&scalar_arg_spec.name, scalar_path.as_deref()),
                    )?;
                }
                Constraint::ScalarCompareScalar {
                    lhs_arg,
                    lhs_path,
                    op,
                    rhs_arg,
                    rhs_path,
                } => {
                    let lhs_arg_spec = constraint_arg(&args, *lhs_arg, "lhs_arg")?;
                    let rhs_arg_spec = constraint_arg(&args, *rhs_arg, "rhs_arg")?;
                    let lhs = constraint_node(lhs_arg_spec, lhs_path.as_deref(), "lhs")?;
                    let rhs = constraint_node(rhs_arg_spec, rhs_path.as_deref(), "rhs")?;
                    if lhs.kind != ArgKind::Scalar || rhs.kind != ArgKind::Scalar {
                        return Err("scalar_compare_scalar args must both be scalar".to_string());
                    }
                    validate_scalar_compare_scalar_domains(
                        lhs,
                        *op,
                        rhs,
                        &path_label(&lhs_arg_spec.name, lhs_path.as_deref()),
                        &path_label(&rhs_arg_spec.name, rhs_path.as_deref()),
                    )?;
                }
                Constraint::ScalarProductLeConst {
                    lhs_arg,
                    lhs_path,
                    rhs_arg,
                    rhs_path,
                    value,
                    repair_arg,
                    repair_path,
                } => {
                    let lhs_arg_spec = constraint_arg(&args, *lhs_arg, "lhs_arg")?;
                    let rhs_arg_spec = constraint_arg(&args, *rhs_arg, "rhs_arg")?;
                    let lhs = constraint_node(lhs_arg_spec, lhs_path.as_deref(), "lhs")?;
                    let rhs = constraint_node(rhs_arg_spec, rhs_path.as_deref(), "rhs")?;
                    if lhs.kind != ArgKind::Scalar || rhs.kind != ArgKind::Scalar {
                        return Err(
                            "scalar_product_le_const operands must both be scalar".to_string()
                        );
                    }
                    let repair_lhs = repair_arg == lhs_arg && repair_path == lhs_path;
                    let repair_rhs = repair_arg == rhs_arg && repair_path == rhs_path;
                    if !repair_lhs && !repair_rhs {
                        return Err(
                            "scalar_product_le_const repair target must match lhs or rhs"
                                .to_string(),
                        );
                    }
                    validate_scalar_product_le_const_domains(lhs, rhs, *value, repair_lhs)?;
                }
                Constraint::CountFitsBuffer {
                    count_arg,
                    count_path,
                    buffer_arg,
                    buffer_path,
                    elem_size_bytes,
                } => {
                    if *elem_size_bytes == 0 {
                        return Err(
                            "count_fits_buffer elem_size_bytes must be non-zero".to_string()
                        );
                    }
                    let count_arg_spec = constraint_arg(&args, *count_arg, "count_arg")?;
                    let buffer_arg_spec = constraint_arg(&args, *buffer_arg, "buffer_arg")?;
                    let count = constraint_node(count_arg_spec, count_path.as_deref(), "count")?;
                    let buffer =
                        constraint_node(buffer_arg_spec, buffer_path.as_deref(), "buffer")?;
                    if count.kind != ArgKind::Scalar {
                        return Err("count_fits_buffer count must be scalar".to_string());
                    }
                    if buffer.kind != ArgKind::Pointer {
                        return Err(
                            "count_fits_buffer buffer must be a payload_buffer pointer".to_string()
                        );
                    }
                    validate_count_fits_buffer_domain(
                        count,
                        buffer,
                        *elem_size_bytes,
                        &path_label(&count_arg_spec.name, count_path.as_deref()),
                        &path_label(&buffer_arg_spec.name, buffer_path.as_deref()),
                    )?;
                }
                Constraint::BufferElementsLtScalar {
                    buffer_arg,
                    buffer_path,
                    scalar_arg,
                    scalar_path,
                    elem_size_bytes,
                } => {
                    let buffer_arg_spec = constraint_arg(&args, *buffer_arg, "buffer_arg")?;
                    let scalar_arg_spec = constraint_arg(&args, *scalar_arg, "scalar_arg")?;
                    let buffer =
                        constraint_node(buffer_arg_spec, buffer_path.as_deref(), "buffer")?;
                    let scalar =
                        constraint_node(scalar_arg_spec, scalar_path.as_deref(), "scalar")?;
                    if buffer.kind != ArgKind::Pointer {
                        return Err(
                            "buffer_elements_lt_scalar buffer must be a payload_buffer pointer"
                                .to_string(),
                        );
                    }
                    if scalar.kind != ArgKind::Scalar {
                        return Err(
                            "buffer_elements_lt_scalar scalar bound must be scalar".to_string()
                        );
                    }
                    buffer_element_constraint_size(buffer, *elem_size_bytes)?;
                }
                Constraint::ExpressionCompare {
                    lhs,
                    op,
                    rhs,
                    repair,
                } => {
                    validate_constraint_expr(&args, lhs)?;
                    validate_constraint_expr(&args, rhs)?;
                    validate_constraint_repair(&args, repair)?;
                    let ConstraintRepair::ResizePayload {
                        arg: repair_arg,
                        path: repair_path,
                    } = repair;
                    let ConstraintExpr::PayloadLen {
                        arg: rhs_arg,
                        path: rhs_path,
                    } = rhs
                    else {
                        return Err(
                            "resize_payload requires payload_len on expression rhs".to_string()
                        );
                    };
                    if *op != ConstraintOp::Le {
                        return Err("resize_payload requires expression op <=".to_string());
                    }
                    if repair_arg != rhs_arg || repair_path != rhs_path {
                        return Err("resize_payload must target compared payload_len".to_string());
                    }
                }
            }
        }

        Ok(Self {
            symbol_name: kernel.symbol_name.clone(),
            display_name: kernel.display_name.clone(),
            args,
            constraints: kernel.constraints.clone(),
            vconfig_bounds,
        })
    }

    pub fn from_manifest_path(path: &Path) -> Result<Self, String> {
        Self::from_manifest(load_manifest(path)?)
    }

    fn pack_current_kernel_args(&self, input: &[u8], output: &[u8], kernel_size: usize) -> Vec<u8> {
        // Convenience encoder for the current kernelmanifest.json subset:
        // first pointer=input, second pointer=output, first scalar=kernel_size.
        let mut values = Vec::with_capacity(self.args.len());
        let mut pointer_seen = 0usize;
        let mut scalar_seen = 0usize;

        for arg in &self.args {
            match arg.kind {
                ArgKind::Pointer => {
                    let bytes = if pointer_seen == 0 { input } else { output };
                    values.push(ArgValue::Pointer(bytes.to_vec()));
                    pointer_seen += 1;
                }
                ArgKind::Scalar => {
                    let value = if scalar_seen == 0 {
                        kernel_size as u64
                    } else {
                        scalar_default(arg)
                    };
                    values.push(ArgValue::Scalar(value));
                    scalar_seen += 1;
                }
                ArgKind::OpaqueVal => {
                    values.push(ArgValue::OpaqueVal(vec![0; arg.size_bytes]));
                }
                ArgKind::OpaqueWithPtr => {
                    values.push(default_value_for_node(arg.as_node()));
                }
            }
        }

        self.pack_values(values)
    }

    fn default_values(&self) -> Vec<ArgValue> {
        let mut values = Vec::with_capacity(self.args.len());
        let mut pointer_seen = 0usize;
        for arg in &self.args {
            match arg.kind {
                ArgKind::Pointer => {
                    let bytes = if pointer_seen == 0 {
                        b"FUZZ".to_vec()
                    } else {
                        vec![0; 4]
                    };
                    values.push(ArgValue::Pointer(bytes));
                    pointer_seen += 1;
                }
                ArgKind::Scalar => values.push(ArgValue::Scalar(scalar_default(arg))),
                ArgKind::OpaqueVal => values.push(ArgValue::OpaqueVal(vec![0; arg.size_bytes])),
                ArgKind::OpaqueWithPtr => values.push(default_value_for_node(arg.as_node())),
            }
        }
        values
    }

    fn default_seed(&self) -> Vec<u8> {
        ARG_PACK_STATS
            .seed_generation_count
            .fetch_add(1, Ordering::Relaxed);
        let values = self.default_values();
        self.pack_values(values)
    }

    fn default_rapid_input(&self) -> Vec<u8> {
        encode_rapid_input(self.vconfig_bounds.default, &self.default_seed())
    }

    fn normalize(&self, raw: &[u8]) -> Vec<u8> {
        self.normalize_for_vconfig(raw, self.vconfig_bounds.default)
    }

    fn normalize_for_vconfig(&self, raw: &[u8], vconfig: RapidVConfig) -> Vec<u8> {
        ARG_PACK_STATS
            .normalize_calls
            .fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        TEST_NORMALIZE_CALLS.set(TEST_NORMALIZE_CALLS.get() + 1);
        if let Some(parsed) = self.parse(raw) {
            let packed = self.pack_values_for_vconfig(parsed.values, vconfig);
            if packed != raw {
                ARG_PACK_STATS
                    .normalize_repack_count
                    .fetch_add(1, Ordering::Relaxed);
            }
            return packed;
        }
        ARG_PACK_STATS
            .invalid_repair_count
            .fetch_add(1, Ordering::Relaxed);
        self.pack_values_for_vconfig(self.default_values(), vconfig)
    }

    fn normalize_rapid_input(&self, raw: &[u8]) -> Vec<u8> {
        let (vconfig, payload) = split_rapid_input_or_repair(raw, self.vconfig_bounds.default);
        let vconfig = clamp_vconfig(vconfig, &self.vconfig_bounds);
        let normalized_payload = if raw.len() < RAPID_TASK_ENVELOPE_HEADER_BYTES {
            self.pack_values_for_vconfig(self.default_values(), vconfig)
        } else {
            self.normalize_for_vconfig(payload, vconfig)
        };
        encode_rapid_input(vconfig, &normalized_payload)
    }

    fn parse(&self, bytes: &[u8]) -> Option<ParsedArgPack> {
        let mut offset = 0usize;
        let mut values = Vec::with_capacity(self.args.len());

        for arg in &self.args {
            values.push(parse_node(arg.as_node(), bytes, &mut offset)?);
        }

        if offset != bytes.len() {
            return None;
        }

        Some(ParsedArgPack { values })
    }

    fn pack_values(&self, values: Vec<ArgValue>) -> Vec<u8> {
        self.pack_values_for_vconfig(values, self.vconfig_bounds.default)
    }

    fn pack_values_for_vconfig(&self, mut values: Vec<ArgValue>, vconfig: RapidVConfig) -> Vec<u8> {
        self.apply_domains(&mut values);
        let max_passes = self.constraints.len().saturating_add(1).max(1);
        for _ in 0..max_passes {
            let before = values.clone();
            self.apply_constraints(&mut values);
            self.apply_launch_constraints(&mut values, vconfig);
            if before == values {
                break;
            }
        }

        let mut out = Vec::new();
        let mut leaves = Vec::new();
        for (arg, value) in self.args.iter().zip(values.iter()) {
            flatten_pack_leaves(&mut leaves, arg.as_node(), Some(value));
        }
        for (idx, (node, value)) in leaves.iter().enumerate() {
            let next = leaves.get(idx + 1).map(|(node, _)| *node);
            pack_node(&mut out, *node, *value, next);
        }
        out
    }

    fn mutatable_leaf_count(&self) -> usize {
        self.args
            .iter()
            .map(|arg| mutatable_leaf_count_for_node(arg.as_node()))
            .sum()
    }

    fn mutate(&self, raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
        ARG_PACK_STATS
            .mutation_calls
            .fetch_add(1, Ordering::Relaxed);
        let mut values = self
            .parse(raw)
            .map(|parsed| parsed.values)
            .unwrap_or_else(|| self.default_values());
        let leaf_count = self.mutatable_leaf_count().max(1);
        let target = (selector as usize) % leaf_count;
        let mut seen = 0usize;
        for (arg, value) in self.args.iter().zip(values.iter_mut()) {
            if mutate_value_for_node(arg.as_node(), value, target, &mut seen, selector, byte) {
                break;
            }
        }
        self.pack_values(values)
    }

    fn mutate_rapid_input_unrepaired(&self, raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
        // RapidInputMutator supplies canonical inputs and repairs once after its mutation plan.
        let envelope = parse_rapid_input_envelope(raw)
            .expect("structured mutation requires a canonical rapid input");
        let mutated_payload = self.mutate(envelope.payload, selector, byte);
        encode_rapid_input(envelope.vconfig, &mutated_payload)
    }

    fn mutate_rapid_input(&self, raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
        let normalized = self.normalize_rapid_input(raw);
        let mutated = self.mutate_rapid_input_unrepaired(&normalized, selector, byte);
        self.normalize_rapid_input(&mutated)
    }

    fn mutate_rapid_input_havoc_unrepaired(&self, raw: &[u8], operations: &[(u64, u8)]) -> Vec<u8> {
        ARG_PACK_STATS
            .mutation_calls
            .fetch_add(1, Ordering::Relaxed);
        let envelope = parse_rapid_input_envelope(raw)
            .expect("havoc mutation requires a canonical rapid input");
        let mut values = self
            .parse(envelope.payload)
            .expect("canonical rapid payload must match the manifest")
            .values;
        let leaf_count = self.mutatable_leaf_count().max(1);

        for &(selector, byte) in operations.iter().take(ARG_PACK_HAVOC_MAX_STACKED_OPS) {
            let target = (selector % leaf_count as u64) as usize;
            let mutation_selector = selector / leaf_count as u64;
            let mut seen = 0usize;
            for (arg, value) in self.args.iter().zip(values.iter_mut()) {
                if mutate_value_havoc_for_node(
                    arg.as_node(),
                    value,
                    target,
                    &mut seen,
                    mutation_selector,
                    byte,
                ) {
                    break;
                }
            }
        }

        let payload = self.pack_values_for_vconfig(values, envelope.vconfig);
        encode_rapid_input(envelope.vconfig, &payload)
    }

    fn mutate_rapid_input_havoc(&self, raw: &[u8], operations: &[(u64, u8)]) -> Vec<u8> {
        let normalized = self.normalize_rapid_input(raw);
        let mutated = self.mutate_rapid_input_havoc_unrepaired(&normalized, operations);
        self.normalize_rapid_input(&mutated)
    }

    fn mutate_rapid_vconfig_unrepaired(&self, raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
        let envelope = parse_rapid_input_envelope(raw)
            .expect("VConfig mutation requires a canonical rapid input");
        let mut vconfig = envelope.vconfig;
        if let Some(candidates) = self.vconfig_bounds.logical_block_candidates.as_deref() {
            let random = (selector >> 8) ^ u64::from(byte);
            let candidate = candidates[(random % candidates.len() as u64) as usize];
            return encode_rapid_input(with_block_shape(vconfig, candidate), envelope.payload);
        }
        let random = ((selector >> 8) as u32) ^ u32::from(byte);
        let bounds = [
            (
                self.vconfig_bounds.min.grid_x,
                self.vconfig_bounds.max.grid_x,
            ),
            (
                self.vconfig_bounds.min.grid_y,
                self.vconfig_bounds.max.grid_y,
            ),
            (
                self.vconfig_bounds.min.grid_z,
                self.vconfig_bounds.max.grid_z,
            ),
            (
                self.vconfig_bounds.min.block_x,
                self.vconfig_bounds.max.block_x,
            ),
            (
                self.vconfig_bounds.min.block_y,
                self.vconfig_bounds.max.block_y,
            ),
            (
                self.vconfig_bounds.min.block_z,
                self.vconfig_bounds.max.block_z,
            ),
        ];
        let mut mutable_dims = [0usize; 6];
        let mut mutable_count = 0usize;
        for (dimension, (min, max)) in bounds.iter().copied().enumerate() {
            if min < max {
                mutable_dims[mutable_count] = dimension;
                mutable_count += 1;
            }
        }
        if mutable_count == 0 {
            return raw.to_vec();
        }

        let dimension = mutable_dims[(selector % mutable_count as u64) as usize];
        match dimension {
            0 => {
                vconfig.grid_x = map_mutated_dim(
                    random,
                    self.vconfig_bounds.min.grid_x,
                    self.vconfig_bounds.max.grid_x,
                    byte,
                    1,
                )
            }
            1 => {
                vconfig.grid_y = map_mutated_dim(
                    random,
                    self.vconfig_bounds.min.grid_y,
                    self.vconfig_bounds.max.grid_y,
                    byte,
                    1,
                )
            }
            2 => {
                vconfig.grid_z = map_mutated_dim(
                    random,
                    self.vconfig_bounds.min.grid_z,
                    self.vconfig_bounds.max.grid_z,
                    byte,
                    1,
                )
            }
            3 => {
                let other_dims = vconfig.block_y.saturating_mul(vconfig.block_z);
                let near_step = if self.vconfig_bounds.require_whole_warps {
                    VCONFIG_WARP_SIZE
                        / greatest_common_divisor(VCONFIG_WARP_SIZE, other_dims.max(1))
                } else {
                    1
                };
                vconfig.block_x = map_mutated_dim(
                    random,
                    self.vconfig_bounds.min.block_x,
                    self.vconfig_bounds.max.block_x,
                    byte,
                    near_step,
                )
            }
            4 => {
                vconfig.block_y = map_mutated_dim(
                    random,
                    self.vconfig_bounds.min.block_y,
                    self.vconfig_bounds.max.block_y,
                    byte,
                    1,
                )
            }
            _ => {
                vconfig.block_z = map_mutated_dim(
                    random,
                    self.vconfig_bounds.min.block_z,
                    self.vconfig_bounds.max.block_z,
                    byte,
                    1,
                )
            }
        }
        encode_rapid_input(
            clamp_vconfig(vconfig, &self.vconfig_bounds),
            envelope.payload,
        )
    }

    fn mutate_rapid_vconfig(&self, raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
        let normalized = self.normalize_rapid_input(raw);
        let mutated = self.mutate_rapid_vconfig_unrepaired(&normalized, selector, byte);
        self.normalize_rapid_input(&mutated)
    }

    fn apply_domains(&self, values: &mut [ArgValue]) {
        for (arg, value) in self.args.iter().zip(values.iter_mut()) {
            apply_domain_for_node(arg.as_node(), value);
        }
    }

    fn apply_launch_constraints(&self, values: &mut [ArgValue], vconfig: RapidVConfig) {
        for constraint in &self.constraints {
            match constraint {
                Constraint::ScalarLeLogicalBlockDim {
                    scalar_arg,
                    dimension,
                } => {
                    let limit = u64::from(logical_vconfig_dimension(vconfig, *dimension));
                    if let Some(ArgValue::Scalar(value)) = values.get_mut(*scalar_arg) {
                        *value = (*value).min(limit);
                    }
                }
                Constraint::ScalarEqLogicalBlockDim {
                    scalar_arg,
                    dimension,
                } => {
                    let selected = u64::from(logical_vconfig_dimension(vconfig, *dimension));
                    if let Some(ArgValue::Scalar(value)) = values.get_mut(*scalar_arg) {
                        *value = selected;
                    }
                }
                _ => {}
            }
        }
    }

    fn apply_constraints(&self, values: &mut [ArgValue]) {
        let max_passes = self.constraints.len().saturating_add(1).max(1);
        for _ in 0..max_passes {
            let before = values.to_vec();
            self.apply_constraints_once(values);
            if before == values {
                break;
            }
        }
    }

    fn apply_constraints_once(&self, values: &mut [ArgValue]) {
        for constraint in &self.constraints {
            if let Constraint::ScalarProductLeConst {
                lhs_arg,
                lhs_path,
                rhs_arg,
                rhs_path,
                value,
                repair_arg,
                repair_path,
            } = constraint
            {
                let Some(lhs_node) = self.args.get(*lhs_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(rhs_node) = self.args.get(*rhs_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let lhs_value = match values
                    .get(*lhs_arg)
                    .and_then(|value| find_value_by_path(value, lhs_node, lhs_path.as_deref()))
                {
                    Some(ArgValue::Scalar(value)) => *value,
                    _ => continue,
                };
                let rhs_value = match values
                    .get(*rhs_arg)
                    .and_then(|value| find_value_by_path(value, rhs_node, rhs_path.as_deref()))
                {
                    Some(ArgValue::Scalar(value)) => *value,
                    _ => continue,
                };
                if u128::from(lhs_value) * u128::from(rhs_value) <= u128::from(*value) {
                    continue;
                }
                let (repair_node, other_value) = if repair_arg == lhs_arg && repair_path == lhs_path
                {
                    (lhs_node, rhs_value)
                } else if repair_arg == rhs_arg && repair_path == rhs_path {
                    (rhs_node, lhs_value)
                } else {
                    continue;
                };
                if other_value == 0 {
                    continue;
                }
                if let Some(ArgValue::Scalar(repair_value)) =
                    values.get_mut(*repair_arg).and_then(|value| {
                        find_value_by_path_mut(value, repair_node, repair_path.as_deref())
                    })
                {
                    *repair_value = (*repair_value).min(*value / other_value);
                }
                continue;
            }

            if let Constraint::ExpressionCompare {
                lhs,
                op,
                rhs,
                repair,
            } = constraint
            {
                let Some(lhs_value) = eval_constraint_expr(&self.args, values, lhs) else {
                    continue;
                };
                let Some(rhs_value) = eval_constraint_expr(&self.args, values, rhs) else {
                    continue;
                };
                if constraint_values_match(lhs_value, *op, rhs_value) {
                    continue;
                }
                if *op != ConstraintOp::Le {
                    continue;
                }
                let Ok(required_len) = usize::try_from(lhs_value) else {
                    continue;
                };
                match repair {
                    ConstraintRepair::ResizePayload { arg, path } => {
                        let Some(arg_node) = self.args.get(*arg).map(ArgSpec::as_node) else {
                            continue;
                        };
                        let Some(payload_node) = find_node_by_path(arg_node, path.as_deref())
                        else {
                            continue;
                        };
                        if let Some(ArgValue::Pointer(bytes)) =
                            values.get_mut(*arg).and_then(|value| {
                                find_value_by_path_mut(value, arg_node, path.as_deref())
                            })
                        {
                            let (_, max_len) = payload_len_bounds(payload_node.domain, bytes.len());
                            let target_len = required_len.min(max_len);
                            if bytes.len() < target_len {
                                bytes.resize(target_len, PAD_BYTE);
                            }
                        }
                    }
                }
                continue;
            }

            if let Constraint::ScalarLeBufferLen {
                scalar_arg,
                scalar_path,
                buffer_arg,
                buffer_path,
                unit,
            } = constraint
            {
                let Some(buffer_arg_node) = self.args.get(*buffer_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(buffer_layout_node) =
                    find_node_by_path(buffer_arg_node, buffer_path.as_deref())
                else {
                    continue;
                };
                let Some(scalar_node) = self.args.get(*scalar_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(scalar_layout_node) =
                    find_node_by_path(scalar_node, scalar_path.as_deref())
                else {
                    continue;
                };
                if let Some(required_len) = positive_scalar_lower_bound(scalar_layout_node)
                    .and_then(|lower| {
                        required_buffer_len_for_scalar(lower, buffer_layout_node, *unit)
                    })
                {
                    if let Some(ArgValue::Pointer(bytes)) =
                        values.get_mut(*buffer_arg).and_then(|value| {
                            find_value_by_path_mut(value, buffer_arg_node, buffer_path.as_deref())
                        })
                    {
                        let (_, max_len) =
                            payload_len_bounds(buffer_layout_node.domain, bytes.len());
                        let target_len = required_len.min(max_len);
                        if bytes.len() < target_len {
                            bytes.resize(target_len, PAD_BYTE);
                        }
                    }
                }
                let buffer_len_bytes = match values.get(*buffer_arg).and_then(|value| {
                    find_value_by_path(value, buffer_arg_node, buffer_path.as_deref())
                }) {
                    Some(ArgValue::Pointer(bytes)) => bytes.len() as u64,
                    _ => continue,
                };
                let scalar_limit = match unit {
                    ConstraintUnit::Bytes => buffer_len_bytes,
                    ConstraintUnit::Elements => {
                        let Some(elem_size) = payload_element_size(buffer_layout_node) else {
                            continue;
                        };
                        buffer_len_bytes / elem_size as u64
                    }
                };
                if let Some(ArgValue::Scalar(value)) =
                    values.get_mut(*scalar_arg).and_then(|value| {
                        find_value_by_path_mut(value, scalar_node, scalar_path.as_deref())
                    })
                {
                    *value = (*value).min(scalar_limit);
                }
                continue;
            }

            if let Constraint::ScalarCompareConst {
                scalar_arg,
                scalar_path,
                op,
                value: constant,
            } = constraint
            {
                let Some(scalar_node) = self.args.get(*scalar_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(scalar_layout_node) =
                    find_node_by_path(scalar_node, scalar_path.as_deref())
                else {
                    continue;
                };
                if let Some(ArgValue::Scalar(value)) =
                    values.get_mut(*scalar_arg).and_then(|value| {
                        find_value_by_path_mut(value, scalar_node, scalar_path.as_deref())
                    })
                {
                    apply_const_constraint(value, scalar_layout_node, *op, *constant);
                }
                continue;
            }

            if let Constraint::ScalarCompareScalar {
                lhs_arg,
                lhs_path,
                op,
                rhs_arg,
                rhs_path,
            } = constraint
            {
                let Some(rhs_node) = self.args.get(*rhs_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(rhs_layout_node) = find_node_by_path(rhs_node, rhs_path.as_deref()) else {
                    continue;
                };
                let rhs_value = match values
                    .get(*rhs_arg)
                    .and_then(|value| find_value_by_path(value, rhs_node, rhs_path.as_deref()))
                {
                    Some(ArgValue::Scalar(value)) => *value,
                    _ => continue,
                };
                let Some(lhs_node) = self.args.get(*lhs_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(lhs_layout_node) = find_node_by_path(lhs_node, lhs_path.as_deref()) else {
                    continue;
                };
                let lhs_value = match values
                    .get(*lhs_arg)
                    .and_then(|value| find_value_by_path(value, lhs_node, lhs_path.as_deref()))
                {
                    Some(ArgValue::Scalar(value)) => *value,
                    _ => continue,
                };
                let (new_lhs, new_rhs) =
                    if matches!(lhs_layout_node.domain, Some(Domain::FloatRange { .. }))
                        && matches!(rhs_layout_node.domain, Some(Domain::FloatRange { .. }))
                    {
                        repair_float_pair_constraint(
                            lhs_value,
                            lhs_layout_node,
                            *op,
                            rhs_value,
                            rhs_layout_node,
                        )
                    } else {
                        repair_scalar_pair_constraint(
                            lhs_value,
                            lhs_layout_node,
                            *op,
                            rhs_value,
                            rhs_layout_node,
                        )
                    };
                if let Some(ArgValue::Scalar(value)) = values
                    .get_mut(*lhs_arg)
                    .and_then(|value| find_value_by_path_mut(value, lhs_node, lhs_path.as_deref()))
                {
                    *value = new_lhs;
                }
                if let Some(ArgValue::Scalar(value)) = values
                    .get_mut(*rhs_arg)
                    .and_then(|value| find_value_by_path_mut(value, rhs_node, rhs_path.as_deref()))
                {
                    *value = new_rhs;
                }
            }

            if let Constraint::CountFitsBuffer {
                count_arg,
                count_path,
                buffer_arg,
                buffer_path,
                elem_size_bytes,
            } = constraint
            {
                if *elem_size_bytes == 0 {
                    continue;
                }
                let Some(buffer_arg_node) = self.args.get(*buffer_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(buffer_node) = find_node_by_path(buffer_arg_node, buffer_path.as_deref())
                else {
                    continue;
                };
                let Some(count_arg_node) = self.args.get(*count_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(count_node) = find_node_by_path(count_arg_node, count_path.as_deref())
                else {
                    continue;
                };
                let current_count = match values.get(*count_arg).and_then(|value| {
                    find_value_by_path(value, count_arg_node, count_path.as_deref())
                }) {
                    Some(ArgValue::Scalar(count)) => *count,
                    _ => continue,
                };
                let current_required_len = current_count
                    .checked_mul(*elem_size_bytes as u64)
                    .and_then(|len| usize::try_from(len).ok());
                let lower_bound_required_len =
                    positive_scalar_lower_bound(count_node).and_then(|lower| {
                        lower
                            .checked_mul(*elem_size_bytes as u64)
                            .and_then(|len| usize::try_from(len).ok())
                    });
                if current_required_len.is_some() || lower_bound_required_len.is_some() {
                    if let Some(ArgValue::Pointer(bytes)) =
                        values.get_mut(*buffer_arg).and_then(|value| {
                            find_value_by_path_mut(value, buffer_arg_node, buffer_path.as_deref())
                        })
                    {
                        let (_, max_len) = payload_len_bounds(buffer_node.domain, bytes.len());
                        let lower_target_len = lower_bound_required_len.unwrap_or(0).min(max_len);
                        if bytes.len() < lower_target_len {
                            bytes.resize(lower_target_len, PAD_BYTE);
                        }
                        if let Some(required_len) = current_required_len {
                            let target_len = required_len.min(max_len);
                            let may_grow = matches!(buffer_node.domain, Some(Domain::Bytes { .. }))
                                || bytes.is_empty();
                            if may_grow && bytes.len() < target_len {
                                bytes.resize(target_len, PAD_BYTE);
                            }
                        }
                    }
                }
                let buffer_len = match values.get(*buffer_arg).and_then(|value| {
                    find_value_by_path(value, buffer_arg_node, buffer_path.as_deref())
                }) {
                    Some(ArgValue::Pointer(bytes)) => bytes.len() as u64,
                    _ => continue,
                };
                let max_count = buffer_len / (*elem_size_bytes as u64);
                if let Some(ArgValue::Scalar(count)) =
                    values.get_mut(*count_arg).and_then(|value| {
                        find_value_by_path_mut(value, count_arg_node, count_path.as_deref())
                    })
                {
                    *count = (*count).min(max_count);
                }
            }

            if let Constraint::BufferElementsLtScalar {
                buffer_arg,
                buffer_path,
                scalar_arg,
                scalar_path,
                elem_size_bytes,
            } = constraint
            {
                let Some(buffer_arg_node) = self.args.get(*buffer_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let Some(buffer_node) = find_node_by_path(buffer_arg_node, buffer_path.as_deref())
                else {
                    continue;
                };
                let Ok(elem_size) = buffer_element_constraint_size(buffer_node, *elem_size_bytes)
                else {
                    continue;
                };
                let Some(scalar_arg_node) = self.args.get(*scalar_arg).map(ArgSpec::as_node) else {
                    continue;
                };
                let scalar_bound = match values.get(*scalar_arg).and_then(|value| {
                    find_value_by_path(value, scalar_arg_node, scalar_path.as_deref())
                }) {
                    Some(ArgValue::Scalar(value)) => *value,
                    _ => continue,
                };
                if let Some(ArgValue::Pointer(bytes)) =
                    values.get_mut(*buffer_arg).and_then(|value| {
                        find_value_by_path_mut(value, buffer_arg_node, buffer_path.as_deref())
                    })
                {
                    repair_buffer_elements_lt_scalar(bytes, elem_size, scalar_bound);
                }
            }
        }
    }
}

fn apply_domain_for_node(node: ArgNode<'_>, value: &mut ArgValue) {
    match (node.kind, value) {
        (ArgKind::Pointer, ArgValue::Pointer(bytes)) => {
            let (min_len, max_len) = payload_len_bounds(node.domain, bytes.len());
            let original_len = bytes.len();
            if bytes.len() > max_len {
                bytes.truncate(max_len);
            }
            if bytes.len() < min_len {
                bytes.resize(min_len, PAD_BYTE);
            }
            if bytes.len() != original_len {
                ARG_PACK_STATS
                    .payload_clamp_count
                    .fetch_add(1, Ordering::Relaxed);
            }
            if let Ok(Some(pattern)) = byte_pattern(node.domain) {
                apply_byte_pattern(bytes, &pattern);
            }
        }
        (ArgKind::Scalar, ArgValue::Scalar(bits)) => {
            *bits = clamp_scalar_to_domain(*bits, node);
        }
        (ArgKind::OpaqueWithPtr, ArgValue::Aggregate(values)) => {
            let Some(children) = node.children() else {
                return;
            };
            for (child, value) in children.nodes().into_iter().zip(values.iter_mut()) {
                apply_domain_for_node(child, value);
            }
        }
        _ => {}
    }
}

fn scalar_default(arg: &ArgSpec) -> u64 {
    scalar_default_for_node(arg.as_node())
}

fn scalar_default_for_node(node: ArgNode<'_>) -> u64 {
    match node.domain {
        Some(Domain::IntRange {
            min: Some(min),
            signed: Some(true),
            ..
        }) => min
            .parse::<i128>()
            .map(|value| scalar_bits_from_i128(value, node.size_bytes))
            .unwrap_or(0),
        Some(Domain::IntRange { min: Some(min), .. }) => min.parse::<u64>().unwrap_or(0),
        Some(Domain::FloatRange { .. }) => float_default_bits(node),
        Some(Domain::Enum { values, .. }) => enum_default_bits(values, node),
        _ => 0,
    }
}

fn mutatable_leaf_count_for_node(node: ArgNode<'_>) -> usize {
    match node.kind {
        ArgKind::Pointer | ArgKind::Scalar | ArgKind::OpaqueVal => 1,
        ArgKind::OpaqueWithPtr => node
            .children()
            .map(|children| {
                children
                    .nodes()
                    .into_iter()
                    .map(mutatable_leaf_count_for_node)
                    .sum::<usize>()
            })
            .unwrap_or(0),
    }
}

fn mutate_value_for_node(
    node: ArgNode<'_>,
    value: &mut ArgValue,
    target: usize,
    seen: &mut usize,
    selector: u64,
    byte: u8,
) -> bool {
    match (node.kind, value) {
        (ArgKind::Pointer, ArgValue::Pointer(bytes)) => {
            let is_target = *seen == target;
            *seen += 1;
            if is_target {
                mutate_pointer_bytes(bytes, node.domain, selector, byte);
            }
            is_target
        }
        (ArgKind::Scalar, ArgValue::Scalar(value)) => {
            let is_target = *seen == target;
            *seen += 1;
            if is_target {
                *value = mutate_scalar_value(*value, node, selector, byte);
            }
            is_target
        }
        (ArgKind::OpaqueVal, ArgValue::OpaqueVal(bytes)) => {
            let is_target = *seen == target;
            *seen += 1;
            if is_target {
                if bytes.is_empty() {
                    bytes.resize(node.size_bytes, PAD_BYTE);
                }
                if !bytes.is_empty() {
                    let idx = (selector as usize) % bytes.len();
                    bytes[idx] ^= byte.max(1);
                }
                bytes.resize(node.size_bytes, PAD_BYTE);
            }
            is_target
        }
        (ArgKind::OpaqueWithPtr, ArgValue::Aggregate(values)) => {
            let Some(children) = node.children() else {
                return false;
            };
            for (child, value) in children.nodes().into_iter().zip(values.iter_mut()) {
                if mutate_value_for_node(child, value, target, seen, selector, byte) {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

fn mutate_value_havoc_for_node(
    node: ArgNode<'_>,
    value: &mut ArgValue,
    target: usize,
    seen: &mut usize,
    selector: u64,
    byte: u8,
) -> bool {
    match (node.kind, value) {
        (ArgKind::Pointer, ArgValue::Pointer(bytes)) => {
            let is_target = *seen == target;
            *seen += 1;
            if is_target {
                mutate_pointer_bytes_havoc(bytes, node.domain, selector, byte);
            }
            is_target
        }
        (ArgKind::Scalar, ArgValue::Scalar(value)) => {
            let is_target = *seen == target;
            *seen += 1;
            if is_target {
                *value = mutate_scalar_value(*value, node, selector, byte);
            }
            is_target
        }
        (ArgKind::OpaqueVal, ArgValue::OpaqueVal(bytes)) => {
            let is_target = *seen == target;
            *seen += 1;
            if is_target {
                bytes.resize(node.size_bytes, PAD_BYTE);
                xor_opaque_value(bytes, selector, byte);
            }
            is_target
        }
        (ArgKind::OpaqueWithPtr, ArgValue::Aggregate(values)) => {
            let Some(children) = node.children() else {
                return false;
            };
            for (child, value) in children.nodes().into_iter().zip(values.iter_mut()) {
                if mutate_value_havoc_for_node(child, value, target, seen, selector, byte) {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

fn mutate_pointer_bytes(bytes: &mut Vec<u8>, domain: Option<&Domain>, selector: u64, byte: u8) {
    let (min_len, max_len) = payload_len_bounds(domain, bytes.len());
    if bytes.len() < min_len {
        bytes.resize(min_len, PAD_BYTE);
    }
    if bytes.len() > max_len {
        bytes.truncate(max_len);
    }
    match selector % 3 {
        0 if bytes.len() < max_len => bytes.push(byte),
        1 if bytes.len() > min_len => {
            let idx = (selector as usize) % bytes.len();
            bytes.remove(idx);
        }
        _ => {
            if bytes.is_empty() && max_len > 0 {
                bytes.push(byte);
            } else if !bytes.is_empty() {
                let idx = (selector as usize) % bytes.len();
                bytes[idx] ^= byte.max(1);
            }
        }
    }
    if bytes.len() < min_len {
        bytes.resize(min_len, PAD_BYTE);
    }
    if bytes.len() > max_len {
        bytes.truncate(max_len);
    }
    if let Ok(Some(pattern)) = byte_pattern(domain) {
        apply_byte_pattern(bytes, &pattern);
    }
}

fn mutate_pointer_bytes_havoc(
    bytes: &mut Vec<u8>,
    domain: Option<&Domain>,
    selector: u64,
    byte: u8,
) {
    let (min_len, max_len) = payload_len_bounds(domain, bytes.len());
    bytes.resize(bytes.len().max(min_len), PAD_BYTE);
    bytes.truncate(max_len);

    let changed = match selector % 6 {
        0 => insert_byte_chunk(bytes, domain, max_len, selector, byte),
        1 => delete_byte_chunk(bytes, domain, min_len, selector),
        2 => {
            xor_byte_span(bytes, selector, byte);
            true
        }
        3 => append_byte_element(bytes, domain, max_len, selector, byte),
        4 => remove_byte_element(bytes, domain, min_len),
        _ => {
            xor_single_byte(bytes, max_len, selector, byte);
            true
        }
    };

    if !changed {
        xor_single_byte(bytes, max_len, selector, byte);
    }
    bytes.resize(bytes.len().max(min_len), PAD_BYTE);
    bytes.truncate(max_len);
    if let Ok(Some(pattern)) = byte_pattern(domain) {
        apply_byte_pattern(bytes, &pattern);
    }
}

fn insert_byte_chunk(
    bytes: &mut Vec<u8>,
    domain: Option<&Domain>,
    max_len: usize,
    selector: u64,
    byte: u8,
) -> bool {
    let available = max_len.saturating_sub(bytes.len());
    let alignment = havoc_element_alignment(domain);
    let Some(chunk_len) = havoc_chunk_len(available, alignment, selector) else {
        return false;
    };
    let position = aligned_havoc_position(selector.rotate_left(19), bytes.len(), alignment);
    let chunk = havoc_bytes(chunk_len, selector, byte);
    bytes.splice(position..position, chunk);
    true
}

fn delete_byte_chunk(
    bytes: &mut Vec<u8>,
    domain: Option<&Domain>,
    min_len: usize,
    selector: u64,
) -> bool {
    let available = bytes.len().saturating_sub(min_len);
    let alignment = havoc_element_alignment(domain);
    let Some(chunk_len) = havoc_chunk_len(available, alignment, selector) else {
        return false;
    };
    let last_start = bytes.len() - chunk_len;
    let position = aligned_havoc_position(selector.rotate_left(29), last_start, alignment);
    bytes.drain(position..position + chunk_len);
    true
}

fn append_byte_element(
    bytes: &mut Vec<u8>,
    domain: Option<&Domain>,
    max_len: usize,
    selector: u64,
    byte: u8,
) -> bool {
    let element_size = havoc_element_alignment(domain);
    if max_len.saturating_sub(bytes.len()) < element_size {
        return false;
    }
    bytes.extend(havoc_bytes(element_size, selector, byte));
    true
}

fn remove_byte_element(bytes: &mut Vec<u8>, domain: Option<&Domain>, min_len: usize) -> bool {
    let element_size = havoc_element_alignment(domain);
    if bytes.len().saturating_sub(min_len) < element_size {
        return false;
    }
    bytes.truncate(bytes.len() - element_size);
    true
}

fn havoc_element_alignment(domain: Option<&Domain>) -> usize {
    match domain {
        Some(Domain::Bytes {
            elem_size_bytes: Some(size),
            ..
        }) if *size > 0 => *size,
        _ => 1,
    }
}

fn havoc_chunk_len(available: usize, alignment: usize, selector: u64) -> Option<usize> {
    let max_len = available.min(HAVOC_MAX_INSERT_BYTES);
    let units = max_len / alignment;
    if units == 0 {
        return None;
    }
    Some((havoc_index(selector.rotate_left(7), units) + 1) * alignment)
}

fn aligned_havoc_position(selector: u64, last_position: usize, alignment: usize) -> usize {
    let positions = last_position / alignment + 1;
    havoc_index(selector, positions) * alignment
}

fn xor_byte_span(bytes: &mut [u8], selector: u64, byte: u8) {
    xor_byte_span_with_min_len(bytes, selector, byte, 1);
}

fn xor_opaque_value(bytes: &mut [u8], selector: u64, byte: u8) {
    let min_len = if bytes.len() > 1 { 2 } else { 1 };
    xor_byte_span_with_min_len(bytes, selector, byte, min_len);
}

fn xor_byte_span_with_min_len(bytes: &mut [u8], selector: u64, byte: u8, min_len: usize) {
    if bytes.is_empty() {
        return;
    }
    let max_len = bytes.len().min(HAVOC_MAX_XOR_BYTES);
    let min_len = min_len.min(max_len);
    let span_len = min_len + havoc_index(selector.rotate_left(11), max_len - min_len + 1);
    let start = havoc_index(selector.rotate_left(37), bytes.len() - span_len + 1);
    let masks = havoc_bytes(span_len, selector.rotate_left(43), byte.max(1));
    for (target, mask) in bytes[start..start + span_len].iter_mut().zip(masks) {
        *target ^= mask.max(1);
    }
}

fn xor_single_byte(bytes: &mut Vec<u8>, max_len: usize, selector: u64, byte: u8) {
    if bytes.is_empty() && max_len > 0 {
        bytes.push(byte);
    } else if !bytes.is_empty() {
        let index = havoc_index(selector, bytes.len());
        bytes[index] ^= byte.max(1);
    }
}

fn havoc_index(selector: u64, count: usize) -> usize {
    debug_assert!(count > 0);
    (selector % count as u64) as usize
}

fn havoc_bytes(len: usize, selector: u64, byte: u8) -> Vec<u8> {
    let mut state = selector ^ (u64::from(byte) << 56) ^ 0x9e37_79b9_7f4a_7c15;
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.push(state as u8);
    }
    bytes
}

fn read_le_element(bytes: &[u8]) -> u64 {
    let mut raw = [0u8; U64_BYTES];
    let copy_len = bytes.len().min(U64_BYTES);
    raw[..copy_len].copy_from_slice(&bytes[..copy_len]);
    u64::from_le_bytes(raw)
}

fn write_le_element(bytes: &mut [u8], value: u64) {
    let raw = value.to_le_bytes();
    let copy_len = bytes.len().min(U64_BYTES);
    bytes[..copy_len].copy_from_slice(&raw[..copy_len]);
}

fn repair_buffer_elements_lt_scalar(bytes: &mut [u8], elem_size: usize, scalar_bound: u64) {
    if elem_size == 0 {
        return;
    }
    for chunk in bytes.chunks_exact_mut(elem_size) {
        let value = read_le_element(chunk);
        let repaired = if scalar_bound == 0 {
            0
        } else if value >= scalar_bound {
            value % scalar_bound
        } else {
            value
        };
        if repaired != value {
            write_le_element(chunk, repaired);
        }
    }
}

fn payload_len_bounds(domain: Option<&Domain>, current_len: usize) -> (usize, usize) {
    let mut min_len = 0usize;
    let mut max_len = DEFAULT_MAX_PAYLOAD_LEN.max(current_len);
    if let Some(Domain::Bytes {
        min_len: min,
        max_len: max,
        ..
    }) = domain
    {
        if let Some(min) = min {
            min_len = min.parse::<usize>().unwrap_or(min_len);
        }
        if let Some(max) = max {
            max_len = max.parse::<usize>().unwrap_or(max_len);
        }
    }
    if max_len < min_len {
        max_len = min_len;
    }
    (min_len, max_len)
}

fn payload_element_size(node: ArgNode<'_>) -> Option<usize> {
    if let Some(Domain::Bytes {
        elem_size_bytes: Some(size),
        ..
    }) = node.domain
    {
        if *size > 0 {
            return Some(*size);
        }
    }
    node.pointee_layout
        .map(|pointee| pointee.size_bytes)
        .filter(|size| *size > 0)
}

fn required_buffer_len_for_scalar(
    scalar_value: u64,
    buffer_node: ArgNode<'_>,
    unit: ConstraintUnit,
) -> Option<usize> {
    match unit {
        ConstraintUnit::Bytes => usize::try_from(scalar_value).ok(),
        ConstraintUnit::Elements => {
            let elem_size = payload_element_size(buffer_node)?;
            scalar_value
                .checked_mul(elem_size as u64)
                .and_then(|len| usize::try_from(len).ok())
        }
    }
}

fn positive_scalar_lower_bound(node: ArgNode<'_>) -> Option<u64> {
    let (lower, _) = int_domain_bounds(node)?;
    u64::try_from(lower).ok().filter(|value| *value > 0)
}

fn scalar_domain_permits_value(node: ArgNode<'_>, value: u32) -> bool {
    let value = u64::from(value);
    if value & scalar_mask(node.size_bytes) != value {
        return false;
    }
    match node.domain {
        None => true,
        Some(Domain::IntRange { signed, .. }) => {
            let value = i128::from(value);
            if signed.unwrap_or(false) && value > signed_max(node.size_bytes) {
                return false;
            }
            int_domain_bounds(node).is_some_and(|(lower, upper)| lower <= value && value <= upper)
        }
        Some(Domain::Enum {
            values,
            allow_unknown,
        }) => {
            allow_unknown.unwrap_or(false)
                || values
                    .iter()
                    .filter_map(|entry| enum_value_bits(entry, node))
                    .any(|allowed| allowed == value)
        }
        Some(Domain::FloatRange { .. } | Domain::Bytes { .. }) => false,
    }
}

fn mutate_scalar_value(current: u64, node: ArgNode<'_>, selector: u64, byte: u8) -> u64 {
    if let Some(Domain::Enum { values, .. }) = node.domain {
        if values.is_empty() {
            return current;
        }
        let idx = (selector as usize) % values.len();
        return enum_value_bits(&values[idx], node).unwrap_or(current);
    }
    let width_bits = node.size_bytes.saturating_mul(8).min(64);
    let mask = if width_bits == 64 {
        u64::MAX
    } else {
        (1u64 << width_bits) - 1
    };
    let mutated = current
        .wrapping_add(selector.rotate_left(13))
        .wrapping_add(byte as u64)
        & mask;
    clamp_scalar_to_domain(mutated, node)
}

fn clamp_scalar_to_domain(value: u64, node: ArgNode<'_>) -> u64 {
    match node.domain {
        Some(Domain::IntRange { min, max, signed }) => {
            if signed.unwrap_or(false) {
                let signed_value = scalar_i128_from_bits(value, node);
                let lower = min
                    .as_ref()
                    .and_then(|v| v.parse::<i128>().ok())
                    .unwrap_or(signed_min(node.size_bytes));
                let upper = max
                    .as_ref()
                    .and_then(|v| v.parse::<i128>().ok())
                    .unwrap_or(signed_max(node.size_bytes));
                scalar_bits_from_i128(signed_value.clamp(lower, upper.max(lower)), node.size_bytes)
            } else {
                let lower = min
                    .as_ref()
                    .and_then(|v| v.parse::<i128>().ok())
                    .map(|v| v.max(0) as u64)
                    .unwrap_or(0);
                let upper = max
                    .as_ref()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(u64::MAX);
                value.clamp(lower, upper.max(lower))
            }
        }
        Some(Domain::FloatRange { .. }) => clamp_float_bits(value, node),
        Some(Domain::Enum {
            values,
            allow_unknown,
        }) => clamp_enum_bits(value, values, allow_unknown.unwrap_or(false), node),
        _ => value,
    }
}

fn default_value_for_node(node: ArgNode<'_>) -> ArgValue {
    match node.kind {
        ArgKind::Pointer => ArgValue::Pointer(Vec::new()),
        ArgKind::Scalar => ArgValue::Scalar(scalar_default_for_node(node)),
        ArgKind::OpaqueVal => ArgValue::OpaqueVal(vec![0; node.size_bytes]),
        ArgKind::OpaqueWithPtr => ArgValue::Aggregate(
            node.children()
                .map(|children| {
                    children
                        .nodes()
                        .into_iter()
                        .map(default_value_for_node)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
        ),
    }
}

fn parse_node(node: ArgNode<'_>, bytes: &[u8], offset: &mut usize) -> Option<ArgValue> {
    match node.kind {
        ArgKind::Pointer => {
            let payload_align = node.payload_align_bytes();
            *offset = payload_len_offset(*offset, payload_align)?;
            if !(*offset).is_multiple_of(U64_BYTES) {
                return None;
            }
            let len = read_le_u64(bytes, offset)? as usize;
            if !(*offset).is_multiple_of(payload_align) {
                return None;
            }
            let end = offset.checked_add(len)?;
            let segment = bytes.get(*offset..end)?;
            *offset = end;
            Some(ArgValue::Pointer(segment.to_vec()))
        }
        ArgKind::Scalar => {
            *offset = align_up(*offset, node.align_bytes)?;
            if !(*offset).is_multiple_of(node.align_bytes) {
                return None;
            }
            let value = read_le_uint(bytes, offset, node.size_bytes)?;
            Some(ArgValue::Scalar(value))
        }
        ArgKind::OpaqueVal => {
            *offset = align_up(*offset, node.align_bytes)?;
            if !(*offset).is_multiple_of(node.align_bytes) {
                return None;
            }
            let end = offset.checked_add(node.size_bytes)?;
            let value = bytes.get(*offset..end)?.to_vec();
            *offset = end;
            Some(ArgValue::OpaqueVal(value))
        }
        ArgKind::OpaqueWithPtr => Some(ArgValue::Aggregate(
            node.children()?
                .nodes()
                .into_iter()
                .map(|child| parse_node(child, bytes, offset))
                .collect::<Option<Vec<_>>>()?,
        )),
    }
}

fn flatten_pack_leaves<'a>(
    out: &mut Vec<(ArgNode<'a>, Option<&'a ArgValue>)>,
    node: ArgNode<'a>,
    value: Option<&'a ArgValue>,
) {
    if node.kind != ArgKind::OpaqueWithPtr {
        out.push((node, value));
        return;
    }

    let Some(children) = node.children() else {
        return;
    };
    let Some(ArgValue::Aggregate(values)) = value else {
        return;
    };
    for (child, child_value) in children.nodes().into_iter().zip(values.iter()) {
        flatten_pack_leaves(out, child, Some(child_value));
    }
}

fn pack_node(
    out: &mut Vec<u8>,
    node: ArgNode<'_>,
    value: Option<&ArgValue>,
    next: Option<ArgNode<'_>>,
) {
    match (node.kind, value) {
        (ArgKind::Pointer, Some(ArgValue::Pointer(bytes))) => {
            append_payload_len_padding(out, node.payload_align_bytes());
            let encoded_len = pointer_encoded_len(out.len(), node, bytes.len(), next);
            append_le_u64(out, encoded_len as u64);
            out.extend_from_slice(bytes);
            let padded_end = out.len().saturating_sub(bytes.len()) + encoded_len;
            while out.len() < padded_end {
                out.push(PAD_BYTE);
            }
        }
        (ArgKind::Scalar, Some(ArgValue::Scalar(value))) => {
            append_aligned_padding(out, node.align_bytes);
            append_le_uint(out, *value, node.size_bytes);
        }
        (ArgKind::OpaqueVal, Some(ArgValue::OpaqueVal(bytes))) => {
            append_aligned_padding(out, node.align_bytes);
            out.extend_from_slice(&fixed_width_bytes(bytes, node.size_bytes));
        }
        (ArgKind::OpaqueWithPtr, Some(ArgValue::Aggregate(values))) => {
            let Some(children) = node.children() else {
                return;
            };
            let nodes = children.nodes();
            for (idx, child) in nodes.iter().enumerate() {
                let child_next = nodes.get(idx + 1).copied();
                pack_node(out, *child, values.get(idx), child_next);
            }
        }
        _ => unreachable!("value shape must match manifest arg kind"),
    }
}

fn pointer_encoded_len(
    len_pos: usize,
    node: ArgNode<'_>,
    data_len: usize,
    next: Option<ArgNode<'_>>,
) -> usize {
    let data_start = len_pos.saturating_add(U64_BYTES);
    debug_assert_eq!(len_pos % U64_BYTES, 0);
    debug_assert_eq!(data_start % node.payload_align_bytes(), 0);

    let Some(next) = next else {
        return data_len;
    };

    match next.kind {
        ArgKind::Pointer => {
            let payload_end = data_start.saturating_add(data_len);
            let next_len_pos =
                payload_len_offset(payload_end, next.payload_align_bytes()).unwrap_or(payload_end);
            next_len_pos.saturating_sub(data_start)
        }
        ArgKind::Scalar | ArgKind::OpaqueVal | ArgKind::OpaqueWithPtr => {
            let next_start_unaligned = data_start.saturating_add(data_len);
            let next_start =
                align_up(next_start_unaligned, next.align_bytes).unwrap_or(next_start_unaligned);
            next_start.saturating_sub(data_start)
        }
    }
}

fn append_payload_len_padding(out: &mut Vec<u8>, payload_align: usize) {
    let aligned = payload_len_offset(out.len(), payload_align).unwrap_or(out.len());
    while out.len() < aligned {
        out.push(PAD_BYTE);
    }
}

fn append_aligned_padding(out: &mut Vec<u8>, align_bytes: usize) {
    let aligned = align_up(out.len(), align_bytes).unwrap_or(out.len());
    while out.len() < aligned {
        out.push(PAD_BYTE);
    }
}

fn fixed_width_bytes(bytes: &[u8], width: usize) -> Vec<u8> {
    let mut out = vec![PAD_BYTE; width];
    let copy_len = bytes.len().min(width);
    out[..copy_len].copy_from_slice(&bytes[..copy_len]);
    out
}

fn apply_const_constraint(value: &mut u64, node: ArgNode<'_>, op: ConstraintOp, constant: i64) {
    if matches!(node.domain, Some(Domain::FloatRange { .. })) {
        *value = repair_float_const_constraint(*value, node, op, constant as f64);
    } else {
        *value = repair_scalar_const_constraint(*value, node, op, constant as i128);
    }
}

fn repair_scalar_const_constraint(
    value_bits: u64,
    node: ArgNode<'_>,
    op: ConstraintOp,
    rhs: i128,
) -> u64 {
    let (lower, upper) = scalar_repair_bounds(node);
    let mut value = scalar_i128_from_bits(value_bits, node).clamp(lower, upper);
    match op {
        ConstraintOp::Le => value = value.min(rhs).clamp(lower, upper),
        ConstraintOp::Lt => value = value.min(rhs.saturating_sub(1)).clamp(lower, upper),
        ConstraintOp::Eq => value = rhs.clamp(lower, upper),
        ConstraintOp::Ne if value == rhs => {
            if value < upper {
                value += 1;
            } else if value > lower {
                value -= 1;
            }
        }
        ConstraintOp::Ne => {}
        ConstraintOp::Ge => value = value.max(rhs).clamp(lower, upper),
        ConstraintOp::Gt => value = value.max(rhs.saturating_add(1)).clamp(lower, upper),
    }
    scalar_bits_from_i128(value, node.size_bytes)
}

fn repair_float_const_constraint(
    value_bits: u64,
    node: ArgNode<'_>,
    op: ConstraintOp,
    rhs: f64,
) -> u64 {
    let (lower, upper) = float_repair_bounds(node);
    let mut value = float_from_bits(value_bits, node.size_bytes);
    if value.is_nan() {
        value = default_float_value(lower, upper);
    }
    value = value.clamp(lower, upper);
    if float_constraint_satisfied(value, op, rhs) {
        return float_bits(value, node.size_bytes);
    }

    let candidate = match op {
        ConstraintOp::Le => value.min(rhs),
        ConstraintOp::Lt => value.min(next_float_down(rhs, node.size_bytes)),
        ConstraintOp::Eq => rhs,
        ConstraintOp::Ne if value == rhs => {
            if lower != rhs {
                lower
            } else {
                upper
            }
        }
        ConstraintOp::Ne => value,
        ConstraintOp::Ge => value.max(rhs),
        ConstraintOp::Gt => value.max(next_float_up(rhs, node.size_bytes)),
    }
    .clamp(lower, upper);
    float_bits(candidate, node.size_bytes)
}

fn float_constraint_satisfied(value: f64, op: ConstraintOp, rhs: f64) -> bool {
    match op {
        ConstraintOp::Le => value <= rhs,
        ConstraintOp::Lt => value < rhs,
        ConstraintOp::Eq => value == rhs,
        ConstraintOp::Ne => value != rhs,
        ConstraintOp::Ge => value >= rhs,
        ConstraintOp::Gt => value > rhs,
    }
}

fn repair_scalar_pair_constraint(
    lhs_bits: u64,
    lhs_node: ArgNode<'_>,
    op: ConstraintOp,
    rhs_bits: u64,
    rhs_node: ArgNode<'_>,
) -> (u64, u64) {
    let (lhs_lower, lhs_upper) = scalar_repair_bounds(lhs_node);
    let (rhs_lower, rhs_upper) = scalar_repair_bounds(rhs_node);
    let mut lhs = scalar_i128_from_bits(lhs_bits, lhs_node).clamp(lhs_lower, lhs_upper);
    let mut rhs = scalar_i128_from_bits(rhs_bits, rhs_node).clamp(rhs_lower, rhs_upper);

    match op {
        ConstraintOp::Le if lhs > rhs => {
            if lhs <= rhs_upper {
                rhs = lhs.max(rhs_lower);
            } else {
                rhs = rhs_upper;
                lhs = rhs.clamp(lhs_lower, lhs_upper);
            }
        }
        ConstraintOp::Lt if lhs >= rhs => {
            if lhs < rhs_upper {
                rhs = lhs.saturating_add(1).clamp(rhs_lower, rhs_upper);
            } else {
                rhs = rhs_upper;
                lhs = rhs.saturating_sub(1).clamp(lhs_lower, lhs_upper);
            }
        }
        ConstraintOp::Eq if lhs != rhs => {
            let target = lhs_lower.max(rhs_lower).min(lhs_upper.min(rhs_upper));
            lhs = target;
            rhs = target;
        }
        ConstraintOp::Ne if lhs == rhs => {
            if lhs < lhs_upper {
                lhs += 1;
            } else if rhs < rhs_upper {
                rhs += 1;
            } else if lhs > lhs_lower {
                lhs -= 1;
            } else if rhs > rhs_lower {
                rhs -= 1;
            }
        }
        ConstraintOp::Ge if lhs < rhs => {
            if rhs <= lhs_upper {
                lhs = rhs.max(lhs_lower);
            } else {
                lhs = lhs_upper;
                rhs = lhs.clamp(rhs_lower, rhs_upper);
            }
        }
        ConstraintOp::Gt if lhs <= rhs => {
            if rhs < lhs_upper {
                lhs = rhs.saturating_add(1).clamp(lhs_lower, lhs_upper);
            } else {
                lhs = lhs_upper;
                rhs = lhs.saturating_sub(1).clamp(rhs_lower, rhs_upper);
            }
        }
        _ => {}
    }

    (
        scalar_bits_from_i128(lhs, lhs_node.size_bytes),
        scalar_bits_from_i128(rhs, rhs_node.size_bytes),
    )
}

fn repair_float_pair_constraint(
    lhs_bits: u64,
    lhs_node: ArgNode<'_>,
    op: ConstraintOp,
    rhs_bits: u64,
    rhs_node: ArgNode<'_>,
) -> (u64, u64) {
    let (lhs_lower, lhs_upper) = float_repair_bounds(lhs_node);
    let (rhs_lower, rhs_upper) = float_repair_bounds(rhs_node);
    let mut lhs = float_from_bits(lhs_bits, lhs_node.size_bytes);
    let mut rhs = float_from_bits(rhs_bits, rhs_node.size_bytes);
    if lhs.is_nan() {
        lhs = default_float_value(lhs_lower, lhs_upper);
    }
    if rhs.is_nan() {
        rhs = default_float_value(rhs_lower, rhs_upper);
    }
    lhs = lhs.clamp(lhs_lower, lhs_upper);
    rhs = rhs.clamp(rhs_lower, rhs_upper);

    match op {
        ConstraintOp::Le if lhs > rhs => {
            if lhs <= rhs_upper {
                rhs = lhs.max(rhs_lower);
            } else {
                rhs = rhs_upper;
                lhs = rhs.clamp(lhs_lower, lhs_upper);
            }
        }
        ConstraintOp::Lt if lhs >= rhs => {
            if lhs < rhs_upper {
                rhs = next_float_up(lhs, rhs_node.size_bytes).clamp(rhs_lower, rhs_upper);
            } else {
                rhs = rhs_upper;
                lhs = next_float_down(rhs, lhs_node.size_bytes).clamp(lhs_lower, lhs_upper);
            }
        }
        ConstraintOp::Eq if lhs != rhs => {
            let target = lhs_lower.max(rhs_lower).min(lhs_upper.min(rhs_upper));
            lhs = target;
            rhs = target;
        }
        ConstraintOp::Ne if lhs == rhs => {
            if lhs < lhs_upper {
                lhs = next_float_up(lhs, lhs_node.size_bytes).clamp(lhs_lower, lhs_upper);
            } else if rhs < rhs_upper {
                rhs = next_float_up(rhs, rhs_node.size_bytes).clamp(rhs_lower, rhs_upper);
            } else if lhs > lhs_lower {
                lhs = next_float_down(lhs, lhs_node.size_bytes).clamp(lhs_lower, lhs_upper);
            } else if rhs > rhs_lower {
                rhs = next_float_down(rhs, rhs_node.size_bytes).clamp(rhs_lower, rhs_upper);
            }
        }
        ConstraintOp::Ge if lhs < rhs => {
            if rhs <= lhs_upper {
                lhs = rhs.max(lhs_lower);
            } else {
                lhs = lhs_upper;
                rhs = lhs.clamp(rhs_lower, rhs_upper);
            }
        }
        ConstraintOp::Gt if lhs <= rhs => {
            if rhs < lhs_upper {
                lhs = next_float_up(rhs, lhs_node.size_bytes).clamp(lhs_lower, lhs_upper);
            } else {
                lhs = lhs_upper;
                rhs = next_float_down(lhs, rhs_node.size_bytes).clamp(rhs_lower, rhs_upper);
            }
        }
        _ => {}
    }

    (
        float_bits(lhs, lhs_node.size_bytes),
        float_bits(rhs, rhs_node.size_bytes),
    )
}

fn scalar_repair_bounds(node: ArgNode<'_>) -> (i128, i128) {
    if let Some(bounds) = int_domain_bounds(node) {
        return bounds;
    }
    if scalar_is_signed(node) {
        (signed_min(node.size_bytes), signed_max(node.size_bytes))
    } else {
        (0, scalar_mask(node.size_bytes) as i128)
    }
}

fn scalar_is_signed(node: ArgNode<'_>) -> bool {
    matches!(
        node.domain,
        Some(Domain::IntRange {
            signed: Some(true),
            ..
        })
    )
}

fn enum_default_bits(values: &[EnumValue], node: ArgNode<'_>) -> u64 {
    values
        .first()
        .and_then(|value| enum_value_bits(value, node))
        .unwrap_or(0)
}

fn clamp_enum_bits(
    value: u64,
    values: &[EnumValue],
    allow_unknown: bool,
    node: ArgNode<'_>,
) -> u64 {
    let masked = value & scalar_mask(node.size_bytes);
    if allow_unknown {
        return masked;
    }
    if values
        .iter()
        .filter_map(|value| enum_value_bits(value, node))
        .any(|value| value == masked)
    {
        return masked;
    }
    enum_default_bits(values, node)
}

fn enum_value_bits(value: &EnumValue, node: ArgNode<'_>) -> Option<u64> {
    parse_enum_discriminant(&value.value).map(|value| scalar_bits_from_i128(value, node.size_bytes))
}

fn parse_enum_discriminant(value: &str) -> Option<i128> {
    value.parse::<i128>().ok()
}

fn float_default_bits(node: ArgNode<'_>) -> u64 {
    let Some(Domain::FloatRange { min, .. }) = node.domain else {
        return 0;
    };
    let value = min.unwrap_or(0.0);
    clamp_float_bits(float_bits(value, node.size_bytes), node)
}

fn clamp_float_bits(bits: u64, node: ArgNode<'_>) -> u64 {
    let Some(Domain::FloatRange { allow_nan, .. }) = node.domain else {
        return bits;
    };

    let allow_nan = allow_nan.unwrap_or(false);
    let (lower, upper) = float_repair_bounds(node);

    let value = float_from_bits(bits, node.size_bytes);
    if value.is_nan() && allow_nan {
        return bits & scalar_mask(node.size_bytes);
    }

    let repaired = if value.is_nan() {
        default_float_value(lower, upper)
    } else {
        value.clamp(lower, upper)
    };
    float_bits(repaired, node.size_bytes)
}

fn float_repair_bounds(node: ArgNode<'_>) -> (f64, f64) {
    float_domain_bounds(node)
}

fn float_domain_bounds(node: ArgNode<'_>) -> (f64, f64) {
    let Some(Domain::FloatRange { min, max, .. }) = node.domain else {
        return (f64::NEG_INFINITY, f64::INFINITY);
    };
    let mut lower = min.unwrap_or(f64::NEG_INFINITY);
    let mut upper = max.unwrap_or(f64::INFINITY);
    if lower.is_nan() {
        lower = f64::NEG_INFINITY;
    }
    if upper.is_nan() {
        upper = f64::INFINITY;
    }
    if upper < lower {
        upper = lower;
    }
    (lower, upper)
}

fn default_float_value(lower: f64, upper: f64) -> f64 {
    if (lower..=upper).contains(&0.0) {
        0.0
    } else if lower.is_finite() {
        lower
    } else if upper.is_finite() {
        upper
    } else {
        0.0
    }
}

fn next_float_down(value: f64, size_bytes: usize) -> f64 {
    match size_bytes {
        4 => next_f32_down(value as f32) as f64,
        8 => next_f64_down(value),
        _ => value,
    }
}

fn next_float_up(value: f64, size_bytes: usize) -> f64 {
    match size_bytes {
        4 => next_f32_up(value as f32) as f64,
        8 => next_f64_up(value),
        _ => value,
    }
}

fn next_f32_down(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::MIN_POSITIVE;
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

fn next_f32_up(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::MIN_POSITIVE;
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

fn next_f64_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::MIN_POSITIVE;
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn next_f64_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::MIN_POSITIVE;
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn float_from_bits(bits: u64, size_bytes: usize) -> f64 {
    match size_bytes {
        4 => f32::from_bits(bits as u32) as f64,
        8 => f64::from_bits(bits),
        _ => 0.0,
    }
}

fn float_bits(value: f64, size_bytes: usize) -> u64 {
    match size_bytes {
        4 => (value as f32).to_bits() as u64,
        8 => value.to_bits(),
        _ => 0,
    }
}

fn scalar_mask(size_bytes: usize) -> u64 {
    let width_bits = size_bytes.saturating_mul(8).min(64);
    if width_bits == 64 {
        u64::MAX
    } else {
        (1u64 << width_bits) - 1
    }
}

fn scalar_i128_from_bits(bits: u64, node: ArgNode<'_>) -> i128 {
    let width_bits = node.size_bytes.saturating_mul(8).min(64);
    let masked = bits & scalar_mask(node.size_bytes);
    if !scalar_is_signed(node) || width_bits == 0 {
        return masked as i128;
    }
    let sign_bit = 1u64 << (width_bits - 1);
    if masked & sign_bit == 0 {
        masked as i128
    } else {
        (masked as i128) - (1i128 << width_bits)
    }
}

fn scalar_bits_from_i128(value: i128, size_bytes: usize) -> u64 {
    (value as u64) & scalar_mask(size_bytes)
}

fn signed_min(size_bytes: usize) -> i128 {
    let width_bits = size_bytes.saturating_mul(8).clamp(1, 64);
    -(1i128 << (width_bits - 1))
}

fn signed_max(size_bytes: usize) -> i128 {
    let width_bits = size_bytes.saturating_mul(8).clamp(1, 64);
    (1i128 << (width_bits - 1)) - 1
}

fn spec() -> &'static ArgPackSpec {
    ARG_PACK_SPEC
        .get()
        .expect("manifest-driven arg-pack spec was not initialized")
        .as_ref()
        .expect("failed to initialize manifest-driven arg-pack spec")
}

pub fn init_arg_pack_manifest(path: &Path) -> Result<(), String> {
    let spec = ArgPackSpec::from_manifest_path(path);
    let result = spec.as_ref().map(|_| ()).map_err(Clone::clone);
    ARG_PACK_SPEC
        .set(spec)
        .map_err(|_| "manifest-driven arg-pack spec already initialized".to_string())?;
    result
}

fn read_le_u64(bytes: &[u8], offset: &mut usize) -> Option<u64> {
    read_le_uint(bytes, offset, U64_BYTES)
}

fn read_le_uint(bytes: &[u8], offset: &mut usize, width: usize) -> Option<u64> {
    let end = offset.checked_add(width)?;
    let chunk = bytes.get(*offset..end)?;
    let mut value = 0u64;
    for (i, b) in chunk.iter().enumerate() {
        value |= (*b as u64) << (i * 8);
    }
    *offset = end;
    Some(value)
}

fn append_le_u64(out: &mut Vec<u8>, value: u64) {
    append_le_uint(out, value, U64_BYTES);
}

fn append_le_uint(out: &mut Vec<u8>, value: u64, width: usize) {
    for i in 0..width {
        out.push(((value >> (i * 8)) & 0xff) as u8);
    }
}

pub fn pack_arg_pack_v1(input: &[u8], output: &[u8], kernel_size: usize) -> Vec<u8> {
    spec().pack_current_kernel_args(input, output, kernel_size)
}

pub fn normalize_arg_pack_v1(raw: &[u8]) -> Vec<u8> {
    spec().normalize(raw)
}

pub fn mutate_arg_pack_v1(raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
    spec().mutate(raw, selector, byte)
}

pub fn default_seed_arg_pack_v1() -> Vec<u8> {
    spec().default_seed()
}

pub fn normalize_rapid_input_v1(raw: &[u8]) -> Vec<u8> {
    spec().normalize_rapid_input(raw)
}

pub fn mutate_rapid_input_v1(raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
    spec().mutate_rapid_input(raw, selector, byte)
}

pub(crate) fn mutate_rapid_input_unrepaired_v1(raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
    spec().mutate_rapid_input_unrepaired(raw, selector, byte)
}

pub fn mutate_rapid_input_havoc_v1(raw: &[u8], operations: &[(u64, u8)]) -> Vec<u8> {
    spec().mutate_rapid_input_havoc(raw, operations)
}

pub(crate) fn mutate_rapid_input_havoc_unrepaired_v1(
    raw: &[u8],
    operations: &[(u64, u8)],
) -> Vec<u8> {
    spec().mutate_rapid_input_havoc_unrepaired(raw, operations)
}

pub fn mutate_rapid_vconfig_v1(raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
    spec().mutate_rapid_vconfig(raw, selector, byte)
}

pub(crate) fn mutate_rapid_vconfig_unrepaired_v1(raw: &[u8], selector: u64, byte: u8) -> Vec<u8> {
    spec().mutate_rapid_vconfig_unrepaired(raw, selector, byte)
}

pub fn default_seed_rapid_input_v1() -> Vec<u8> {
    spec().default_rapid_input()
}

pub fn arg_pack_manifest_summary_v1() -> String {
    let spec = spec();
    format!(
        "kernel={}, display_name={}, args={}, constraints={}",
        spec.symbol_name,
        spec.display_name,
        spec.args.len(),
        spec.constraints.len()
    )
}

pub fn arg_pack_stats_v1() -> ArgPackStats {
    ArgPackStats {
        normalize_calls: ARG_PACK_STATS.normalize_calls.load(Ordering::Relaxed),
        normalize_repack_count: ARG_PACK_STATS
            .normalize_repack_count
            .load(Ordering::Relaxed),
        invalid_repair_count: ARG_PACK_STATS.invalid_repair_count.load(Ordering::Relaxed),
        mutation_calls: ARG_PACK_STATS.mutation_calls.load(Ordering::Relaxed),
        seed_generation_count: ARG_PACK_STATS.seed_generation_count.load(Ordering::Relaxed),
        payload_clamp_count: ARG_PACK_STATS.payload_clamp_count.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
pub(crate) fn test_normalize_calls_v1() -> u64 {
    TEST_NORMALIZE_CALLS.get()
}

#[cfg(test)]
pub(crate) struct ArgPackHavocTestSpec(ArgPackSpec);

#[cfg(test)]
impl ArgPackHavocTestSpec {
    pub(crate) fn from_manifest_json(manifest_json: &str) -> Self {
        let manifest: KernelManifest = serde_json::from_str(manifest_json).unwrap();
        Self(ArgPackSpec::from_manifest(manifest).unwrap())
    }

    pub(crate) fn default_rapid_input(&self) -> Vec<u8> {
        self.0.default_rapid_input()
    }

    pub(crate) fn mutate_rapid_input_havoc(&self, raw: &[u8], operations: &[(u64, u8)]) -> Vec<u8> {
        self.0.mutate_rapid_input_havoc(raw, operations)
    }

    fn parse_values(&self, raw: &[u8]) -> Vec<ArgValue> {
        let envelope = parse_rapid_input_envelope(raw).unwrap();
        self.0.parse(envelope.payload).unwrap().values
    }

    pub(crate) fn is_canonical_rapid_input(&self, raw: &[u8]) -> bool {
        let Some(envelope) = parse_rapid_input_envelope(raw) else {
            return false;
        };
        self.0.parse(envelope.payload).is_some() && self.0.normalize_rapid_input(raw) == raw
    }

    pub(crate) fn pointer_len(&self, raw: &[u8], index: usize) -> usize {
        match &self.parse_values(raw)[index] {
            ArgValue::Pointer(bytes) => bytes.len(),
            other => panic!("expected pointer at index {index}, got {other:?}"),
        }
    }

    pub(crate) fn scalar(&self, raw: &[u8], index: usize) -> u64 {
        match self.parse_values(raw)[index] {
            ArgValue::Scalar(value) => value,
            ref other => panic!("expected scalar at index {index}, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MANIFEST: &str = r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "vulnerable_kernel",
      "display_name": "vulnerable_kernel",
      "args": [
        {
          "index": 0,
          "name": "input",
          "type": "volatile uint8_t *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "pointee_layout": {
            "index": "input.*",
            "name": "$pointee",
            "type": "uint8_t",
            "kind": "scalar",
            "size_bytes": 1,
            "align_bytes": 1
          },
          "size_bytes": 8,
          "align_bytes": 8
        },
        {
          "index": 1,
          "name": "output",
          "type": "volatile uint8_t *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "pointee_layout": {
            "index": "output.*",
            "name": "$pointee",
            "type": "uint8_t",
            "kind": "scalar",
            "size_bytes": 1,
            "align_bytes": 1
          },
          "size_bytes": 8,
          "align_bytes": 8
        },
        {
          "index": 2,
          "name": "size",
          "type": "size_t",
          "kind": "scalar",
          "size_bytes": 8,
          "align_bytes": 8,
          "domain": {
            "kind": "int_range",
            "min": "0",
            "signed": false
          }
        }
      ],
      "constraints": [
        {
          "kind": "scalar_le_buffer_len",
          "scalar_arg": 2,
          "buffer_arg": 0,
          "unit": "bytes"
        },
        {
          "kind": "scalar_le_buffer_len",
          "scalar_arg": 2,
          "buffer_arg": 1,
          "unit": "bytes"
        }
      ]
    }
  ]
}
"#;

    fn test_spec() -> ArgPackSpec {
        let manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        ArgPackSpec::from_manifest(manifest).unwrap()
    }

    fn scalar(values: &[ArgValue], idx: usize) -> u64 {
        match &values[idx] {
            ArgValue::Scalar(value) => *value,
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    fn pointer(values: &[ArgValue], idx: usize) -> &[u8] {
        match &values[idx] {
            ArgValue::Pointer(value) => value,
            other => panic!("expected pointer, got {other:?}"),
        }
    }

    fn float_payload_arg(index: usize, name: &str, max_len: usize) -> ArgSpec {
        serde_json::from_value(serde_json::json!({
            "index": index,
            "name": name,
            "type": "float *",
            "kind": "pointer",
            "pointer_role": "payload_buffer",
            "pointee_layout": {
                "index": format!("{name}.*"),
                "name": "$pointee",
                "type": "float",
                "kind": "scalar",
                "size_bytes": 4,
                "align_bytes": 4
            },
            "size_bytes": 8,
            "align_bytes": 8,
            "domain": {
                "kind": "bytes",
                "min_len": "0",
                "max_len": max_len.to_string(),
                "elem_size_bytes": 4,
                "nullable": false
            }
        }))
        .unwrap()
    }

    fn opaque_val(values: &[ArgValue], idx: usize) -> &[u8] {
        match &values[idx] {
            ArgValue::OpaqueVal(value) => value,
            other => panic!("expected opaque_val, got {other:?}"),
        }
    }

    fn aggregate(values: &[ArgValue], idx: usize) -> &[ArgValue] {
        match &values[idx] {
            ArgValue::Aggregate(value) => value,
            other => panic!("expected aggregate, got {other:?}"),
        }
    }

    fn assert_whole_warp_vconfig(vconfig: RapidVConfig) {
        let threads =
            u64::from(vconfig.block_x) * u64::from(vconfig.block_y) * u64::from(vconfig.block_z);
        assert!(
            threads >= u64::from(VCONFIG_WARP_SIZE),
            "expected at least one whole warp, got {threads} threads from {vconfig:?}"
        );
        assert_eq!(
            threads % u64::from(VCONFIG_WARP_SIZE),
            0,
            "expected whole-warp logical block, got {threads} threads from {vconfig:?}"
        );
    }

    #[test]
    fn manifest_loads_current_subset() {
        let spec = test_spec();
        assert_eq!(spec.args.len(), 3);
        assert_eq!(spec.args[0].name, "input");
        assert_eq!(spec.args[0].kind, ArgKind::Pointer);
        assert_eq!(spec.args[0].pointer_role, Some(PointerRole::PayloadBuffer));
        assert_eq!(spec.args[0].align_bytes, 8);
        assert_eq!(spec.args[1].name, "output");
        assert_eq!(spec.args[1].kind, ArgKind::Pointer);
        assert_eq!(spec.args[2].name, "size");
        assert_eq!(spec.args[2].kind, ArgKind::Scalar);
        assert_eq!(spec.args[2].size_bytes, 8);
        assert_eq!(spec.constraints.len(), 2);
    }

    #[test]
    fn pack_layout_matches_current_envelope() {
        let spec = test_spec();
        let packed = spec.pack_current_kernel_args(b"ABC", b"XY", 3);

        assert_eq!(&packed[0..8], &(8u64.to_le_bytes()));
        assert_eq!(&packed[8..11], b"ABC");
        assert_eq!(&packed[16..24], &(8u64.to_le_bytes()));
        assert_eq!(&packed[24..26], b"XY");
        assert_eq!(&packed[32..40], &(2u64.to_le_bytes()));

        let parsed = spec.parse(&packed).unwrap();
        assert_eq!(pointer(&parsed.values, 0), b"ABC\0\0\0\0\0");
        assert_eq!(pointer(&parsed.values, 1), b"XY\0\0\0\0\0\0");
        assert_eq!(scalar(&parsed.values, 2), 2);
    }

    #[test]
    fn pointer_payload_start_uses_pointee_alignment_when_wider_than_pointer() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "wide".to_string(),
            type_name: "uint4 *".to_string(),
            kind: ArgKind::Pointer,
            pointer_role: Some(PointerRole::PayloadBuffer),
            pointee_layout: Some(LayoutNode {
                index: "wide.*".to_string(),
                name: "$pointee".to_string(),
                type_name: "uint4".to_string(),
                kind: ArgKind::OpaqueVal,
                pointer_role: None,
                pointee_layout: None,
                size_bytes: 16,
                align_bytes: 16,
                domain: None,
                fields: vec![],
                element: None,
                element_count: None,
            }),
            type_layout: None,
            size_bytes: 8,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints.clear();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let packed = spec.pack_values(vec![ArgValue::Pointer(b"ABCD".to_vec())]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(packed.len(), 20);
        assert_eq!(&packed[8..16], &4u64.to_le_bytes());
        assert_eq!(&packed[16..20], b"ABCD");
        assert_eq!(pointer(&parsed.values, 0), b"ABCD");
    }

    #[test]
    fn nested_pointer_payload_start_uses_wider_pointee_alignment() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![LayoutNode {
                    index: "params.data".to_string(),
                    name: "data".to_string(),
                    type_name: "uint4 *".to_string(),
                    kind: ArgKind::Pointer,
                    pointer_role: Some(PointerRole::PayloadBuffer),
                    pointee_layout: Some(Box::new(LayoutNode {
                        index: "params.data.*".to_string(),
                        name: "$pointee".to_string(),
                        type_name: "uint4".to_string(),
                        kind: ArgKind::OpaqueVal,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 16,
                        align_bytes: 16,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    })),
                    size_bytes: 8,
                    align_bytes: 8,
                    domain: None,
                    fields: vec![],
                    element: None,
                    element_count: None,
                }],
                element: None,
                element_count: None,
            }),
            size_bytes: 8,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints.clear();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![ArgValue::Pointer(
            b"ABCD".to_vec(),
        )])]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(packed.len(), 20);
        assert_eq!(&packed[8..16], &4u64.to_le_bytes());
        assert_eq!(&packed[16..20], b"ABCD");
        assert_eq!(pointer(aggregate(&parsed.values, 0), 0), b"ABCD");
    }

    #[test]
    fn normalize_canonical_repack_clamps_constraints() {
        let spec = test_spec();
        let mut raw = spec.pack_current_kernel_args(b"ABCD", b"XY", 2);
        raw[32..40].copy_from_slice(&99u64.to_le_bytes());
        let normalized = spec.normalize(&raw);

        assert_ne!(normalized, raw);
        let parsed = spec.parse(&normalized).unwrap();
        assert_eq!(scalar(&parsed.values, 2), 8);
        assert_eq!(normalized, spec.pack_values(parsed.values));
    }

    #[test]
    fn invalid_normalize_uses_manifest_default_seed() {
        let spec = test_spec();
        let normalized = spec.normalize(b"abcdefg");
        assert_eq!(normalized, spec.default_seed());
    }

    #[test]
    fn default_seed_is_manifest_driven_and_valid() {
        let spec = test_spec();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();
        assert_eq!(pointer(&parsed.values, 0), b"FUZZ\0\0\0\0");
        assert_eq!(pointer(&parsed.values, 1), &[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(scalar(&parsed.values, 2), 0);
    }

    #[test]
    fn rapid_input_default_seed_is_full_envelope() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([1, 1, 1]),
            block_candidates: Some(vec![32, 128, 64]),
            physical_block_max: Some(96),
            logical_grid: None,
            logical_block: None,
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: None,
        });
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let seed = spec.default_rapid_input();

        assert!(seed.len() > RAPID_TASK_ENVELOPE_HEADER_BYTES);
        let envelope = parse_rapid_input_envelope(&seed).unwrap();
        assert_eq!(
            envelope.vconfig,
            RapidVConfig {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                block_x: 64,
                block_y: 1,
                block_z: 1,
            }
        );
        assert_eq!(
            envelope.payload_size as usize,
            seed.len() - RAPID_TASK_ENVELOPE_HEADER_BYTES
        );
        assert_eq!(spec.normalize(envelope.payload), envelope.payload);
    }

    #[test]
    fn rapid_input_uses_explicit_logical_vconfig_shape() {
        let mut manifest_value: serde_json::Value = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest_value["kernels"][0]["launch_policy"] = serde_json::json!({
            "grid": [1, 1, 1],
            "block_candidates": [512, 256, 128],
            "physical_block_max": 512,
            "logical_grid": [1, 1, 1],
            "logical_block": [33, 1, 1],
            "target_dynamic_shared_bytes": 0,
            "coverage_memory": "global",
            "vconfig_reserved": true,
            "vconfig_mutation": false
        });
        let manifest: KernelManifest = serde_json::from_value(manifest_value).unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let seed = spec.default_rapid_input();
        let envelope = parse_rapid_input_envelope(&seed).unwrap();

        assert_eq!(
            envelope.vconfig,
            RapidVConfig {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                block_x: 33,
                block_y: 1,
                block_z: 1,
            }
        );
        for selector in 0..16 {
            let mutated = spec.mutate_rapid_vconfig(&seed, selector, 0xff);
            assert_eq!(
                parse_rapid_input_envelope(&mutated).unwrap().vconfig,
                envelope.vconfig
            );
        }
    }

    fn manifest_with_logical_block_candidates(
        candidates: serde_json::Value,
        logical_block: [u32; 3],
    ) -> KernelManifest {
        let mut manifest_value: serde_json::Value = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest_value["kernels"][0]["launch_policy"] = serde_json::json!({
            "grid": [1, 1, 1],
            "block_candidates": [256],
            "physical_block_max": 256,
            "logical_grid": [1, 1, 1],
            "logical_block": logical_block,
            "logical_block_candidates": candidates,
            "target_dynamic_shared_bytes": 0,
            "coverage_memory": "global",
            "vconfig_reserved": true,
            "vconfig_mutation": true
        });
        serde_json::from_value(manifest_value).unwrap()
    }

    #[test]
    fn rapid_input_logical_block_candidates_normalize_to_nearest_complete_shape() {
        let manifest = manifest_with_logical_block_candidates(
            serde_json::json!([[32, 1, 1], [64, 1, 1], [128, 1, 1], [256, 1, 1]]),
            [256, 1, 1],
        );
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        for (raw_block_x, expected_block_x) in [
            (32, 32),
            (48, 32),
            (64, 64),
            (80, 64),
            (128, 128),
            (256, 256),
        ] {
            let raw = encode_rapid_input(
                RapidVConfig {
                    grid_x: 1,
                    grid_y: 1,
                    grid_z: 1,
                    block_x: raw_block_x,
                    block_y: 1,
                    block_z: 1,
                },
                &spec.default_seed(),
            );
            let normalized = spec.normalize_rapid_input(&raw);
            let block = parse_rapid_input_envelope(&normalized).unwrap().vconfig;
            assert_eq!(
                [block.block_x, block.block_y, block.block_z],
                [expected_block_x, 1, 1]
            );
        }
    }

    #[test]
    fn rapid_input_radix_candidates_raise_saved_subdomain_blocks_to_minimum() {
        let manifest = manifest_with_logical_block_candidates(
            serde_json::json!([[64, 1, 1], [96, 1, 1], [128, 1, 1]]),
            [128, 1, 1],
        );
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let raw = encode_rapid_input(
            RapidVConfig {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                block_x: 32,
                block_y: 1,
                block_z: 1,
            },
            &spec.default_seed(),
        );

        let normalized = spec.normalize_rapid_input(&raw);
        let block = parse_rapid_input_envelope(&normalized).unwrap().vconfig;

        assert_eq!([block.block_x, block.block_y, block.block_z], [64, 1, 1]);
    }

    #[test]
    fn rapid_input_logical_block_candidates_mutate_only_whole_declared_shapes() {
        let candidates = [[32, 1, 1], [64, 1, 1], [128, 1, 1], [256, 1, 1]];
        let manifest =
            manifest_with_logical_block_candidates(serde_json::json!(candidates), [256, 1, 1]);
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_rapid_input();

        for selector in 0..128 {
            for byte in [0, 1, 2, 3, 0xff] {
                let mutated = spec.mutate_rapid_vconfig(&seed, selector, byte);
                let block = parse_rapid_input_envelope(&mutated).unwrap().vconfig;
                assert!(candidates.contains(&[block.block_x, block.block_y, block.block_z]));
            }
        }
    }

    #[test]
    fn rapid_input_logical_block_candidates_reject_invalid_domains() {
        let cases = [
            (serde_json::json!([]), [256, 1, 1], "must be non-empty"),
            (
                serde_json::json!([[32, 1, 1], [32, 1, 1]]),
                [32, 1, 1],
                "must be unique",
            ),
            (
                serde_json::json!([[0, 1, 1], [256, 1, 1]]),
                [256, 1, 1],
                "dimensions must be positive",
            ),
            (
                serde_json::json!([[32, 1, 1], [512, 1, 1]]),
                [32, 1, 1],
                "exceed physical block envelope",
            ),
            (
                serde_json::json!([[32, 1, 1], [128, 1, 1]]),
                [256, 1, 1],
                "must contain logical_block",
            ),
            (
                serde_json::json!([[32, 1, 1], [48, 1, 1], [256, 1, 1]]),
                [256, 1, 1],
                "whole-warp",
            ),
        ];

        for (candidates, logical_block, expected) in cases {
            let manifest = manifest_with_logical_block_candidates(candidates, logical_block);
            let error = ArgPackSpec::from_manifest(manifest).unwrap_err();
            assert!(
                error.contains(expected),
                "expected {expected:?} in {error:?}"
            );
        }
    }

    fn manifest_value_with_scalar_le_logical_block_dim(
        scalar_arg: usize,
        dimension: &str,
    ) -> serde_json::Value {
        let mut manifest_value: serde_json::Value = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest_value["kernels"][0]["launch_policy"] = serde_json::json!({
            "grid": [1, 1, 1],
            "block_candidates": [256],
            "physical_block_max": 256,
            "logical_grid": [1, 1, 1],
            "logical_block": [256, 1, 1],
            "logical_block_candidates": [[32,1,1],[64,1,1],[128,1,1],[256,1,1]],
            "target_dynamic_shared_bytes": 0,
            "coverage_memory": "global",
            "vconfig_reserved": true,
            "vconfig_mutation": true
        });
        manifest_value["kernels"][0]["args"][2]["domain"] = serde_json::json!({
            "kind": "int_range",
            "min": "1",
            "max": "256",
            "signed": false
        });
        manifest_value["kernels"][0]["constraints"] = serde_json::json!([{
            "kind": "scalar_le_logical_block_dim",
            "scalar_arg": scalar_arg,
            "dimension": dimension
        }]);
        manifest_value
    }

    fn manifest_with_scalar_le_logical_block_dim(
        scalar_arg: usize,
        dimension: &str,
    ) -> KernelManifest {
        serde_json::from_value(manifest_value_with_scalar_le_logical_block_dim(
            scalar_arg, dimension,
        ))
        .unwrap()
    }

    #[test]
    fn rapid_input_scalar_le_logical_block_dim_repairs_after_normalize_and_mutation() {
        let manifest = manifest_with_scalar_le_logical_block_dim(2, "x");
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let payload = spec.pack_current_kernel_args(&vec![0; 256], &vec![0; 256], 256);
        let raw = encode_rapid_input(
            RapidVConfig {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                block_x: 32,
                block_y: 1,
                block_z: 1,
            },
            &payload,
        );

        let normalized = spec.normalize_rapid_input(&raw);
        let envelope = parse_rapid_input_envelope(&normalized).unwrap();
        assert_eq!(scalar(&spec.parse(envelope.payload).unwrap().values, 2), 32);

        for selector in 0..256 {
            let mutated_payload = spec.mutate_rapid_input(&normalized, selector, selector as u8);
            let payload_envelope = parse_rapid_input_envelope(&mutated_payload).unwrap();
            assert!(
                scalar(&spec.parse(payload_envelope.payload).unwrap().values, 2)
                    <= u64::from(payload_envelope.vconfig.block_x)
            );
        }

        let default = encode_rapid_input(
            RapidVConfig {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                block_x: 256,
                block_y: 1,
                block_z: 1,
            },
            &payload,
        );
        assert_eq!(
            scalar(&spec.parse(&spec.normalize(&payload)).unwrap().values, 2),
            256
        );
        for selector in 0..256 {
            let mutated_vconfig = spec.mutate_rapid_vconfig(&default, selector, selector as u8);
            let vconfig_envelope = parse_rapid_input_envelope(&mutated_vconfig).unwrap();
            assert!(
                scalar(&spec.parse(vconfig_envelope.payload).unwrap().values, 2)
                    <= u64::from(vconfig_envelope.vconfig.block_x)
            );
        }
    }

    #[test]
    fn rapid_input_scalar_eq_logical_block_dim_follows_selected_shape() {
        let mut manifest_value: serde_json::Value = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest_value["kernels"][0]["launch_policy"] = serde_json::json!({
            "grid": [1, 1, 1],
            "block_candidates": [256],
            "physical_block_max": 256,
            "logical_grid": [1, 1, 1],
            "logical_block": [32, 8, 1],
            "logical_block_candidates": [[8, 8, 1], [16, 16, 1], [32, 8, 1]],
            "target_dynamic_shared_bytes": 0,
            "coverage_memory": "global",
            "vconfig_reserved": true,
            "vconfig_mutation": true
        });
        manifest_value["kernels"][0]["args"][2]["domain"] = serde_json::json!({
            "kind": "enum",
            "values": [
                {"name": "tile_8", "value": "8"},
                {"name": "tile_16", "value": "16"},
                {"name": "tile_32", "value": "32"}
            ],
            "allow_unknown": false
        });
        manifest_value["kernels"][0]["constraints"] = serde_json::json!([{
            "kind": "scalar_eq_logical_block_dim",
            "scalar_arg": 2,
            "dimension": "x"
        }]);
        let mut incompatible_manifest_value = manifest_value.clone();
        incompatible_manifest_value["kernels"][0]["args"][2]["domain"]["values"] = serde_json::json!([
            {"name": "tile_8", "value": "8"},
            {"name": "tile_16", "value": "16"}
        ]);
        let manifest: KernelManifest = serde_json::from_value(manifest_value).unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let payload = spec.pack_current_kernel_args(&vec![0; 256], &vec![0; 256], 32);
        let raw = encode_rapid_input(
            RapidVConfig {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                block_x: 9,
                block_y: 8,
                block_z: 1,
            },
            &payload,
        );

        let normalized = spec.normalize_rapid_input(&raw);
        let envelope = parse_rapid_input_envelope(&normalized).unwrap();
        assert_eq!(block_shape(envelope.vconfig), [8, 8, 1]);
        assert_eq!(scalar(&spec.parse(envelope.payload).unwrap().values, 2), 8);

        for selector in 0..64 {
            let mutated = spec.mutate_rapid_vconfig(&normalized, selector, selector as u8);
            let envelope = parse_rapid_input_envelope(&mutated).unwrap();
            assert!([[8, 8, 1], [16, 16, 1], [32, 8, 1]].contains(&block_shape(envelope.vconfig)));
            assert_eq!(
                scalar(&spec.parse(envelope.payload).unwrap().values, 2),
                u64::from(envelope.vconfig.block_x)
            );
        }

        let incompatible_manifest: KernelManifest =
            serde_json::from_value(incompatible_manifest_value).unwrap();
        let error = ArgPackSpec::from_manifest(incompatible_manifest).unwrap_err();
        assert!(
            error.contains("does not permit logical block dimension 32"),
            "{error}"
        );
    }

    #[test]
    fn rapid_input_scalar_le_logical_block_dim_rejects_non_scalar_argument() {
        let manifest = manifest_with_scalar_le_logical_block_dim(0, "x");
        let error = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(error.contains("scalar_arg input is not scalar"), "{error}");
    }

    #[test]
    fn rapid_input_scalar_le_logical_block_dim_rejects_out_of_range_argument() {
        let manifest = manifest_with_scalar_le_logical_block_dim(99, "x");
        let error = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(error.contains("missing scalar_arg 99"), "{error}");
    }

    #[test]
    fn rapid_input_scalar_le_logical_block_dim_rejects_unknown_dimension() {
        let manifest = manifest_value_with_scalar_le_logical_block_dim(2, "w");
        let error = serde_json::from_value::<KernelManifest>(manifest).unwrap_err();

        assert!(error.to_string().contains("unknown variant `w`"), "{error}");
    }

    #[test]
    fn rapid_input_scalar_le_logical_block_dim_rejects_incompatible_minimum() {
        let mut manifest_value = manifest_value_with_scalar_le_logical_block_dim(2, "x");
        manifest_value["kernels"][0]["args"][2]["domain"]["min"] = serde_json::json!("64");
        let manifest = serde_json::from_value::<KernelManifest>(manifest_value).unwrap();

        let error = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(
            error.contains("positive minimum exceeds smallest logical block dimension"),
            "{error}"
        );
    }

    #[test]
    fn rapid_input_rejects_logical_block_larger_than_physical_envelope() {
        let mut manifest_value: serde_json::Value = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest_value["kernels"][0]["launch_policy"] = serde_json::json!({
            "grid": [1, 1, 1],
            "block_candidates": [128, 64, 32],
            "physical_block_max": 128,
            "logical_block": [16, 16, 1],
            "coverage_memory": "global"
        });
        let manifest: KernelManifest = serde_json::from_value(manifest_value).unwrap();

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("logical block threads"));
    }

    #[test]
    fn manifest_rejects_unknown_kernel_field_during_decode() {
        let err = serde_json::from_str::<KernelManifest>(
            r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "unknown_field_kernel",
      "display_name": "unknown_field_kernel",
      "unexpected_launch": true,
      "args": [],
      "constraints": []
    }
  ]
}
"#,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("unexpected_launch"));
    }

    #[test]
    fn manifest_rejects_kernel_id_field_during_decode() {
        let err = serde_json::from_str::<KernelManifest>(
            r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "kernel_id": "kernel__12345678",
      "symbol_name": "kernel",
      "display_name": "kernel",
      "args": [],
      "constraints": []
    }
  ]
}
"#,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("kernel_id"));
    }

    #[test]
    fn rapid_input_normalize_repairs_payload_size_and_payload() {
        let spec = test_spec();
        let payload = spec.default_seed();
        let mut raw = encode_rapid_input(
            RapidVConfig {
                grid_x: 0,
                grid_y: 0,
                grid_z: 1,
                block_x: 2048,
                block_y: 2,
                block_z: 2,
            },
            &payload,
        );
        raw[24..32].copy_from_slice(&1u64.to_le_bytes());

        let normalized = spec.normalize_rapid_input(&raw);

        let envelope = parse_rapid_input_envelope(&normalized).unwrap();
        assert_eq!(
            envelope.payload_size as usize,
            normalized.len() - RAPID_TASK_ENVELOPE_HEADER_BYTES
        );
        assert_eq!(envelope.vconfig.grid_x, 1);
        assert_eq!(envelope.vconfig.grid_y, 1);
        assert_eq!(envelope.vconfig.block_x, 1024);
        assert_eq!(envelope.vconfig.block_y, 1);
        assert_eq!(spec.normalize(envelope.payload), envelope.payload);
    }

    #[test]
    fn rapid_input_normalize_snaps_block_x_to_launch_candidate() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([1, 1, 1]),
            block_candidates: Some(vec![128, 96, 64, 32]),
            physical_block_max: Some(128),
            logical_grid: None,
            logical_block: None,
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: None,
        });
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let raw = encode_rapid_input(
            RapidVConfig {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                block_x: 33,
                block_y: 1,
                block_z: 1,
            },
            &spec.default_seed(),
        );

        let normalized = spec.normalize_rapid_input(&raw);

        let envelope = parse_rapid_input_envelope(&normalized).unwrap();
        assert_eq!(envelope.vconfig.block_x, 64);
    }

    #[test]
    fn rapid_input_vconfig_mutation_clamps_to_launch_shape() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([1, 1, 1]),
            block_candidates: Some(vec![128, 64, 32]),
            physical_block_max: Some(128),
            logical_grid: None,
            logical_block: None,
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: None,
        });
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let mut input = spec.default_rapid_input();

        for selector in 0..32 {
            input = spec.mutate_rapid_vconfig(&input, selector, 0xff);
            let envelope = parse_rapid_input_envelope(&input).unwrap();
            assert_eq!(envelope.vconfig.grid_x, 1);
            assert_eq!(envelope.vconfig.grid_y, 1);
            assert_eq!(envelope.vconfig.grid_z, 1);
            assert!((32..=128).contains(&envelope.vconfig.block_x));
            assert_eq!(envelope.vconfig.block_x % 32, 0);
            assert_eq!(envelope.vconfig.block_y, 1);
            assert_eq!(envelope.vconfig.block_z, 1);
        }
    }

    #[test]
    fn rapid_input_vconfig_default_bounds_mutate_only_whole_warps() {
        let spec = test_spec();
        let seed = spec.default_rapid_input();

        for selector in 0..64 {
            for byte in [0, 1, 2, 3, 0xff] {
                let mutated = spec.mutate_rapid_vconfig(&seed, selector, byte);
                let vconfig = parse_rapid_input_envelope(&mutated).unwrap().vconfig;

                assert_whole_warp_vconfig(vconfig);
            }
        }
    }

    #[test]
    fn rapid_input_vconfig_mutation_aligns_non_warp_default_launch_policy() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([1, 1, 1]),
            block_candidates: Some(vec![64, 32]),
            physical_block_max: Some(64),
            logical_grid: Some([1, 1, 1]),
            logical_block: Some([33, 1, 1]),
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: Some(true),
        });
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_rapid_input();

        for selector in 0..64 {
            for byte in [0, 1, 2, 3, 0xff] {
                let mutated = spec.mutate_rapid_vconfig(&seed, selector, byte);
                let vconfig = parse_rapid_input_envelope(&mutated).unwrap().vconfig;

                assert_whole_warp_vconfig(vconfig);
            }
        }
    }

    #[test]
    fn rapid_input_vconfig_mutation_targets_legal_block_boundaries() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([1, 1, 1]),
            block_candidates: Some(vec![256]),
            physical_block_max: Some(256),
            logical_grid: None,
            logical_block: None,
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: Some(true),
        });
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_rapid_input();

        for (selector, strategy, expected_block_x) in
            [(0, 1, 32), (1, 2, 256), (2, 3, 224), (5, 1, 32)]
        {
            let mutated = spec.mutate_rapid_vconfig(&seed, selector, strategy);
            let vconfig = parse_rapid_input_envelope(&mutated).unwrap().vconfig;
            let threads = vconfig.block_x * vconfig.block_y * vconfig.block_z;

            assert_eq!(vconfig.block_x, expected_block_x);
            assert_eq!(threads % VCONFIG_WARP_SIZE, 0);
            assert!(threads <= 256);
        }
    }

    #[test]
    fn rapid_input_vconfig_mutation_keeps_multidimensional_warp_groups_whole() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([1, 1, 1]),
            block_candidates: Some(vec![64, 32]),
            physical_block_max: Some(64),
            logical_grid: Some([1, 1, 1]),
            logical_block: Some([16, 4, 1]),
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: Some(true),
        });
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let mut input = spec.default_rapid_input();

        for selector in 0..256 {
            input = spec.mutate_rapid_vconfig(&input, selector, 0);
            let envelope = parse_rapid_input_envelope(&input).unwrap();
            let threads =
                envelope.vconfig.block_x * envelope.vconfig.block_y * envelope.vconfig.block_z;
            assert!(threads <= 64);
            assert_eq!(threads % VCONFIG_WARP_SIZE, 0);
        }
    }

    #[test]
    fn rapid_input_vconfig_mutation_maps_large_random_values_into_bounds() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([1, 1, 1]),
            block_candidates: Some(vec![256]),
            physical_block_max: Some(256),
            logical_grid: Some([1, 1, 1]),
            logical_block: Some([32, 8, 1]),
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: Some(true),
        });
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_rapid_input();
        let selector = (u64::from(u32::MAX - 1) << 8) + 2;

        let mutated = spec.mutate_rapid_vconfig(&seed, selector, 0);
        let envelope = parse_rapid_input_envelope(&mutated).unwrap();
        let threads =
            envelope.vconfig.block_x * envelope.vconfig.block_y * envelope.vconfig.block_z;

        assert_ne!(mutated, seed);
        assert!(threads <= 256);
        assert_eq!(threads % VCONFIG_WARP_SIZE, 0);
    }

    #[test]
    fn rapid_input_rejects_multi_block_physical_grid_until_runtime_support_exists() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].launch_policy = Some(LaunchPolicySpec {
            grid: Some([2, 1, 1]),
            block_candidates: Some(vec![128, 64, 32]),
            physical_block_max: Some(128),
            logical_grid: None,
            logical_block: None,
            logical_block_candidates: None,
            target_dynamic_shared_bytes: Some(0),
            coverage_memory: Some("global".to_string()),
            vconfig_reserved: Some(true),
            vconfig_mutation: None,
        });

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("physical grid [1, 1, 1]"));
    }

    #[test]
    fn normalize_is_stable_after_pointer_alignment_filler_is_parsed() {
        let spec = test_spec();
        let seed = spec.default_seed();

        let once = spec.normalize(&seed);
        let twice = spec.normalize(&once);

        assert_eq!(once, seed);
        assert_eq!(twice, once);
    }

    #[test]
    fn opaque_val_is_fixed_width_payload() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args.insert(
            2,
            ArgSpec {
                index: 2,
                name: "cfg".to_string(),
                type_name: "Config".to_string(),
                kind: ArgKind::OpaqueVal,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: None,
            },
        );
        manifest.kernels[0].args[3].index = 3;
        for constraint in &mut manifest.kernels[0].constraints {
            if let Constraint::ScalarLeBufferLen { scalar_arg, .. } = constraint {
                *scalar_arg = 3;
            }
        }
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let packed = spec.pack_values(vec![
            ArgValue::Pointer(b"abc".to_vec()),
            ArgValue::Pointer(b"def".to_vec()),
            ArgValue::OpaqueVal(b"ghi".to_vec()),
            ArgValue::Scalar(0),
        ]);
        let parsed = spec.parse(&packed).unwrap();
        assert_eq!(pointer(&parsed.values, 0), b"abc\0\0\0\0\0");
        assert_eq!(pointer(&parsed.values, 1), b"def\0");
        assert_eq!(opaque_val(&parsed.values, 2), b"ghi\0");
        assert_eq!(scalar(&parsed.values, 3), 0);
    }

    #[test]
    fn signed_scalar_compare_const_encodes_negative_bit_pattern() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "signed_kernel",
                  "display_name": "signed_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "x",
                      "type": "int",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "int_range",
                        "min": "-8",
                        "max": "8",
                        "signed": true
                      }
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "scalar_compare_const",
                      "scalar_arg": 0,
                      "op": "==",
                      "value": -1
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), u32::MAX as u64);
    }

    #[test]
    fn signed_scalar_default_uses_manifest_width() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "signed_default_kernel",
                  "display_name": "signed_default_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "x",
                      "type": "int",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "int_range",
                        "min": "-1",
                        "max": "8",
                        "signed": true
                      }
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();

        assert_eq!(seed, u32::MAX.to_le_bytes());
    }

    #[test]
    fn float_range_seed_and_mutation_stay_in_range() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "float_kernel",
                  "display_name": "float_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "scale",
                      "type": "float",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "float_range",
                        "min": 1.5,
                        "max": 2.5,
                        "allow_nan": false
                      }
                    },
                    {
                      "index": 1,
                      "name": "bias",
                      "type": "double",
                      "kind": "scalar",
                      "size_bytes": 8,
                      "align_bytes": 8,
                      "domain": {
                        "kind": "float_range",
                        "min": -2.0,
                        "max": -1.0,
                        "allow_nan": false
                      }
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();
        assert_eq!(scalar(&parsed.values, 0), 1.5f32.to_bits() as u64);
        assert_eq!(scalar(&parsed.values, 1), (-2.0f64).to_bits());

        let mutated = spec.mutate(&seed, 0, 0xff);
        let parsed = spec.parse(&mutated).unwrap();
        let scale = f32::from_bits(scalar(&parsed.values, 0) as u32);
        let bias = f64::from_bits(scalar(&parsed.values, 1));
        assert!((1.5..=2.5).contains(&scale), "{scale}");
        assert!((-2.0..=-1.0).contains(&bias), "{bias}");
        assert!(!scale.is_nan());
        assert!(!bias.is_nan());
    }

    #[test]
    fn float_scalar_compare_const_repairs_numeric_value_not_bits() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "float_constraint_kernel",
                  "display_name": "float_constraint_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "scale",
                      "type": "float",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "float_range",
                        "min": 0.0,
                        "max": 4.0,
                        "allow_nan": false
                      }
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "scalar_compare_const",
                      "scalar_arg": 0,
                      "op": ">=",
                      "value": 1
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Scalar(0.0f32.to_bits() as u64)]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 1.0f32.to_bits() as u64);
    }

    #[test]
    fn manifest_rejects_unsatisfiable_float_compare_const_domain() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "bad_float_constraint_kernel",
                  "display_name": "bad_float_constraint_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "scale",
                      "type": "float",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "float_range",
                        "min": 0.0,
                        "max": 4.0,
                        "allow_nan": false
                      }
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "scalar_compare_const",
                      "scalar_arg": 0,
                      "op": ">=",
                      "value": 10
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("unsatisfiable scalar_compare_const"));
        assert!(err.contains("float"));
    }

    #[test]
    fn float_scalar_compare_scalar_repairs_numeric_values() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "float_pair_constraint_kernel",
                  "display_name": "float_pair_constraint_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "lhs",
                      "type": "float",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "float_range",
                        "min": 0.0,
                        "max": 10.0,
                        "allow_nan": false
                      }
                    },
                    {
                      "index": 1,
                      "name": "rhs",
                      "type": "double",
                      "kind": "scalar",
                      "size_bytes": 8,
                      "align_bytes": 8,
                      "domain": {
                        "kind": "float_range",
                        "min": 9.0,
                        "max": 20.0,
                        "allow_nan": false
                      }
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "scalar_compare_scalar",
                      "lhs_arg": 0,
                      "op": ">=",
                      "rhs_arg": 1
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![
            ArgValue::Scalar(0.0f32.to_bits() as u64),
            ArgValue::Scalar(20.0f64.to_bits()),
        ]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 10.0f32.to_bits() as u64);
        assert_eq!(scalar(&parsed.values, 1), 10.0f64.to_bits());
    }

    #[test]
    fn manifest_rejects_unsatisfiable_float_compare_scalar_domains() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "bad_float_pair_constraint_kernel",
                  "display_name": "bad_float_pair_constraint_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "lhs",
                      "type": "float",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "float_range",
                        "min": 0.0,
                        "max": 4.0,
                        "allow_nan": false
                      }
                    },
                    {
                      "index": 1,
                      "name": "rhs",
                      "type": "float",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "float_range",
                        "min": 9.0,
                        "max": 12.0,
                        "allow_nan": false
                      }
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "scalar_compare_scalar",
                      "lhs_arg": 0,
                      "op": ">=",
                      "rhs_arg": 1
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("unsatisfiable scalar_compare_scalar"));
        assert!(err.contains("float"));
    }

    #[test]
    fn manifest_rejects_mixed_float_integer_compare_scalar() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "mixed_scalar_constraint_kernel",
                  "display_name": "mixed_scalar_constraint_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "scale",
                      "type": "float",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "float_range",
                        "min": 0.0,
                        "max": 10.0,
                        "allow_nan": false
                      }
                    },
                    {
                      "index": 1,
                      "name": "count",
                      "type": "int",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "int_range",
                        "min": "0",
                        "max": "10",
                        "signed": true
                      }
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "scalar_compare_scalar",
                      "lhs_arg": 0,
                      "op": ">=",
                      "rhs_arg": 1
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("mixed float/integer scalar_compare_scalar"));
    }

    #[test]
    fn unsupported_float_range_width_fails_fast() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "bad_float_kernel",
                  "display_name": "bad_float_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "x",
                      "type": "__half",
                      "kind": "scalar",
                      "size_bytes": 2,
                      "align_bytes": 2,
                      "domain": {
                        "kind": "float_range",
                        "min": 0.0,
                        "max": 1.0
                      }
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("float_range"));
        assert!(err.contains("size_bytes=2"));
    }

    #[test]
    fn enum_domain_seed_and_normalize_use_declared_values() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "enum_kernel",
                  "display_name": "enum_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "mode",
                      "type": "Mode",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "enum",
                        "values": [
                          {"name": "ModeA", "value": "7"},
                          {"name": "ModeB", "value": "9"}
                        ],
                        "allow_unknown": false
                      }
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();
        assert_eq!(scalar(&parsed.values, 0), 7);

        let packed = spec.pack_values(vec![ArgValue::Scalar(99)]);
        let parsed = spec.parse(&packed).unwrap();
        assert_eq!(scalar(&parsed.values, 0), 7);
    }

    #[test]
    fn enum_domain_encodes_negative_discriminant_by_manifest_width() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "enum_negative_kernel",
                  "display_name": "enum_negative_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "kind",
                      "type": "Kind",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "enum",
                        "values": [
                          {"name": "Unknown", "value": "-1"},
                          {"name": "Known", "value": "1"}
                        ]
                      }
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();

        assert_eq!(seed, u32::MAX.to_le_bytes());
    }

    #[test]
    fn enum_domain_allow_unknown_preserves_unknown_value() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "enum_unknown_kernel",
                  "display_name": "enum_unknown_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "mode",
                      "type": "Mode",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {
                        "kind": "enum",
                        "values": [
                          {"name": "ModeA", "value": "7"}
                        ],
                        "allow_unknown": true
                      }
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Scalar(99)]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 99);
    }

    #[test]
    fn pointer_without_pointee_layout_fails_fast() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args[0].pointee_layout = None;
        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("missing pointee_layout"));
    }

    #[test]
    fn pointer_with_opaque_void_pointee_layout_loads() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args[0].type_name = "void *".to_string();
        manifest.kernels[0].args[0].pointee_layout = Some(LayoutNode {
            index: "input.*".to_string(),
            name: "$pointee".to_string(),
            type_name: "void".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            size_bytes: 0,
            align_bytes: 1,
            domain: None,
            fields: vec![],
            element: None,
            element_count: None,
        });

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_current_kernel_args(b"ABC", b"XY", 3);
        let parsed = spec.parse(&packed).unwrap();
        assert_eq!(pointer(&parsed.values, 0), b"ABC\0\0\0\0\0");
    }

    #[test]
    fn opaque_with_ptr_complete_layout_encodes_recursively() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints.clear();

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![
            ArgValue::Scalar(0),
            ArgValue::Pointer(b"abcdefgh".to_vec()),
        ])]);
        let parsed = spec.parse(&packed).unwrap();
        let fields = aggregate(&parsed.values, 0);

        assert_eq!(scalar(fields, 0), 0);
        assert_eq!(pointer(fields, 1), b"abcdefgh");
    }

    #[test]
    fn nested_opaque_with_ptr_array_encodes_pointer_payloads() {
        let pointer_element = LayoutNode {
            index: "params.inner.lanes[]".to_string(),
            name: "lanes[]".to_string(),
            type_name: "float *".to_string(),
            kind: ArgKind::Pointer,
            pointer_role: Some(PointerRole::PayloadBuffer),
            pointee_layout: Some(Box::new(LayoutNode {
                index: "params.inner.lanes[].*".to_string(),
                name: "$pointee".to_string(),
                type_name: "float".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: None,
                fields: vec![],
                element: None,
                element_count: None,
            })),
            size_bytes: 8,
            align_bytes: 8,
            domain: None,
            fields: vec![],
            element: None,
            element_count: None,
        };
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.seed".to_string(),
                        name: "seed".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.inner".to_string(),
                        name: "inner".to_string(),
                        type_name: "Inner".to_string(),
                        kind: ArgKind::OpaqueWithPtr,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 24,
                        align_bytes: 8,
                        domain: None,
                        fields: vec![
                            LayoutNode {
                                index: "params.inner.count".to_string(),
                                name: "count".to_string(),
                                type_name: "int".to_string(),
                                kind: ArgKind::Scalar,
                                pointer_role: None,
                                pointee_layout: None,
                                size_bytes: 4,
                                align_bytes: 4,
                                domain: None,
                                fields: vec![],
                                element: None,
                                element_count: None,
                            },
                            LayoutNode {
                                index: "params.inner.lanes".to_string(),
                                name: "lanes".to_string(),
                                type_name: "float *[2]".to_string(),
                                kind: ArgKind::OpaqueWithPtr,
                                pointer_role: None,
                                pointee_layout: None,
                                size_bytes: 16,
                                align_bytes: 8,
                                domain: None,
                                fields: vec![],
                                element: Some(Box::new(pointer_element)),
                                element_count: Some(2),
                            },
                        ],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 32,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints.clear();

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![
            ArgValue::Scalar(0),
            ArgValue::Aggregate(vec![
                ArgValue::Scalar(0),
                ArgValue::Aggregate(vec![
                    ArgValue::Pointer(b"abcd".to_vec()),
                    ArgValue::Pointer(b"efgh".to_vec()),
                ]),
            ]),
        ])]);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);
        let inner = aggregate(params, 1);
        let lanes = aggregate(inner, 1);

        assert_eq!(scalar(params, 0), 0);
        assert_eq!(scalar(inner, 0), 0);
        assert_eq!(pointer(lanes, 0), b"abcd\0\0\0\0");
        assert_eq!(pointer(lanes, 1), b"efgh");
    }

    #[test]
    fn array_of_struct_elements_with_pointer_encodes_next_outer_element_alignment() {
        let scalar_field = |index: &str, name: &str| LayoutNode {
            index: index.to_string(),
            name: name.to_string(),
            type_name: "int".to_string(),
            kind: ArgKind::Scalar,
            pointer_role: None,
            pointee_layout: None,
            size_bytes: 4,
            align_bytes: 4,
            domain: None,
            fields: vec![],
            element: None,
            element_count: None,
        };
        let pointer_field = |index: &str, name: &str| LayoutNode {
            index: index.to_string(),
            name: name.to_string(),
            type_name: "uint8_t *".to_string(),
            kind: ArgKind::Pointer,
            pointer_role: Some(PointerRole::PayloadBuffer),
            pointee_layout: Some(Box::new(LayoutNode {
                index: format!("{index}.*"),
                name: "$pointee".to_string(),
                type_name: "uint8_t".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                size_bytes: 1,
                align_bytes: 1,
                domain: None,
                fields: vec![],
                element: None,
                element_count: None,
            })),
            size_bytes: 8,
            align_bytes: 8,
            domain: None,
            fields: vec![],
            element: None,
            element_count: None,
        };
        let item_element = LayoutNode {
            index: "params.items[]".to_string(),
            name: "$element".to_string(),
            type_name: "Item".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
            fields: vec![
                scalar_field("params.items[].id", "id"),
                pointer_field("params.items[].ptr", "ptr"),
            ],
            element: None,
            element_count: None,
        };
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![LayoutNode {
                    index: "params.items".to_string(),
                    name: "items".to_string(),
                    type_name: "Item[2]".to_string(),
                    kind: ArgKind::OpaqueWithPtr,
                    pointer_role: None,
                    pointee_layout: None,
                    size_bytes: 32,
                    align_bytes: 8,
                    domain: None,
                    fields: vec![],
                    element: Some(Box::new(item_element)),
                    element_count: Some(2),
                }],
                element: None,
                element_count: None,
            }),
            size_bytes: 32,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints.clear();

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let values = vec![ArgValue::Aggregate(vec![ArgValue::Aggregate(vec![
            ArgValue::Aggregate(vec![
                ArgValue::Scalar(7),
                ArgValue::Pointer(b"abc".to_vec()),
            ]),
            ArgValue::Aggregate(vec![ArgValue::Scalar(9), ArgValue::Pointer(b"xy".to_vec())]),
        ])])];
        let packed = spec.pack_values(values);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);
        let items = aggregate(params, 0);
        let item0 = aggregate(items, 0);
        let item1 = aggregate(items, 1);

        assert_eq!(scalar(item0, 0), 7);
        assert_eq!(pointer(item0, 1), b"abc\0");
        assert_eq!(scalar(item1, 0), 9);
        assert_eq!(pointer(item1, 1), b"xy");
    }

    #[test]
    fn nested_constraints_repair_recursive_fields() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.nested".to_string(),
                        name: "nested".to_string(),
                        type_name: "Nested".to_string(),
                        kind: ArgKind::OpaqueWithPtr,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 8,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![LayoutNode {
                            index: "params.nested.inner_count".to_string(),
                            name: "inner_count".to_string(),
                            type_name: "int".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        }],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 24,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![
            Constraint::ScalarLeBufferLen {
                scalar_arg: 0,
                scalar_path: Some(vec!["count".to_string()]),
                buffer_arg: 0,
                buffer_path: Some(vec!["data".to_string()]),
                unit: ConstraintUnit::Bytes,
            },
            Constraint::ScalarCompareConst {
                scalar_arg: 0,
                scalar_path: Some(vec!["nested".to_string(), "inner_count".to_string()]),
                op: ConstraintOp::Ge,
                value: 3,
            },
        ];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let values = vec![ArgValue::Aggregate(vec![
            ArgValue::Scalar(99),
            ArgValue::Pointer(b"abcd".to_vec()),
            ArgValue::Aggregate(vec![ArgValue::Scalar(0)]),
        ])];
        let packed = spec.pack_values(values);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);
        let nested = aggregate(params, 2);

        assert_eq!(scalar(params, 0), 4);
        assert_eq!(pointer(params, 1), b"abcd");
        assert_eq!(scalar(nested, 0), 3);
    }

    #[test]
    fn scalar_le_buffer_len_elements_uses_pointee_size() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![
            ArgSpec {
                index: 0,
                name: "count".to_string(),
                type_name: "int".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: None,
            },
            ArgSpec {
                index: 1,
                name: "data".to_string(),
                type_name: "float *".to_string(),
                kind: ArgKind::Pointer,
                pointer_role: Some(PointerRole::PayloadBuffer),
                pointee_layout: Some(LayoutNode {
                    index: "data.*".to_string(),
                    name: "$pointee".to_string(),
                    type_name: "float".to_string(),
                    kind: ArgKind::Scalar,
                    pointer_role: None,
                    pointee_layout: None,
                    size_bytes: 4,
                    align_bytes: 4,
                    domain: None,
                    fields: vec![],
                    element: None,
                    element_count: None,
                }),
                type_layout: None,
                size_bytes: 8,
                align_bytes: 8,
                domain: None,
            },
        ];
        manifest.kernels[0].constraints = vec![Constraint::ScalarLeBufferLen {
            scalar_arg: 0,
            scalar_path: None,
            buffer_arg: 1,
            buffer_path: None,
            unit: ConstraintUnit::Elements,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![
            ArgValue::Scalar(99),
            ArgValue::Pointer(b"abcdefgh".to_vec()),
        ]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 2);
        assert_eq!(pointer(&parsed.values, 1), b"abcdefgh");
    }

    #[test]
    fn buffer_elements_lt_scalar_repairs_index_tables() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "index_kernel",
                  "display_name": "index_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "index_map",
                      "type": "uint64_t *",
                      "kind": "pointer",
                      "pointer_role": "payload_buffer",
                      "pointee_layout": {
                        "index": "index_map.*",
                        "name": "$pointee",
                        "type": "uint64_t",
                        "kind": "scalar",
                        "size_bytes": 8,
                        "align_bytes": 8
                      },
                      "size_bytes": 8,
                      "align_bytes": 8,
                      "domain": {
                        "kind": "bytes",
                        "min_len": "24",
                        "max_len": "24",
                        "elem_size_bytes": 8,
                        "nullable": false
                      }
                    },
                    {
                      "index": 1,
                      "name": "slots",
                      "type": "uint64_t",
                      "kind": "scalar",
                      "size_bytes": 8,
                      "align_bytes": 8,
                      "domain": {
                        "kind": "int_range",
                        "min": "3",
                        "max": "3",
                        "signed": false
                      }
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "buffer_elements_lt_scalar",
                      "buffer_arg": 0,
                      "scalar_arg": 1,
                      "elem_size_bytes": 8
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let mut raw_index_map = Vec::new();
        raw_index_map.extend_from_slice(&0u64.to_le_bytes());
        raw_index_map.extend_from_slice(&3u64.to_le_bytes());
        raw_index_map.extend_from_slice(&u64::MAX.to_le_bytes());

        let packed = spec.pack_values(vec![ArgValue::Pointer(raw_index_map), ArgValue::Scalar(3)]);
        let parsed = spec.parse(&packed).unwrap();
        let repaired = pointer(&parsed.values, 0);

        for chunk in repaired.chunks_exact(8) {
            let value = u64::from_le_bytes(chunk.try_into().unwrap());
            assert!(value < 3, "index table element was not repaired: {value}");
        }

        let mutated = spec.mutate(&packed, 0, 0xff);
        let parsed_mutated = spec.parse(&mutated).unwrap();
        for chunk in pointer(&parsed_mutated.values, 0).chunks_exact(8) {
            let value = u64::from_le_bytes(chunk.try_into().unwrap());
            assert!(
                value < 3,
                "mutated index table element escaped repair: {value}"
            );
        }
    }

    #[test]
    fn nested_scalar_le_buffer_len_elements_uses_domain_elem_size() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "uint8_t *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "uint8_t".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 1,
                            align_bytes: 1,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: Some(Domain::Bytes {
                            min_len: None,
                            max_len: None,
                            elem_size_bytes: Some(2),
                            nullable: Some(false),
                            pattern_hex: None,
                        }),
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![Constraint::ScalarLeBufferLen {
            scalar_arg: 0,
            scalar_path: Some(vec!["count".to_string()]),
            buffer_arg: 0,
            buffer_path: Some(vec!["data".to_string()]),
            unit: ConstraintUnit::Elements,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![
            ArgValue::Scalar(99),
            ArgValue::Pointer(b"abcdef".to_vec()),
        ])]);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), 3);
        assert_eq!(pointer(params, 1), b"abcdef");
    }

    #[test]
    fn nested_scalar_le_buffer_len_grows_buffer_for_scalar_path_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: Some(Domain::IntRange {
                            min: Some("2".to_string()),
                            max: Some("8".to_string()),
                            signed: Some(false),
                        }),
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: Some(Domain::Bytes {
                            min_len: None,
                            max_len: Some("16".to_string()),
                            elem_size_bytes: Some(4),
                            nullable: Some(false),
                            pattern_hex: None,
                        }),
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![Constraint::ScalarLeBufferLen {
            scalar_arg: 0,
            scalar_path: Some(vec!["count".to_string()]),
            buffer_arg: 0,
            buffer_path: Some(vec!["data".to_string()]),
            unit: ConstraintUnit::Elements,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), 2);
        assert_eq!(pointer(params, 1), &[0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn count_fits_buffer_repairs_nested_count() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![Constraint::CountFitsBuffer {
            count_arg: 0,
            count_path: Some(vec!["count".to_string()]),
            buffer_arg: 0,
            buffer_path: Some(vec!["data".to_string()]),
            elem_size_bytes: 4,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let values = vec![ArgValue::Aggregate(vec![
            ArgValue::Scalar(99),
            ArgValue::Pointer(b"abcdefgh".to_vec()),
        ])];
        let packed = spec.pack_values(values);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), 2);
        assert_eq!(pointer(params, 1), b"abcdefgh");
    }

    #[test]
    fn array_indexed_count_fits_buffer_repairs_only_target_element() {
        let scalar_layout = |name: &str, index: &str| LayoutNode {
            index: index.to_string(),
            name: name.to_string(),
            type_name: "int".to_string(),
            kind: ArgKind::Scalar,
            pointer_role: None,
            pointee_layout: None,
            size_bytes: 4,
            align_bytes: 4,
            domain: None,
            fields: vec![],
            element: None,
            element_count: None,
        };
        let pointer_layout = |name: &str, index: &str| LayoutNode {
            index: index.to_string(),
            name: name.to_string(),
            type_name: "float *".to_string(),
            kind: ArgKind::Pointer,
            pointer_role: Some(PointerRole::PayloadBuffer),
            pointee_layout: Some(Box::new(LayoutNode {
                index: format!("{index}.*"),
                name: "$pointee".to_string(),
                type_name: "float".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: None,
                fields: vec![],
                element: None,
                element_count: None,
            })),
            size_bytes: 8,
            align_bytes: 8,
            domain: None,
            fields: vec![],
            element: None,
            element_count: None,
        };
        let inner = LayoutNode {
            index: "cfg.items[].inner".to_string(),
            name: "inner".to_string(),
            type_name: "Inner".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
            fields: vec![
                scalar_layout("count", "cfg.items[].inner.count"),
                pointer_layout("data", "cfg.items[].inner.data"),
            ],
            element: None,
            element_count: None,
        };
        let item = LayoutNode {
            index: "cfg.items[]".to_string(),
            name: "$element".to_string(),
            type_name: "Item".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            size_bytes: 24,
            align_bytes: 8,
            domain: None,
            fields: vec![inner, scalar_layout("scale", "cfg.items[].scale")],
            element: None,
            element_count: None,
        };

        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "cfg".to_string(),
            type_name: "Cfg".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![LayoutNode {
                    index: "cfg.items".to_string(),
                    name: "items".to_string(),
                    type_name: "Item[2]".to_string(),
                    kind: ArgKind::OpaqueWithPtr,
                    pointer_role: None,
                    pointee_layout: None,
                    size_bytes: 48,
                    align_bytes: 8,
                    domain: None,
                    fields: vec![],
                    element: Some(Box::new(item)),
                    element_count: Some(2),
                }],
                element: None,
                element_count: None,
            }),
            size_bytes: 48,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![
            Constraint::CountFitsBuffer {
                count_arg: 0,
                count_path: Some(vec![
                    "items".to_string(),
                    "1".to_string(),
                    "inner".to_string(),
                    "count".to_string(),
                ]),
                buffer_arg: 0,
                buffer_path: Some(vec![
                    "items".to_string(),
                    "1".to_string(),
                    "inner".to_string(),
                    "data".to_string(),
                ]),
                elem_size_bytes: 4,
            },
            Constraint::ScalarCompareConst {
                scalar_arg: 0,
                scalar_path: Some(vec![
                    "items".to_string(),
                    "1".to_string(),
                    "inner".to_string(),
                    "count".to_string(),
                ]),
                op: ConstraintOp::Ge,
                value: 1,
            },
        ];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed_seed = spec.parse(&seed).unwrap();
        let cfg = aggregate(&parsed_seed.values, 0);
        let items = aggregate(cfg, 0);
        let item1 = aggregate(items, 1);
        let item1_inner = aggregate(item1, 0);

        assert_eq!(scalar(item1_inner, 0), 1);
        assert_eq!(pointer(item1_inner, 1), &[0, 0, 0, 0]);

        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![ArgValue::Aggregate(vec![
            ArgValue::Aggregate(vec![
                ArgValue::Aggregate(vec![
                    ArgValue::Scalar(77),
                    ArgValue::Pointer(b"abcd".to_vec()),
                ]),
                ArgValue::Scalar(0),
            ]),
            ArgValue::Aggregate(vec![
                ArgValue::Aggregate(vec![
                    ArgValue::Scalar(99),
                    ArgValue::Pointer(b"abcdefgh".to_vec()),
                ]),
                ArgValue::Scalar(0),
            ]),
        ])])]);
        let parsed = spec.parse(&packed).unwrap();
        let cfg = aggregate(&parsed.values, 0);
        let items = aggregate(cfg, 0);
        let item0_inner = aggregate(aggregate(items, 0), 0);
        let item1_inner = aggregate(aggregate(items, 1), 0);

        assert_eq!(scalar(item0_inner, 0), 77);
        assert_eq!(pointer(item0_inner, 1), b"abcd");
        assert_eq!(scalar(item1_inner, 0), 2);
        assert_eq!(pointer(item1_inner, 1), b"abcdefgh");
    }

    #[test]
    fn scalar_le_buffer_len_grows_mutable_buffer_to_preserve_scalar_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![
            ArgSpec {
                index: 0,
                name: "count".to_string(),
                type_name: "int".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: Some(Domain::IntRange {
                    min: Some("8".to_string()),
                    max: Some("16".to_string()),
                    signed: Some(false),
                }),
            },
            ArgSpec {
                index: 1,
                name: "data".to_string(),
                type_name: "uint8_t *".to_string(),
                kind: ArgKind::Pointer,
                pointer_role: Some(PointerRole::PayloadBuffer),
                pointee_layout: Some(LayoutNode {
                    index: "data.*".to_string(),
                    name: "$pointee".to_string(),
                    type_name: "uint8_t".to_string(),
                    kind: ArgKind::Scalar,
                    pointer_role: None,
                    pointee_layout: None,
                    size_bytes: 1,
                    align_bytes: 1,
                    domain: None,
                    fields: vec![],
                    element: None,
                    element_count: None,
                }),
                type_layout: None,
                size_bytes: 8,
                align_bytes: 8,
                domain: Some(Domain::Bytes {
                    min_len: None,
                    max_len: Some("16".to_string()),
                    elem_size_bytes: Some(1),
                    nullable: Some(false),
                    pattern_hex: None,
                }),
            },
        ];
        manifest.kernels[0].constraints = vec![Constraint::ScalarLeBufferLen {
            scalar_arg: 0,
            scalar_path: None,
            buffer_arg: 1,
            buffer_path: None,
            unit: ConstraintUnit::Bytes,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![
            ArgValue::Scalar(8),
            ArgValue::Pointer(b"abcd".to_vec()),
        ]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 8);
        assert_eq!(pointer(&parsed.values, 1), b"abcd\0\0\0\0");
    }

    #[test]
    fn scalar_le_buffer_len_grows_existing_buffer_without_bytes_domain_for_scalar_lower_bound() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![
            ArgSpec {
                index: 0,
                name: "count".to_string(),
                type_name: "int".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: Some(Domain::IntRange {
                    min: Some("8".to_string()),
                    max: Some("16".to_string()),
                    signed: Some(false),
                }),
            },
            ArgSpec {
                index: 1,
                name: "data".to_string(),
                type_name: "uint8_t *".to_string(),
                kind: ArgKind::Pointer,
                pointer_role: Some(PointerRole::PayloadBuffer),
                pointee_layout: Some(LayoutNode {
                    index: "data.*".to_string(),
                    name: "$pointee".to_string(),
                    type_name: "uint8_t".to_string(),
                    kind: ArgKind::Scalar,
                    pointer_role: None,
                    pointee_layout: None,
                    size_bytes: 1,
                    align_bytes: 1,
                    domain: None,
                    fields: vec![],
                    element: None,
                    element_count: None,
                }),
                type_layout: None,
                size_bytes: 8,
                align_bytes: 8,
                domain: None,
            },
        ];
        manifest.kernels[0].constraints = vec![Constraint::ScalarLeBufferLen {
            scalar_arg: 0,
            scalar_path: None,
            buffer_arg: 1,
            buffer_path: None,
            unit: ConstraintUnit::Bytes,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![
            ArgValue::Scalar(8),
            ArgValue::Pointer(b"abcd".to_vec()),
        ]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 8);
        assert_eq!(pointer(&parsed.values, 1), b"abcd\0\0\0\0");
    }

    #[test]
    fn count_fits_buffer_grows_existing_buffer_without_bytes_domain_for_count_lower_bound() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![
            ArgSpec {
                index: 0,
                name: "count".to_string(),
                type_name: "int".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: Some(Domain::IntRange {
                    min: Some("2".to_string()),
                    max: Some("8".to_string()),
                    signed: Some(false),
                }),
            },
            ArgSpec {
                index: 1,
                name: "data".to_string(),
                type_name: "float *".to_string(),
                kind: ArgKind::Pointer,
                pointer_role: Some(PointerRole::PayloadBuffer),
                pointee_layout: Some(LayoutNode {
                    index: "data.*".to_string(),
                    name: "$pointee".to_string(),
                    type_name: "float".to_string(),
                    kind: ArgKind::Scalar,
                    pointer_role: None,
                    pointee_layout: None,
                    size_bytes: 4,
                    align_bytes: 4,
                    domain: None,
                    fields: vec![],
                    element: None,
                    element_count: None,
                }),
                type_layout: None,
                size_bytes: 8,
                align_bytes: 8,
                domain: None,
            },
        ];
        manifest.kernels[0].constraints = vec![Constraint::CountFitsBuffer {
            count_arg: 0,
            count_path: None,
            buffer_arg: 1,
            buffer_path: None,
            elem_size_bytes: 4,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![
            ArgValue::Scalar(2),
            ArgValue::Pointer(b"abcd".to_vec()),
        ]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 2);
        assert_eq!(pointer(&parsed.values, 1), b"abcd\0\0\0\0");
    }

    #[test]
    fn default_seed_grows_nested_buffer_to_satisfy_count_lower_bound() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: Some(Domain::IntRange {
                            min: Some("1".to_string()),
                            max: None,
                            signed: Some(false),
                        }),
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: Some(Domain::Bytes {
                            min_len: None,
                            max_len: Some("16".to_string()),
                            elem_size_bytes: Some(4),
                            nullable: Some(false),
                            pattern_hex: None,
                        }),
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![
            Constraint::ScalarCompareConst {
                scalar_arg: 0,
                scalar_path: Some(vec!["count".to_string()]),
                op: ConstraintOp::Ge,
                value: 1,
            },
            Constraint::CountFitsBuffer {
                count_arg: 0,
                count_path: Some(vec!["count".to_string()]),
                buffer_arg: 0,
                buffer_path: Some(vec!["data".to_string()]),
                elem_size_bytes: 4,
            },
        ];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), 1);
        assert_eq!(pointer(params, 1), &[0, 0, 0, 0]);
    }

    #[test]
    fn default_seed_grows_buffer_without_bytes_domain_to_satisfy_count_lower_bound() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: Some(Domain::IntRange {
                            min: Some("1".to_string()),
                            max: None,
                            signed: Some(false),
                        }),
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![
            Constraint::ScalarCompareConst {
                scalar_arg: 0,
                scalar_path: Some(vec!["count".to_string()]),
                op: ConstraintOp::Ge,
                value: 1,
            },
            Constraint::CountFitsBuffer {
                count_arg: 0,
                count_path: Some(vec!["count".to_string()]),
                buffer_arg: 0,
                buffer_path: Some(vec!["data".to_string()]),
                elem_size_bytes: 4,
            },
        ];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), 1);
        assert_eq!(pointer(params, 1), &[0, 0, 0, 0]);
    }

    #[test]
    fn constraint_repair_revisits_count_fits_after_later_scalar_lower_bound() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: Some(Domain::Bytes {
                            min_len: None,
                            max_len: Some("16".to_string()),
                            elem_size_bytes: Some(4),
                            nullable: Some(false),
                            pattern_hex: None,
                        }),
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![
            Constraint::CountFitsBuffer {
                count_arg: 0,
                count_path: Some(vec!["count".to_string()]),
                buffer_arg: 0,
                buffer_path: Some(vec!["data".to_string()]),
                elem_size_bytes: 4,
            },
            Constraint::ScalarCompareConst {
                scalar_arg: 0,
                scalar_path: Some(vec!["count".to_string()]),
                op: ConstraintOp::Ge,
                value: 1,
            },
        ];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.default_seed();
        let parsed = spec.parse(&seed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), 1);
        assert_eq!(pointer(params, 1), &[0, 0, 0, 0]);
    }

    #[test]
    fn manifest_rejects_unsatisfiable_scalar_const_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "count".to_string(),
            type_name: "uint32_t".to_string(),
            kind: ArgKind::Scalar,
            pointer_role: None,
            pointee_layout: None,
            type_layout: None,
            size_bytes: 4,
            align_bytes: 4,
            domain: Some(Domain::IntRange {
                min: Some("0".to_string()),
                max: Some("4".to_string()),
                signed: Some(false),
            }),
        }];
        manifest.kernels[0].constraints = vec![Constraint::ScalarCompareConst {
            scalar_arg: 0,
            scalar_path: None,
            op: ConstraintOp::Ge,
            value: 9,
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("unsatisfiable scalar_compare_const"));
    }

    #[test]
    fn scalar_compare_const_ne_repair_stays_in_unsigned_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "value".to_string(),
            type_name: "uint32_t".to_string(),
            kind: ArgKind::Scalar,
            pointer_role: None,
            pointee_layout: None,
            type_layout: None,
            size_bytes: 4,
            align_bytes: 4,
            domain: Some(Domain::IntRange {
                min: Some("0".to_string()),
                max: Some("10".to_string()),
                signed: Some(false),
            }),
        }];
        manifest.kernels[0].constraints = vec![Constraint::ScalarCompareConst {
            scalar_arg: 0,
            scalar_path: None,
            op: ConstraintOp::Ne,
            value: 10,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Scalar(10)]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 9);
    }

    #[test]
    fn scalar_compare_const_ne_repair_stays_in_signed_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "value".to_string(),
            type_name: "int32_t".to_string(),
            kind: ArgKind::Scalar,
            pointer_role: None,
            pointee_layout: None,
            type_layout: None,
            size_bytes: 4,
            align_bytes: 4,
            domain: Some(Domain::IntRange {
                min: Some("-3".to_string()),
                max: Some("3".to_string()),
                signed: Some(true),
            }),
        }];
        manifest.kernels[0].constraints = vec![Constraint::ScalarCompareConst {
            scalar_arg: 0,
            scalar_path: None,
            op: ConstraintOp::Ne,
            value: 3,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Scalar(scalar_bits_from_i128(3, 4))]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(
            scalar_i128_from_bits(scalar(&parsed.values, 0), spec.args[0].as_node()),
            2
        );
    }

    #[test]
    fn scalar_compare_const_repair_uses_nested_signed_field_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![LayoutNode {
                    index: "params.delta".to_string(),
                    name: "delta".to_string(),
                    type_name: "int32_t".to_string(),
                    kind: ArgKind::Scalar,
                    pointer_role: None,
                    pointee_layout: None,
                    size_bytes: 4,
                    align_bytes: 4,
                    domain: Some(Domain::IntRange {
                        min: Some("-8".to_string()),
                        max: Some("8".to_string()),
                        signed: Some(true),
                    }),
                    fields: vec![],
                    element: None,
                    element_count: None,
                }],
                element: None,
                element_count: None,
            }),
            size_bytes: 4,
            align_bytes: 4,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![Constraint::ScalarCompareConst {
            scalar_arg: 0,
            scalar_path: Some(vec!["delta".to_string()]),
            op: ConstraintOp::Ge,
            value: -1,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![ArgValue::Scalar(
            scalar_bits_from_i128(-8, 4),
        )])]);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), u32::MAX as u64);
    }

    #[test]
    fn manifest_rejects_unsatisfiable_scalar_buffer_len_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args[2].domain = Some(Domain::IntRange {
            min: Some("8".to_string()),
            max: Some("16".to_string()),
            signed: Some(false),
        });
        manifest.kernels[0].args[0].domain = Some(Domain::Bytes {
            min_len: None,
            max_len: Some("4".to_string()),
            elem_size_bytes: Some(1),
            nullable: Some(false),
            pattern_hex: None,
        });
        manifest.kernels[0].constraints = vec![Constraint::ScalarLeBufferLen {
            scalar_arg: 2,
            scalar_path: None,
            buffer_arg: 0,
            buffer_path: None,
            unit: ConstraintUnit::Bytes,
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("unsatisfiable scalar_le_buffer_len"));
    }

    #[test]
    fn manifest_rejects_unsatisfiable_count_fits_buffer_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args[2].domain = Some(Domain::IntRange {
            min: Some("2".to_string()),
            max: Some("8".to_string()),
            signed: Some(false),
        });
        manifest.kernels[0].args[0].domain = Some(Domain::Bytes {
            min_len: None,
            max_len: Some("4".to_string()),
            elem_size_bytes: Some(4),
            nullable: Some(false),
            pattern_hex: None,
        });
        manifest.kernels[0].constraints = vec![Constraint::CountFitsBuffer {
            count_arg: 2,
            count_path: None,
            buffer_arg: 0,
            buffer_path: None,
            elem_size_bytes: 4,
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("unsatisfiable count_fits_buffer"));
    }

    #[test]
    fn manifest_rejects_unsatisfiable_scalar_compare_scalar_domains() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![
            ArgSpec {
                index: 0,
                name: "lhs".to_string(),
                type_name: "uint32_t".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: Some(Domain::IntRange {
                    min: Some("0".to_string()),
                    max: Some("4".to_string()),
                    signed: Some(false),
                }),
            },
            ArgSpec {
                index: 1,
                name: "rhs".to_string(),
                type_name: "uint32_t".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: Some(Domain::IntRange {
                    min: Some("9".to_string()),
                    max: Some("12".to_string()),
                    signed: Some(false),
                }),
            },
        ];
        manifest.kernels[0].constraints = vec![Constraint::ScalarCompareScalar {
            lhs_arg: 0,
            lhs_path: None,
            op: ConstraintOp::Ge,
            rhs_arg: 1,
            rhs_path: None,
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("unsatisfiable scalar_compare_scalar"));
    }

    #[test]
    fn scalar_compare_scalar_repair_keeps_both_domains() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![
            ArgSpec {
                index: 0,
                name: "lhs".to_string(),
                type_name: "uint32_t".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: Some(Domain::IntRange {
                    min: Some("0".to_string()),
                    max: Some("10".to_string()),
                    signed: Some(false),
                }),
            },
            ArgSpec {
                index: 1,
                name: "rhs".to_string(),
                type_name: "uint32_t".to_string(),
                kind: ArgKind::Scalar,
                pointer_role: None,
                pointee_layout: None,
                type_layout: None,
                size_bytes: 4,
                align_bytes: 4,
                domain: Some(Domain::IntRange {
                    min: Some("9".to_string()),
                    max: Some("20".to_string()),
                    signed: Some(false),
                }),
            },
        ];
        manifest.kernels[0].constraints = vec![Constraint::ScalarCompareScalar {
            lhs_arg: 0,
            lhs_path: None,
            op: ConstraintOp::Ge,
            rhs_arg: 1,
            rhs_path: None,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Scalar(0), ArgValue::Scalar(20)]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(scalar(&parsed.values, 0), 10);
        assert_eq!(scalar(&parsed.values, 1), 10);
    }

    #[test]
    fn scalar_compare_scalar_repair_uses_nested_field_domains() {
        let field = |name: &str, min: &str, max: &str| LayoutNode {
            index: format!("params.{name}"),
            name: name.to_string(),
            type_name: "uint32_t".to_string(),
            kind: ArgKind::Scalar,
            pointer_role: None,
            pointee_layout: None,
            size_bytes: 4,
            align_bytes: 4,
            domain: Some(Domain::IntRange {
                min: Some(min.to_string()),
                max: Some(max.to_string()),
                signed: Some(false),
            }),
            fields: vec![],
            element: None,
            element_count: None,
        };
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![field("lhs", "0", "10"), field("rhs", "9", "20")],
                element: None,
                element_count: None,
            }),
            size_bytes: 8,
            align_bytes: 4,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![Constraint::ScalarCompareScalar {
            lhs_arg: 0,
            lhs_path: Some(vec!["lhs".to_string()]),
            op: ConstraintOp::Ge,
            rhs_arg: 0,
            rhs_path: Some(vec!["rhs".to_string()]),
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![
            ArgValue::Scalar(0),
            ArgValue::Scalar(20),
        ])]);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(scalar(params, 0), 10);
        assert_eq!(scalar(params, 1), 10);
    }

    #[test]
    fn structured_mutation_repairs_nested_count_fits_after_payload_change() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![
                    LayoutNode {
                        index: "params.count".to_string(),
                        name: "count".to_string(),
                        type_name: "int".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 4,
                        align_bytes: 4,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                    LayoutNode {
                        index: "params.data".to_string(),
                        name: "data".to_string(),
                        type_name: "float *".to_string(),
                        kind: ArgKind::Pointer,
                        pointer_role: Some(PointerRole::PayloadBuffer),
                        pointee_layout: Some(Box::new(LayoutNode {
                            index: "params.data.*".to_string(),
                            name: "$pointee".to_string(),
                            type_name: "float".to_string(),
                            kind: ArgKind::Scalar,
                            pointer_role: None,
                            pointee_layout: None,
                            size_bytes: 4,
                            align_bytes: 4,
                            domain: None,
                            fields: vec![],
                            element: None,
                            element_count: None,
                        })),
                        size_bytes: 8,
                        align_bytes: 8,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    },
                ],
                element: None,
                element_count: None,
            }),
            size_bytes: 16,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints = vec![Constraint::CountFitsBuffer {
            count_arg: 0,
            count_path: Some(vec!["count".to_string()]),
            buffer_arg: 0,
            buffer_path: Some(vec!["data".to_string()]),
            elem_size_bytes: 4,
        }];

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let seed = spec.pack_values(vec![ArgValue::Aggregate(vec![
            ArgValue::Scalar(1),
            ArgValue::Pointer(b"abcd".to_vec()),
        ])]);
        let mutated = spec.mutate(&seed, 1, 0x41);
        let parsed = spec.parse(&mutated).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_ne!(mutated, seed);
        assert_eq!(pointer(params, 1).len(), 3);
        assert_eq!(scalar(params, 0), 0);
    }

    #[test]
    fn pack_values_clamps_nested_pointer_payload_to_domain() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args = vec![ArgSpec {
            index: 0,
            name: "params".to_string(),
            type_name: "Params".to_string(),
            kind: ArgKind::OpaqueWithPtr,
            pointer_role: None,
            pointee_layout: None,
            type_layout: Some(TypeLayout {
                layout_status: LayoutStatus::Complete,
                fields: vec![LayoutNode {
                    index: "params.data".to_string(),
                    name: "data".to_string(),
                    type_name: "uint8_t *".to_string(),
                    kind: ArgKind::Pointer,
                    pointer_role: Some(PointerRole::PayloadBuffer),
                    pointee_layout: Some(Box::new(LayoutNode {
                        index: "params.data.*".to_string(),
                        name: "$pointee".to_string(),
                        type_name: "uint8_t".to_string(),
                        kind: ArgKind::Scalar,
                        pointer_role: None,
                        pointee_layout: None,
                        size_bytes: 1,
                        align_bytes: 1,
                        domain: None,
                        fields: vec![],
                        element: None,
                        element_count: None,
                    })),
                    size_bytes: 8,
                    align_bytes: 8,
                    domain: Some(Domain::Bytes {
                        min_len: None,
                        max_len: Some("4".to_string()),
                        elem_size_bytes: Some(1),
                        nullable: Some(true),
                        pattern_hex: None,
                    }),
                    fields: vec![],
                    element: None,
                    element_count: None,
                }],
                element: None,
                element_count: None,
            }),
            size_bytes: 8,
            align_bytes: 8,
            domain: None,
        }];
        manifest.kernels[0].constraints.clear();

        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        let packed = spec.pack_values(vec![ArgValue::Aggregate(vec![ArgValue::Pointer(
            b"abcdefgh".to_vec(),
        )])]);
        let parsed = spec.parse(&packed).unwrap();
        let params = aggregate(&parsed.values, 0);

        assert_eq!(pointer(params, 0), b"abcd");
    }

    #[test]
    fn structured_mutation_preserves_canonical_constraints() {
        let spec = test_spec();
        let seed = spec.pack_current_kernel_args(b"ABCD", b"XYZW", 999);
        let mutated = spec.mutate(&seed, 2, 0x41);
        let parsed = spec.parse(&mutated).unwrap();

        assert_ne!(mutated, seed);
        assert_eq!(mutated, spec.pack_values(parsed.values.clone()));
        assert!(scalar(&parsed.values, 2) <= pointer(&parsed.values, 0).len() as u64);
        assert!(scalar(&parsed.values, 2) <= pointer(&parsed.values, 1).len() as u64);
    }

    #[test]
    fn unsupported_pointer_role_fails_fast() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args[0].pointer_role = Some(PointerRole::DerivedPointer);
        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("unsupported pointer_role"));
        assert!(err.contains("payload_buffer"));
    }

    #[test]
    fn missing_pointer_role_fails_fast() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args[0].pointer_role = None;
        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("missing pointer_role"));
    }

    #[test]
    fn multiple_kernels_fail_fast() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels.push(manifest.kernels[0].clone());
        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();
        assert!(err.contains("exactly one kernel"));
    }

    #[test]
    fn zero_arg_kernel_uses_empty_payload() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "empty",
                  "display_name": "empty",
                  "args": []
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();
        assert!(spec.default_seed().is_empty());
        assert!(spec.normalize(b"junk").is_empty());
        assert!(spec.parse(b"").unwrap().values.is_empty());
    }

    #[test]
    fn expression_compare_resizes_payload_for_scalar_product() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "matrix_kernel",
                  "display_name": "matrix_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "buffer",
                      "type": "float *",
                      "kind": "pointer",
                      "pointer_role": "payload_buffer",
                      "pointee_layout": {
                        "index": "buffer.*",
                        "name": "$pointee",
                        "type": "float",
                        "kind": "scalar",
                        "size_bytes": 4,
                        "align_bytes": 4
                      },
                      "size_bytes": 8,
                      "align_bytes": 8,
                      "domain": {
                        "kind": "bytes",
                        "min_len": "0",
                        "max_len": "256",
                        "elem_size_bytes": 4,
                        "nullable": false
                      }
                    },
                    {
                      "index": 1,
                      "name": "rows",
                      "type": "int",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {"kind": "int_range", "min": "1", "max": "8", "signed": true}
                    },
                    {
                      "index": 2,
                      "name": "cols",
                      "type": "int",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {"kind": "int_range", "min": "1", "max": "8", "signed": true}
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "expression_compare",
                      "lhs": {
                        "kind": "binary",
                        "op": "*",
                        "lhs": {
                          "kind": "binary",
                          "op": "*",
                          "lhs": {"kind": "arg_value", "arg": 1},
                          "rhs": {"kind": "arg_value", "arg": 2}
                        },
                        "rhs": {"kind": "const", "value": 4}
                      },
                      "op": "<=",
                      "rhs": {"kind": "payload_len", "arg": 0},
                      "repair": {"kind": "resize_payload", "arg": 0}
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let packed = spec.pack_values(vec![
            ArgValue::Pointer(Vec::new()),
            ArgValue::Scalar(3),
            ArgValue::Scalar(4),
        ]);
        let parsed = spec.parse(&packed).unwrap();

        assert_eq!(pointer(&parsed.values, 0).len(), 48);
        assert_eq!(scalar(&parsed.values, 1), 3);
        assert_eq!(scalar(&parsed.values, 2), 4);
        assert_eq!(packed, spec.pack_values(parsed.values));
    }

    fn scalar_product_manifest() -> KernelManifest {
        serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "bounded_product",
                  "display_name": "bounded_product",
                  "args": [
                    {
                      "index": 0,
                      "name": "batch",
                      "type": "uint32_t",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {"kind": "int_range", "min": "1", "max": "32", "signed": false}
                    },
                    {
                      "index": 1,
                      "name": "channels",
                      "type": "uint32_t",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4,
                      "domain": {"kind": "int_range", "min": "1", "max": "256", "signed": false}
                    }
                  ],
                  "constraints": [
                    {
                      "kind": "scalar_product_le_const",
                      "lhs_arg": 0,
                      "rhs_arg": 1,
                      "value": 2048,
                      "repair_arg": 1
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn scalar_product_le_const_repairs_selected_dimension() {
        let manifest = scalar_product_manifest();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let repaired = spec.pack_values(vec![ArgValue::Scalar(32), ArgValue::Scalar(256)]);
        let repaired_values = spec.parse(&repaired).unwrap().values;
        assert_eq!(scalar(&repaired_values, 0), 32);
        assert_eq!(scalar(&repaired_values, 1), 64);
        assert_eq!(repaired, spec.pack_values(repaired_values));

        let unchanged = spec.pack_values(vec![ArgValue::Scalar(8), ArgValue::Scalar(256)]);
        let unchanged_values = spec.parse(&unchanged).unwrap().values;
        assert_eq!(scalar(&unchanged_values, 0), 8);
        assert_eq!(scalar(&unchanged_values, 1), 256);

        let mut zero_manifest = scalar_product_manifest();
        zero_manifest.kernels[0].args[0].domain = Some(Domain::IntRange {
            min: Some("0".to_string()),
            max: Some("32".to_string()),
            signed: Some(false),
        });
        let zero_spec = ArgPackSpec::from_manifest(zero_manifest).unwrap();
        let zero = zero_spec.pack_values(vec![ArgValue::Scalar(0), ArgValue::Scalar(256)]);
        let zero_values = zero_spec.parse(&zero).unwrap().values;
        assert_eq!(scalar(&zero_values, 0), 0);
        assert_eq!(scalar(&zero_values, 1), 256);
    }

    #[test]
    fn scalar_product_repair_composes_with_payload_length_repairs() {
        let mut manifest = scalar_product_manifest();
        manifest.kernels[0].args.extend([
            float_payload_arg(2, "y", 128),
            float_payload_arg(3, "diff", 8192),
            float_payload_arg(4, "dist_sq", 128),
            float_payload_arg(5, "bottom_diff", 8192),
        ]);
        let batch_bytes = ConstraintExpr::Binary {
            op: ConstraintBinaryOp::Mul,
            lhs: Box::new(ConstraintExpr::ArgValue { arg: 0, path: None }),
            rhs: Box::new(ConstraintExpr::Const { value: 4 }),
        };
        let count_bytes = ConstraintExpr::Binary {
            op: ConstraintBinaryOp::Mul,
            lhs: Box::new(ConstraintExpr::Binary {
                op: ConstraintBinaryOp::Mul,
                lhs: Box::new(ConstraintExpr::ArgValue { arg: 0, path: None }),
                rhs: Box::new(ConstraintExpr::ArgValue { arg: 1, path: None }),
            }),
            rhs: Box::new(ConstraintExpr::Const { value: 4 }),
        };
        manifest.kernels[0].constraints = vec![
            Constraint::ScalarProductLeConst {
                lhs_arg: 0,
                lhs_path: None,
                rhs_arg: 1,
                rhs_path: None,
                value: 2048,
                repair_arg: 1,
                repair_path: None,
            },
            Constraint::ExpressionCompare {
                lhs: batch_bytes.clone(),
                op: ConstraintOp::Le,
                rhs: ConstraintExpr::PayloadLen { arg: 2, path: None },
                repair: ConstraintRepair::ResizePayload { arg: 2, path: None },
            },
            Constraint::ExpressionCompare {
                lhs: count_bytes.clone(),
                op: ConstraintOp::Le,
                rhs: ConstraintExpr::PayloadLen { arg: 3, path: None },
                repair: ConstraintRepair::ResizePayload { arg: 3, path: None },
            },
            Constraint::ExpressionCompare {
                lhs: batch_bytes,
                op: ConstraintOp::Le,
                rhs: ConstraintExpr::PayloadLen { arg: 4, path: None },
                repair: ConstraintRepair::ResizePayload { arg: 4, path: None },
            },
            Constraint::ExpressionCompare {
                lhs: count_bytes,
                op: ConstraintOp::Le,
                rhs: ConstraintExpr::PayloadLen { arg: 5, path: None },
                repair: ConstraintRepair::ResizePayload { arg: 5, path: None },
            },
        ];
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let packed = spec.pack_values(vec![
            ArgValue::Scalar(32),
            ArgValue::Scalar(256),
            ArgValue::Pointer(Vec::new()),
            ArgValue::Pointer(Vec::new()),
            ArgValue::Pointer(Vec::new()),
            ArgValue::Pointer(Vec::new()),
        ]);
        let values = spec.parse(&packed).unwrap().values;

        assert_eq!(scalar(&values, 0), 32);
        assert_eq!(scalar(&values, 1), 64);
        assert_eq!(pointer(&values, 2).len(), 128);
        assert_eq!(pointer(&values, 3).len(), 8192);
        assert_eq!(pointer(&values, 4).len(), 128);
        assert_eq!(pointer(&values, 5).len(), 8192);
        assert_eq!(packed, spec.pack_values(values));
    }

    #[test]
    fn scalar_product_le_const_rejects_non_scalar_operand() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].constraints = vec![Constraint::ScalarProductLeConst {
            lhs_arg: 0,
            lhs_path: None,
            rhs_arg: 2,
            rhs_path: None,
            value: 2048,
            repair_arg: 2,
            repair_path: None,
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("scalar_product_le_const operands must both be scalar"));
    }

    #[test]
    fn scalar_product_le_const_rejects_non_integer_repair_domain() {
        for domain in [
            Domain::FloatRange {
                min: Some(1.0),
                max: Some(256.0),
                allow_nan: Some(false),
            },
            Domain::Enum {
                values: vec![
                    EnumValue {
                        name: "one".to_string(),
                        value: "1".to_string(),
                    },
                    EnumValue {
                        name: "two".to_string(),
                        value: "2".to_string(),
                    },
                ],
                allow_unknown: Some(false),
            },
        ] {
            let mut manifest = scalar_product_manifest();
            manifest.kernels[0].args[1].domain = Some(domain);

            let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

            assert!(err.contains("scalar_product_le_const requires int_range domains"));
        }
    }

    #[test]
    fn scalar_product_le_const_rejects_negative_integer_domain() {
        let mut manifest = scalar_product_manifest();
        manifest.kernels[0].args[0].domain = Some(Domain::IntRange {
            min: Some("-1".to_string()),
            max: Some("32".to_string()),
            signed: Some(true),
        });

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("scalar_product_le_const requires non-negative domains"));
    }

    #[test]
    fn scalar_product_le_const_rejects_unrelated_repair_target() {
        let mut manifest = scalar_product_manifest();
        let mut extra = manifest.kernels[0].args[1].clone();
        extra.index = 2;
        extra.name = "other".to_string();
        manifest.kernels[0].args.push(extra);
        manifest.kernels[0].constraints = vec![Constraint::ScalarProductLeConst {
            lhs_arg: 0,
            lhs_path: None,
            rhs_arg: 1,
            rhs_path: None,
            value: 2048,
            repair_arg: 2,
            repair_path: None,
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("repair target must match lhs or rhs"));
    }

    #[test]
    fn scalar_product_le_const_rejects_unsatisfiable_lower_bounds() {
        let mut manifest = scalar_product_manifest();
        manifest.kernels[0].constraints = vec![Constraint::ScalarProductLeConst {
            lhs_arg: 0,
            lhs_path: None,
            rhs_arg: 1,
            rhs_path: None,
            value: 0,
            repair_arg: 1,
            repair_path: None,
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("unsatisfiable scalar_product_le_const"));
    }

    #[test]
    fn expression_compare_rejects_mismatched_resize_target() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].constraints = vec![Constraint::ExpressionCompare {
            lhs: ConstraintExpr::ArgValue { arg: 2, path: None },
            op: ConstraintOp::Le,
            rhs: ConstraintExpr::PayloadLen { arg: 0, path: None },
            repair: ConstraintRepair::ResizePayload { arg: 1, path: None },
        }];

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("resize_payload must target compared payload_len"));
    }

    #[test]
    fn bytes_pattern_is_reapplied_after_pointer_mutation() {
        let manifest: KernelManifest = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "kernels": [
                {
                  "symbol_name": "pattern_kernel",
                  "display_name": "pattern_kernel",
                  "args": [
                    {
                      "index": 0,
                      "name": "buffer",
                      "type": "uint8_t *",
                      "kind": "pointer",
                      "pointer_role": "payload_buffer",
                      "pointee_layout": {
                        "index": "buffer.*",
                        "name": "$pointee",
                        "type": "uint8_t",
                        "kind": "scalar",
                        "size_bytes": 1,
                        "align_bytes": 1
                      },
                      "size_bytes": 8,
                      "align_bytes": 8,
                      "domain": {
                        "kind": "bytes",
                        "min_len": "8",
                        "max_len": "8",
                        "elem_size_bytes": 1,
                        "nullable": false,
                        "pattern_hex": "aA05"
                      }
                    }
                  ]
                }
              ]
            }"#,
        )
        .unwrap();
        let spec = ArgPackSpec::from_manifest(manifest).unwrap();

        let seed = spec.pack_values(vec![ArgValue::Pointer(Vec::new())]);
        let seed_values = spec.parse(&seed).unwrap();
        assert_eq!(
            pointer(&seed_values.values, 0),
            &[0xaa, 0x05, 0xaa, 0x05, 0xaa, 0x05, 0xaa, 0x05]
        );

        let mutated = spec.mutate(&seed, 2, 0xff);
        let mutated_values = spec.parse(&mutated).unwrap();
        assert_eq!(
            pointer(&mutated_values.values, 0),
            &[0xaa, 0x05, 0xaa, 0x05, 0xaa, 0x05, 0xaa, 0x05]
        );
    }

    #[test]
    fn bytes_pattern_rejects_invalid_hex() {
        let mut manifest: KernelManifest = serde_json::from_str(TEST_MANIFEST).unwrap();
        manifest.kernels[0].args[0].domain = serde_json::from_str(
            r#"{"kind":"bytes","min_len":"4","max_len":"8","pattern_hex":"xyz"}"#,
        )
        .unwrap();

        let err = ArgPackSpec::from_manifest(manifest).unwrap_err();

        assert!(err.contains("pattern_hex"));
    }

    #[test]
    fn public_pack_wrapper_uses_manifest_constraints() {
        let manifest_path =
            std::env::temp_dir().join(format!("rapid-arg-pack-test-{}.json", std::process::id()));
        std::fs::write(&manifest_path, TEST_MANIFEST).unwrap();
        let init_result = init_arg_pack_manifest(&manifest_path);
        let packed = pack_arg_pack_v1(b"ABC", b"XY", 99);
        let _ = std::fs::remove_file(manifest_path);
        assert_eq!(normalize_arg_pack_v1(&packed), packed);
        if init_result.is_ok() {
            assert_eq!(&packed[32..40], &(2u64.to_le_bytes()));
        }
    }
}
