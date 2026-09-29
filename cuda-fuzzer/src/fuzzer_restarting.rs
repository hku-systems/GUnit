use libafl::{
    events::{EventConfig, LlmpRestartingEventManager, LlmpShouldSaveState, RestartingMgr},
    monitors::Monitor,
    Error,
};
use libafl_bolts::{
    shmem::{ShMemProvider, StdShMem, StdShMemProvider},
    tuples::tuple_list,
};
use serde::{de::DeserializeOwned, Serialize};

pub const RESTART_STATE_POLICY: LlmpShouldSaveState = LlmpShouldSaveState::OOMSafeOnRestart;

#[expect(clippy::type_complexity)]
pub fn setup_oom_safe_restarting_mgr<I, MT, S>(
    monitor: MT,
    broker_port: u16,
    configuration: EventConfig,
) -> Result<
    (
        Option<S>,
        LlmpRestartingEventManager<(), I, S, StdShMem, StdShMemProvider>,
    ),
    Error,
>
where
    I: DeserializeOwned,
    MT: Monitor + Clone,
    S: Serialize + DeserializeOwned,
{
    RestartingMgr::builder()
        .shmem_provider(StdShMemProvider::new()?)
        .monitor(Some(monitor))
        .broker_port(broker_port)
        .configuration(configuration)
        .serialize_state(RESTART_STATE_POLICY)
        .hooks(tuple_list!())
        .build()
        .launch()
}
