use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
    time::Duration,
};

use libafl::{
    corpus::Testcase,
    executors::ExitKind,
    feedbacks::{Feedback, StateInitializer},
    Error,
};
use libafl_bolts::Named;

#[derive(Debug)]
struct TaskTimeState {
    pending: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct TaskTimeFeedback {
    state: Arc<Mutex<TaskTimeState>>,
    name: Cow<'static, str>,
}

impl TaskTimeFeedback {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(TaskTimeState { pending: None })),
            name: Cow::Borrowed("task_time"),
        }
    }

    pub fn observe_ns(&self, exec_time_ns: u64) -> anyhow::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("task time feedback state lock is poisoned"))?;
        state.pending = Some(Duration::from_nanos(exec_time_ns));
        Ok(())
    }

    pub fn take_pending_duration(&self) -> anyhow::Result<Duration> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("task time feedback state lock is poisoned"))?;
        state
            .pending
            .take()
            .ok_or_else(|| anyhow::anyhow!("no pending task execution time"))
    }
}

impl Default for TaskTimeFeedback {
    fn default() -> Self {
        Self::new()
    }
}

impl Named for TaskTimeFeedback {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

impl<S> StateInitializer<S> for TaskTimeFeedback {}

impl<EM, I, OT, S> Feedback<EM, I, OT, S> for TaskTimeFeedback {
    #[cfg(feature = "track_hit_feedbacks")]
    fn last_result(&self) -> Result<bool, Error> {
        Ok(false)
    }

    fn is_interesting(
        &mut self,
        _state: &mut S,
        _manager: &mut EM,
        _input: &I,
        _observers: &OT,
        _exit_kind: &ExitKind,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn append_metadata(
        &mut self,
        _state: &mut S,
        _manager: &mut EM,
        _observers: &OT,
        testcase: &mut Testcase<I>,
    ) -> Result<(), Error> {
        let duration = self
            .take_pending_duration()
            .map_err(|err| Error::illegal_state(err.to_string()))?;
        *testcase.exec_time_mut() = Some(duration);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::TaskTimeFeedback;

    #[test]
    fn task_time_uses_the_duration_from_each_completion() {
        let feedback = TaskTimeFeedback::new();

        feedback.observe_ns(41).unwrap();
        assert_eq!(
            feedback.take_pending_duration().unwrap(),
            Duration::from_nanos(41)
        );

        feedback.observe_ns(97).unwrap();
        assert_eq!(
            feedback.take_pending_duration().unwrap(),
            Duration::from_nanos(97)
        );
    }

    #[test]
    fn task_time_consumes_each_completion_once() {
        let feedback = TaskTimeFeedback::new();
        feedback.observe_ns(11).unwrap();

        assert_eq!(
            feedback.take_pending_duration().unwrap(),
            Duration::from_nanos(11)
        );
        assert!(feedback.take_pending_duration().is_err());
    }
}
