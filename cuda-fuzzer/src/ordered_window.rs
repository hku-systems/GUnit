use std::{
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
};

#[derive(Debug)]
pub struct OrderedWindow<I, R> {
    capacity: NonZeroUsize,
    pending: VecDeque<(u64, I)>,
    ready: HashMap<u64, R>,
}

impl<I, R> OrderedWindow<I, R> {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            pending: VecDeque::with_capacity(capacity.get()),
            ready: HashMap::with_capacity(capacity.get()),
        }
    }

    pub fn can_submit(&self) -> bool {
        self.pending.len() < self.capacity.get()
    }

    pub fn push_pending(&mut self, task_id: u64, input: I) -> Result<(), String> {
        if task_id == 0 {
            return Err("task ID must be nonzero".to_string());
        }
        if !self.can_submit() {
            return Err("ordered window is full".to_string());
        }
        if self
            .pending
            .iter()
            .any(|(pending_id, _)| *pending_id == task_id)
        {
            return Err(format!("duplicate pending task ID {task_id}"));
        }

        self.pending.push_back((task_id, input));
        Ok(())
    }

    pub fn record_ready(&mut self, task_id: u64, result: R) -> Result<(), String> {
        if !self
            .pending
            .iter()
            .any(|(pending_id, _)| *pending_id == task_id)
        {
            return Err(format!("completion references unknown task ID {task_id}"));
        }
        if self.ready.contains_key(&task_id) {
            return Err(format!("duplicate completion for task ID {task_id}"));
        }

        self.ready.insert(task_id, result);
        Ok(())
    }

    pub fn pop_ready_head(&mut self) -> Option<(u64, I, R)> {
        let task_id = self.pending.front().map(|(task_id, _)| *task_id)?;
        let result = self.ready.remove(&task_id)?;
        let (pending_id, input) = self
            .pending
            .pop_front()
            .expect("ordered window head disappeared");
        debug_assert_eq!(pending_id, task_id);
        Some((task_id, input, result))
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}
