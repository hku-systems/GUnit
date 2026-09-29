use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SubmissionContext {
    #[default]
    Unmarked,
    Canonical,
    Supply,
}

impl SubmissionContext {
    pub(crate) fn is_canonical(self) -> bool {
        !matches!(self, Self::Unmarked)
    }

    pub(crate) fn is_supply(self) -> bool {
        matches!(self, Self::Supply)
    }
}

thread_local! {
    static CURRENT: Cell<SubmissionContext> = const { Cell::new(SubmissionContext::Unmarked) };
}

pub(crate) fn current() -> SubmissionContext {
    CURRENT.get()
}

pub(crate) fn with_submission_context<T>(context: SubmissionContext, f: impl FnOnce() -> T) -> T {
    // The stage and executor run on the same submit thread; restore on unwind
    // so a canonical marker cannot leak into the next evaluation.
    let previous = CURRENT.replace(context);
    struct Restore(SubmissionContext);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.set(self.0);
        }
    }
    let _restore = Restore(previous);
    f()
}
