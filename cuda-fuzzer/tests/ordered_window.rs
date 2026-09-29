use std::num::NonZeroUsize;

use cuda_fuzzer::ordered_window::OrderedWindow;

#[test]
fn window_opens_one_slot_only_after_head_retirement() {
    let mut window = OrderedWindow::new(NonZeroUsize::new(2).unwrap());

    window.push_pending(1, b"one".to_vec()).unwrap();
    window.push_pending(2, b"two".to_vec()).unwrap();
    assert!(!window.can_submit());

    window.record_ready(2, 22).unwrap();
    assert_eq!(window.pop_ready_head(), None);
    assert!(!window.can_submit());

    window.record_ready(1, 11).unwrap();
    assert_eq!(window.pop_ready_head(), Some((1, b"one".to_vec(), 11)));
    assert!(window.can_submit());
    assert_eq!(window.pending_len(), 1);
}

#[test]
fn one_slot_window_is_lockstep() {
    let mut window = OrderedWindow::new(NonZeroUsize::new(1).unwrap());

    window.push_pending(7, b"x".to_vec()).unwrap();
    assert!(!window.can_submit());

    window.record_ready(7, 70).unwrap();
    assert_eq!(window.pop_ready_head(), Some((7, b"x".to_vec(), 70)));
    assert!(window.can_submit());
    assert!(window.is_empty());
}

#[test]
fn window_rejects_invalid_task_transitions() {
    let mut window = OrderedWindow::new(NonZeroUsize::new(1).unwrap());

    assert_eq!(
        window.push_pending(0, b"zero".to_vec()).unwrap_err(),
        "task ID must be nonzero"
    );
    window.push_pending(1, b"one".to_vec()).unwrap();
    assert_eq!(
        window.push_pending(2, b"two".to_vec()).unwrap_err(),
        "ordered window is full"
    );
    assert_eq!(
        window.record_ready(2, 22).unwrap_err(),
        "completion references unknown task ID 2"
    );
    window.record_ready(1, 11).unwrap();
    assert_eq!(
        window.record_ready(1, 111).unwrap_err(),
        "duplicate completion for task ID 1"
    );
}
