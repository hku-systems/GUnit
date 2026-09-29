use std::{
    env,
    fs::{File, OpenOptions},
    io::ErrorKind,
    net::{Ipv4Addr, TcpListener},
    path::PathBuf,
};

use fs2::FileExt;

#[derive(Debug)]
pub struct GpuClientGuard {
    _lock: File,
}

fn lock_path(identity: &str) -> PathBuf {
    let encoded = identity
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    env::temp_dir().join(format!("rapid-fuzzer-async-gpu-{encoded}.lock"))
}

pub(crate) fn try_acquire_for_identity(identity: &str) -> Result<GpuClientGuard, String> {
    let path = lock_path(identity);
    // Keep the file after unlock: unlinking an advisory-lock path lets racing
    // clients open different inodes and both acquire an exclusive lock.
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|error| format!("failed to open GPU client lock {}: {error}", path.display()))?;
    lock.try_lock_exclusive().map_err(|error| {
        format!(
            "GPU {identity} already has a RAPID async client; stop it or select a different CUDA_VISIBLE_DEVICES device ({error})"
        )
    })?;
    Ok(GpuClientGuard { _lock: lock })
}

pub fn try_acquire_for_existing_broker(broker_port: u16) -> Result<Option<GpuClientGuard>, String> {
    match TcpListener::bind((Ipv4Addr::LOCALHOST, broker_port)) {
        Ok(listener) => {
            drop(listener);
            Ok(None)
        }
        Err(error) if error.kind() == ErrorKind::AddrInUse => {
            let visible = env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "default".into());
            let identity = visible.split(',').next().unwrap_or("default").trim();
            try_acquire_for_identity(identity).map(Some)
        }
        Err(error) => Err(format!(
            "failed to probe restarting-manager port {broker_port}: {error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{lock_path, try_acquire_for_identity};

    #[test]
    fn second_client_for_same_gpu_is_rejected_until_first_exits() {
        let identity = format!(
            "test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let path = lock_path(&identity);
        let first = try_acquire_for_identity(&identity).unwrap();
        assert!(try_acquire_for_identity(&identity).is_err());
        drop(first);
        let second = try_acquire_for_identity(&identity).unwrap();
        drop(second);
        let _ = std::fs::remove_file(path);
    }
}
