use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static WORKDIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct FuzzerWorkDir {
    root: PathBuf,
    corpus_dir: PathBuf,
    crashes_dir: PathBuf,
}

impl FuzzerWorkDir {
    pub fn new(label: &str) -> io::Result<Self> {
        let counter = WORKDIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rapid-fuzzer-{label}-{}-{counter}",
            std::process::id()
        ));
        let corpus_dir = root.join("corpus");
        let crashes_dir = root.join("crashes");
        fs::create_dir_all(&corpus_dir)?;
        fs::create_dir_all(&crashes_dir)?;
        Ok(Self {
            root,
            corpus_dir,
            crashes_dir,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn corpus_dir(&self) -> &Path {
        &self.corpus_dir
    }

    pub fn crashes_dir(&self) -> &Path {
        &self.crashes_dir
    }
}

impl Drop for FuzzerWorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_workdir_uses_temp_corpus_and_crashes_dirs() {
        let root;
        {
            let workdir = FuzzerWorkDir::new("unit").unwrap();

            assert!(workdir.root().starts_with(std::env::temp_dir()));
            assert_ne!(workdir.corpus_dir(), std::path::Path::new("./corpus"));
            assert_ne!(workdir.crashes_dir(), std::path::Path::new("./crashes"));
            assert!(workdir.corpus_dir().is_dir());
            assert!(workdir.crashes_dir().is_dir());
            root = workdir.root().to_path_buf();
        }
        assert!(!root.exists());
    }
}
