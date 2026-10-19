use dashmap::DashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bounded concurrent string-keyed cache backed by DashMap shard locks.
#[pyclass]
struct Int8Cache {
    map: DashMap<String, Vec<i8>>,
    capacity: usize,
    size: AtomicUsize,
}

#[pymethods]
impl Int8Cache {
    #[new]
    #[pyo3(signature = (capacity = 1024))]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be positive"));
        }
        Ok(Int8Cache {
            map: DashMap::with_capacity(capacity),
            capacity,
            size: AtomicUsize::new(0),
        })
    }

    /// Store a vector under a key; evicts an arbitrary entry when full.
    pub fn put(&self, key: String, vector: Vec<i8>) -> PyResult<()> {
        if self.map.contains_key(&key) {
            self.map.insert(key, vector);
            return Ok(());
        }
        if self.size.load(Ordering::Relaxed) >= self.capacity {
            let victim = self.map.iter().next().map(|e| e.key().clone());
            if let Some(key) = victim {
                self.map.remove(&key);
                self.size.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.map.insert(key, vector);
        self.size.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Fetch the vector stored under a key, or None.
    pub fn get(&self, key: &str) -> Option<Vec<i8>> {
        self.map.get(key).map(|v| v.value().clone())
    }

    /// Current number of stored entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Maximum capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Int8Cache>()?;
    Ok(())
}
