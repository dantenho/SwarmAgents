mod cache;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute int8 cosine similarity between two vectors.
#[pyfunction]
fn cosine_similarity_int8(a: Vec<i8>, b: Vec<i8>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(
            "vectors must have the same length",
        ));
    }
    let mut dot = 0i64;
    let mut na = 0i64;
    let mut nb = 0i64;
    for i in 0..a.len() {
        dot += a[i] as i64 * b[i] as i64;
        na += a[i] as i64 * a[i] as i64;
        nb += b[i] as i64 * b[i] as i64;
    }
    if na == 0 || nb == 0 {
        return Ok(0.0);
    }
    Ok((dot as f64) / ((na as f64).sqrt() * (nb as f64).sqrt()))
}

/// Compute int8 cosine similarity for aligned batches, rayon-parallel.
#[pyfunction]
fn cosine_batch_int8(a: Vec<Vec<i8>>, b: Vec<Vec<i8>>) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("batches must be aligned"));
    }
    Ok(a.par_iter()
        .zip(&b)
        .map(|(x, y)| {
            let mut dot = 0i64;
            let mut nx = 0i64;
            let mut ny = 0i64;
            for i in 0..x.len() {
                dot += x[i] as i64 * y[i] as i64;
                nx += x[i] as i64 * x[i] as i64;
                ny += y[i] as i64 * y[i] as i64;
            }
            if nx == 0 || ny == 0 {
                0.0
            } else {
                (dot as f64) / ((nx as f64).sqrt() * (ny as f64).sqrt())
            }
        })
        .collect())
}

#[pymodule]
fn swarmagents_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity_int8, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_batch_int8, m)?)?;
    cache::register(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_one() {
        let v = vec![1i8, 2, 3, 4];
        let s = cosine_similarity_int8(v.clone(), v).unwrap();
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_vector_is_zero() {
        let z = vec![0i8; 4];
        let o = vec![1i8, 2, 3, 4];
        assert_eq!(cosine_similarity_int8(z, o).unwrap(), 0.0);
    }

    #[test]
    fn length_mismatch_errors() {
        let a = vec![1i8, 2];
        let b = vec![1i8, 2, 3];
        assert!(cosine_similarity_int8(a, b).is_err());
    }

    #[test]
    fn batch_matches_single() {
        let a = vec![vec![1i8, 2, 3], vec![4, 5, 6]];
        let b = vec![vec![3i8, 2, 1], vec![6, 5, 4]];
        let out = cosine_batch_int8(a.clone(), b.clone()).unwrap();
        for i in 0..2 {
            let single = cosine_similarity_int8(a[i].clone(), b[i].clone()).unwrap();
            assert!((out[i] - single).abs() < 1e-12);
        }
    }

    #[test]
    fn cache_put_get_evict() {
        let c = crate::cache::Int8Cache::new(2).unwrap();
        c.put("a".into(), vec![1, 2]).unwrap();
        c.put("b".into(), vec![3, 4]).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.get("a"), Some(vec![1, 2]));
        c.put("c".into(), vec![5, 6]).unwrap();
        assert!(c.len() <= 2);
        assert_eq!(c.capacity(), 2);
    }
}
