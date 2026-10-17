use pyo3::prelude::*;

/// Compute int8 cosine similarity between two 384-dim vectors.
#[pyfunction]
fn cosine_similarity_int8(a: Vec<i8>, b: Vec<i8>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
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

#[pymodule]
fn swarmagents_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity_int8, m)?)?;
    Ok(())
}
