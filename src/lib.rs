//#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod bubble_sort;
mod bidirectional_solver;

#[pyfunction]
fn add_one(a: i32, n: u32) -> i32 {
    let mut b = a;
    for _ in 0..n {
        b += 1;
    }
    b
}


#[pyfunction]
fn maskify(cc: &str) -> String {
    let mask_length = cc.len().saturating_sub(4);
    "#".repeat(mask_length) + &cc[mask_length..]
}

#[pymodule]
fn rust(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(add_one, module)?)?;
    module.add_function(wrap_pyfunction!(maskify, module)?)?;

    // Register modules
    bubble_sort::register(py, module)?;
    bidirectional_solver::register(py, module)?;
    Ok(())
}
