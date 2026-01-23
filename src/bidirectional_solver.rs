//#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

#[pyfunction]
fn bidirectional_solver(
    initial_permutation: Vec<u32>,
    actions: Vec<Vec<u32>>,  // Enumerated 0..n_actions
    pattern: Vec<u8>,
    min_search_depth: u32,
    max_search_depth: u32,
    n_solutions: u32,
) -> Option<Vec<Vec<u32>>> {
    println!("Initial permutation: {:?}", initial_permutation);
    println!("Actions: {:?}", actions);
    println!("Pattern: {:?}", pattern);
    println!("Search depth: {min_search_depth} - {max_search_depth}");
    println!("Number of solutions: {:?}", n_solutions);
    Some(vec![vec![0; 1]])
}


pub fn register(_py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bidirectional_solver, module)?)?;
    Ok(())
}
