//#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;


#[pyfunction]
fn bubble_sort(mut arr: Vec<u64>) -> Vec<u64> {
    let n = arr.len();
    if n < 2 {
        return arr;
    }

    for i in 0..n {
        let mut swapped = false;

        // The last i elements are already in place
        for j in 0..(n - i - 1) {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                swapped = true;
            }
        }

        // If no swaps occurred, the list is already sorted
        if !swapped {
            break;
        }
    }

    arr
}


pub fn register(_py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bubble_sort, module)?)?;
    Ok(())
}
