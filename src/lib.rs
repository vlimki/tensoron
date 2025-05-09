#![allow(internal_features, incomplete_features)]
#![feature(generic_const_exprs, core_intrinsics)]

use cust::context::Context;
use cust::module::Module;
use cust::stream::*;
use lazy_static::lazy_static;
use std::ffi::CString;
use std::sync::Mutex;
use std::env;

pub const VECTOR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vector.ptx"));
pub const MATRIX_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/matrix.ptx"));
pub const TENSOR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tensor.ptx"));

pub mod matrix;
pub mod ops;
mod tensor;
pub mod vector;
pub mod kernel;

pub(crate) use kernel::Kernel;
pub use matrix::Matrix;
pub use tensor::{Tensor, TensorView};
pub use vector::Vector;

pub struct CudaCtx {
    stream: Stream,
    vector: Module,
    tensor: Module,
    matrix: Module,
    _ctx: Context,
}


lazy_static! {
    pub static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
}

impl Default for CudaCtx {
    fn default() -> Self {
        let context = cust::quick_init().unwrap();

        let module_vector = Module::from_ptx_cstr(&CString::new(VECTOR_PTX).unwrap(), &[]).unwrap();
        let module_matrix = Module::from_ptx_cstr(&CString::new(MATRIX_PTX).unwrap(), &[]).unwrap();
        let module_tensor = Module::from_ptx_cstr(&CString::new(TENSOR_PTX).unwrap(), &[]).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        Self {
            vector: module_vector,
            matrix: module_matrix,
            tensor: module_tensor,
            stream,
            _ctx: context,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{ops::ML, Vector};
    use crate::{tensor, Tensor};
    use rand::distr::Uniform;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    fn generate_data() -> Vec<f32> {
        let size: usize = 10_000_000;
        let mut rng = StdRng::from_os_rng();
        let uniform = Uniform::new(-1.0f32, 1.0).unwrap();

        (0..size).map(|_| rng.sample(&uniform)).collect()
    }

    #[test]
    fn vectors() {
        let v1 = tensor!([3][1.0f32, 2.0, 3.0]);
        let v2 = tensor!([3][2.0, 4.0, 6.0]);

        let v3 = (&v1 + &v2).cpu();
        assert_eq!(v3, tensor!([3][3.0, 6.0, 9.0]));

        // Chain operations + ML functions like tanh, relu, sigmoid
        assert_eq!(
            v3.clone().scale(10.0).tanh().cpu(),
            tensor!([3][30.0, 60.0, 90.0]).tanh().cpu()
        );

        let v4: f32 = v1 * v2;
        assert_eq!(v4, 28.0);

        // Bigger vectors
        // The library also has support for multiple number types (currently just f64 and f32)
        // It will compile a CUDA kernel for each type the user is using.
        let v5: Vector<f32> = Vector::from(generate_data());
        let v6: Vector<f32> = Vector::from(generate_data());
        let v7 = v5 * v6;

        println!("{:#?}", v7);
    }

    #[test]
    fn matrices() {
        let m1 = tensor!([2, 2][1.0f32, 2.0, 3.0, 4.0]);
        let m2 = tensor!([2, 2][2.0f32, 3.0, 4.0, 5.0]);

        let m3 = (&m1 * &m2).cpu();
        println!("{:#?}", m3);
        assert_eq!(m3, tensor!([2, 2][10.0, 13.0, 22.0, 29.0]));
        assert_eq!(m3.view().at([0, 1]).value(), 13.0);

        let m4 = (m1 + m2).cpu();
        println!("{:#?}", m4.clone().transpose().cpu());
        assert_eq!(m4.transpose().cpu(), tensor!([2, 2][3.0, 7.0, 5.0, 9.0]));
    }
}
