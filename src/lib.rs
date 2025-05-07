#![feature(generic_const_exprs, core_intrinsics)]

use cust::context::Context;
use cust::memory::DeviceCopy;
use cust::module::Module;
use cust::stream::*;
use lazy_static::lazy_static;
use std::any::TypeId;
use std::ffi::CString;
use std::intrinsics::type_id;
use std::{env, fs};
use std::path::PathBuf;
use std::sync::Mutex;

pub mod matrix;
pub mod vector;
pub mod ops;
mod tensor;

pub use matrix::Matrix;
pub use tensor::Tensor;
pub use vector::Vector;

struct CudaCtx {
    stream: Stream,
    vector: Module,
    tensor: Module,
    matrix: Module,
    _ctx: Context,
}

type Dimension = (u32, u32, u32);

lazy_static! {
    pub(crate) static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
}

fn load_ptx(src: &str) -> String {
    let out_dir = env::var("OUT_DIR").unwrap();
    let ptx_path = PathBuf::from(format!("{}/{}", out_dir, src));
    fs::read_to_string(ptx_path).expect("Failed to read compiled PTX")
}

pub(crate) fn get_cuda_type<T: DeviceCopy + 'static>() -> &'static str {
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() {
        return "_f32";
    }

    if t == TypeId::of::<f64>() {
        return "_f64";
    }

    return "_f32"
}

impl Default for CudaCtx {
    fn default() -> Self {
        let context = cust::quick_init().unwrap();

        let ptx_vector = CString::new(load_ptx("vector.ptx")).unwrap();
        let module_vector = Module::from_ptx_cstr(&ptx_vector, &[]).unwrap();

        let ptx_matrix = CString::new(load_ptx("matrix.ptx")).unwrap();
        let module_matrix = Module::from_ptx_cstr(&ptx_matrix, &[]).unwrap();

        let ptx_tensor = CString::new(load_ptx("tensor.ptx")).unwrap();
        let module_tensor = Module::from_ptx_cstr(&ptx_tensor, &[]).unwrap();

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

pub(crate) fn calc_grid_size<T>(t1: &Tensor<T, 2>, t2: &Tensor<T, 2>) -> (Dimension, Dimension)
where
    T: DeviceCopy,
{
    let bs = (16, 16, 1);

    let s1 = t1.shape();
    let s2 = t2.shape();
    let gs = (
        (s2[1] as usize + bs.0 as usize - 1) as u32 / bs.0,
        (s1[0] as usize + bs.1 as usize - 1) as u32 / bs.1,
        1
    );
    (bs, gs)
}

#[cfg(test)]
mod tests {
    use crate::Vector;
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

        let v3 = (v1.clone() + v2.clone()).cpu();
        assert_eq!(v3, tensor!([3][3.0, 6.0, 9.0]));
        assert_eq!(v3.clone().scale(10.0).cpu(), tensor!([3][30.0, 60.0, 90.0]));

        // Currently returns a scalar, maybe I'll change it to a Tensor<T, 1> later.
        let v4 = v1 * v2;
        assert_eq!(v4, 28.0);

        // Bigger vectors
        let v5 = Vector::from(generate_data());
        let v6 = Vector::from(generate_data());
        let v7 = v5 * v6;

        println!("{:#?}", v7);
    }

    #[test]
    fn matrices() {
        let m1 = tensor!([2, 2][1.0f32, 2.0, 3.0, 4.0]);
        let m2 = tensor!([2, 2][2.0, 3.0, 4.0, 5.0]);

        let m3 = (m1.clone() * m2.clone()).cpu();
        assert_eq!(m3, tensor!([2, 2][10.0, 13.0, 22.0, 29.0]));
        assert_eq!(m3.at([0, 1]).value(), 13.0);

        let m4 = (m1 + m2).cpu();
        println!("{:#?}", m4);
        assert_eq!(m4, tensor!([2, 2][3.0, 5.0, 7.0, 9.0]));
    } 

    #[test]
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = tensor!([3, 1][1.0, 2.0, 3.0]);
        let v2 = tensor!([2, 1][2.0, 4.0]);

        let _ = v1 + v2;
    }
}
