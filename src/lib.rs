use cust::context::Context;
use cust::memory::DeviceCopy;
use cust::module::Module;
use cust::stream::*;
use lazy_static::lazy_static;
use std::ffi::CString;
use std::sync::Mutex;

pub mod matrix;
mod tensor;
pub mod vector;
pub use matrix::Matrix;
pub use tensor::Tensor;
pub use vector::Vector;

struct CudaCtx {
    stream: Stream,
    vector: Module,
    matrix: Module,
    _ctx: Context,
}

type Dimension = (u32, u32, u32);

lazy_static! {
    pub(crate) static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
}

impl Default for CudaCtx {
    fn default() -> Self {
        let context = cust::quick_init().unwrap();

        let ptx_vector = CString::new(include_str!("../kernels/vector.ptx")).unwrap();
        let module_vector = Module::from_ptx_cstr(&ptx_vector, &[]).unwrap();
        let ptx_matrix = CString::new(include_str!("../kernels/matrix.ptx")).unwrap();
        let module_matrix = Module::from_ptx_cstr(&ptx_matrix, &[]).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        Self {
            vector: module_vector,
            matrix: module_matrix,
            stream,
            _ctx: context,
        }
    }
}

//pub type Matrix<T> = Tensor<T, 2>;

pub(crate) fn calc_grid_size<T>(t1: &Tensor<T, 2>, t2: &Tensor<T, 2>) -> (Dimension, Dimension)
where
    T: DeviceCopy,
{
    match t1.shape() {
        [1, _] | [_, 1] => {

            let bs = 256;
            let gs = (t1.inner().len() as u32 + bs - 1) / bs;
            return ((bs, 1, 1), (gs, 1, 1))
        }
        _ =>  {
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
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor, Tensor, Vector};
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
        let v1 = tensor!([3, 1][1.0f32, 2.0, 3.0]);
        let v2 = tensor!([3, 1][2.0, 4.0, 6.0]);

        let v3 = v1.clone() + v2.clone();

        let v4 = v1 * v2.transpose();

        assert_eq!(v3, tensor!([3,1][3.0, 6.0, 9.0]));
        assert_eq!(v3.scale(10.0), tensor!([3,1][30.0, 60.0, 90.0]));
        assert_eq!(v4, tensor!([1, 1][28.0]));

        // Bigger vectors
        let v5 = Vector::from(generate_data());
        let v6 = Vector::from(generate_data());
        let _ = v5 * v6.transpose();
    }

    #[test]
    fn matrices() {
        let m1 = tensor!([2, 2][1.0f32, 2.0, 3.0, 4.0]);
        let m2 = tensor!([2, 2][2.0, 3.0, 4.0, 5.0]);

        let m3 = m1 * m2;
        println!("{:#?}", m3)
    } 

    #[test]
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = tensor!([3, 1][1.0, 2.0, 3.0]);
        let v2 = tensor!([2, 1][2.0, 4.0]);

        let _ = v1 + v2;
    }
}
