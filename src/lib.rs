use cust::context::Context;
use cust::memory::DeviceCopy;
use cust::module::Module;
use cust::stream::*;
use lazy_static::lazy_static;
use std::ffi::CString;
use std::sync::Mutex;

mod tensor;
pub mod vector;
pub mod matrix;
pub use tensor::Tensor;
pub use vector::Vector;
pub use matrix::Matrix;

struct CudaCtx {
    stream: Stream,
    vector: Module,
    _ctx: Context,
}

lazy_static! {
    pub(crate) static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
}

impl Default for CudaCtx {
    fn default() -> Self {
        let context = cust::quick_init().unwrap();
        let ptx = CString::new(include_str!("../kernels/vector.ptx")).unwrap();
        let module = Module::from_ptx_cstr(&ptx, &[]).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        Self {
            vector: module,
            stream,
            _ctx: context,
        }
    }
}

//pub type Matrix<T> = Tensor<T, 2>;

pub(crate) fn calc_grid_size<T, const R: usize>(t: &Tensor<T, R>) -> (u32, u32)
where T: DeviceCopy
{
    let block_size = 256; // or 128, 512 based on occupancy tuning
    let grid_size = (t.inner().len() as u32 + block_size - 1) / block_size;

    (block_size, grid_size)
}


#[cfg(test)]
mod tests {
    use crate::{Tensor, tensor, Vector};
    use rand::distr::Uniform;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand::Rng;


    fn generate_data() -> Vec<f32> {
        let size: usize = 10_000_000;
        let mut rng = StdRng::from_os_rng();
        let uniform = Uniform::new(-1.0f32, 1.0).unwrap();

        (0..size)
            .map(|_| rng.sample(&uniform))
            .collect()
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
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = tensor!([3, 1][1.0, 2.0, 3.0]);
        let v2 = tensor!([2, 1][2.0, 4.0]);

        let _ = v1 + v2;
    }
}
