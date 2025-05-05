use cust::context::Context;
use cust::module::Module;
use cust::stream::*;
use lazy_static::lazy_static;
use std::ffi::CString;
use std::sync::Mutex;

pub mod tensor;
pub mod vector;
pub use tensor::Tensor;
pub use vector::Vector;

struct CudaCtx {
    stream: Stream,
    module: Module,
    _ctx: Context,
}

lazy_static! {
    pub(crate) static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
}

impl Default for CudaCtx {
    fn default() -> Self {
        let context = cust::quick_init().unwrap();
        let ptx = CString::new(include_str!("../kernels/vec_add.ptx")).unwrap();
        let module = Module::from_ptx_cstr(&ptx, &[]).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        Self {
            module,
            stream,
            _ctx: context,
        }
    }
}

pub type Matrix<T> = Tensor<T, 2>;

#[macro_export]
macro_rules! tensor {
    ([$($shape:expr),*] [ $($elem:expr),* $(,)? ]) => {{
        let data = vec![$($elem),*];
        const SHAPE: &[usize] = &[$($shape),*];
        Tensor::<_, { SHAPE.len() }>::from(([ $($shape),* ], data))
    }};
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn vec_add() {
        let v1 = tensor!([3, 1][1.0f32, 2.0, 3.0]);
        let v2 = tensor!([3, 1][2.0, 4.0, 6.0]);

        let v3 = v1 + v2;
        assert_eq!(v3, tensor!([3,1][3.0, 6.0, 9.0]));

        //let s1: Scalar<f32> = 10.0.into();
        //assert_eq!(s1 * v3, tensor!([3,1][30.0, 60.0, 90.0]));
    }

    #[test]
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = tensor!([3, 1][1.0, 2.0, 3.0]);
        let v2 = tensor!([2, 1][2.0, 4.0]);

        let _ = v1 + v2;
    }
}
