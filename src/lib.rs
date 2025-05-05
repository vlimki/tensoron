use bytemuck::Zeroable;
use cust::context::Context;
use cust::launch;
use cust::memory::*;
use cust::module::Module;
use cust::stream::*;
use lazy_static::lazy_static;
use std::ffi::CString;
use std::ops::Add;
use std::sync::Mutex;

pub mod tensor;
pub use tensor::Tensor;

struct CudaCtx {
    stream: Stream,
    module: Module,
    _ctx: Context,
}

lazy_static! {
    static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
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

/// To separate column vectors and row vectors, we have the dimensions as either [1, N] or [N, 1].
pub type Vector<T> = Tensor<T, 2>;

impl<T: DeviceCopy> From<Vec<T>> for Vector<T> {
    fn from(v: Vec<T>) -> Self {
        Tensor::from(([v.len(), 1], v))
    }
}

impl<T> Add for Vector<T>
where
    T: DeviceCopy + Zeroable,
{
    type Output = Vector<T>;

    fn add(mut self, mut other: Self) -> Self {
        assert_eq!(self.shape(), other.shape());

        let _ctx = cust::quick_init().unwrap();

        self.to_device();
        other.to_device();

        let len = self.shape()[0].max(self.shape()[1]);
        let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(len).unwrap();

        let ctx = CUDA_CTX.lock().unwrap();
        let CudaCtx { ref module, ref stream, .. } = *ctx;

        unsafe {
            let result = launch!(module.vec_add<<<1, 3, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                other.device_ptr().as_ref().unwrap().as_device_ptr(),
                c_out.as_device_ptr(),
                len as std::os::raw::c_int,
            ));
            result.unwrap()
        }

        let mut host_out = vec![T::zeroed(); len];
        c_out.copy_to(&mut host_out[..]).unwrap();

        ctx.stream.synchronize().unwrap();

        return Tensor::from((self.shape(), Vec::from(host_out)));
    }
}

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
