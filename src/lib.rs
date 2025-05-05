use bytemuck::Zeroable;
use cust::context::Context;
use cust::launch;
use cust::memory::*;
use cust::module::Module;
use cust::stream::*;
use lazy_static::lazy_static;
use std::ffi::CString;
use std::ops::Add;
use std::ops::Mul;
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

pub type Scalar<T> = Tensor<T, 0>;

impl<T: DeviceCopy> From<Vec<T>> for Vector<T> {
    fn from(v: Vec<T>) -> Self {
        Tensor::from(([v.len(), 1], v))
    }
}

impl<T: DeviceCopy + PartialEq> PartialEq for Vector<T> {
    fn eq(&self, other: &Self) -> bool {
        self._inner == other._inner
    }
}

impl<T> Mul<Scalar<T>> for Vector<T> 
where 
    T: DeviceCopy + Mul<Output = T>
{
    type Output = Self;
    fn mul(mut self, rhs: Scalar<T>) -> Self::Output {
        self._inner = self._inner.iter().map(|x| *x * rhs._inner[0]).collect();
        self
    }
}

impl<T> Add for Vector<T>
where
    T: DeviceCopy + Zeroable,
{
    type Output = Vector<T>;

    fn add(mut self, mut other: Self) -> Self {
        assert_eq!(self._shape, other._shape);

        let _ctx = cust::quick_init().unwrap();

        self._device_ptr = Some(DeviceBuffer::from_slice(&self._inner).unwrap());
        other._device_ptr = Some(DeviceBuffer::from_slice(&other._inner).unwrap());

        let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(self._inner.len()).unwrap();

        let ctx = CUDA_CTX.lock().unwrap();
        let CudaCtx { ref module, ref stream, .. } = *ctx;

        unsafe {
            let result = launch!(module.vec_add<<<1, 3, 0, stream>>>(
                self._device_ptr.as_ref().unwrap().as_device_ptr(),
                other._device_ptr.as_ref().unwrap().as_device_ptr(),
                c_out.as_device_ptr(),
                self._inner.len() as std::os::raw::c_int,
            ));
            result.unwrap()
        }

        let mut host_out = vec![T::zeroed(); self._inner.len()];
        c_out.copy_to(&mut host_out[..]).unwrap();

        ctx.stream.synchronize().unwrap();

        return Self {
            _device_ptr: None,
            _shape: self._shape,
            _inner: Vec::from(host_out),
        };
    }
}

impl<T: DeviceCopy> Vector<T> {
    pub fn with_shape(shape: Vec<usize>, data: Vec<T>) -> Self {
        assert_eq!(shape.len(), 2, "Vectors should be constructed with two dimensions.");
        Self {
            _device_ptr: None,
            _shape: [shape[0], shape[1]],
            _inner: data,
        }
    }
}

#[macro_export]
macro_rules! vector {
    ([$($shape:expr),*] [ $($elem:expr),* $(,)? ]) => {{
        let tmp = vec![$($elem),*];
        let shapes = vec![$($shape),*];
        Vector::with_shape(shapes, tmp)
    }};
}

#[cfg(test)]
mod tests {
    use crate::Vector;

    #[test]
    fn vec_add() {
        let v1 = vector!([3, 1][1.0f32, 2.0, 3.0]);
        let v2 = vector!([3, 1][2.0, 4.0, 6.0]);

        let v3 = v1 + v2;
        assert_eq!(v3, vector!([3,1][3.0, 6.0, 9.0]));
    }

    #[test]
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = vector!([3, 1][1.0, 2.0, 3.0]);
        let v2 = vector!([2, 1][2.0, 4.0]);

        let _ = v1 + v2;
    }
}
