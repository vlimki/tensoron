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

#[derive(Debug)]
pub struct Vector<T>
where
    T: DeviceCopy,
{
    _device_ptr: Option<DeviceBuffer<T>>,
    _inner: Vec<T>,
}

impl<T: DeviceCopy> From<Vec<T>> for Vector<T> {
    fn from(v: Vec<T>) -> Self {
        Self {
            _inner: v,
            _device_ptr: None,
        }
    }
}

impl<T: DeviceCopy + PartialEq> PartialEq for Vector<T> {
    fn eq(&self, other: &Self) -> bool {
        self._inner == other._inner
    }
}

impl<T: DeviceCopy> Drop for Vector<T> {
    fn drop(&mut self) {
        let _ = self._device_ptr.take();
    }
}

impl<T> Add for Vector<T>
where
    T: DeviceCopy + Zeroable,
{
    type Output = Vector<T>;

    fn add(mut self, mut other: Self) -> Self {
        assert_eq!(self._inner.len(), other._inner.len());
        let _ctx = cust::quick_init().unwrap();

        self._device_ptr = Some(DeviceBuffer::from_slice(&self._inner).unwrap());
        other._device_ptr = Some(DeviceBuffer::from_slice(&other._inner).unwrap());

        let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(self._inner.len()).unwrap();

        let ctx = CUDA_CTX.lock().unwrap();
        let (module, stream) = (&ctx.module, &ctx.stream);

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

        return Vector {
            _device_ptr: None,
            _inner: Vec::from(host_out),
        };
    }
}

#[macro_export]
macro_rules! vector {
    [ $($elem:expr),* $(,)? ] => {{
        let tmp = vec![$($elem),*];
        Vector::from(tmp)
    }};

    ($elem:expr ; $count:expr) => {{
        let tmp = vec![$elem; $count];
        Vector::from(tmp)
    }};
}

#[cfg(test)]
mod tests {
    use crate::Vector;

    #[test]
    fn vec_add() {
        let v1 = vector![1.0f32, 2.0, 3.0];
        let v2 = vector![2.0, 4.0, 6.0];

        let v3 = v1 + v2;
        assert_eq!(v3, vector![3.0, 6.0, 9.0]);
    }

    #[test]
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = vector![1.0, 2.0, 3.0];
        let v2 = vector![2.0, 4.0];

        let v3 = v1 + v2;
    }
}
