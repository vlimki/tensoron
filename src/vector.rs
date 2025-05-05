use std::ops::Add;
use crate::tensor::Tensor;
use bytemuck::Zeroable;
use cust::launch;
use cust::memory::*;
use crate::{CudaCtx, CUDA_CTX};

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
