use crate::tensor::Tensor;
use crate::{calc_grid_size, CudaCtx, CUDA_CTX};
use bytemuck::Zeroable;
use cust::launch;
use cust::memory::*;
use std::ops::{Add, Mul};

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

        let ctx = CUDA_CTX.lock().unwrap();

        self.to_device();
        other.to_device();

        let len = self.shape()[0].max(self.shape()[1]);
        let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(len).unwrap();

        let CudaCtx {
            ref vector,
            ref stream,
            ..
        } = *ctx;

        let (bs, gs) = calc_grid_size(&self);

        unsafe {
            let result = launch!(vector.add<<<gs, bs, 0, stream>>>(
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

impl<T> Mul for Vector<T>
where
    T: DeviceCopy + Zeroable,
{
    type Output = T;

    fn mul(mut self, mut other: Self) -> T {
        assert_eq!(self.shape(), other.shape());

        let ctx = CUDA_CTX.lock().unwrap();

        self.to_device();
        other.to_device();

        let len = self.shape()[0].max(self.shape()[1]);
        let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(1).unwrap();

        let CudaCtx {
            ref vector,
            ref stream,
            ..
        } = *ctx;

        let (bs, gs) = calc_grid_size(&self);

        unsafe {
            let result = launch!(vector.dot_product<<<gs, bs, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                other.device_ptr().as_ref().unwrap().as_device_ptr(),
                c_out.as_device_ptr(),
                len as std::os::raw::c_int,
            ));
            result.unwrap()
        }

        let mut host_out = vec![T::zeroed(); 1];
        c_out.copy_to(&mut host_out[..]).unwrap();

        ctx.stream.synchronize().unwrap();

        return host_out[0];
    }
}
