use std::ops::Mul;
use cust::memory::bytemuck::Zeroable;
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy};

use crate::{ops::*, Kernel};
use crate::{CudaCtx, Tensor, CUDA_CTX};

pub type Vector<T> = Tensor<T, 1>;

impl<T: DeviceCopy + Zeroable + 'static> GpuMul for &Vector<T> {
    type Output = T;

    fn gpu_mul(self, rhs: Self) -> T {
        let ctx = CUDA_CTX.lock().unwrap();

        let a_dev = self.ptr();
        let b_dev = rhs.ptr();

        let output: DeviceBuffer<T> = DeviceBuffer::zeroed(1).unwrap();

        let CudaCtx {
            ref vector,
            ref stream,
            ..
        } = *ctx;

        let mut k: Kernel<'_, T> = Kernel::new(vector, "mul");
        k.tune_1d(self.shape()[0]);

        unsafe {
            k.launch(&stream,
                (a_dev.as_device_ptr(),
                b_dev.as_device_ptr(),
                output.as_device_ptr(),
                self.shape()[0])
            )
        }

        let mut host = vec![T::zeroed(); 1];

        output.copy_to(&mut host).unwrap();
        host[0]
    }
}

impl<T: DeviceCopy + Zeroable + 'static> Mul for &Vector<T> {
    type Output = T;

    fn mul(self, rhs: Self) -> Self::Output {
        self.gpu_mul(rhs)
    }
}
impl<T: DeviceCopy + Zeroable + 'static> Mul for Vector<T> {
    type Output = T;

    fn mul(self, rhs: Self) -> Self::Output {
        self.gpu_mul(&rhs)
    }
}


impl<T: DeviceCopy> From<Vec<T>> for Vector<T> {
    fn from(v: Vec<T>) -> Self {
        Tensor::from(([v.len()], v))
    }
}
