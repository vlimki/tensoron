use std::ops::Mul;

use cust::launch;
use cust::memory::bytemuck::Zeroable;
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy};

use crate::{get_cuda_type, CudaCtx, Tensor, CUDA_CTX};
use crate::ops::*;

pub type Vector<T> = Tensor<T, 1>;

impl<T: DeviceCopy + Zeroable + 'static> GpuMul for Vector<T> {
    type Output = T;

    fn gpu_mul(mut self, mut rhs: Self) -> T {
        let ctx = CUDA_CTX.lock().unwrap();

        self.gpu();
        rhs.gpu();

        let len = self.shape()[0];
        let bs = 256;
        let gs = (self.shape()[0] as u32 + bs - 1) / bs;

        let output: DeviceBuffer<T> = DeviceBuffer::zeroed(1).unwrap();
        let CudaCtx { ref vector, ref stream, .. } = *ctx;

        let t = get_cuda_type::<T>();
        let f = vector.get_function(format!("mul_{}", t)).unwrap();

        unsafe {
            launch!(f<<<gs, bs, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                rhs.device_ptr().as_ref().unwrap().as_device_ptr(),
                output.as_device_ptr(),
                len,
            )).unwrap()
        }

        let mut host = vec![T::zeroed(); 1];

        output.copy_to(&mut host).unwrap();
        host[0]
    }
}

impl<T: DeviceCopy + Zeroable + 'static> Mul for Vector<T> {
    type Output = T;

    fn mul(self, rhs: Self) -> Self::Output {
        self.gpu_mul(rhs)
    }
}

impl<T: DeviceCopy> From<Vec<T>> for Vector<T> {
    fn from(v: Vec<T>) -> Self {
        Tensor::from(([v.len()], v))
    }
}

