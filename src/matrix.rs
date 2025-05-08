use std::ops::Mul;

use crate::{calc_grid_size, get_cuda_type, ops::GpuMul, CudaCtx, Tensor, CUDA_CTX};
use cust::{
    launch,
    memory::{bytemuck::Zeroable, *},
};

pub type Matrix<T> = Tensor<T, 2>;

#[repr(C)]
#[derive(DeviceCopy, Copy, Clone, Debug)]
pub(crate) struct Dimensions {
    pub rows: u32,
    pub cols: u32,
}

impl Dimensions {
    pub fn from_shape<T: DeviceCopy>(t1: &Matrix<T>) -> Self {
        return Self {
            rows: t1.shape()[0] as u32,
            cols: t1.shape()[1] as u32,
        };
    }
}

impl<T: DeviceCopy> From<Vec<T>> for Matrix<T> {
    fn from(v: Vec<T>) -> Self {
        Tensor::from(([v.len(), 1], v))
    }
}
impl<T> GpuMul for Matrix<T>
where
    T: DeviceCopy + Zeroable + 'static,
{
    type Output = Self;
    fn gpu_mul(mut self, mut rhs: Self) -> Self::Output {
        let ctx = CUDA_CTX.lock().unwrap();

        self.gpu();
        rhs.gpu();

        let CudaCtx {
            ref matrix,
            ref stream,
            ..
        } = *ctx;

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape()[0] * rhs.shape()[1]).unwrap();
        let dims_self = Dimensions::from_shape(&self);
        let dims_rhs = Dimensions::from_shape(&rhs);
        let (bs, gs) = calc_grid_size(&self, &rhs);

        let t = get_cuda_type::<T>();
        let f = matrix.get_function(format!("mul_{}", t)).unwrap();

        unsafe {
            launch!(f<<<gs, bs, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                rhs.device_ptr().as_ref().unwrap().as_device_ptr(),
                output.as_device_ptr(),
                dims_self,
                dims_rhs
            ))
            .unwrap()
        }
        return Tensor {
            _device_ptr: Some(output),
            _inner: None,
            _shape: [self.shape()[0], rhs.shape()[1]],
        };
    }
}

impl<T> Matrix<T>
where
    T: DeviceCopy + Zeroable + 'static
{
    pub fn transpose(mut self) -> Self {
        let ctx = CUDA_CTX.lock().unwrap();

        self.gpu();

        let CudaCtx {
            ref matrix,
            ref stream,
            ..
        } = *ctx;

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape()[0] * self.shape()[1]).unwrap();

        let dims = Dimensions::from_shape(&self);

        let bs = 256;
        let gs = (self.shape()[0] as u32 + bs - 1) / bs;


        let t = get_cuda_type::<T>();
        let f = matrix.get_function(format!("transpose_{}", t)).unwrap();

        unsafe {
            launch!(f<<<gs, bs, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                output.as_device_ptr(),
                dims,
            ))
            .unwrap()
        }

        return Tensor {
            _device_ptr: Some(output),
            _inner: None,
            _shape: [self.shape()[1], self.shape()[0]],
        };
    }
}

impl<T> Mul for Matrix<T>
where
    T: DeviceCopy + Zeroable + 'static,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.gpu_mul(rhs)
    }
}
