use std::{ops::Mul, sync::Arc};

use crate::{ops::GpuMul, CudaCtx, Kernel, Tensor, CUDA_CTX};
use cust::memory::{bytemuck::Zeroable, *};

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

impl<T> GpuMul for &Matrix<T>
where
    T: DeviceCopy + Zeroable + 'static,
{
    type Output = Matrix<T>;
    fn gpu_mul(self, rhs: Self) -> Self::Output {
        let ctx = CUDA_CTX.lock().unwrap();

        let a_dev = self.ptr();
        let b_dev = rhs.ptr();

        let CudaCtx {
            ref matrix,
            ref stream,
            ..
        } = *ctx;

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape()[0] * rhs.shape()[1]).unwrap();

        let mut k = Kernel::new(&matrix, "mul");

        let dims_self = Dimensions::from_shape(&self);
        let dims_rhs = Dimensions::from_shape(&rhs);
        k.tune_2d(&self, &rhs);

        unsafe {
            k.launch(&stream, (
                    a_dev.as_device_ptr(),
                    b_dev.as_device_ptr(),
                    output.as_device_ptr(),
                    dims_self,
                    dims_rhs
                    )
            );
        }
        return Tensor {
            _device_ptr: Some(Arc::new(output)),
            _inner: None,
            _shape: [self.shape()[0], rhs.shape()[1]],
        };
    }
}

impl<T> Matrix<T>
where
    T: DeviceCopy + Zeroable + 'static
{
    pub fn transpose(&self) -> Self {
        let ctx = CUDA_CTX.lock().unwrap();

        let a_dev = self.ptr();

        let CudaCtx {
            ref matrix,
            ref stream,
            ..
        } = *ctx;

        let mut k: Kernel<'_, T> = Kernel::new(matrix, "transpose");

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape()[0] * self.shape()[1]).unwrap();

        let dims = Dimensions::from_shape(&self);
        k.tune_1d(self.shape()[0]);

        unsafe {
            k.launch(&stream,
                (a_dev.as_device_ptr(),
                output.as_device_ptr(),
                dims)
            )
        }

        return Tensor {
            _device_ptr: Some(Arc::new(output)),
            _inner: None,
            _shape: [self.shape()[1], self.shape()[0]],
        };
    }
}

impl<'a, 'b, T> Mul<&'b Matrix<T>> for &'a Matrix<T>
where
    T: DeviceCopy + Zeroable + 'static,
{
    type Output = Matrix<T>;
    fn mul(self, rhs: &'b Matrix<T>) -> Self::Output {
        self.gpu_mul(rhs)
    }
}

impl<T> Mul for Matrix<T>
where
    T: DeviceCopy + Zeroable + 'static,
{
    type Output = Matrix<T>;
    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self.gpu_mul(&rhs)
    }
}
