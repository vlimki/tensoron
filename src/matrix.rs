use std::ops::Mul;

use crate::{calc_grid_size, get_cuda_type, ops::GpuMul, CudaCtx, Tensor, CUDA_CTX};
use cust::{launch, memory::{bytemuck::Zeroable, *}};

pub type Matrix<T> = Tensor<T, 2>;

#[repr(C)]
#[derive(DeviceCopy, Copy, Clone, Debug)]
pub(crate) struct Dimensions {
    pub m1_rows: u32,
    pub m1_cols: u32,
    pub m2_rows: u32,
    pub m2_cols: u32,
}

impl Dimensions {
    pub fn from_shapes<T: DeviceCopy>(t1: &Matrix<T>, t2: &Matrix<T>) -> Self {
        return Self {
            m1_rows: t1.shape()[0] as u32,
            m1_cols: t1.shape()[1] as u32,
            m2_rows: t2.shape()[0] as u32,
            m2_cols: t2.shape()[1] as u32
        }
    }
}

impl<T: DeviceCopy> From<Vec<T>> for Matrix<T> {
    fn from(v: Vec<T>) -> Self {
        Tensor::from(([v.len(), 1], v))
    }
}
impl<T> GpuMul for Matrix<T>
where T: DeviceCopy + Zeroable + 'static {
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

        let output: DeviceBuffer<T> = DeviceBuffer::zeroed(self.shape()[0] * rhs.shape()[1]).unwrap();
        let dims = Dimensions::from_shapes(&self, &rhs);
        let (bs, gs) = calc_grid_size(&self, &rhs);

        let t = get_cuda_type::<T>();
        let f = matrix.get_function(format!("mul_{}", t)).unwrap();

        unsafe {
            launch!(f<<<gs, bs, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                rhs.device_ptr().as_ref().unwrap().as_device_ptr(),
                output.as_device_ptr(),
                dims
            )).unwrap()
        }
        return Tensor {
            _device_ptr: Some(output),
            _inner: vec![],
            _shape: [self.shape()[0], rhs.shape()[1]],
            _strides: [rhs.shape()[1], 1]
        }
    }
}

// Impl later
impl<T> Matrix<T>
where
    T: DeviceCopy,
{
    pub fn transpose(self) -> Self {
        let s = self.shape();
        match s {
            _ => unimplemented!(),
        }
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
