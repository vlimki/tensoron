use crate::tensor::Tensor;
use crate::{calc_grid_size, CudaCtx, CUDA_CTX};
use bytemuck::Zeroable;
use cust::launch;
use cust::memory::*;
use std::ops::{Add, Mul};

pub type Matrix<T> = Tensor<T, 2>;

#[repr(C)]
#[derive(DeviceCopy, Copy, Clone, Debug)]
struct Dimensions {
    m1_rows: u32,
    m1_cols: u32,
    m2_rows: u32,
    m2_cols: u32
}

// Do this soon
/*pub fn mul
impl<T> Mul for Matrix<T>
where T: DeviceCopy + Zeroable {
    type Output = Tensor<T, 2>
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape()[1], rhs.shape()[0]);
    }
}*/
