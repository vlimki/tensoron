use crate::tensor::Tensor;
use crate::{calc_grid_size, CudaCtx, CUDA_CTX};
use cust::memory::*;

pub type Matrix<T> = Tensor<T, 2>;

#[repr(C)]
#[derive(DeviceCopy, Copy, Clone, Debug)]
pub(crate) struct Dimensions {
    pub m1_rows: u32,
    pub m1_cols: u32,
    pub m2_rows: u32,
    pub m2_cols: u32,
}
