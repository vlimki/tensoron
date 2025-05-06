use crate::tensor::Tensor;
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

impl Dimensions {
    pub fn from_shapes<T: DeviceCopy>(t1: &Tensor<T, 2>, t2: &Tensor<T, 2>) -> Self {
        return Self {
            m1_rows: t1.shape()[0] as u32,
            m1_cols: t1.shape()[1] as u32,
            m2_rows: t2.shape()[0] as u32,
            m2_cols: t2.shape()[1] as u32
        }
    }
}
