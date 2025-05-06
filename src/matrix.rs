use std::ops::{Add, Mul};

use crate::{execute_operation, tensor, Tensor, Operation};
use cust::memory::{bytemuck::Zeroable, *};

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
    T: DeviceCopy + Zeroable,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        execute_operation(self, rhs, Operation::Mul)
    }
}

impl<T> Add for Matrix<T>
where
    T: DeviceCopy + Zeroable,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        execute_operation(self, rhs, Operation::Add)
    }
}

impl<T> Matrix<T>
where
    T: DeviceCopy + Mul<Output = T> + Zeroable
{
    pub fn scale(self, value: T) -> Self {
        execute_operation(self, tensor!([1,1][value]), Operation::Scale)
    }
}
