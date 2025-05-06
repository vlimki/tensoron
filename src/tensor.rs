use crate::{execute_operation, matrix, vector, Operation};
use std::ops::{Add, Mul};

use bytemuck::Zeroable;
use cust::memory::*;
//use std::ops::Mul;

pub type Scalar<T> = Tensor<T, 0>;

impl<T: DeviceCopy> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Tensor::from(([], vec![value]))
    }
}

impl<T: DeviceCopy> Scalar<T> {
    pub fn value(&self) -> T {
        self.inner()[0]
    }
}

#[derive(Debug)]
pub struct Tensor<T, const R: usize>
where
    T: DeviceCopy,
{
    pub(crate) _device_ptr: Option<DeviceBuffer<T>>,
    pub(crate) _inner: Vec<T>,
    pub(crate) _shape: [usize; R],
}

#[macro_export]
macro_rules! tensor {
    ([$($shape:expr),*] [ $($elem:expr),* $(,)? ]) => {{
        let data = vec![$($elem),*];
        const SHAPE: &[usize] = &[$($shape),*];
        Tensor::<_, { SHAPE.len() }>::from(([ $($shape),* ], data))
    }};
}

/*
 * RANK-R TENSOR FUNCTIONS
 */

impl<T, const R: usize> Clone for Tensor<T, R>
where
    T: DeviceCopy + Clone,
{
    /// Safe because it discards the device pointer.
    /// Will be removing once when making GPU-local. OR will alternatively allocate it on a new
    /// pointer.
    fn clone(&self) -> Self {
        Self {
            _device_ptr: None,
            _inner: self._inner.clone(),
            _shape: self._shape.clone(),
        }
    }
}

impl<T, const R: usize> Drop for Tensor<T, R>
where
    T: DeviceCopy,
{
    fn drop(&mut self) {
        let _ = self._device_ptr.take();
    }
}

impl<T> Tensor<T, 2>
where
    T: DeviceCopy + Mul<Output = T> + Zeroable
{
    pub fn scale(self, value: T) -> Self {
        execute_operation(self, tensor!([1,1][value]), Operation::Scale)
    }
}

impl<T, const R: usize> Tensor<T, R>
where
    T: DeviceCopy,
{
    pub fn shape(&self) -> [usize; R] {
        self._shape
    }

    pub fn device_ptr(&self) -> &Option<DeviceBuffer<T>> {
        &self._device_ptr
    }

    pub(crate) fn inner(&self) -> &Vec<T> {
        &self._inner
    }

    pub(crate) fn to_device(&mut self) {
        if let None = self._device_ptr {
            self._device_ptr = Some(DeviceBuffer::from_slice(&self._inner).unwrap());
        }
    }

    // Element-wise map. Note that this discards the device pointer.
    pub fn map<U: DeviceCopy>(&self, f: impl Fn(&T) -> U) -> Tensor<U, R> {
        let data = self._inner.iter().map(f).collect::<Vec<_>>();
        Tensor::<U, R>::from((self._shape, data))
    }
}

impl<T, const R: usize> From<([usize; R], Vec<T>)> for Tensor<T, R>
where
    T: DeviceCopy,
{
    fn from(value: ([usize; R], Vec<T>)) -> Self {
        Self {
            _device_ptr: None,
            _shape: value.0,
            _inner: value.1,
        }
    }
}

impl<T, const R: usize> PartialEq for Tensor<T, R>
where
    T: DeviceCopy + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self._inner == other._inner && self._shape == other._shape
    }
}

/*
 * RANK-2 TENSOR FUNCTIONS
 */
impl<T> Tensor<T, 2>
where
    T: DeviceCopy,
{
    pub fn transpose(self) -> Self {
        let s = self.shape();
        match s {
            [_, 1] | [1, _] => vector::transpose(self),
            _ => unimplemented!(),
        }
    }
}

impl<T> Mul for Tensor<T, 2>
where
    T: DeviceCopy + Zeroable,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let s = self.shape();
        match s {
            [_, 1] | [1, _] => execute_operation(self, rhs, Operation::VecMul),
            _ => execute_operation(self, rhs, Operation::MatMul),
        }
    }
}

impl<T> Add for Tensor<T, 2>
where
    T: DeviceCopy + Zeroable,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let s = self.shape();
        match s {
            [_, 1] | [1, _] => execute_operation(self, rhs, Operation::VecAdd),
            _ => execute_operation(self, rhs, Operation::MatAdd),
        }
    }
}
