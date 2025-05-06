use crate::{execute_operation, matrix, vector, Operation};
use std::ops::{Add, Index, Mul};

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
    pub(crate) _strides: [usize; R],
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
            _strides: self._strides.clone()
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
impl<T: DeviceCopy, const R: usize> Tensor<T, R> {
    pub fn shape(&self) -> [usize; R] {
        self._shape
    }

    pub fn device_ptr(&self) -> &Option<DeviceBuffer<T>> {
        &self._device_ptr
    }

    pub(crate) fn inner(&self) -> &Vec<T> {
        &self._inner
    }

    pub fn map<U: DeviceCopy>(&self, f: impl Fn(&T) -> U) -> Tensor<U, R> {
        let data = self._inner.iter().map(f).collect::<Vec<_>>();
        Tensor::<U, R>::from((self._shape, data))
    }
}


impl<T, const R: usize> Tensor<T, R>
where
    T: DeviceCopy + Zeroable,
{
    pub fn to_device(&mut self) {
        if let None = self._device_ptr {
            self._device_ptr = Some(DeviceBuffer::from_slice(&self._inner).unwrap());
            self._inner = vec![];
        }
    }

    pub fn slice<const N: usize>(&self, index: [usize; N]) -> Tensor<T, R>
        where T: DeviceCopy, [(); R - N]: {
        let offset = Iterator::zip(index.iter(), self._strides.iter()).map(|(a, b)| a * b).sum();

        let mut new_shape = self._shape;

        let new_strides = self._strides;

        for i in 0..N {
            new_shape[i] = 1;
        }

        let len: usize = new_shape.iter().product();
        let data = &self.inner()[offset..offset + len];

        Tensor {
            _device_ptr: None,
            _inner: data.to_vec(),
            _shape: new_shape,
            _strides: new_strides,
        }
    }

    pub fn at<const N: usize>(&self, index: [usize; N]) -> Tensor<T, {R - N}>
        where T: DeviceCopy, [(); R - N]: {
        let offset = Iterator::zip(index.iter(), self._strides.iter()).map(|(a, b)| a * b).sum();

        let new_shape = {
            let mut s = [0; R - N];
            s.copy_from_slice(&self._shape[N..]);
            s
        };

        let new_strides = {
            let mut s = [0; R - N];
            s.copy_from_slice(&self._shape[N..]);
            s
        };

        let len: usize = new_shape.iter().product();
        let data = &self.inner()[offset..offset + len];

        Tensor {
            _device_ptr: None,
            _inner: data.to_vec(),
            _shape: new_shape,
            _strides: new_strides,
        }
    }



    pub fn to_host(mut self) -> Self {
        if let Some(ref ptr) = self._device_ptr {
            let mut host_out = vec![T::zeroed(); self.shape()[0] * self.shape()[1]];
            ptr.copy_to(&mut host_out[..]).unwrap();
            self._inner = host_out;
        }
        self
    }

    // Element-wise map. Note that this discards the device pointer.
}

impl<T, const R: usize> From<([usize; R], Vec<T>)> for Tensor<T, R>
where
    T: DeviceCopy,
{
    fn from(value: ([usize; R], Vec<T>)) -> Self {
        let mut strides = [0; R];

        strides[R - 1] = 1;

        for i in (0..R - 1).rev() {
            strides[i] = strides[i + 1] * value.0[i + 1];
        }

        Self {
            _device_ptr: None,
            _shape: value.0,
            _inner: value.1,
            _strides: strides
        }
    }
}

impl<T, const R: usize> PartialEq for Tensor<T, R>
where
    T: DeviceCopy + PartialEq + Zeroable
{
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.inner() == other.inner()
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
        execute_operation(self, rhs, Operation::Mul)
    }
}

impl<T> Add for Tensor<T, 2>
where
    T: DeviceCopy + Zeroable,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        execute_operation(self, rhs, Operation::Add)
    }
}
