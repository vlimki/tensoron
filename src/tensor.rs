use std::ops::Mul;

use cust::memory::*;
//use std::ops::Mul;

/*pub type Scalar<T> = Tensor<T, 0>;

impl<T: DeviceCopy> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Tensor::from(([], vec![value]))
    }
}

impl<T: DeviceCopy> Scalar<T> {
    pub fn value(&self) -> T {
        self.inner()[0]
    }
}*/

#[derive(Debug)]
pub struct Tensor<T, const R: usize>
where
    T: DeviceCopy,
{
    _device_ptr: Option<DeviceBuffer<T>>,
    _inner: Vec<T>,
    _shape: [usize; R],
}

impl<T, const R: usize> Clone for Tensor<T, R>
where
    T: DeviceCopy + Clone
{
    /// Safe because it discards the device pointer.
    fn clone(&self) -> Self {
        Self {
            _device_ptr: None,
            _inner: self._inner.clone(),
            _shape: self._shape.clone()
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

impl<T, const R: usize> Tensor<T, R>
where
    T: DeviceCopy + Mul<Output = T>
{
    pub fn scale(&self, value: T) -> Self {
        self.map(|x| *x * value)
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
