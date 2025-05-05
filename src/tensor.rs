use cust::memory::*;

#[derive(Debug)]
pub struct Tensor<T, const R: usize>
where
    T: DeviceCopy
{
    _device_ptr: Option<DeviceBuffer<T>>,
    _inner: Vec<T>,
    _shape: [usize; R],
}

impl<T, const R: usize> Drop for Tensor<T, R>
where
    T: DeviceCopy
{
    fn drop(&mut self) {
        let _ = self._device_ptr.take();
    }
}

impl<T, const R: usize> Tensor<T, R>
where T: DeviceCopy 
{
    fn shape(&self) -> [usize; R] {
        self._shape
    }
}

impl<T, const R: usize> From<([usize; R], Vec<T>)> for Tensor<T, R>
where T: DeviceCopy
{
    fn from(value: ([usize; R], Vec<T>)) -> Self {
        Self {
            _device_ptr: None,
            _shape: value.0,
            _inner: value.1,
        }
    }
}
