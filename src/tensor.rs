use std::ops::Add;

use bytemuck::Zeroable;
use cust::{launch, memory::*};

use crate::{get_cuda_type, ops::*, CudaCtx, CUDA_CTX};

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

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> GpuAdd<T> for Tensor<T, R> {
    type Output = Self;

    fn gpu_add(mut self, mut rhs: Self) -> Self {
        assert_eq!(self.shape(), rhs.shape());
        let ctx = CUDA_CTX.lock().unwrap();

        self.gpu();
        rhs.gpu();

        // Make a proper grid calc function eventually
        let len: usize = self.shape().iter().product();
        let bs = 256;
        let gs = (self.shape()[0] as u32 + bs - 1) / bs;

        let CudaCtx { ref tensor, ref stream, .. } = *ctx;

        let t = get_cuda_type::<T>();
        let f = tensor.get_function(format!("add_{}", t)).unwrap();

        unsafe {
            launch!(f<<<gs, bs, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                rhs.device_ptr().as_ref().unwrap().as_device_ptr(),
                len as i32,
            )).unwrap()
        }

        self
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> Add for Tensor<T, R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.gpu_add(rhs)
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> GpuScale<T> for Tensor<T, R> {
    type Output = Self;

    fn gpu_scale(mut self, rhs: T) -> Self {
        let ctx = CUDA_CTX.lock().unwrap();

        let mut scalar_tensor = tensor!([1, 1][rhs]);

        self.gpu();
        scalar_tensor.gpu();

        // Make a proper grid calc function eventually
        let len: usize = self.shape().iter().product();
        let bs = 256;
        let gs = (self.shape()[0] as u32 + bs - 1) / bs;

        let CudaCtx { ref tensor, ref stream, .. } = *ctx;

        let t = get_cuda_type::<T>();
        let f = tensor.get_function(format!("scale_{}", t)).unwrap();

        unsafe {
            launch!(f<<<gs, bs, 0, stream>>>(
                self.device_ptr().as_ref().unwrap().as_device_ptr(),
                scalar_tensor.device_ptr().as_ref().unwrap().as_device_ptr(),
                len as i32,
            )).unwrap()
        }

        self
    }
}

impl<T, const R: usize> Tensor<T, R>
where
    T: DeviceCopy + Zeroable + 'static
{
    pub fn gpu(&mut self) {
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

    pub fn scale(self, rhs: T) -> Self {
        self.gpu_scale(rhs)
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

    pub fn cpu(mut self) -> Self {
        if let Some(ref ptr) = self._device_ptr {
            let mut host_out = vec![T::zeroed(); self.shape().iter().product()];
            ptr.copy_to(&mut host_out[..]).unwrap();
            self._inner = host_out;
        }
        self
    }
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
