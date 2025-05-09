use std::ops::Sub;
use std::{fmt::Debug, ops::Add};
use std::sync::Arc;

use bytemuck::Zeroable;
use cust::memory::*;

mod view;
pub use view::TensorView;

use crate::{ops::*, CudaCtx, Kernel, CUDA_CTX};

pub type Scalar<T> = Tensor<T, 0>;

impl<T: DeviceCopy> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Tensor::from(([], vec![value]))
    }
}

impl<T: DeviceCopy> Scalar<T> {
    pub fn value(&self) -> T {
        self.inner()
            .as_ref()
            .expect("Calling `.value()` on unsynchronized data; call `.cpu()` on the tensor first.")
            [0]
    }
}

pub struct Tensor<T, const R: usize>
where
    T: DeviceCopy,
{
    pub(crate) _device_ptr: Option<Arc<DeviceBuffer<T>>>,
    pub(crate) _inner: Option<Vec<T>>,
    pub(crate) _shape: [usize; R],
}

impl<T: DeviceCopy + Debug, const R: usize> Debug for Tensor<T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}{:?}", self._shape, self.inner().as_ref().expect("Trying to print out a tensor with unsynchronized data; call .cpu() on the tensor first."))
    }
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
    fn clone(&self) -> Self {
        Self {
            _device_ptr: self._device_ptr.clone(),
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

impl<T: DeviceCopy, const R: usize> Tensor<T, R> {
    pub fn shape(&self) -> [usize; R] {
        self._shape
    }

    pub(crate) fn device_ptr(&self) -> &Option<Arc<DeviceBuffer<T>>> {
        &self._device_ptr
    }


    pub fn inner(&self) -> &Option<Vec<T>> {
        &self._inner
    }

    pub fn view<'a>(&'a self) -> TensorView<'a, T, R> {
        TensorView::from(self)
    }

    pub fn map<U: DeviceCopy>(&self, f: impl Fn(&T) -> U) -> Tensor<U, R> {
        let data = self
            ._inner
            .as_ref()
            .expect("Calling `map` with unsynchronized data; call `.cpu()` on the tensor first.")
            .iter()
            .map(f)
            .collect::<Vec<_>>();
        Tensor::<U, R>::from((self._shape, data))
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> GpuAdd<T> for &Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn gpu_add(self, rhs: Self) -> Tensor<T, R> {
        assert_eq!(self.shape(), rhs.shape());
        let ctx = CUDA_CTX.lock().unwrap();

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape().iter().product()).unwrap();

        let a_dev = self.ptr();
        let b_dev = rhs.ptr();

        // Make a proper grid calc function eventually
        let len: usize = self.shape().iter().product();

        let CudaCtx {
            ref tensor,
            ref stream,
            ..
        } = *ctx;

        let mut k: Kernel<'_, T> = Kernel::new(tensor, "add");
        k.tune_1d(len);

        unsafe {
            k.launch(&stream,
                (a_dev.as_device_ptr(),
                b_dev.as_device_ptr(),
                output.as_device_ptr(),
                len as i32)
            )
        }

        return Tensor {
            _device_ptr: Some(Arc::new(output)),
            _inner: None,
            _shape: self.shape()
        }
    }

    fn gpu_sub(self, rhs: Self) -> Tensor<T, R> {
        assert_eq!(self.shape(), rhs.shape());
        let ctx = CUDA_CTX.lock().unwrap();
        let len = self.shape().iter().product();

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape().iter().product()).unwrap();

        let a_dev = self.ptr();
        let b_dev = rhs.ptr();

        let CudaCtx {
            ref tensor,
            ref stream,
            ..
        } = *ctx;

        let mut k: Kernel<'_, T> = Kernel::new(tensor, "sub");
        k.tune_1d(len);

        unsafe {
            k.launch(&stream,
                (a_dev.as_device_ptr(),
                b_dev.as_device_ptr(),
                output.as_device_ptr(),
                len as i32)
            )
        }

        return Tensor {
            _device_ptr: Some(Arc::new(output)),
            _inner: None,
            _shape: self.shape()
        }
    }

    fn gpu_cmul(self, rhs: Self) -> Tensor<T, R> {
        assert_eq!(self.shape(), rhs.shape());
        let ctx = CUDA_CTX.lock().unwrap();

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape().iter().product()).unwrap();

        let a_dev = self.ptr();
        let b_dev = rhs.ptr();

        let len: usize = self.shape().iter().product();


        let CudaCtx {
            ref tensor,
            ref stream,
            ..
        } = *ctx;

        let mut k: Kernel<'_, T> = Kernel::new(tensor, "cmul");
        k.tune_1d(len);

        unsafe {
            k.launch(&stream,
                (a_dev.as_device_ptr(),
                b_dev.as_device_ptr(),
                output.as_device_ptr(),
                len as i32)
            )
        }

        return Tensor {
            _device_ptr: Some(Arc::new(output)),
            _inner: None,
            _shape: self.shape()
        }
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> Add for &Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn add(self, rhs: Self) -> Self::Output {
        self.gpu_add(rhs)
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> Add for Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn add(self, rhs: Self) -> Self::Output {
        self.gpu_add(&rhs)
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> Sub for &Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.gpu_sub(rhs)
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> Sub for Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.gpu_sub(&rhs)
    }
}


impl<T: DeviceCopy + Zeroable + 'static, const R: usize> GpuScale<T> for &Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn gpu_scale(self, rhs: T) -> Self::Output {
        let ctx = CUDA_CTX.lock().unwrap();

        let mut scalar_tensor = tensor!([1, 1][rhs]);

        let output: DeviceBuffer<T> =
            DeviceBuffer::zeroed(self.shape().iter().product()).unwrap();

        let a_dev = self.ptr();
        scalar_tensor.gpu();

        let len: usize = self.shape().iter().product();

        let CudaCtx {
            ref tensor,
            ref stream,
            ..
        } = *ctx;

        let mut k: Kernel<'_, T> = Kernel::new(tensor, "scale");
        k.tune_1d(len);

        unsafe {
            k.launch(&stream,
                (a_dev.as_device_ptr(),
                scalar_tensor.device_ptr().as_ref().unwrap().as_device_ptr(),
                output.as_device_ptr(),
                len as i32)
            )
        }

        return Tensor {
            _device_ptr: Some(Arc::new(output)),
            _inner: None,
            _shape: self.shape()
        }
    }
}

fn call_ml_function<T, const R: usize>(getter: &'static str, t: &Tensor<T, R>) -> Tensor<T, R>
where
    T: DeviceCopy + Zeroable + 'static,
{
    let ctx = CUDA_CTX.lock().unwrap();

    let output: DeviceBuffer<T> =
        DeviceBuffer::zeroed(t.shape().iter().product()).unwrap();

    let a_dev = t.ptr();

    let len: usize = t.shape().iter().product();

    let CudaCtx {
        ref tensor,
        ref stream,
        ..
    } = *ctx;

    let mut k: Kernel<'_, T> = Kernel::new(tensor, getter);

    k.tune_1d(len);

    unsafe {
        k.launch(&stream,
            (a_dev.as_device_ptr(),
            output.as_device_ptr(),
            len as i32)
        )
    }

    return Tensor {
        _device_ptr: Some(Arc::new(output)),
        _inner: None,
        _shape: t.shape()
    }
}

impl<T: DeviceCopy + Zeroable + 'static, const R: usize> ML<T> for &Tensor<T, R> {
    type Output = Tensor<T, R>;
    fn relu(self) -> Tensor<T, R> {
        call_ml_function("relu", self)
    }
    fn tanh(self) -> Tensor<T, R> {
        call_ml_function("tanh", self)
    }

    fn sigmoid(self) -> Tensor<T, R> {
        call_ml_function("sigmoid", self)
    }

    fn sigmoid_derivative(self) -> Tensor<T, R> {
        call_ml_function("sigmoid_derivative", self)
    }
    fn relu_derivative(self) -> Self::Output {
        call_ml_function("sigmoid_derivative", self)
    }
}

impl<T, const R: usize> Tensor<T, R>
where
    T: DeviceCopy + Zeroable + 'static,
{
    pub fn gpu(&mut self) {
        if let None = self._device_ptr {
            // .unwrap() is fine here, since the device_ptr and self._inner cannot both be None
            self._device_ptr =
                Some(Arc::new(DeviceBuffer::from_slice(self._inner.as_ref().unwrap()).unwrap()));
            self._inner = None;
        }
    }

    pub fn zeros(shape: [usize; R]) -> Self {
        Self {
            _device_ptr: None,
            _shape: shape,
            _inner: Some(vec![T::zeroed(); shape.iter().product()]),
        }
    }

    pub fn ptr(&self) -> Arc<DeviceBuffer<T>> {
        self._device_ptr.as_ref().cloned().unwrap_or_else(|| Arc::new(DeviceBuffer::from_slice(self.inner().as_ref().unwrap()).unwrap()))
    }

    pub fn scale(&self, rhs: T) -> Self {
        self.gpu_scale(rhs)
    }

    pub fn cpu(mut self) -> Self {
        if let Some(ref ptr) = self._device_ptr {
            let mut host_out = vec![T::zeroed(); self.shape().iter().product()];
            ptr.copy_to(&mut host_out[..]).unwrap();
            self._inner = Some(host_out);
        } else {
            panic!("Calling .cpu() with no device ptr");
        }
        self
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
            _inner: Some(value.1),
        }
    }
}


impl<T, const R: usize> PartialEq for Tensor<T, R>
where
    T: DeviceCopy + PartialEq + Zeroable,
{
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.inner() == other.inner()
    }
}
