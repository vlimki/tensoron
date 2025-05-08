use cust::memory::DeviceCopy;

use super::Tensor;

#[derive(Debug, Clone)]
pub struct TensorView<'a, T, const R: usize> {
    pub(crate) _shape: [usize; R],
    pub(crate) _data: &'a [T],
    pub(crate) _strides: [usize; R],
}

impl<'a, T, const R: usize> From<&'a Tensor<T, R>> for TensorView<'a, T, R>
where
    T: DeviceCopy,
{
    fn from(t: &'a Tensor<T, R>) -> Self {
        let mut strides = [0; R];

        strides[R - 1] = 1;

        for i in (0..R - 1).rev() {
            strides[i] = strides[i + 1] * t.shape()[i + 1];
        }

        Self {
            _shape: t.shape(),
            _data: t.inner().as_ref().expect("Trying to create a tensor view with unsynchronized data; call `.cpu()` on the tensor first."),
            _strides: strides,
        }
    }
}

impl<'a, T, const R: usize> TensorView<'a, T, R>
where
    T: DeviceCopy,
{
    pub fn at<const N: usize>(&self, index: [usize; N]) -> TensorView<'a, T, { R - N }>
    where
        T: DeviceCopy,
        [(); R - N]:,
    {
        let offset = Iterator::zip(index.iter(), self._strides.iter())
            .map(|(a, b)| a * b)
            .sum();

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
        let data = &self._data[offset..offset + len];

        TensorView {
            _data: data,
            _shape: new_shape,
            _strides: new_strides,
        }
    }

    pub fn data(&self) -> &'a [T] {
        self._data
    }

    pub fn slice<const N: usize>(&self, index: [usize; N]) -> TensorView<'a, T, R>
    where
        T: DeviceCopy,
        [(); R - N]:,
    {
        let offset = Iterator::zip(index.iter(), self._strides.iter())
            .map(|(a, b)| a * b)
            .sum();

        let mut new_shape = self._shape;

        let new_strides = self._strides;

        for i in 0..N {
            new_shape[i] = 1;
        }

        let len: usize = new_shape.iter().product();
        let data = &self._data[offset..offset + len];

        TensorView {
            _data: data,
            _shape: new_shape,
            _strides: new_strides,
        }
    }
}

impl<'a, T: Clone> TensorView<'a, T, 0> {
    pub fn value(&self) -> T {
        self._data[0].clone()
    }
}
