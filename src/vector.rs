use crate::Tensor;
use cust::memory::*;

/// To separate column vectors and row vectors, we have the dimensions as either [1, N] or [N, 1].
pub type Vector<T> = Tensor<T, 2>;

impl<T: DeviceCopy> From<Vec<T>> for Vector<T> {
    fn from(v: Vec<T>) -> Self {
        Tensor::from(([v.len(), 1], v))
    }
}

pub(crate) fn transpose<T>(mut v1: Vector<T>) -> Vector<T>
where
    T: DeviceCopy,
{
    let s = v1.shape();
    v1._shape = [s[1], s[0]];
    v1
}
