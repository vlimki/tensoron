use crate::tensor;
use crate::Tensor;
use crate::{calc_grid_size, CudaCtx, CUDA_CTX};
use bytemuck::Zeroable;
use cust::launch;
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

pub(crate) fn add<T>(mut v1: Vector<T>, mut v2: Vector<T>) -> Vector<T>
where
    T: DeviceCopy + Zeroable,
{
    assert_eq!(v1.shape(), v2.shape());

    let ctx = CUDA_CTX.lock().unwrap();

    v1.to_device();
    v2.to_device();

    let len = v1.shape()[0].max(v1.shape()[1]);
    let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(len).unwrap();

    let CudaCtx {
        ref vector,
        ref stream,
        ..
    } = *ctx;

    let (bs, gs) = calc_grid_size(&v1, &v2);

    unsafe {
        let result = launch!(vector.add<<<gs, bs, 0, stream>>>(
            v1.device_ptr().as_ref().unwrap().as_device_ptr(),
            v2.device_ptr().as_ref().unwrap().as_device_ptr(),
            c_out.as_device_ptr(),
            len as std::os::raw::c_int,
        ));
        result.unwrap()
    }

    let mut host_out = vec![T::zeroed(); len];
    c_out.copy_to(&mut host_out[..]).unwrap();

    ctx.stream.synchronize().unwrap();

    return Tensor::from((v1.shape(), Vec::from(host_out)));
}

pub(crate) fn mul<T>(mut v1: Vector<T>, mut v2: Vector<T>) -> Vector<T>
where
    T: DeviceCopy + Zeroable,
{
    assert_eq!(v1.shape()[0], v2.shape()[1]);
    assert_eq!(v1.shape()[1], v2.shape()[0]);

    let ctx = CUDA_CTX.lock().unwrap();

    v1.to_device();
    v2.to_device();

    let len = v1.shape()[0].max(v1.shape()[1]);
    let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(1).unwrap();

    let CudaCtx {
        ref vector,
        ref stream,
        ..
    } = *ctx;

    let (bs, gs) = calc_grid_size(&v1, &v2);

    unsafe {
        let result = launch!(vector.dot_product<<<gs, bs, 0, stream>>>(
            v1.device_ptr().as_ref().unwrap().as_device_ptr(),
            v2.device_ptr().as_ref().unwrap().as_device_ptr(),
            c_out.as_device_ptr(),
            len as std::os::raw::c_int,
        ));
        result.unwrap()
    }

    let mut host_out = vec![T::zeroed(); 1];
    c_out.copy_to(&mut host_out[..]).unwrap();

    ctx.stream.synchronize().unwrap();

    // 1x1 tensor
    return tensor!([1, 1][host_out[0]]);
}
