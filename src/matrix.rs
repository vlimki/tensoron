use crate::tensor::Tensor;
use crate::{calc_grid_size, CudaCtx, CUDA_CTX};
use bytemuck::Zeroable;
use cust::launch;
use cust::memory::*;

pub type Matrix<T> = Tensor<T, 2>;

#[repr(C)]
#[derive(DeviceCopy, Copy, Clone, Debug)]
struct Dimensions {
    m1_rows: u32,
    m1_cols: u32,
    m2_rows: u32,
    m2_cols: u32,
}

pub(crate) fn mul<T>(mut m1: Matrix<T>, mut m2: Matrix<T>) -> Matrix<T>
where
    T: DeviceCopy + Zeroable,
{
    assert_eq!(m1.shape()[1], m2.shape()[0]);
    
    let sz_new = m1.shape()[0] * m2.shape()[1];

    let ctx = CUDA_CTX.lock().unwrap();

    m1.to_device();
    m2.to_device();

    let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(sz_new).unwrap();

    let CudaCtx {
        ref matrix,
        ref stream,
        ..
    } = *ctx;

    let dims = Dimensions {
        m1_rows: m1.shape()[0] as u32,
        m1_cols: m1.shape()[1] as u32,
        m2_rows: m2.shape()[0] as u32,
        m2_cols: m2.shape()[1] as u32
    };

    let (bs, gs) = calc_grid_size(&m1, &m2);

    unsafe {
        let result = launch!(matrix.matmul_kernel<<<gs, bs, 0, stream>>>(
            m1.device_ptr().as_ref().unwrap().as_device_ptr(),
            m2.device_ptr().as_ref().unwrap().as_device_ptr(),
            c_out.as_device_ptr(),
            dims
        ));
        result.unwrap()
    }

    let mut host_out = vec![T::zeroed(); sz_new];
    c_out.copy_to(&mut host_out[..]).unwrap();

    ctx.stream.synchronize().unwrap();

    return Tensor::from(([m1.shape()[0], m2.shape()[1]], host_out));
}
