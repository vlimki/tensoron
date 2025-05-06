use cust::context::Context;
use crate::matrix::Dimensions;
use cust::memory::bytemuck::Zeroable;
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy};
use cust::module::Module;
use cust::{launch, stream::*};
use lazy_static::lazy_static;
use std::ffi::CString;
use std::sync::Mutex;

pub mod matrix;
mod tensor;
pub mod vector;
pub use matrix::Matrix;
pub use tensor::Tensor;
pub use vector::Vector;

struct CudaCtx {
    stream: Stream,
    vector: Module,
    matrix: Module,
    _ctx: Context,
}

pub(crate) enum Operation {
    VecMul,
    VecAdd,
    MatMul,
    MatAdd,
}

type Dimension = (u32, u32, u32);

lazy_static! {
    pub(crate) static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
}

impl Default for CudaCtx {
    fn default() -> Self {
        let context = cust::quick_init().unwrap();

        let ptx_vector = CString::new(include_str!("../kernels/vector.ptx")).unwrap();
        let module_vector = Module::from_ptx_cstr(&ptx_vector, &[]).unwrap();
        let ptx_matrix = CString::new(include_str!("../kernels/matrix.ptx")).unwrap();
        let module_matrix = Module::from_ptx_cstr(&ptx_matrix, &[]).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        Self {
            vector: module_vector,
            matrix: module_matrix,
            stream,
            _ctx: context,
        }
    }
}

pub(crate) fn calc_grid_size<T>(t1: &Tensor<T, 2>, t2: &Tensor<T, 2>) -> (Dimension, Dimension)
where
    T: DeviceCopy,
{
    match t1.shape() {
        [1, _] | [_, 1] => {

            let bs = 256;
            let gs = (t1.inner().len() as u32 + bs - 1) / bs;
            return ((bs, 1, 1), (gs, 1, 1))
        }
        _ =>  {
            let bs = (16, 16, 1);

            let s1 = t1.shape();
            let s2 = t2.shape();
            let gs = (
                (s2[1] as usize + bs.0 as usize - 1) as u32 / bs.0,
                (s1[0] as usize + bs.1 as usize - 1) as u32 / bs.1,
                1
            );
            (bs, gs)

        }
    }
}

pub(crate) fn execute_operation<T>(mut t1: Tensor<T, 2>, mut t2: Tensor<T, 2>, op: Operation) -> Tensor<T, 2>
where T: DeviceCopy + Zeroable
{
    use Operation::*;

    let sz_new = match op {
        MatMul => {
            assert_eq!(t1.shape()[1], t2.shape()[0]);
            t1.shape()[0] * t2.shape()[1]

        }
        VecAdd | MatAdd => {
            assert_eq!(t1.shape(), t2.shape());
            t1.shape()[0].max(t1.shape()[0])
        }
        VecMul => {
            assert_eq!(t1.shape()[1], t2.shape()[0]);
            assert_eq!(t1.shape()[0], t2.shape()[1]);
            1
        }
    };

    let ctx = CUDA_CTX.lock().unwrap();

    t1.to_device();
    t2.to_device();

    let c_out: DeviceBuffer<T> = DeviceBuffer::zeroed(sz_new).unwrap();
    let CudaCtx {
        ref matrix,
        ref vector,
        ref stream,
        ..
    } = *ctx;

    match op {
        MatMul => {
            let (bs, gs) = calc_grid_size(&t1, &t2);
            let dims = Dimensions {
                m1_rows: t1.shape()[0] as u32,
                m1_cols: t1.shape()[1] as u32,
                m2_rows: t2.shape()[0] as u32,
                m2_cols: t2.shape()[1] as u32
            };
            unsafe {
                launch!(matrix.matmul_kernel<<<gs, bs, 0, stream>>>(
                    t1.device_ptr().as_ref().unwrap().as_device_ptr(),
                    t1.device_ptr().as_ref().unwrap().as_device_ptr(),
                    c_out.as_device_ptr(),
                    dims
                )).unwrap()
            }
        }
        VecMul => {
            let (bs, gs) = calc_grid_size(&t1, &t2);
            let len = t1.shape()[0].max(t1.shape()[1]);

            unsafe {
                launch!(vector.dot_product<<<gs, bs, 0, stream>>>(
                    t1.device_ptr().as_ref().unwrap().as_device_ptr(),
                    t2.device_ptr().as_ref().unwrap().as_device_ptr(),
                    c_out.as_device_ptr(),
                    len as std::os::raw::c_int,
                )).unwrap()
            }
        }
        VecAdd => {
            let len = t1.shape()[0].max(t1.shape()[1]);
            let (bs, gs) = calc_grid_size(&t1, &t2);

            unsafe {
                launch!(vector.add<<<gs, bs, 0, stream>>>(
                    t1.device_ptr().as_ref().unwrap().as_device_ptr(),
                    t2.device_ptr().as_ref().unwrap().as_device_ptr(),
                    c_out.as_device_ptr(),
                    len as std::os::raw::c_int,
                )).unwrap()
            }

        }
        _ => unimplemented!()
    }

    let mut host_out = vec![T::zeroed(); sz_new];
    c_out.copy_to(&mut host_out[..]).unwrap();

    ctx.stream.synchronize().unwrap();
    
    let shape = match op {
        MatMul => [t1.shape()[0], t2.shape()[1]],
        VecAdd => t1.shape(),
        MatAdd => t1.shape(),
        VecMul => [1, 1]
    };

    return Tensor::from((shape, host_out));
}

#[cfg(test)]
mod tests {
    use crate::{tensor, Tensor, Vector};
    use rand::distr::Uniform;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    fn generate_data() -> Vec<f32> {
        let size: usize = 10_000_000;
        let mut rng = StdRng::from_os_rng();
        let uniform = Uniform::new(-1.0f32, 1.0).unwrap();

        (0..size).map(|_| rng.sample(&uniform)).collect()
    }

    #[test]
    fn vectors() {
        let v1 = tensor!([3, 1][1.0f32, 2.0, 3.0]);
        let v2 = tensor!([3, 1][2.0, 4.0, 6.0]);

        let v3 = v1.clone() + v2.clone();

        let v4 = v1 * v2.transpose();

        assert_eq!(v3, tensor!([3,1][3.0, 6.0, 9.0]));
        assert_eq!(v3.scale(10.0), tensor!([3,1][30.0, 60.0, 90.0]));
        assert_eq!(v4, tensor!([1, 1][28.0]));

        // Bigger vectors
        let v5 = Vector::from(generate_data());
        let v6 = Vector::from(generate_data());
        let _ = v5 * v6.transpose();
    }

    #[test]
    fn matrices() {
        let m1 = tensor!([2, 2][1.0f32, 2.0, 3.0, 4.0]);
        let m2 = tensor!([2, 2][2.0, 3.0, 4.0, 5.0]);

        let m3 = m1 * m2;
        println!("{:#?}", m3)
    } 

    #[test]
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = tensor!([3, 1][1.0, 2.0, 3.0]);
        let v2 = tensor!([2, 1][2.0, 4.0]);

        let _ = v1 + v2;
    }
}
