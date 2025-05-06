use cust::context::Context;
use crate::matrix::Dimensions;
use cust::memory::bytemuck::Zeroable;
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy};
use cust::module::Module;
use cust::{launch, stream::*};
use lazy_static::lazy_static;
use std::ffi::CString;
use std::{env, fs};
use std::path::PathBuf;
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
    tensor: Module,
    matrix: Module,
    _ctx: Context,
}

#[derive(Debug, Clone)]
pub(crate) enum Operation {
    Add,
    Mul,
    Scale,
}

type Dimension = (u32, u32, u32);

lazy_static! {
    pub(crate) static ref CUDA_CTX: Mutex<CudaCtx> = Mutex::new(CudaCtx::default());
}

fn load_ptx(src: &str) -> String {
    let out_dir = env::var("OUT_DIR").unwrap();
    let ptx_path = PathBuf::from(format!("{}/{}", out_dir, src));
    fs::read_to_string(ptx_path).expect("Failed to read compiled PTX")
}


impl Default for CudaCtx {
    fn default() -> Self {
        let context = cust::quick_init().unwrap();

        let ptx_vector = CString::new(load_ptx("vector.ptx")).unwrap();
        let module_vector = Module::from_ptx_cstr(&ptx_vector, &[]).unwrap();

        let ptx_matrix = CString::new(load_ptx("matrix.ptx")).unwrap();
        let module_matrix = Module::from_ptx_cstr(&ptx_matrix, &[]).unwrap();

        let ptx_tensor = CString::new(load_ptx("tensor.ptx")).unwrap();
        let module_tensor = Module::from_ptx_cstr(&ptx_tensor, &[]).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        Self {
            vector: module_vector,
            matrix: module_matrix,
            tensor: module_tensor,
            stream,
            _ctx: context,
        }
    }
}

pub(crate) fn calc_grid_size<T>(t1: &Tensor<T, 2>, t2: &Tensor<T, 2>, op: Operation) -> (Dimension, Dimension)
where
    T: DeviceCopy,
{
    use Operation::*;
    match op {
        Add | Scale => {
            let bs = 256;
            let gs = ((t1.shape()[0] * t1.shape()[1]) as u32 + bs - 1) / bs;
            return ((bs, 1, 1), (gs, 1, 1))
        }
        Mul => {
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

    let shape = match op {
        Mul => [t1.shape()[0], t2.shape()[1]],
        Add | Scale => t1.shape(),
    };

    let (bs, gs) = calc_grid_size(&t1, &t2, op.clone());
    let len = shape[0] * shape[1];

    let ctx = CUDA_CTX.lock().unwrap();

    t1.to_device();
    t2.to_device();

    let CudaCtx {
        ref matrix,
        ref tensor,
        ref stream,
        ..
    } = *ctx;

    let output = match op {
        Mul => {
            let output: DeviceBuffer<T> = DeviceBuffer::zeroed(t1.shape()[0] * t2.shape()[1]).unwrap();
            let dims = Dimensions::from_shapes(&t1, &t2);

            unsafe {
                launch!(matrix.mul<<<gs, bs, 0, stream>>>(
                    t1.device_ptr().as_ref().unwrap().as_device_ptr(),
                    t2.device_ptr().as_ref().unwrap().as_device_ptr(),
                    output.as_device_ptr(),
                    dims
                )).unwrap()
            }
            return Tensor {
                _device_ptr: Some(output),
                _inner: vec![],
                _shape: [t1.shape()[0], t2.shape()[1]]
            }
        }
        Add => {
            assert_eq!(t1.shape(), t2.shape());

            unsafe {
                launch!(tensor.add<<<gs, bs, 0, stream>>>(
                    t1.device_ptr().as_ref().unwrap().as_device_ptr(),
                    t2.device_ptr().as_ref().unwrap().as_device_ptr(),
                    len
                )).unwrap()
            }
            t1
        }
        Scale => {
            unsafe {
                launch!(tensor.scale<<<gs, bs, 0, stream>>>(
                    t1.device_ptr().as_ref().unwrap().as_device_ptr(),
                    t2.device_ptr().as_ref().unwrap().as_device_ptr(),
                    len as std::os::raw::c_int,
                )).unwrap()
            }
            t1
        }
    };

    stream.synchronize().unwrap();

    output
}

#[cfg(test)]
mod tests {
    use crate::{tensor, Tensor, Vector};
    use rand::distr::Uniform;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    fn generate_data() -> Vec<f32> {
        let size: usize = 100_000;
        let mut rng = StdRng::from_os_rng();
        let uniform = Uniform::new(-1.0f32, 1.0).unwrap();

        (0..size).map(|_| rng.sample(&uniform)).collect()
    }

    #[test]
    fn vectors() {
        let v1 = tensor!([3, 1][1.0f32, 2.0, 3.0]);
        let v2 = tensor!([3, 1][2.0, 4.0, 6.0]);

        let v3 = (v1.clone() + v2.clone()).to_host();
        assert_eq!(v3, tensor!([3,1][3.0, 6.0, 9.0]));
        assert_eq!(v3.clone().scale(10.0), tensor!([3,1][30.0, 60.0, 90.0]));

        let v4 = (v1.transpose() * v2).to_host();
        assert_eq!(v4, tensor!([1, 1][28.0]));

        // Bigger vectors
        let v5 = Vector::from(generate_data());
        let v6 = Vector::from(generate_data());
        let v7 = (v5.transpose() * v6).to_host();

        println!("{:#?}", v7);
    }

    #[test]
    fn matrices() {
        let m1 = tensor!([2, 2][1.0f32, 2.0, 3.0, 4.0]);
        let m2 = tensor!([2, 2][2.0, 3.0, 4.0, 5.0]);

        let m3 = (m1.clone() * m2.clone()).to_host();
        assert_eq!(m3, tensor!([2, 2][10.0, 13.0, 22.0, 29.0]));

        let m4 = (m1 + m2).to_host();
        println!("{:#?}", m4);
        assert_eq!(m4, tensor!([2, 2][3.0, 5.0, 7.0, 9.0]));
    } 

    #[test]
    #[should_panic]
    fn vec_add_illegal() {
        let v1 = tensor!([3, 1][1.0, 2.0, 3.0]);
        let v2 = tensor!([2, 1][2.0, 4.0]);

        let _ = v1 + v2;
    }
}
