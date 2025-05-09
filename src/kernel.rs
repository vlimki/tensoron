#![allow(non_snake_case)]

use std::{any::TypeId, marker::PhantomData};

use cust::{launch, memory::DeviceCopy, module::Module, stream::Stream};
use cust::function::Function;
use crate::Tensor;

pub trait KernelArgs {
    unsafe fn invoke(self, f: &Function, stream: &Stream, grid: (u32,u32,u32), block: (u32,u32,u32));
}

macro_rules! impl_kernel_args {
    ($($T:ident),+) => {
        impl<$($T),+> KernelArgs for ($($T,)+)
        where
            $($T: DeviceCopy,)+
        {
            unsafe fn invoke(self, f: &Function, stream: &Stream, grid: (u32,u32,u32), block: (u32,u32,u32)) {
                let ($($T,)+) = self;
                launch!(f<<< grid, block, 0, stream >>>($($T),+)).unwrap();
            }
        }
    };
}

impl_kernel_args!(A);
impl_kernel_args!(A, B);
impl_kernel_args!(A, B, C);
impl_kernel_args!(A, B, C, D);
impl_kernel_args!(A, B, C, D, E);


pub struct Kernel<'m, T> {
    module: &'m Module,
    name: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    _t: PhantomData<T>,
}

impl<'m, T> Kernel<'m, T>
where T: DeviceCopy + 'static
{
    pub fn new(module: &'m Module, name: &'static str) -> Self {
        Self {
            module,
            name,
            block: (256, 1, 1),
            grid: (256, 1, 1),
            _t: PhantomData,
        }
    }

    pub fn tune_1d(&mut self, n: usize) {
        let bs = self.block.0;
        self.grid = ((n as u32 + bs - 1) / bs, 1, 1);
    }

    pub fn tune_2d(&mut self, t1: &Tensor<T, 2>, t2: &Tensor<T, 2>) {
        let bs = (16, 16, 1);

        let s1 = t1.shape();
        let s2 = t2.shape();
        let gs = (
            (s2[1] as usize + bs.0 as usize - 1) as u32 / bs.0,
            (s1[0] as usize + bs.1 as usize - 1) as u32 / bs.1,
            1,
        );
        self.block = bs;
        self.grid = gs;
    }


    pub fn get_cuda_type(&self) -> &'static str {
        let t = TypeId::of::<T>();

        if t == TypeId::of::<f32>() {
            return "float";
        }

        if t == TypeId::of::<f64>() {
            return "double";
        }

        panic!("Calling CUDA operations with unsupported types. Supported types: f32, f64");
    }

    pub unsafe fn launch<Args: KernelArgs + DeviceCopy>(&self, stream: &Stream, args: Args) {
        let t = self.get_cuda_type();
        let name = format!("{}_{}", self.name, t);
        let f = self.module.get_function(name).unwrap();
        unsafe { 
            args.invoke(&f, stream, self.grid, self.block);
        }

    }
}
