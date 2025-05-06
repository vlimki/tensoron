pub trait GpuAdd<Rhs = Self> {
    fn gpu_add(self, rhs: Self) -> Self;
}

pub trait GpuMul<Rhs = Self> {
    type Output;
    fn gpu_mul(self, rhs: Self) -> Self::Output;
}

pub trait GpuScale<Rhs = Self> {
    fn gpu_scale(self, rhs: Self) -> Self;
}
