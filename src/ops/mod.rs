pub trait GpuAdd<Rhs = Self> {
    type Output;
    fn gpu_add(self, rhs: Self) -> Self::Output;
}

pub trait GpuMul<Rhs = Self> {
    type Output;
    fn gpu_mul(self, rhs: Self) -> Self::Output;
}

pub trait GpuScale<T> {
    type Output;
    fn gpu_scale(self, rhs: T) -> Self::Output;
}

pub trait ML<T> {
    fn relu(self) -> Self;
    fn tanh(self) -> Self;
    fn sigmoid(self) -> Self;
}
