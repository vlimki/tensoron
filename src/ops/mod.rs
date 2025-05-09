pub trait GpuAdd<Rhs = Self> {
    type Output;
    fn gpu_add(self, rhs: Self) -> Self::Output;
    fn gpu_cmul(self, rhs: Self) -> Self::Output;
    fn gpu_sub(self, rhs: Self) -> Self::Output;
}

pub trait GpuMul<Rhs = Self> {
    type Output;
    fn gpu_mul(self, rhs: Self) -> Self::Output;
}

pub trait GpuScale<T> {
    type Output;
    fn gpu_scale(self, rhs: T) -> Self::Output;
}

pub trait GpuTranspose<T> {
    type Output;
    fn gpu_transpose(self) -> Self::Output;
}

pub trait ML<T> {
    type Output;
    fn relu(self) -> Self::Output;
    fn relu_derivative(self) -> Self::Output;
    fn tanh(self) -> Self::Output;
    fn sigmoid(self) -> Self::Output;
    fn sigmoid_derivative(self) -> Self::Output;
}
