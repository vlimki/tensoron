# tensoron - CUDA-Accelerated Linear Algebra in Rust


### Todo
- [ ] (!) Actually learn CUDA and write good kernels
- [ ] Compile CUDA at build-time (build.rs)
    - [ ] Support other types besides f32 by replacing float* with whatever type is needed in the CUDA code (build.rs)
- [ ] Matrix type and its operations

- [x] Basic vector operations: multiply by scalar, dot product, etc.
- [x] (!) Dynamically calculate grid sizes; don't just hardcode 3 threads with 1 block
