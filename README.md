# tensoron - CUDA-Accelerated Linear Algebra in Rust

### Todo
- [ ] (!) Actually learn CUDA and write good kernels
- [ ] Make the library GPU-local
    - [ ] Treat device pointers as authoritative data; don't discard them
    - [ ] Do every operation on the GPU
- [ ] Compile CUDA at build-time (build.rs)
    - [ ] Support other types besides f32 by replacing float* with whatever type is needed in the CUDA code (build.rs)
- [ ] `src/matrix.rs` comment, line 48
- [ ] Matrix addition ~~Matrix type and its operations~~

- [x] Basic vector operations: multiply by scalar, dot product, etc.
- [x] (!) Dynamically calculate grid sizes; don't just hardcode 3 threads with 1 block
