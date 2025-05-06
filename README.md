# tensoron - CUDA-Accelerated Linear Algebra in Rust

### Example
```rust
// Vectors ((1 x N) or (N x 1) matrices)
let v1 = tensor!([3, 1][1.0f32, 2.0, 3.0]);
let v2 = tensor!([3, 1][2.0, 4.0, 6.0]);

// The library is GPU-local. Always call to_host() before evaluating anything.
let v3 = (v1.clone() + v2.clone()).to_host();

assert_eq!(v3, tensor!([3,1][3.0, 6.0, 9.0]));
assert_eq!(v3.scale(10.0), tensor!([3,1][30.0, 60.0, 90.0]));

let v4 = (v1.transpose() * v2).to_host();
assert_eq!(v4, tensor!([1, 1][28.0]));

// Matrices
let m1 = tensor!([2, 2][1.0f32, 2.0, 3.0, 4.0]);
let m2 = tensor!([2, 2][2.0, 3.0, 4.0, 5.0]);

let m3 = (m1.clone() * m2.clone()).to_host();
assert_eq!(m3, tensor!([2, 2][10.0, 13.0, 22.0, 29.0]));

// Indexing
assert_eq!(m3.at([0, 1]).value(), 13.0);

let m4 = (m1 + m2).to_host();
assert_eq!(m4, tensor!([2, 2][3.0, 5.0, 7.0, 9.0]));
```


### Todo
- [ ] (!) Actually learn CUDA and write good kernels
- [ ] GPU operation traits: GPUMul, GPUAdd, etc. and then implement them for Tensor<T, 2> and Tensor<T, 1>
    - [ ] This way `execute_operation` can be gotten rid of
    - [ ] Treat Vector<T> as Tensor<T, 1> exclusively; stop using that name for Tensor<T, 2> altogether

- [x] Make the library GPU-local
    - Treat device pointers as authoritative data; don't discard them.
    - [ ] Do every operation on the GPU
    - [x] Redesign the `execute_operation` function to be more flexible and to support GPU locality (it's ugly right now)

- [x] Compile CUDA at build-time (build.rs)
    - [ ] Support other types besides f32 by replacing float* with whatever type is needed in the CUDA code (build.rs)

- [x] Basic vector operations: multiply by scalar, dot product, etc.
- [x] (!) Dynamically calculate grid sizes; don't just hardcode 3 threads with 1 block
- [x] `src/matrix.rs` comment, line 48
- [x] Operation enum and a wrapper for launching kernels to reduce redundancy
- [x] Matrix addition ~~Matrix type and its operations~~
- [x] Tensor indexing
