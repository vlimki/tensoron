# tensoron - CUDA-Accelerated Linear Algebra in Rust

### Example
```rust
use crate::tensor;

let v1 = tensor!([3, 1][1.0f32, 2.0, 3.0]);
let v2 = tensor!([3, 1][2.0, 4.0, 6.0]);

let v3 = v1.clone() + v2.clone();

let v4 = v1 * v2.transpose();

assert_eq!(v3, tensor!([3,1][3.0, 6.0, 9.0]));
assert_eq!(v3.scale(10.0), tensor!([3,1][30.0, 60.0, 90.0]));
assert_eq!(v4, tensor!([1, 1][28.0]));
```


### Todo
- [ ] (!) Actually learn CUDA and write good kernels
- [ ] Make the library GPU-local
    - Treat device pointers as authoritative data; don't discard them.
    - [ ] Do every operation on the GPU
    - [ ] Redesign the execute_operation function to be more flexible and to support GPU locality (it's ugly right now)
- [x] Compile CUDA at build-time (build.rs)
    - [ ] Support other types besides f32 by replacing float* with whatever type is needed in the CUDA code (build.rs)

- [x] Basic vector operations: multiply by scalar, dot product, etc.
- [x] (!) Dynamically calculate grid sizes; don't just hardcode 3 threads with 1 block
- [x] `src/matrix.rs` comment, line 48
- [x] Operation enum and a wrapper for launching kernels to reduce redundancy
- [x] Matrix addition ~~Matrix type and its operations~~
