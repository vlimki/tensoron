# tensoron - CUDA-Accelerated Tensor Computation with Rust

- [Development log](https://vlimki.dev/writing/tensoron)

### Features
- **Support for multiple number types**: the library will dynamically compile a CUDA kernel at build time for each type the user is using in their code.
- **GPU locality**

### Example
```rust
// Vectors (Vector<T>)
let v1 = tensor!([3][1.0f32, 2.0, 3.0]);
let v2 = tensor!([3][2.0, 4.0, 6.0]);

let v3 = (v1.clone() + v2.clone()).cpu();
assert_eq!(v3, tensor!([3][3.0, 6.0, 9.0]));
assert_eq!(v3.clone().scale(10.0).cpu(), tensor!([3][30.0, 60.0, 90.0]));

let v4 = v1 * v2;
assert_eq!(v4, 28.0);

// Matrices (Matrix<T>)
let m1 = tensor!([2, 2][1.0f32, 2.0, 3.0, 4.0]);
let m2 = tensor!([2, 2][2.0, 3.0, 4.0, 5.0]);

let m3 = (m1 * m2).cpu();
assert_eq!(m3, tensor!([2, 2][10.0, 13.0, 22.0, 29.0]));

// `.at()` returns a Tensor<T, R - N>, where N is the length of the index array.
// A 2D index on a 2D tensor returns a Tensor<T, 0>, i.e. a Scalar<T>, on which you can call `.value()`.
assert_eq!(m3.at([0, 1]).value(), 13.0);
```

### Todo
- [ ] (!) Actually learn CUDA and write good kernels
    - [ ] Not just good kernels, but also hyper-optimize the crap out of them
- [ ] Tensor views
    - [ ] Proper indexing
    - [ ] Make strides actually do something
    - [ ] Constructing tensors from tensor views
- [x] Nicer API around calling kernels
- [ ] Start using GitHub issues instead of this TODO list lol
- [x] Change tensor.inner to an Option
- [x] Make the library GPU-local
    - Treat device pointers as authoritative data; don't discard them.
    - [x] Do every operation on the GPU
    - [x] Redesign the `execute_operation` function to be more flexible and to support GPU locality (it's ugly right now)
- [x] Transposition operator (for matrices)
