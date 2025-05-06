use std::{env, process::Command};

fn compile_ptx(src: &str, out: &str) {
    let status = Command::new("nvcc")
        .args([
            "-ptx",
            src,
            "-o",
            out,
            "--use_fast_math",
            "-arch=sm_75", // or adjust based on target GPU
        ])
        .status()
        .expect("Failed to run nvcc");

    if !status.success() {
        panic!("nvcc failed to compile {}", src);
    }
}
fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("mkdir")
        .arg(format!("{}/out", out_dir)).output().unwrap();
    


    for kernel in ["vector", "matrix"] {
        let src = format!("kernels/{kernel}.cu");
        let dst = format!("{out_dir}/{kernel}.ptx");
        #[cfg(debug_assertions)]
        std::fs::write("debug.txt", &dst).unwrap();

        compile_ptx(&src, &dst);

        println!("cargo:rerun-if-changed={}", src);
        println!("cargo:rerun-if-changed=build.rs");
    }
}
