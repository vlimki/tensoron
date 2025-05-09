use std::{env, fs::{self}, path::Path, process::Command};

fn compile_ptx(src: &str, out: &str) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("nvcc")
        .args(["-ptx", src, "-o", out, "--use_fast_math", "-arch=sm_70"])
        .status()?;

    if !status.success() {
        return Err("nvcc failed".into());
    }
    Ok(())
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let mut constants = String::new();


    for kernel in ["vector", "matrix", "tensor"] {
        let src = format!("kernels/{kernel}.cu");
        let dst = format!("{out_dir}/{kernel}.ptx");

        println!("cargo:rerun-if-changed={}", src);

        match compile_ptx(&src, &dst) {
            Ok(_) => (),
            Err(_) => {
                panic!("Failed to compile kernels to PTX. Do you have a viable GPU and `nvcc` installed?");
            }
        }

        let ptx = fs::read_to_string(&dst).unwrap();

        constants.push_str(&format!(
            "pub const {}_PTX: &str = r#\"{}\"#;\n",
            kernel.to_uppercase(),
            ptx.replace(r#"""#, r#"\""#)
        ));

    }

    let dest = Path::new(&out_dir).join("kernels.rs");
    fs::write(dest, constants).unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}
