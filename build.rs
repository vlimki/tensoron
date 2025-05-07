use std::{env, fs::{self, OpenOptions}, io::Write, process::Command};

fn compile_ptx(src: &str, out: &str) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("nvcc")
        .args([
            "-ptx",
            src,
            "-o",
            out,
            "--use_fast_math",
            "-arch=sm_70",
        ])
        .status()?;

    if !status.success() {
        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Not coompiling")));
    }

    Ok(())
}

fn enabled_types() -> Vec<&'static str> {
    let mut types = vec![];

    if env::var("CARGO_FEATURE_F32").is_ok() {
        types.push("float");
    }

    if env::var("CARGO_FEATURE_F64").is_ok() {
        types.push("double");
    }
    
    types
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("mkdir")
        .arg(format!("{}/out", out_dir)).output().unwrap();

    for kernel in ["vector", "matrix", "tensor"] {
        let src = format!("kernels/{kernel}.cu");
        let dst = format!("{out_dir}/{kernel}.ptx");
        let mut replaced: Vec<String> = vec![];

        let contents = fs::read_to_string(&src).expect("Failed to read file");
        let content = contents.split("// KERNELS").collect::<Vec<&str>>()[1];

        for t in enabled_types() {
            if t != "float" {
                replaced.push(content.replace("float", t));
            }
        }

        println!("cargo:rerun-if-changed={}", src);

        let mut file = OpenOptions::new().append(true).open(&src).unwrap();
        file.write(replaced.join("\n").as_bytes()).unwrap();

        match compile_ptx(&src, &dst) {
            Ok(_) => {
                // Return the CUDA file to its original state.
                fs::write(&src, contents.as_bytes()).unwrap();
            }
            Err(_) => {
                fs::write(&src, contents.as_bytes()).unwrap();
                panic!("Failed to compile kernels to PTX. Do you have a viable GPU and `nvcc` installed?");
            }
        }
        println!("cargo:rerun-if-changed=build.rs");

    }
}
