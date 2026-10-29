//! Embeds the kernel libraries (libnextcuda-<kind>.so and stable-diffusion.cpp's libstable-diffusion.so, from
//! NS_LIB_DIST, else dist/) into the binary: `LIBS`, each as (file name, bytes). nvcc is on the GPU host, not here: the
//! libraries are there to embed only when they were built first and brought back (tooling/deploy.sh with EMBED=1 does);
//! without them the set is empty and the libraries are opened from files (lib.rs). Silos stay files: they open on
//! demand.
use std::fmt::Write;
use std::path::PathBuf;

const LIBS: &[&str] = &["libnextcuda-llm.so", "libnextcuda-image.so", "libnextcuda-video.so", "libnextcuda-audio.so", "libnextcuda-model3d.so",
                        "libstable-diffusion.so"];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    println!("cargo:rerun-if-env-changed=NS_LIB_DIST");
    let dist = std::env::var("NS_LIB_DIST").map(PathBuf::from).unwrap_or_else(|_| manifest.join("../../dist"));
    let mut src = String::from("pub static LIBS: &[(&str, &[u8])] = &[\n");
    for name in LIBS {
        let p = dist.join(name);
        println!("cargo:rerun-if-changed={}", p.display());
        if p.is_file() {
            let abs = std::fs::canonicalize(&p).unwrap_or(p);
            writeln!(src, "    ({name:?}, include_bytes!({:?})),", abs.display().to_string()).expect("string write");
        }
    }
    src.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it")).join("libs_embedded.rs");
    std::fs::write(out, src).expect("write libs_embedded.rs");
}
