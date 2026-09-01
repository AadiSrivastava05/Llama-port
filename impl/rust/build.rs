// Capture the rustc version at build time so the benchmark report can name the
// toolchain the way the C++ port names g++.

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "rustc (unknown version)".to_string());
    println!("cargo:rustc-env=MICROGPT_RUSTC={version}");
    println!("cargo:rerun-if-changed=build.rs");
}
