//! The architecture's rules, checked on every `cargo test` (CONTRIBUTING.md explains them):
//!
//! 1. Layers: the foundation (crates/) depends only on the foundation; a kind's contract (<kind>/contract) on the
//!    foundation; an engine (<kind>/<arch>) on the foundation and its own kind's contract - never on another kind,
//!    the glue or the program; the glue (glue/) on the foundation and the contracts, never on an engine.
//! 2. One GPU runtime: CUDA. No crate that brings another (SYCL, Vulkan, OpenCL, Metal, wgpu, ONNX Runtime,
//!    PyTorch, candle, burn...) anywhere in Cargo.lock. (Phase 1 reaches CUDA only through the python workers and
//!    sd-cli; phase 2's Rust runtime may bring `cudarc`.)
//! 3. The project's own kernels (kernels/ns, kernels/<kind>/<arch>) include no other runtime's headers.
//! 4. Every environment variable the code reads has a row in docs/environment.md.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Every workspace crate: its directory (relative to the root) and its package name
fn crates() -> BTreeMap<String, String> {
    let root = root();
    let ws = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let members = ws.split("members = [").nth(1).unwrap().split(']').next().unwrap();
    let mut out = BTreeMap::new();
    for dir in members.split(',').map(|m| m.trim().trim_matches('"')).filter(|m| !m.is_empty()) {
        let toml = std::fs::read_to_string(root.join(dir).join("Cargo.toml")).unwrap();
        let name = toml.lines().find_map(|l| l.strip_prefix("name = \"")).unwrap().trim_end_matches('"').to_string();
        out.insert(name, dir.to_string());
    }
    out
}

/// A crate's dependencies' names (the [dependencies] table)
fn deps(dir: &str) -> Vec<String> {
    let toml = std::fs::read_to_string(root().join(dir).join("Cargo.toml")).unwrap();
    let mut on = false;
    let mut out = Vec::new();
    for l in toml.lines() {
        let l = l.trim();
        if l.starts_with('[') {
            on = l == "[dependencies]";
            continue;
        }
        if on && !l.is_empty() && !l.starts_with('#') {
            out.push(l.split(['=', '.', ' ']).next().unwrap().to_string());
        }
    }
    out
}

/// The layer of a crate directory: ("foundation" | "contract" | "engine" | "glue" | "program", its kind)
fn layer(dir: &str) -> (&'static str, String) {
    let mut parts = dir.split('/');
    let (top, sub) = (parts.next().unwrap(), parts.next().unwrap_or(""));
    match top {
        "crates" => ("foundation", String::new()),
        "glue" => ("glue", String::new()),
        "cli" => ("program", String::new()),
        kind if sub == "contract" => ("contract", kind.to_string()),
        kind => ("engine", kind.to_string()),
    }
}

#[test]
fn layers_depend_only_downward() {
    let crates = crates();
    let mut bad = Vec::new();
    for (name, dir) in &crates {
        let (l, kind) = layer(dir);
        for d in deps(dir) {
            let Some(ddir) = crates.get(&d) else { continue }; // a third-party crate (rule 2 checks those)
            let (dl, dkind) = layer(ddir);
            let ok = match l {
                "foundation" => dl == "foundation",
                "contract" => dl == "foundation",
                "engine" => dl == "foundation" || (dl == "contract" && dkind == kind),
                "glue" => dl == "foundation" || dl == "contract",
                _ => true,
            };
            if !ok {
                bad.push(format!("{name} ({l}{}) depends on {d} ({dl}{})", if kind.is_empty() { String::new() } else { format!(" {kind}") },
                                 if dkind.is_empty() { String::new() } else { format!(" {dkind}") }));
            }
        }
    }
    assert!(bad.is_empty(), "layering broken (CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

#[test]
fn cuda_is_the_only_gpu_runtime() {
    const OTHERS: &[&str] = &["wgpu", "vulkano", "ash", "opencl3", "ocl", "cl-sys", "metal", "candle-core", "tch", "sycl-sys",
                              "torch-sys", "ort", "onnxruntime", "burn", "burn-wgpu", "burn-cuda", "hip-sys", "rocm", "vulkanalia", "naga"];
    let lock = std::fs::read_to_string(root().join("Cargo.lock")).unwrap();
    let found: Vec<&str> = lock.lines().filter_map(|l| l.strip_prefix("name = \"")).map(|n| n.trim_end_matches('"'))
        .filter(|n| OTHERS.contains(n)).collect();
    assert!(found.is_empty(), "crates that bring another GPU runtime (only CUDA is allowed: CONTRIBUTING.md): {found:?}");
}

#[test]
fn own_kernels_include_no_other_runtime() {
    const OTHERS: &[&str] = &["sycl/", "CL/sycl", "vulkan", "CL/", "hip/", "Metal/", "level_zero/"];
    let k = root().join("kernels");
    let mut dirs = vec![k.join("ns")];
    for kind in ["llm", "image", "video", "audio"] {
        if let Ok(rd) = std::fs::read_dir(k.join(kind)) {
            dirs.extend(rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
        }
    }
    let mut bad = Vec::new();
    for d in dirs {
        for f in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = f.path();
            if !matches!(p.extension().and_then(|e| e.to_str()), Some("cpp" | "hpp" | "h")) {
                continue;
            }
            for (n, line) in std::fs::read_to_string(&p).unwrap_or_default().lines().enumerate() {
                let l = line.trim_start();
                if let Some(inc) = l.strip_prefix("#include") {
                    let inc = inc.trim().trim_start_matches(['<', '"']);
                    if OTHERS.iter().any(|o| inc.starts_with(o)) {
                        bad.push(format!("{}:{}: {}", p.display(), n + 1, line.trim()));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "another runtime's headers in the project's kernels (CUDA only: CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

/// Every environment variable the code reads, each with the first place it is read (file:line): `env::var("X")`,
/// `getenv("X")`, `option_env!("X")`, the settings file's `cfg.get("X")` / `cfg.or("X", ..)` / `with_prefix(..)`,
/// an engine's `setting("X")` and its option table's `env: "X"`, the settings page's `st("X", ..)`, the helpers that
/// wrap those (`port`, `peak`, `env_on`, `env_flag`, `lanes_env`, `cfg, "X"`, `("--flag", "X", ..)`), and the build
/// scripts' `${X:-default}`. Over *.rs, *.cpp, *.hpp, *.h and *.sh, kernels/strata included; target/ (build output),
/// dist/ (the built program) and node_modules/ are not sources. reference/ is ours (the parity tools) and is scanned.
fn env_reads() -> BTreeMap<String, String> {
    const CALLS: &[&str] = &["env::var(", "env::var_os(", "getenv(", "option_env!(", ".get(", ".or(", ".setting(", "with_prefix(",
                             "env_on(", "env_flag(", "lanes_env(", "port(", "peak(", "st(", "env:", "cfg,"];
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let name_at = |s: &str| -> String { s.chars().take_while(|&c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_').collect() };
    let root = root();
    let mut out = BTreeMap::new();
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            let file = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if !matches!(file.as_str(), "target" | "dist" | "node_modules") && !file.starts_with('.') {
                    stack.push(p);
                }
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext, "rs" | "cpp" | "hpp" | "h" | "sh") {
                continue;
            }
            let rel = p.strip_prefix(&root).unwrap_or(&p).display().to_string();
            for (n, line) in std::fs::read_to_string(&p).unwrap_or_default().lines().enumerate() {
                let mut found = Vec::new();
                if ext == "sh" {
                    for (i, _) in line.match_indices("${") {
                        let name = name_at(&line[i + 2..]);
                        // not ${1:-all}: an argument
                        if name.starts_with(|c: char| c.is_ascii_uppercase()) && line[i + 2 + name.len()..].starts_with([':', '-', '+', '=']) {
                            found.push(name);
                        }
                    }
                } else {
                    for (i, _) in line.match_indices('"') {
                        let name = name_at(&line[i + 1..]);
                        if name.len() < 2 || !name.starts_with(|c: char| c.is_ascii_uppercase()) || !line[i + 1 + name.len()..].starts_with('"') {
                            continue;
                        }
                        let before = line[..i].trim_end();
                        let call = CALLS.iter().any(|&c| {
                            before.strip_suffix(c).is_some_and(|b| c.starts_with(['.', ':']) || !b.ends_with(ident))
                        });
                        // ("--flag", "X", default): a flag and the setting it falls back to
                        let flag = before.strip_suffix(',').map(str::trim_end).and_then(|b| b.strip_suffix('"'))
                            .and_then(|b| b.rfind('"').map(|q| &b[q + 1..])).is_some_and(|f| f.starts_with("--"));
                        if call || flag {
                            found.push(name);
                        }
                    }
                }
                for name in found {
                    out.entry(name).or_insert_with(|| format!("{rel}:{}", n + 1));
                }
            }
        }
    }
    out
}

#[test]
fn every_environment_variable_is_documented() {
    let doc = std::fs::read_to_string(root().join("docs/environment.md")).unwrap();
    // the doc's words: a variable is documented when its name is one (NS_QW_ matches the pattern row `NS_QW_*`)
    let words: std::collections::BTreeSet<&str> = doc.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|w| !w.is_empty()).collect();
    let reads = env_reads();
    assert!(reads.len() > 20, "the scan found only {} variables: is the pattern list broken?", reads.len());
    let missing: Vec<String> = reads.iter().filter(|(name, _)| !words.contains(name.as_str())).map(|(name, at)| format!("{name} ({at})")).collect();
    assert!(missing.is_empty(), "environment variables read but not in docs/environment.md (a row each: CONTRIBUTING.md):\n  {}", missing.join("\n  "));
}

/// The Rust sources of the engines, the foundation and the glue (llm/, image/, video/, audio/, model3d/, crates/,
/// glue/), each with its lines up to its test module (`#[cfg(test)]` then `mod`), comments cut: (path relative to the
/// root, [(line number, code)])
fn engine_and_glue_code() -> Vec<(String, Vec<(usize, String)>)> {
    let root = root();
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = ["llm", "image", "video", "audio", "model3d", "crates", "glue"].iter().map(|d| root.join(d)).collect();
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                // (tests/, benches and examples are not engine code)
                if !matches!(name.as_str(), "target" | "tests" | "examples" | "benches" | "node_modules") && !name.starts_with('.') {
                    stack.push(p);
                }
                continue;
            }
            // (a test module of its own file: `#[cfg(test)] mod x_tests;`)
            if p.extension().and_then(|x| x.to_str()) != Some("rs") || name.ends_with("_tests.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            let mut code = Vec::new();
            for (n, l) in lines.iter().enumerate() {
                if l.trim() == "#[cfg(test)]" && lines.get(n + 1).is_some_and(|x| x.trim_start().starts_with("mod ")) {
                    break;
                }
                // the line without its comment: `//` outside a string literal
                let (mut in_str, mut cut) = (false, l.len());
                let b = l.as_bytes();
                for i in 0..b.len() {
                    match b[i] {
                        b'"' if i == 0 || b[i - 1] != b'\\' => in_str = !in_str,
                        b'/' if !in_str && b.get(i + 1) == Some(&b'/') => {
                            cut = i;
                            break;
                        }
                        _ => {}
                    }
                }
                code.push((n + 1, l[..cut].to_string()));
            }
            out.push((p.strip_prefix(&root).unwrap().display().to_string(), code));
        }
    }
    out
}

/// The string literals of a line of code
fn literals(code: &str) -> Vec<&str> {
    code.split('"').skip(1).step_by(2).collect()
}

/// What the device reports, not a card's name or size: no GPU name in a string literal of the engines', the
/// foundation's and the glue's code, and no VRAM size written as a number, outside comments, tests and this list (each
/// with why)
const CARD_ALLOWED: &[(&str, &str)] = &[
    // a refusal's words name the cards shorter ("RTX 3060", not "NVIDIA GeForce RTX 3060"): display only, nothing plans
    ("glue/serve/src/home/pulls.rs", "replace(\"NVIDIA GeForce \", \"\")"),
    // an option's help line says where int8 pays (a GeForce card's int8 rate): words, the option is a flag
    ("image/qwenimage21/src/rust/mod.rs", "the card's int8 tensor cores"),
];

/// Rule 5 (memory): engines plan from what the device reports (`Gpu::free_or_guess`, `memory()`, `gpu_units`), never a
/// card's name or an assumed size
#[test]
fn engines_plan_from_the_device_not_a_cards_name_or_size() {
    const NAMES: &[&str] = &["RTX 5070", "RTX 3060", "RTX 4090", "GeForce", "Blackwell", "Ampere"];
    // a card's size as a number: MiB on a line of card memory (`vram`, `_mb`), GiB shifted or multiplied on a line that
    // reads the card's memory (a model's own need, or a host-memory figure, is on neither)
    const MIB: &[&str] = &["8192", "11264", "12227", "12288", "16384", "20480", "24576", "32768", "49152"];
    let mib_line = |l: &str| { let l = l.to_lowercase(); l.contains("vram") || l.contains("_mb") };
    let gib_line = |l: &str| { let l = l.to_lowercase(); ["vram", ".memory()", "card_mb", "total_mb", "mem_gib", "free_or_guess", "mem_cap", "mem_free"].iter().any(|h| l.contains(h)) };
    let gib = |code: &str| -> bool {
        let c = code.replace(' ', "");
        (6..=96).any(|n: u32| ["<<30", "*GIB", "u64<<30", "usize<<30"].iter().any(|op| {
            c.match_indices(&format!("{n}{op}")).any(|(i, _)| i == 0 || !c.as_bytes()[i - 1].is_ascii_alphanumeric())
        }))
    };
    let allowed = |path: &str, code: &str| CARD_ALLOWED.iter().any(|(p, what)| *p == path && code.contains(what));
    let mut bad = Vec::new();
    for (path, lines) in engine_and_glue_code() {
        for (n, code) in &lines {
            if allowed(&path, code) {
                continue;
            }
            if let Some(name) = literals(code).iter().find_map(|s| NAMES.iter().find(|x| s.contains(*x))) {
                bad.push(format!("{path}:{n}: the card name {name:?} (plan from gpu_units / gpu.memory(), not a name)"));
            }
            let mib = MIB.iter().find(|m| code.match_indices(*m).any(|(i, _)| {
                let b = code.as_bytes();
                (i == 0 || !b[i - 1].is_ascii_alphanumeric()) && b.get(i + m.len()).is_none_or(|c| !c.is_ascii_digit())
            }));
            if (mib.is_some() && mib_line(code)) || (gib(code) && gib_line(code)) {
                bad.push(format!("{path}:{n}: a card's memory as a number: {}", code.trim()));
            }
        }
    }
    assert!(bad.is_empty(), "engines and the glue plan from what the device reports (CONTRIBUTING rule 5, docs/architecture.md); \
                             a needed exception goes in CARD_ALLOWED with why:\n  {}", bad.join("\n  "));
}

/// Rule 5: a silo is per model architecture, never per card - no `per-card` build, no NS_SILO_SM, no
/// libnextcuda-<arch>.sm<NN>.so - and its engine opens it (nextcuda_core::silo / silo_for), so without the file it
/// runs on the shared kernels
#[test]
fn silos_are_per_architecture_and_their_engines_own() {
    let root = root();
    let mut bad = Vec::new();
    let mut silos = 0;
    for kind in ["llm", "image", "video", "audio", "model3d"] {
        for e in std::fs::read_dir(root.join("kernels").join(kind)).into_iter().flatten().flatten() {
            let silo = e.path().join("silo");
            if !silo.is_dir() {
                continue;
            }
            silos += 1;
            let arch = e.file_name().to_string_lossy().to_string();
            if silo.join("per-card").exists() {
                bad.push(format!("kernels/{kind}/{arch}/silo/per-card: a silo is one library for every card (a class of card: a variant)"));
            }
            for f in std::fs::read_dir(&silo).into_iter().flatten().flatten() {
                if std::fs::read_to_string(f.path()).unwrap_or_default().contains("NS_SILO_SM") {
                    bad.push(format!("kernels/{kind}/{arch}/silo/{}: NS_SILO_SM (a build a card): decide by what the card reports, or a variant", f.file_name().to_string_lossy()));
                }
            }
            let src = root.join(kind).join(&arch).join("src");
            let opens = std::fs::read_dir(&src).into_iter().flatten().flatten().any(|f| {
                let t = std::fs::read_to_string(f.path()).unwrap_or_default();
                t.contains("nextcuda_core::silo(") || t.contains("nextcuda_core::silo_for(")
            });
            if !opens {
                bad.push(format!("kernels/{kind}/{arch}/silo exists but {kind}/{arch}/src never opens it (nextcuda_core::silo)"));
            }
        }
    }
    assert!(silos > 0, "no silos found: is the scan broken?");
    let build = std::fs::read_to_string(root.join("kernels/build.sh")).unwrap();
    assert!(!build.contains("NS_SILO_SM") && !build.contains(".sm$a.so"), "kernels/build.sh builds a silo a card at a time");
    assert!(bad.is_empty(), "silos broken (rule 5, CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

/// Every silo variant: (kind, the silo's arch, its name, its directory)
fn variant_dirs() -> Vec<(String, String, String, PathBuf)> {
    let root = root();
    let mut out = Vec::new();
    for kind in ["llm", "image", "video", "audio", "model3d"] {
        for e in std::fs::read_dir(root.join("kernels").join(kind)).into_iter().flatten().flatten() {
            let arch = e.file_name().to_string_lossy().to_string();
            for v in std::fs::read_dir(e.path().join("silo/variants")).into_iter().flatten().flatten() {
                if v.path().is_dir() {
                    out.push((kind.to_string(), arch.clone(), v.file_name().to_string_lossy().to_string(), v.path()));
                }
            }
        }
    }
    out.sort();
    out
}

/// Every silo variant whose manifest reads: (kind, arch, name, manifest) - one that does not read is left to
/// `silo_variants_are_deltas_off_their_silo`
fn silo_variants() -> Vec<(String, String, String, nextcuda_core::variant::Manifest)> {
    variant_dirs().into_iter().filter_map(|(k, a, n, d)| {
        let m = nextcuda_core::variant::parse_manifest(&std::fs::read_to_string(d.join("variant.yml")).ok()?).ok()?;
        Some((k, a, n, m))
    }).collect()
}

/// Rule 5: a silo variant is a delta off its silo - a manifest that reads, a name of [a-z0-9-], and only ops its silo has
#[test]
fn silo_variants_are_deltas_off_their_silo() {
    let root = root();
    let mut bad = Vec::new();
    for (kind, arch, name, dir) in variant_dirs() {
        let at = format!("kernels/{kind}/{arch}/silo/variants/{name}");
        // the silo's ops: every ns_* its headers declare or its sources define
        let mut ops = std::collections::BTreeSet::new();
        for f in std::fs::read_dir(root.join("kernels").join(&kind).join(&arch).join("silo")).into_iter().flatten().flatten() {
            let t = std::fs::read_to_string(f.path()).unwrap_or_default();
            for (i, _) in t.match_indices("ns_") {
                let id: String = t[i..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
                if t[i + id.len()..].starts_with('(') {
                    ops.insert(id);
                }
            }
        }
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
            bad.push(format!("{at}: a variant's name is [a-z0-9-] (it is part of the library's file name)"));
        }
        match std::fs::read_to_string(dir.join("variant.yml")) {
            Ok(t) => {
                if let Err(x) = nextcuda_core::variant::parse_manifest(&t) {
                    bad.push(format!("{at}/variant.yml: {x}"));
                }
            }
            Err(_) => bad.push(format!("{at}: no variant.yml")),
        }
        let mut exports = 0;
        for f in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let p = f.path();
            if !matches!(p.extension().and_then(|x| x.to_str()), Some("cu" | "cuh" | "h")) {
                continue;
            }
            let t = std::fs::read_to_string(&p).unwrap_or_default();
            let file = format!("{at}/{}", f.file_name().to_string_lossy());
            // its exports: the C names of an extern "C" block or line
            let mut in_c = false;
            for l in t.lines() {
                if l.starts_with("extern \"C\" {") {
                    in_c = true;
                } else if in_c && l.starts_with('}') {
                    in_c = false;
                }
                if !(in_c || l.starts_with("extern \"C\"")) || l.starts_with(' ') || l.starts_with('\t') {
                    continue;
                }
                let Some(i) = l.find("ns_") else { continue };
                let id: String = l[i..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
                if !l[i + id.len()..].starts_with('(') {
                    continue;
                }
                exports += 1;
                if !ops.contains(&id) {
                    bad.push(format!("{file}: {id} is not an op of the {arch} silo (a variant is a delta, not a fork)"));
                }
            }
        }
        if exports == 0 {
            bad.push(format!("{at}: no op (an empty variant)"));
        }
    }
    assert!(bad.is_empty(), "silo variants broken (rule 5, CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

/// The options an engine declares, read from its sources (every `EngineOption { name: "..", env: "..", value: "..",
/// .. at: At::.. }`) - the engine of silo `kernels/<kind>/<arch>`: the crate `<kind>/<arch>` - with its kind's common
/// ones (`<kind>/contract`). None: no engine found.
fn engine_options_of(kind: &str, arch: &str) -> Option<Vec<nextcuda_core::EngineOption>> {
    let root = root();
    let sources = |dir: &Path| -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![dir.join("src")];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(std::fs::read_to_string(&p).unwrap_or_default());
                }
            }
        }
        out
    };
    let own = root.join(kind).join(arch);
    if !own.join("Cargo.toml").exists() {
        return None;
    }
    let quoted = |s: &str| -> String {
        let mut out = String::new();
        let mut esc = false;
        for c in s.chars() {
            match (esc, c) {
                (true, c) => {
                    out.push(c);
                    esc = false;
                }
                (false, '\\') => esc = true,
                (false, '"') => break,
                (false, c) => out.push(c),
            }
        }
        out
    };
    let field = |block: &str, f: &str| block.find(&format!("{f}: \"")).map(|i| quoted(&block[i + f.len() + 3..]));
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let mut out = Vec::new();
    for c in [own, root.join(kind).join("contract")] {
        for t in sources(&c) {
            for (i, _) in t.match_indices("EngineOption {") {
                let block = &t[i..];
                let Some(a) = block.find("at: At::") else { continue };
                let at = match block[a + 8..].get(..4) {
                    Some("Load") => nextcuda_core::At::Load,
                    Some("Requ") => nextcuda_core::At::Request,
                    _ => nextcuda_core::At::Both,
                };
                let block = &block[..a];
                let Some(name) = field(block, "name") else { continue };
                out.push(nextcuda_core::EngineOption { name: leak(name), env: leak(field(block, "env").unwrap_or_default()),
                                                       value: leak(field(block, "value").unwrap_or_default()), help: "", at });
            }
        }
    }
    Some(out)
}

/// Rule 5: a variant's `settings` are options its engine declares, at load, with values their shapes take and no
/// memory size (`nextcuda_core::options::variant_settings`, what the load itself checks)
#[test]
fn variant_settings_are_options_their_engine_takes() {
    let mut bad = Vec::new();
    for (kind, arch, name, m) in &silo_variants() {
        if m.settings.is_empty() {
            continue;
        }
        let at = format!("kernels/{kind}/{arch}/silo/variants/{name}/variant.yml");
        match engine_options_of(kind, arch) {
            None => bad.push(format!("{at}: settings, but no engine crate {kind}/{arch}/")),
            Some(opts) => {
                if let Err(e) = nextcuda_core::options::variant_settings(&m.settings, &opts, name) {
                    bad.push(format!("{at}: {e}"));
                }
            }
        }
    }
    // the scan reads an engine's table as the program has it (qwen35's own options and the llm kind's common ones)
    let q = engine_options_of("llm", "qwen35").unwrap();
    for o in nextcuda_llm_qwen35::kind().options {
        assert!(q.iter().any(|x| x.name == o.name && x.env == o.env && x.value == o.value && x.at == o.at), "the option scan misread {}", o.name);
    }
    assert!(bad.is_empty(), "silo variant settings broken (rule 5, CONTRIBUTING.md):\n  {}", bad.join("\n  "));
}

/// Rule 5: two variants of one silo never match the same card - their `match` ranges must not meet on some number
/// (`Match::overlaps`: a name never separates two rules). The fix is a tighter `match`.
#[test]
fn variants_of_one_silo_never_match_the_same_card() {
    let vs = silo_variants();
    let mut bad = Vec::new();
    for (i, a) in vs.iter().enumerate() {
        for b in &vs[i + 1..] {
            if (&a.0, &a.1) == (&b.0, &b.1) && a.3.rule.overlaps(&b.3.rule) {
                bad.push(format!("kernels/{}/{}/silo/variants: {} and {} can both match one card ({:?} / {:?})", a.0, a.1, a.2, b.2, a.3.rule, b.3.rule));
            }
        }
    }
    assert!(bad.is_empty(), "ambiguous silo variants (rule 5, CONTRIBUTING.md: give one a tighter match):\n  {}", bad.join("\n  "));
}

/// A baseline card's `device` (one card's JSON, or a GPU set's list) as a variant's rule reads it
fn devices(d: &serde_json::Value) -> Vec<nextcuda_core::variant::DeviceInfo> {
    let one = |d: &serde_json::Value| nextcuda_core::variant::DeviceInfo {
        index: 0,
        name: d["name"].as_str().unwrap_or("").to_string(),
        units: d["units"].as_u64().unwrap_or(0) as u32,
        mhz: d["mhz"].as_u64().unwrap_or(0) as u32,
        vram_mb: d["vram_mb"].as_u64(),
        cc: d["cc"].as_u64().map(|c| c as u32),
    };
    match d.as_array() {
        Some(a) => a.iter().map(one).collect(),
        None => vec![one(d)],
    }
}

/// The gate: every card recorded says what it is, under the id the gate gives it; every variant has a baseline
/// (tooling/gate-baselines.json, its engine `<kind>/<arch>`) on at least one card its `match` accepts - recorded on
/// that card (`NS_GATE_RECORD=1 tooling/gate.sh`) and committed in the variant's pull request
#[test]
fn every_silo_variant_has_a_gate_baseline_on_a_card_it_matches() {
    let b: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root().join("tooling/gate-baselines.json")).unwrap()).unwrap();
    let mut bad = Vec::new();
    for (eng, e) in b["engines"].as_object().unwrap() {
        for (id, c) in e["cards"].as_object().into_iter().flatten() {
            let ds = devices(&c["device"]);
            if ds.iter().any(|d| d.name.is_empty() || d.units == 0) {
                bad.push(format!("gate-baselines.json {eng} card {id}: no device (what the card reports: tooling/gate.sh records it)"));
                continue;
            }
            let ids: Vec<String> = ds.iter().map(|d| d.card_id()).collect();
            if ids.join("+") != *id {
                bad.push(format!("gate-baselines.json {eng} card {id}: its device's id is {}", ids.join("+")));
            }
        }
    }
    for (kind, arch, name, m) in silo_variants() {
        let engine = format!("{kind}/{arch}");
        let cards: Vec<&String> = b["engines"].as_object().unwrap().iter().filter(|(k, _)| k.split('#').next() == Some(engine.as_str()))
            .flat_map(|(_, e)| e["cards"].as_object().into_iter().flatten())
            .filter(|(_, c)| devices(&c["device"]).iter().any(|d| m.rule.matches(d))).map(|(id, _)| id).collect();
        if cards.is_empty() {
            bad.push(format!("kernels/{kind}/{arch}/silo/variants/{name}: no gate baseline for {engine} on a card it matches \
                              (record one on that card: NS_GATE_HOST=local NS_GATE_RECORD=1 tooling/gate.sh HEAD HEAD {engine})"));
        }
    }
    assert!(bad.is_empty(), "gate baselines missing (CONTRIBUTING.md, the merge gate):\n  {}", bad.join("\n  "));
}
