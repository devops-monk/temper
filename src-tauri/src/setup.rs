//! First-run setup: installs uv, a uv-managed Python and the engine's packages into app data.
//!
//! Nothing outside the app-data folder is touched. The installed state is recorded with a hash
//! of the bundled engine, so an app update that changes the engine re-syncs it automatically.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const UV_VERSION: &str = "0.12.4";
const PYTHON_VERSION: &str = "3.12";
const STATE_FILE: &str = "engine-state.json";
const LOG_LINES: usize = 400;

/// uv release archives and their published SHA-256 checksums, per target.
const UV_ARCHIVES: &[(&str, &str, &str)] = &[
    ("aarch64-apple-darwin", "tar.gz", "99a913b606194867b43086404412c1afe079547fee72ecfb6af7e7b0dd54b0c6"),
    ("x86_64-apple-darwin", "tar.gz", "e603f1eb634ca97a2a125539b983891f53235e901511ed10c32c08c86e253ecd"),
    ("aarch64-unknown-linux-gnu", "tar.gz", "49d881b3403187e1f1789720881e77e4251ad4259d86c4844862657d2a35d13f"),
    ("x86_64-unknown-linux-gnu", "tar.gz", "c8c60f47e6f88d18dbf6f33d7279fb1fbf7ae76631768152cf5578c3d65729b4"),
    ("x86_64-pc-windows-msvc", "zip", "4f3b7d63cd81fca0da5a655d973d20affca89ce6e5f9a29fd0183cc4204a7639"),
    ("aarch64-pc-windows-msvc", "zip", "3290abffee78c30e3113f5113e26684fd057287e89124a588dcdcdd6ceec0fea"),
];

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// GGUF chat through llama.cpp; no PyTorch.
    Chat,
    /// Chat plus fine-tuning: MLX on Apple Silicon, PyTorch on NVIDIA/AMD.
    Train,
}

impl std::str::FromStr for Profile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "chat" => Ok(Profile::Chat),
            "train" => Ok(Profile::Train),
            _ => Err(format!("unknown profile {s:?}; use chat or train")),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Hardware {
    /// apple | nvidia | amd | none
    pub kind: &'static str,
    pub name: Option<String>,
    pub can_train: bool,
}

#[derive(Serialize, Clone)]
pub struct ProfileOption {
    pub id: Profile,
    pub label: String,
    pub description: String,
    pub size: String,
    pub available: bool,
    pub recommended: bool,
}

#[derive(Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StepState {
    Pending,
    Running,
    Done,
    Failed,
}

#[derive(Serialize, Clone)]
pub struct Step {
    pub id: &'static str,
    pub label: String,
    pub state: StepState,
}

/// Everything the setup screen shows. The UI polls it through `engine_info`.
#[derive(Serialize, Clone, Default)]
pub struct SetupState {
    pub hardware: Option<Hardware>,
    pub options: Vec<ProfileOption>,
    /// What is installed now, if anything.
    pub installed: Option<Profile>,
    /// What is being installed.
    pub profile: Option<Profile>,
    pub steps: Vec<Step>,
    /// Fraction of the current step, when it can be measured honestly.
    pub progress: Option<f64>,
    pub log: Vec<String>,
    pub error: Option<String>,
}

pub type Shared = Arc<Mutex<SetupState>>;

#[derive(Clone)]
pub struct Layout {
    pub data: PathBuf,
    /// The engine project: bundled resources in release, `../engine` in development.
    pub project: PathBuf,
}

impl Layout {
    fn bin(&self) -> PathBuf {
        self.data.join("bin")
    }
    fn uv(&self) -> PathBuf {
        self.bin().join(exe("uv"))
    }
    fn python_dir(&self) -> PathBuf {
        self.data.join("python")
    }
    pub fn venv(&self) -> PathBuf {
        self.data.join("engine-venv")
    }
    fn venv_bin(&self) -> PathBuf {
        self.venv().join(if cfg!(windows) { "Scripts" } else { "bin" })
    }
    fn venv_python(&self) -> PathBuf {
        self.venv_bin().join(exe("python"))
    }
    pub fn engine_exe(&self) -> PathBuf {
        self.venv_bin().join(exe("temper-engine"))
    }
    fn state_file(&self) -> PathBuf {
        self.data.join(STATE_FILE)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Installed {
    pub app_version: String,
    pub engine_hash: String,
    pub profile: Profile,
}

pub enum Check {
    Ready,
    FirstRun,
    /// Installed before, but the app was updated or the environment is damaged; reinstall this profile.
    Outdated(Profile),
}

pub fn installed(layout: &Layout) -> Option<Installed> {
    serde_json::from_slice(&fs::read(layout.state_file()).ok()?).ok()
}

pub fn check(layout: &Layout, app_version: &str) -> Check {
    match installed(layout) {
        None => Check::FirstRun,
        Some(i) if i.app_version == app_version && i.engine_hash == engine_hash(&layout.project) && layout.engine_exe().is_file() => Check::Ready,
        Some(i) => Check::Outdated(i.profile),
    }
}

// ------------------------------------------------------------------ hardware

pub fn detect() -> Hardware {
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        let name = output("sysctl", &["-n", "machdep.cpu.brand_string"]);
        return Hardware { kind: "apple", name, can_train: true };
    }
    if let Some(out) = output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]) {
        let name = out.lines().next().map(str::to_string);
        return Hardware { kind: "nvidia", name, can_train: true };
    }
    // GPU training supports AMD through ROCm on Linux only.
    if cfg!(target_os = "linux") && Path::new("/dev/kfd").exists() {
        let name = output("rocm-smi", &["--showproductname"]).and_then(|o| {
            o.lines().find(|l| l.contains("Card Series")).map(|l| l.rsplit(':').next().unwrap_or(l).trim().to_string())
        });
        return Hardware { kind: "amd", name, can_train: true };
    }
    Hardware { kind: "none", name: None, can_train: false }
}

pub fn options(hw: &Hardware) -> Vec<ProfileOption> {
    let (train_desc, train_size) = match hw.kind {
        "apple" => ("Adds MLX for fine-tuning models on Apple Silicon.", "about 600 MB"),
        _ => ("Adds PyTorch and training libraries for fine-tuning models on your GPU.", "about 6 GB"),
    };
    vec![
        ProfileOption {
            id: Profile::Train,
            label: "Chat and training".into(),
            description: if hw.can_train {
                train_desc.into()
            } else {
                "Needs an NVIDIA GPU, an AMD GPU on Linux, or an Apple Silicon Mac.".into()
            },
            size: train_size.into(),
            available: hw.can_train,
            recommended: hw.can_train,
        },
        ProfileOption {
            id: Profile::Chat,
            label: "Chat only".into(),
            description: "Run GGUF models with llama.cpp. You can add training later in Settings.".into(),
            size: "about 150 MB".into(),
            available: true,
            recommended: !hw.can_train,
        },
    ]
}

// ------------------------------------------------------------------ install

pub fn install(layout: &Layout, profile: Profile, hw: &Hardware, app_version: &str, shared: &Shared, cancel: &AtomicBool) -> Result<(), String> {
    let apple = hw.kind == "apple";
    let gpu_training = profile == Profile::Train && !apple;
    let mut steps = vec![
        ("uv", "Download the package manager".to_string()),
        ("python", format!("Install Python {PYTHON_VERSION}")),
        ("engine", if profile == Profile::Train && apple { "Install the engine and MLX" } else { "Install the engine" }.to_string()),
    ];
    if gpu_training {
        steps.push(("training", "Install PyTorch and training libraries".to_string()));
    }
    {
        let mut s = shared.lock().unwrap();
        s.profile = Some(profile);
        s.steps = steps.into_iter().map(|(id, label)| Step { id, label, state: StepState::Pending }).collect();
        s.progress = None;
        s.log.clear();
        s.error = None;
    }
    fs::create_dir_all(&layout.data).map_err(err)?;
    check_disk(layout, profile, apple, shared)?;

    let uv = run_step(shared, "uv", || ensure_uv(layout, shared, cancel))?;

    run_step(shared, "python", || {
        let mut cmd = uv_command(&uv, layout);
        cmd.args(["python", "install", PYTHON_VERSION]);
        run_logged(cmd, shared, cancel)
    })?;

    run_step(shared, "engine", || {
        let mut cmd = uv_command(&uv, layout);
        // --frozen: install exactly the shipped lock and never rewrite it (the bundle is read-only).
        // --no-editable: copy the engine into the venv; AppImage mounts the bundle at a new path each run.
        cmd.args(["sync", "--frozen", "--no-dev", "--no-editable", "--python", PYTHON_VERSION, "--project"]).arg(&layout.project);
        if profile == Profile::Train && apple {
            cmd.args(["--extra", "mlx"]);
        }
        run_logged(cmd, shared, cancel)
    })?;

    if gpu_training {
        run_step(shared, "training", || {
            // `uv sync` cannot pick a PyTorch build per machine; --torch-backend=auto matches the
            // installed CUDA driver or ROCm. On Windows torchao has to come from PyPI first, because
            // the PyTorch index has no recent Windows build of it.
            if cfg!(windows) {
                let mut cmd = uv_command(&uv, layout);
                cmd.args(["pip", "install", "--python"]).arg(layout.venv_python()).arg("torchao");
                run_logged(cmd, shared, cancel)?;
            }
            let mut cmd = uv_command(&uv, layout);
            cmd.args(["pip", "install", "--python"]).arg(layout.venv_python()).args(["unsloth", "--torch-backend=auto"]);
            run_logged(cmd, shared, cancel)
        })?;
    }

    let state = Installed { app_version: app_version.into(), engine_hash: engine_hash(&layout.project), profile };
    let tmp = layout.state_file().with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(&state).map_err(err)?).map_err(err)?;
    fs::rename(&tmp, layout.state_file()).map_err(err)?;
    let mut s = shared.lock().unwrap();
    s.installed = Some(profile);
    s.progress = None;
    log_locked(&mut s, "Setup finished".into());
    Ok(())
}

/// Stops before downloading anything when the disk can't hold the install. Needs about twice the
/// installed size, because uv keeps a download cache next to the environment.
fn check_disk(layout: &Layout, profile: Profile, apple: bool, shared: &Shared) -> Result<(), String> {
    const GB: u64 = 1 << 30;
    let needed = match profile {
        Profile::Chat => GB / 2,
        Profile::Train if apple => 3 * GB / 2,
        Profile::Train => 12 * GB,
    };
    // If the free space can't be read, let the install run; a full disk still gets a clear error.
    let Ok(free) = fs4::available_space(&layout.data) else { return Ok(()) };
    if free >= needed {
        return Ok(());
    }
    let gb = |b: u64| format!("{:.1} GB", b as f64 / GB as f64);
    let message = format!(
        "Temper needs about {} of free disk space for this setup, and {} is free on the disk that holds {}. Free up some space and try again.",
        gb(needed),
        gb(free),
        layout.data.display()
    );
    let mut s = shared.lock().unwrap();
    log_locked(&mut s, message.clone());
    s.error = Some(message.clone());
    Err(message)
}

fn run_step<T>(shared: &Shared, id: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    set_step(shared, id, StepState::Running);
    let result = f();
    set_step(shared, id, if result.is_ok() { StepState::Done } else { StepState::Failed });
    result.map_err(|e| {
        let mut s = shared.lock().unwrap();
        let label = s.steps.iter().find(|st| st.id == id).map(|st| st.label.clone()).unwrap_or_default();
        let message = format!("{label} failed. {}", explain(&s.log, &e));
        s.error = Some(message.clone());
        s.progress = None;
        message
    })
}

/// Turns the usual failure causes into something a person can act on.
fn explain(log: &[String], raw: &str) -> String {
    let tail = log.iter().rev().take(40).map(|l| l.to_lowercase()).collect::<Vec<_>>().join("\n");
    let has = |needles: &[&str]| needles.iter().any(|n| tail.contains(n) || raw.to_lowercase().contains(n));
    if has(&["no space left", "disk full", "os error 28", "os error 112"]) {
        "Your disk is full. Free up some space and try again.".into()
    } else if has(&["dns error", "failed to fetch", "connection refused", "timed out", "network", "could not download", "tls", "certificate"]) {
        "Temper couldn't download what it needs. Check your internet connection or proxy settings and try again.".into()
    } else if has(&["permission denied", "access is denied", "os error 13", "os error 5"]) {
        "Temper wasn't allowed to write to its folder. Check the folder's permissions and try again.".into()
    } else if raw == "Setup was cancelled" {
        raw.into()
    } else {
        format!("{raw} The details below show why.")
    }
}

fn uv_command(uv: &Path, layout: &Layout) -> Command {
    let mut cmd = Command::new(uv);
    cmd.args(["--color", "never", "--no-progress"])
        .env("UV_PYTHON_INSTALL_DIR", layout.python_dir())
        // Never pick up a system Python: every install gets the same interpreter.
        .env("UV_PYTHON_PREFERENCE", "only-managed")
        .env("UV_PROJECT_ENVIRONMENT", layout.venv())
        .env_remove("VIRTUAL_ENV")
        .env_remove("CONDA_PREFIX");
    cmd
}

fn run_logged(mut cmd: Command, shared: &Shared, cancel: &AtomicBool) -> Result<(), String> {
    let shown = describe(&cmd);
    log(shared, format!("$ {shown}"));
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    no_window(&mut cmd);
    let mut child = cmd.spawn().map_err(|e| format!("Could not run {shown}: {e}"))?;
    let readers = [
        pipe_to_log(child.stdout.take().unwrap(), shared.clone()),
        pipe_to_log(child.stderr.take().unwrap(), shared.clone()),
    ];
    let status = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Setup was cancelled".into());
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(200)),
            Err(e) => return Err(e.to_string()),
        }
    };
    for r in readers {
        let _ = r.join();
    }
    if status.success() {
        Ok(())
    } else {
        Err(format!("The installer exited with {status}."))
    }
}

fn pipe_to_log<R: Read + Send + 'static>(reader: R, shared: Shared) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(reader).lines().map_while(Result::ok) {
            let line = line.trim_end();
            if !line.is_empty() {
                log(&shared, line.to_string());
            }
        }
    })
}

// ------------------------------------------------------------------ uv

fn ensure_uv(layout: &Layout, shared: &Shared, cancel: &AtomicBool) -> Result<PathBuf, String> {
    let uv = layout.uv();
    if uv.is_file() && uv_version(&uv).as_deref() == Some(UV_VERSION) {
        log(shared, format!("uv {UV_VERSION} is already installed"));
        return Ok(uv);
    }
    match download_uv(layout, shared, cancel) {
        Ok(()) => Ok(uv),
        Err(e) if !cancel.load(Ordering::Relaxed) => match crate::engine::find_uv() {
            Some(system) => {
                log(shared, format!("{e}. Using the uv already installed at {}", system.display()));
                Ok(system)
            }
            None => Err(e),
        },
        Err(e) => Err(e),
    }
}

fn uv_version(uv: &Path) -> Option<String> {
    let out = output(uv.to_str()?, &["--version"])?;
    out.split_whitespace().nth(1).map(str::to_string)
}

fn download_uv(layout: &Layout, shared: &Shared, cancel: &AtomicBool) -> Result<(), String> {
    let triple = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        ("windows", "aarch64") => "aarch64-pc-windows-msvc",
        (os, arch) => return Err(format!("There is no uv build for {os} {arch}")),
    };
    let &(_, ext, sha) = UV_ARCHIVES.iter().find(|(t, _, _)| *t == triple).expect("every triple has an archive");
    let url = format!("https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-{triple}.{ext}");
    log(shared, format!("Downloading {url}"));

    let mut resp = ureq::get(&url).call().map_err(|e| format!("Could not download uv: {e}"))?;
    let total = resp.body().content_length();
    let mut reader = resp.body_mut().as_reader();
    let mut data = Vec::with_capacity(total.unwrap_or(0) as usize);
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("Setup was cancelled".into());
        }
        let n = reader.read(&mut chunk).map_err(|e| format!("Could not download uv: {e}"))?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&chunk[..n]);
        if let Some(total) = total {
            shared.lock().unwrap().progress = Some(data.len() as f64 / total as f64);
        }
    }

    let digest = hex::encode(Sha256::digest(&data));
    if digest != sha {
        return Err(format!("The uv download is corrupt (checksum {digest}). Check your connection and try again."));
    }
    let name = exe("uv");
    let binary = if ext == "zip" { from_zip(&data, &name)? } else { from_tar_gz(&data, &name)? };
    fs::create_dir_all(layout.bin()).map_err(err)?;
    let tmp = layout.bin().join(format!("{name}.download"));
    fs::write(&tmp, binary).map_err(err)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).map_err(err)?;
    }
    fs::rename(&tmp, layout.uv()).map_err(err)?;
    shared.lock().unwrap().progress = None;
    log(shared, format!("Installed uv {UV_VERSION}"));
    Ok(())
}

fn from_tar_gz(data: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(data));
    for entry in archive.entries().map_err(err)? {
        let mut entry = entry.map_err(err)?;
        if entry.path().map_err(err)?.file_name().and_then(|n| n.to_str()) == Some(name) {
            let mut out = Vec::new();
            entry.read_to_end(&mut out).map_err(err)?;
            return Ok(out);
        }
    }
    Err(format!("{name} is missing from the uv archive"))
}

fn from_zip(data: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let mut archive = zip::ZipArchive::new(Cursor::new(data)).map_err(err)?;
    let mut file = archive.by_name(name).map_err(|_| format!("{name} is missing from the uv archive"))?;
    let mut out = Vec::new();
    file.read_to_end(&mut out).map_err(err)?;
    Ok(out)
}

// ------------------------------------------------------------------ helpers

/// Hash of everything that ends up in the engine venv, so app updates that change it trigger a re-sync.
fn engine_hash(project: &Path) -> String {
    fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.file_name().is_some_and(|n| n != "__pycache__") {
                collect(&path, out);
            } else if path.extension().is_some_and(|e| e == "py") {
                out.push(path);
            }
        }
    }
    let mut hasher = Sha256::new();
    for file in ["pyproject.toml", "uv.lock"] {
        hasher.update(fs::read(project.join(file)).unwrap_or_default());
    }
    let mut files = Vec::new();
    collect(&project.join("temper_engine"), &mut files);
    files.sort();
    for file in files {
        hasher.update(file.strip_prefix(project).unwrap_or(&file).to_string_lossy().replace('\\', "/").as_bytes());
        hasher.update(fs::read(&file).unwrap_or_default());
    }
    hex::encode(hasher.finalize())
}

fn log(shared: &Shared, line: String) {
    log_locked(&mut shared.lock().unwrap(), line);
}

fn log_locked(s: &mut SetupState, line: String) {
    if std::env::var_os("TEMPER_SETUP_ECHO").is_some() {
        println!("{line}");
    }
    s.log.push(line);
    if s.log.len() > LOG_LINES {
        let extra = s.log.len() - LOG_LINES;
        s.log.drain(..extra);
    }
}

fn set_step(shared: &Shared, id: &str, state: StepState) {
    let mut s = shared.lock().unwrap();
    if let Some(step) = s.steps.iter_mut().find(|st| st.id == id) {
        step.state = state;
        if state == StepState::Running {
            let label = step.label.clone();
            s.progress = None;
            log_locked(&mut s, format!("== {label}"));
        }
    }
}

fn describe(cmd: &Command) -> String {
    let program = Path::new(cmd.get_program()).file_name().unwrap_or_default().to_string_lossy().into_owned();
    std::iter::once(program)
        .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn output(program: &str, args: &[&str]) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args).stdin(Stdio::null()).stderr(Stdio::null());
    no_window(&mut cmd);
    let out = cmd.output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

pub fn no_window(_cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        _cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}
