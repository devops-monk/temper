//! Starts and supervises the Python engine (`temper-engine`).
//!
//! Installed builds run the engine from the venv that first-run setup created in app data
//! (see setup.rs). Development builds run it straight from `../engine` with `uv run`, unless
//! TEMPER_BOOTSTRAP is set to exercise the setup flow.
//!
//! The engine listens on a loopback port and requires a random bearer token, so other local
//! processes cannot drive it. It also watches our PID and exits if the app dies uncleanly.

use crate::setup::{self, Check, Layout, Profile, SetupState, Shared};
use serde::Serialize;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

const MAX_RESTARTS: u32 = 5;

#[derive(Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EngineStatus {
    Starting,
    Running,
    Crashed,
    /// Development only: `uv` was not found.
    MissingUv,
    Stopped,
    /// First launch: waiting for the user to pick what to install.
    NeedsSetup,
    Installing,
    SetupFailed,
}

#[derive(Clone, Serialize)]
pub struct EngineInfo {
    pub url: String,
    pub token: String,
    pub status: EngineStatus,
    pub detail: Option<String>,
    /// Present when this build manages its own engine install.
    pub setup: Option<SetupState>,
}

#[derive(Clone)]
enum Launcher {
    Dev { project: PathBuf },
    Venv { exe: PathBuf },
}

struct Inner {
    port: u16,
    token: String,
    status: EngineStatus,
    detail: Option<String>,
    child: Option<Child>,
    generation: u64,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Mutex<Inner>>,
    setup: Shared,
    cancel_setup: Arc<AtomicBool>,
}

/// Installed builds always manage their engine; dev builds opt in with TEMPER_BOOTSTRAP=1.
pub fn uses_setup() -> bool {
    !cfg!(debug_assertions) || std::env::var_os("TEMPER_BOOTSTRAP").is_some()
}

/// What `temper doctor` runs: the engine's checks, printed as text.
pub fn doctor_command(app: &AppHandle) -> Result<Command, String> {
    let mut cmd = if uses_setup() {
        let layout = layout(app);
        match setup::check(&layout, &app.package_info().version.to_string()) {
            Check::Ready => Command::new(layout.engine_exe()),
            Check::FirstRun => return Err("Temper isn't set up yet. Open the app once, or run: temper --setup chat".into()),
            Check::Outdated(p) => {
                let p = if p == Profile::Train { "train" } else { "chat" };
                return Err(format!("Temper's engine is out of date. Open the app once to update it, or run: temper --setup {p}"));
            }
        }
    } else {
        let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine");
        let mut c = Command::new(find_uv().ok_or("uv isn't installed")?);
        c.arg("run").arg("--quiet").arg("--project").arg(project).arg("temper-engine");
        c
    };
    cmd.arg("doctor");
    if let Ok(data) = app.path().app_data_dir() {
        cmd.env("TEMPER_HOME", data);
    }
    Ok(cmd)
}

pub fn layout(app: &AppHandle) -> Layout {
    let data = app.path().app_data_dir().expect("no app data directory");
    let project = if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine")
    } else {
        app.path().resource_dir().map(|d| d.join("engine")).unwrap_or_default()
    };
    Layout { data, project }
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            inner: Arc::new(Mutex::new(Inner {
                port: 0,
                token: uuid::Uuid::new_v4().simple().to_string(),
                status: EngineStatus::Stopped,
                detail: None,
                child: None,
                generation: 0,
            })),
            setup: Arc::new(Mutex::new(SetupState::default())),
            cancel_setup: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn info(&self) -> EngineInfo {
        let s = self.inner.lock().unwrap();
        EngineInfo {
            url: format!("http://127.0.0.1:{}", s.port),
            token: s.token.clone(),
            status: s.status.clone(),
            detail: s.detail.clone(),
            setup: uses_setup().then(|| self.setup.lock().unwrap().clone()),
        }
    }

    pub fn start(&self, app: &AppHandle) {
        if !uses_setup() {
            let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine");
            return self.supervise(app, Launcher::Dev { project });
        }
        let layout = layout(app);
        let hardware = setup::detect();
        {
            let mut s = self.setup.lock().unwrap();
            s.options = setup::options(&hardware);
            s.hardware = Some(hardware);
            s.installed = setup::installed(&layout).map(|i| i.profile);
        }
        match setup::check(&layout, &app.package_info().version.to_string()) {
            Check::Ready => self.supervise(app, Launcher::Venv { exe: layout.engine_exe() }),
            Check::FirstRun => self.set_status(EngineStatus::NeedsSetup, None),
            Check::Outdated(profile) => self.install(app, profile, false),
        }
    }

    /// Installs (or re-syncs) the engine for `profile`, then starts it. `clean` deletes the venv first.
    pub fn install(&self, app: &AppHandle, profile: Profile, clean: bool) {
        if self.inner.lock().unwrap().status == EngineStatus::Installing {
            return;
        }
        self.stop();
        self.cancel_setup.store(false, Ordering::Relaxed);
        self.set_status(EngineStatus::Installing, None);
        let engine = self.clone();
        let app = app.clone();
        thread::spawn(move || {
            let layout = layout(&app);
            if clean {
                let _ = std::fs::remove_dir_all(layout.venv());
            }
            let hardware = engine.setup.lock().unwrap().hardware.clone().unwrap_or_else(setup::detect);
            let version = app.package_info().version.to_string();
            match setup::install(&layout, profile, &hardware, &version, &engine.setup, &engine.cancel_setup) {
                Ok(()) => engine.supervise(&app, Launcher::Venv { exe: layout.engine_exe() }),
                Err(_) if engine.cancel_setup.load(Ordering::Relaxed) => {}
                Err(e) => engine.set_status(EngineStatus::SetupFailed, Some(e)),
            }
        });
    }

    pub fn restart(&self, app: &AppHandle) {
        self.stop();
        self.start(app);
    }

    pub fn stop(&self) {
        self.cancel_setup.store(true, Ordering::Relaxed);
        let mut s = self.inner.lock().unwrap();
        s.generation += 1; // tells the running supervisor to quit
        s.status = EngineStatus::Stopped;
        if let Some(mut child) = s.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn set_status(&self, status: EngineStatus, detail: Option<String>) {
        set_status(&self.inner, status, detail);
    }

    fn supervise(&self, app: &AppHandle, launcher: Launcher) {
        let generation = {
            let mut s = self.inner.lock().unwrap();
            s.generation += 1;
            s.status = EngineStatus::Starting;
            s.generation
        };
        let inner = self.inner.clone();
        let app = app.clone();
        thread::spawn(move || supervise(inner, app, generation, launcher));
    }
}

fn supervise(inner: Arc<Mutex<Inner>>, app: AppHandle, generation: u64, launcher: Launcher) {
    let mut restarts = 0;
    loop {
        let port = free_port();
        let token = {
            let mut s = inner.lock().unwrap();
            if s.generation != generation {
                return;
            }
            s.port = port;
            s.status = EngineStatus::Starting;
            s.detail = None;
            s.token.clone()
        };

        let Some(mut cmd) = engine_command(&app, &launcher, port, &token) else {
            set_status(&inner, EngineStatus::MissingUv, Some("Install uv from https://docs.astral.sh/uv/".into()));
            return;
        };
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                set_status(&inner, EngineStatus::Crashed, Some(format!("Could not start engine: {e}")));
                return;
            }
        };
        inner.lock().unwrap().child = Some(child);

        let started = Instant::now();
        let mut announced = false;
        let exit = loop {
            thread::sleep(Duration::from_millis(250));
            let mut s = inner.lock().unwrap();
            if s.generation != generation {
                return; // stop() already killed the child
            }
            let Some(child) = s.child.as_mut() else { return };
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {}
                Err(_) => break std::process::ExitStatus::default(),
            }
            drop(s);
            if !announced && port_open(port) {
                announced = true;
                set_status(&inner, EngineStatus::Running, None);
            }
        };

        inner.lock().unwrap().child = None;
        // A run that lasted a while resets the crash budget.
        if started.elapsed() > Duration::from_secs(60) {
            restarts = 0;
        }
        restarts += 1;
        if restarts > MAX_RESTARTS {
            set_status(&inner, EngineStatus::Crashed, Some(format!("Engine exited ({exit}) {MAX_RESTARTS} times")));
            return;
        }
        set_status(&inner, EngineStatus::Starting, Some(format!("Engine exited ({exit}); restarting")));
        thread::sleep(Duration::from_secs(restarts as u64));
    }
}

fn engine_command(app: &AppHandle, launcher: &Launcher, port: u16, token: &str) -> Option<Command> {
    let mut cmd = match launcher {
        Launcher::Dev { project } => {
            let mut c = Command::new(find_uv()?);
            c.arg("run").arg("--project").arg(project).arg("temper-engine");
            c
        }
        Launcher::Venv { exe } => Command::new(exe),
    };
    cmd.args(["--port", &port.to_string(), "--parent-pid", &std::process::id().to_string()])
        .env("TEMPER_TOKEN", token)
        .stdin(Stdio::null());
    if let Some(hf) = crate::secrets::hf_token() {
        cmd.env("HF_TOKEN", hf);
    }
    if let Ok(data) = app.path().app_data_dir() {
        cmd.env("TEMPER_HOME", data);
    }
    setup::no_window(&mut cmd);
    Some(cmd)
}

/// GUI apps on macOS and Linux do not inherit the shell PATH, so check the usual install spots too.
pub(crate) fn find_uv() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    if let Some(h) = home {
        dirs.push(h.join(".local/bin"));
        dirs.push(h.join(".cargo/bin"));
    }
    dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from));
    dirs.into_iter().map(|d| d.join(exe)).find(|p| p.is_file())
}

/// Prefers 8765 so agent configs that point at the engine keep working across launches.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:8765")
        .or_else(|_| TcpListener::bind("127.0.0.1:0"))
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8765)
}

fn port_open(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(200)).is_ok()
}

fn set_status(inner: &Arc<Mutex<Inner>>, status: EngineStatus, detail: Option<String>) {
    let mut s = inner.lock().unwrap();
    s.status = status;
    s.detail = detail;
}
