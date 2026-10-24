//! Starts and supervises the Python engine (`temper-engine`) through `uv`.
//!
//! The engine listens on a free loopback port and requires a random bearer token,
//! so other local processes cannot drive it. It also watches our PID and exits
//! if the app dies without a clean shutdown.

use serde::Serialize;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
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
    /// `uv` was not found; the UI shows setup instructions.
    MissingUv,
    Stopped,
}

#[derive(Clone, Serialize)]
pub struct EngineInfo {
    pub url: String,
    pub token: String,
    pub status: EngineStatus,
    pub detail: Option<String>,
}

struct Inner {
    port: u16,
    token: String,
    status: EngineStatus,
    detail: Option<String>,
    child: Option<Child>,
    generation: u64,
}

pub struct Engine(Arc<Mutex<Inner>>);

impl Engine {
    pub fn new() -> Self {
        Engine(Arc::new(Mutex::new(Inner {
            port: 0,
            token: uuid::Uuid::new_v4().simple().to_string(),
            status: EngineStatus::Stopped,
            detail: None,
            child: None,
            generation: 0,
        })))
    }

    pub fn info(&self) -> EngineInfo {
        let s = self.0.lock().unwrap();
        EngineInfo {
            url: format!("http://127.0.0.1:{}", s.port),
            token: s.token.clone(),
            status: s.status.clone(),
            detail: s.detail.clone(),
        }
    }

    pub fn start(&self, app: &AppHandle) {
        let generation = {
            let mut s = self.0.lock().unwrap();
            s.generation += 1;
            s.generation
        };
        let inner = self.0.clone();
        let app = app.clone();
        thread::spawn(move || supervise(inner, app, generation));
    }

    pub fn restart(&self, app: &AppHandle) {
        self.stop();
        self.start(app);
    }

    pub fn stop(&self) {
        let mut s = self.0.lock().unwrap();
        s.generation += 1; // tells the running supervisor to quit
        s.status = EngineStatus::Stopped;
        if let Some(mut child) = s.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn supervise(inner: Arc<Mutex<Inner>>, app: AppHandle, generation: u64) {
    let mut restarts = 0;
    loop {
        let Some(uv) = find_uv() else {
            set_status(&inner, EngineStatus::MissingUv, Some("Install uv from https://docs.astral.sh/uv/".into()));
            return;
        };
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

        let child = match engine_command(&app, &uv, port, &token).spawn() {
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

fn engine_command(app: &AppHandle, uv: &PathBuf, port: u16, token: &str) -> Command {
    let mut cmd = Command::new(uv);
    cmd.arg("run").arg("--project").arg(engine_dir(app));
    if !cfg!(debug_assertions) {
        // The app bundle is read-only (and signed): install from the shipped lock without rewriting
        // it, and keep the virtualenv in app data.
        cmd.arg("--frozen");
        if let Ok(data) = app.path().app_data_dir() {
            cmd.env("UV_PROJECT_ENVIRONMENT", data.join("engine-venv"));
        }
    }
    cmd.args(["temper-engine", "--port", &port.to_string(), "--parent-pid", &std::process::id().to_string()])
        .env("TEMPER_TOKEN", token)
        .stdin(Stdio::null());
    if let Ok(data) = app.path().app_data_dir() {
        cmd.env("TEMPER_HOME", data);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn engine_dir(app: &AppHandle) -> PathBuf {
    if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine")
    } else {
        app.path().resource_dir().map(|d| d.join("engine")).unwrap_or_default()
    }
}

/// GUI apps on macOS and Linux do not inherit the shell PATH, so check the usual install spots too.
fn find_uv() -> Option<PathBuf> {
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
