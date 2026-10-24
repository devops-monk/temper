mod engine;
mod setup;

use engine::{Engine, EngineInfo};
use setup::Profile;
use std::sync::atomic::AtomicBool;
use tauri::{Manager, RunEvent, State};

#[tauri::command]
fn engine_info(engine: State<'_, Engine>) -> EngineInfo {
    engine.info()
}

#[tauri::command]
fn engine_restart(app: tauri::AppHandle, engine: State<'_, Engine>) -> EngineInfo {
    engine.restart(&app);
    engine.info()
}

/// Installs or changes the engine environment. `clean` deletes the existing one first (repair).
#[tauri::command]
fn setup_install(app: tauri::AppHandle, engine: State<'_, Engine>, profile: Profile, clean: bool) -> EngineInfo {
    engine.install(&app, profile, clean);
    engine.info()
}

/// `--setup chat|train [--clean]` installs the engine without showing a window, printing the log,
/// and exits with 0 on success. Useful for scripted installs and CI smoke tests.
fn headless_setup() -> Option<(Profile, bool)> {
    let args: Vec<String> = std::env::args().collect();
    let i = args.iter().position(|a| a == "--setup")?;
    let profile = match args.get(i + 1).map(|p| p.parse::<Profile>()) {
        Some(Ok(p)) => p,
        Some(Err(e)) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
        None => Profile::Chat,
    };
    Some((profile, args.iter().any(|a| a == "--clean")))
}

/// Release builds on Windows have no console of their own; borrow the terminal that started us
/// so `temper doctor` and `temper --setup` can print.
fn attach_console() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let headless = headless_setup();
    let doctor = std::env::args().nth(1).as_deref() == Some("doctor");
    if headless.is_some() || doctor {
        attach_console();
    }
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(Engine::new())
        .setup(move |app| {
            if doctor {
                let handle = app.handle().clone();
                std::thread::spawn(move || {
                    let code = match engine::doctor_command(&handle).and_then(|mut c| c.status().map_err(|e| e.to_string())) {
                        Ok(status) => status.code().unwrap_or(1),
                        Err(e) => {
                            eprintln!("{e}");
                            1
                        }
                    };
                    handle.exit(code);
                });
                return Ok(());
            }
            if let Some((profile, clean)) = headless {
                let handle = app.handle().clone();
                std::thread::spawn(move || {
                    std::env::set_var("TEMPER_SETUP_ECHO", "1");
                    let layout = engine::layout(&handle);
                    if clean {
                        let _ = std::fs::remove_dir_all(layout.venv());
                    }
                    let hardware = setup::detect();
                    println!("Installing the {profile:?} profile into {} (hardware: {hardware:?})", layout.data.display());
                    let version = handle.package_info().version.to_string();
                    let code = match setup::install(&layout, profile, &hardware, &version, &Default::default(), &AtomicBool::new(false)) {
                        Ok(()) => 0,
                        Err(e) => {
                            eprintln!("Setup failed: {e}");
                            1
                        }
                    };
                    handle.exit(code);
                });
                return Ok(());
            }
            // The window starts hidden (tauri.conf.json) so headless setup never flashes it.
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
            }
            app.state::<Engine>().start(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![engine_info, engine_restart, setup_install])
        .build(tauri::generate_context!())
        .expect("failed to build Temper");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            handle.state::<Engine>().stop();
        }
    });
}
