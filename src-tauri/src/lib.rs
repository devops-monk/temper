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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let headless = headless_setup();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Engine::new())
        .setup(move |app| {
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
