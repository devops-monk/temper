mod desktop;
mod engine;
mod secrets;
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

/// Saves the Hugging Face token in the OS keychain; the engine reads it as HF_TOKEN on its next start
/// (the UI also hands it to the running engine, so nothing restarts).
#[tauri::command]
fn hf_token_save(token: String) -> Result<(), String> {
    secrets::set_hf_token(token.trim())
}

#[tauri::command]
fn hf_token_delete() -> Result<(), String> {
    secrets::delete_hf_token()
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
/// so `temper doctor`, `temper chat` and the other commands, and `temper --setup`, can print.
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
    // The `temper` subcommands (doctor, start, mcp, models, chat, train, ...) are the engine's to answer.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = matches!(
        args.first().map(String::as_str),
        Some("doctor" | "start" | "mcp" | "models" | "load" | "ps" | "unload" | "status" | "version" | "get" | "import" | "log" | "daemon" | "engines" | "create" | "devices" | "transcribe" | "speech" | "say" | "stems" | "image" | "chat" | "train" | "runs" | "export" | "serve")
    );
    if headless.is_some() || cli {
        attach_console();
    }
    let window = headless.is_none() && !cli;
    let mut builder = tauri::Builder::default();
    if window {
        // Must come first: a second launch (a file or link opened while Temper runs) hands over its arguments and exits.
        builder = builder.plugin(tauri_plugin_single_instance::init(|app, argv, cwd| {
            desktop::open_args(app, argv.iter().skip(1), std::path::Path::new(&cwd));
            desktop::show(app);
        }));
    }
    let app = builder
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(Engine::new())
        .manage(desktop::Opens::default())
        .setup(move |app| {
            if cli {
                let handle = app.handle().clone();
                std::thread::spawn(move || {
                    let code = match engine::cli_command(&handle, &args).and_then(|mut c| c.status().map_err(|e| e.to_string())) {
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
            if let Err(e) = desktop::tray(app.handle()) {
                eprintln!("No tray icon: {e}");
            }
            #[cfg(target_os = "linux")]
            {
                use tauri_plugin_deep_link::DeepLinkExt;
                let _ = app.deep_link().register_all(); // AppImages have no installer to register temper://
            }
            // Windows and Linux pass opened files and links as arguments; macOS sends RunEvent::Opened.
            let cwd = std::env::current_dir().unwrap_or_default();
            desktop::open_args(app.handle(), std::env::args().skip(1), &cwd);
            Ok(())
        })
        .on_window_event(|window, event| {
            // In background mode closing the window keeps the API and the loaded model up, in the tray.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if desktop::background(window.app_handle()) {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            engine_info,
            engine_restart,
            setup_install,
            hf_token_save,
            hf_token_delete,
            desktop::take_opens,
            desktop::get_background,
            desktop::set_background
        ])
        .build(tauri::generate_context!())
        .expect("failed to build Temper");

    app.run(|handle, event| match event {
        RunEvent::Exit => handle.state::<Engine>().stop(),
        #[cfg(target_os = "macos")]
        RunEvent::Opened { urls } => {
            let cwd = std::env::current_dir().unwrap_or_default();
            desktop::open_args(handle, urls.iter().map(|u| u.to_string()), &cwd);
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => desktop::show(handle),
        _ => {}
    });
}
