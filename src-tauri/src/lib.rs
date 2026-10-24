mod engine;

use engine::{Engine, EngineInfo};
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Engine::new())
        .setup(|app| {
            app.state::<Engine>().start(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![engine_info, engine_restart])
        .build(tauri::generate_context!())
        .expect("failed to build Temper");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            handle.state::<Engine>().stop();
        }
    });
}
