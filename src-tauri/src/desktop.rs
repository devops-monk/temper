//! Desktop integration: files opened with Temper, `temper://` links, the tray icon and running in the background.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State};

const PAGES: &[&str] = &["chat", "research", "models", "library", "train", "datasets", "recipes", "export", "connect", "settings"];
const MODEL_FILES: &[&str] = &["gguf"];
const DATASET_FILES: &[&str] = &["jsonl", "json", "csv", "parquet"];

/// Something the operating system asked Temper to open; the interface takes these with `take_opens`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Open {
    /// A page, from `temper://<page>`; `temper://model/<owner>/<name>` searches Models for that repo.
    Page { page: String, query: Option<String> },
    /// A model file to add to the library.
    Model { path: String },
    /// A file to turn into a dataset.
    Dataset { path: String },
}

/// What one argument or URL asks for: a `temper://` link, a `file://` URL or a path (relative to `cwd`).
pub fn route(arg: &str, cwd: &Path) -> Option<Open> {
    if let Some(rest) = arg.strip_prefix("temper://") {
        let rest = rest.split(['?', '#']).next().unwrap_or("").trim_matches('/');
        let mut parts = rest.splitn(2, '/');
        let page = parts.next().unwrap_or("").to_ascii_lowercase();
        let tail = parts.next().filter(|t| !t.is_empty()).map(decode);
        return match page.as_str() {
            "model" => Some(Open::Page { page: "models".into(), query: tail }),
            "" => Some(Open::Page { page: "chat".into(), query: None }),
            p if PAGES.contains(&p) => Some(Open::Page { page: p.into(), query: tail }),
            _ => None,
        };
    }
    let path = match arg.strip_prefix("file://") {
        Some(p) => PathBuf::from(decode(p)),
        None if arg.starts_with('-') || arg.contains("://") => return None,
        None => cwd.join(arg),
    };
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let path = path.to_string_lossy().into_owned();
    if MODEL_FILES.contains(&ext.as_str()) {
        Some(Open::Model { path })
    } else if DATASET_FILES.contains(&ext.as_str()) {
        Some(Open::Dataset { path })
    } else {
        None
    }
}

/// Undoes %XX escapes (spaces in file URLs, slashes in repo names).
fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(n) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(n);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[derive(Default)]
pub struct Opens(Mutex<Vec<Open>>);

/// Queues what the arguments ask for and tells the interface; it may not be listening yet, so it also
/// takes the queue when it starts.
pub fn open_args<S: AsRef<str>>(app: &AppHandle, args: impl IntoIterator<Item = S>, cwd: &Path) {
    let found: Vec<Open> = args.into_iter().filter_map(|a| route(a.as_ref(), cwd)).collect();
    if found.is_empty() {
        return;
    }
    app.state::<Opens>().0.lock().unwrap().extend(found);
    let _ = app.emit("temper-open", ());
    show(app);
}

#[tauri::command]
pub fn take_opens(opens: State<'_, Opens>) -> Vec<Open> {
    std::mem::take(&mut *opens.0.lock().unwrap())
}

pub fn show(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

/// Shell settings that must be known before the engine starts; kept apart from the engine's settings.
#[derive(Default, Serialize, Deserialize)]
struct Shell {
    #[serde(default)]
    background: bool,
}

fn shell_file(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join("shell.json"))
}

fn load(app: &AppHandle) -> Shell {
    shell_file(app).and_then(|f| std::fs::read(f).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// Whether closing the window keeps Temper running in the tray (the API and loaded model stay up).
pub fn background(app: &AppHandle) -> bool {
    load(app).background
}

#[tauri::command]
pub fn get_background(app: AppHandle) -> bool {
    background(&app)
}

#[tauri::command]
pub fn set_background(app: AppHandle, on: bool) -> Result<(), String> {
    let file = shell_file(&app).ok_or("no app data folder")?;
    let mut shell = load(&app);
    shell.background = on;
    std::fs::create_dir_all(file.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(file, serde_json::to_vec_pretty(&shell).unwrap()).map_err(|e| e.to_string())
}

/// The menu-bar (tray) icon: click to show the window; its menu shows or quits Temper.
pub fn tray(app: &AppHandle) -> tauri::Result<()> {
    let show_item = MenuItem::with_id(app, "show", "Show Temper", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Temper", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_item, &quit])?;
    let mut builder = TrayIconBuilder::with_id("temper")
        .tooltip("Temper")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                show(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(p: &str, q: Option<&str>) -> Option<Open> {
        Some(Open::Page { page: p.into(), query: q.map(Into::into) })
    }

    #[test]
    fn links_open_pages() {
        let cwd = Path::new("/home/me");
        assert_eq!(route("temper://train", cwd), page("train", None));
        assert_eq!(route("temper://Settings/", cwd), page("settings", None));
        assert_eq!(route("temper://", cwd), page("chat", None));
        assert_eq!(route("temper://model/Qwen/Qwen3-4B", cwd), page("models", Some("Qwen/Qwen3-4B")));
        assert_eq!(route("temper://model/Qwen%2FQwen3-4B?ref=site", cwd), page("models", Some("Qwen/Qwen3-4B")));
        assert_eq!(route("temper://nowhere", cwd), None);
    }

    #[test]
    fn files_are_models_or_datasets() {
        let cwd = Path::new("/home/me");
        assert_eq!(route("m.GGUF", cwd), Some(Open::Model { path: "/home/me/m.GGUF".into() }));
        assert_eq!(route("/data/qa.jsonl", cwd), Some(Open::Dataset { path: "/data/qa.jsonl".into() }));
        assert_eq!(route("file:///data/my%20set.csv", cwd), Some(Open::Dataset { path: "/data/my set.csv".into() }));
        assert_eq!(route("notes.txt", cwd), None);
        assert_eq!(route("--flag.json", cwd), None);
        assert_eq!(route("https://x.org/a.gguf", cwd), None);
        assert_eq!(route("/usr/bin/temper", cwd), None);
    }
}
