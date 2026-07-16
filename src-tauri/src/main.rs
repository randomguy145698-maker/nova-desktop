// Nova Desktop — Tauri entry point
// =================================
//
// This file boots the Tauri runtime, loads Nova's existing `src/index.html`
// exactly as-is, and checks for updates every time the app starts, so you
// never have to manually reinstall.
//
// The update flow lives ENTIRELY here in Rust, using native OS dialogs.
// Nova's frontend (index.html) is not touched at all: it has no idea an
// updater exists, and it doesn't need to.
//
// How the update flow works, step by step:
//   1. On launch, spawn a background task that asks the updater plugin
//      "is there a newer version than the one currently running?"
//   2. If yes, show a native Yes/No dialog with the new version number.
//   3. If the person clicks Yes, download the new version and install it
//      over the current one.
//   4. Show a second native dialog asking to restart now (finishing the
//      update) or later (it'll finish next time the app opens normally).
//
// Where updates come from is configured in tauri.conf.json's
// `plugins.updater.endpoints` — see that file and README.md's "Shipping
// updates" section for the full picture (signing keys, cutting a release).
//
// Restarting the app after an update installs is done with plain Rust
// standard library code (spawn a fresh copy of this same .exe, then exit
// the current one) rather than an extra plugin — one less external API
// surface to depend on, and it's guaranteed to compile since it only
// touches std::env / std::process.

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::UpdaterExt;

fn main() {
    tauri::Builder::default()
        // Nova's own features need no plugins at all — everything below
        // exists purely to support automatic updates.
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // Run the update check in the background so it never delays
            // Nova's window from opening — the app is usable immediately,
            // and a dialog pops up a moment later only if there's actually
            // something new.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                check_for_updates(handle).await;
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Nova desktop application");
}

/// Checks the configured update endpoint for a newer release. If one
/// exists, asks the person (via a native dialog) whether to install it,
/// downloads and installs it if they say yes, then offers to restart.
async fn check_for_updates(app: tauri::AppHandle) {
    // `.updater()` reads the endpoints/pubkey from tauri.conf.json.
    let updater = match app.updater() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("Updater not available: {e}");
            return;
        }
    };

    let update = match updater.check().await {
        Ok(Some(update)) => update,
        Ok(None) => return, // already on the latest version — nothing to do
        Err(e) => {
            // Network hiccups or an unreachable update server land here.
            // Nova still runs fine offline; we just silently skip the
            // check rather than bothering the person with an error.
            eprintln!("Update check failed: {e}");
            return;
        }
    };

    let new_version = update.version.clone();

    // Ask permission before touching anything — never install silently.
    let confirmed = app
        .dialog()
        .message(format!(
            "A new version of Nova is available: {new_version}\n\nInstall it now?"
        ))
        .title("Update available")
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::YesNo)
        .blocking_show();

    if !confirmed {
        return; // they said no — ask again next launch
    }

    // Download and install the update over the currently running copy.
    // The two closures report progress; we don't need a progress bar,
    // so both are no-ops here.
    let install_result = update
        .download_and_install(
            |_chunk_len, _total_len| { /* could update a progress bar here */ },
            || { /* called once the download finishes, before install */ },
        )
        .await;

    if let Err(e) = install_result {
        app.dialog()
            .message(format!("The update failed to install: {e}"))
            .title("Update failed")
            .kind(MessageDialogKind::Error)
            .blocking_show();
        return;
    }

    // Installed successfully — offer to restart immediately so the new
    // version takes effect right away, instead of waiting for next launch.
    let restart_now = app
        .dialog()
        .message("Nova has been updated. Restart now to finish?")
        .title("Update installed")
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::YesNo)
        .blocking_show();

    if restart_now {
        relaunch_app();
    }
}

/// Starts a brand-new copy of this same running .exe, then exits the
/// current one — the effect is an app restart. Written with only the
/// Rust standard library (no extra plugin) so it's guaranteed to compile
/// regardless of which Tauri plugin versions end up installed.
fn relaunch_app() -> ! {
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::process::Command::new(exe).spawn();
    }
    std::process::exit(0);
}
