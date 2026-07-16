// Nova Desktop — Tauri entry point
// =================================
//
// This file boots the Tauri runtime, loads Nova's existing `src/index.html`
// exactly as-is, checks for updates on launch, and exposes two native,
// Rust-side neural networks the frontend can optionally train and use:
//
//   nn.rs  — a topic CLASSIFIER (bag-of-words -> tanh -> tanh -> softmax
//             over topics), used by the 🚀 Native Network section and as
//             an optional cross-check for the 🔮 Sorter agent.
//
//   gen.rs — a text GENERATOR (same network shape, but predicting the
//             next word instead of a topic), used by the 🧠 Native
//             Generator section and as an optional bonus follow-up when
//             using ✨ Dream / ✍️ Write / 💻 Code in the desktop app.
//
// Everything Nova already did (chat, learning, the lightweight in-browser
// topic router used for every message, the deterministic code builder)
// is untouched — both native networks are purely additive, explicitly
// triggered from the frontend, and never replace the existing, already
// -tested synchronous behavior.

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod embed;
mod gen;
mod nn;

use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::UpdaterExt;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        // Both native networks start empty (None) until the frontend
        // explicitly asks to train one.
        .manage(nn::NetState(std::sync::Mutex::new(None)))
        .manage(gen::GenState(std::sync::Mutex::new(None)))
        .manage(embed::EmbedState(std::sync::Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            nn::train_native_network,
            nn::predict_native_topic,
            gen::train_native_generator,
            gen::generate_native_text,
            embed::train_native_embeddings,
            embed::semantic_rank,
            embed::nearest_words
        ])
        .setup(|app| {
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
    let updater = match app.updater() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("Updater not available: {e}");
            return;
        }
    };

    let update = match updater.check().await {
        Ok(Some(update)) => update,
        Ok(None) => return,
        Err(e) => {
            eprintln!("Update check failed: {e}");
            return;
        }
    };

    let new_version = update.version.clone();

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
        return;
    }

    let install_result = update
        .download_and_install(
            |_chunk_len, _total_len| {},
            || {},
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
/// Rust standard library so it's guaranteed to compile regardless of
/// which Tauri plugin versions end up installed.
fn relaunch_app() -> ! {
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::process::Command::new(exe).spawn();
    }
    std::process::exit(0);
}
