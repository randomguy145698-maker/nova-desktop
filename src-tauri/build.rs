// Tauri's build script. This runs before compilation and embeds the
// settings from tauri.conf.json (window config, icon, bundle identifier)
// into the binary. Every Tauri project needs this file — it is not
// Nova-specific and should not need to change.
fn main() {
    tauri_build::build()
}
