# Nova Desktop

Nova, packaged as a real desktop application using [Tauri](https://tauri.app) —
same app, same brain, same everything, just with a proper window, an icon,
and a Windows installer instead of "open this HTML file in your browser."

**Nothing about how Nova works has changed.** `src/index.html` is the exact
file you already had — every chat feature, the 12 agents, the neural
network, the memory map, Wikipedia/Stack Exchange/arXiv lookups, all of it,
byte-for-byte identical. Tauri just gives it a native window to live in.

---

## Project structure

```
nova-desktop/
├── src/
│   └── index.html              ← Nova itself — unmodified
├── src-tauri/                  ← the native shell (Rust + config)
│   ├── src/
│   │   └── main.rs             ← native side of the app + the auto-updater logic
│   ├── icons/                  ← app icon, all required sizes/formats
│   ├── capabilities/
│   │   └── default.json        ← native permissions Nova is allowed to use
│   ├── Cargo.toml               ← Rust package manifest (now incl. updater plugins)
│   ├── build.rs                 ← Tauri's required build hook
│   └── tauri.conf.json          ← window settings, dark theme, CSP, updater config
├── .github/workflows/
│   ├── build-windows.yml       ← quick test builds on every push to main
│   └── release.yml             ← signed, versioned releases that power auto-update
├── package.json                 ← npm scripts that drive the Tauri CLI
├── .gitignore
└── README.md                    ← you are here
```

## What every file does

| File | Purpose |
|---|---|
| **`src/index.html`** | Nova, completely unchanged. Same JS, same CSS, same 12 agents, same neural network. |
| **`src-tauri/src/main.rs`** | The native side of the app: opens the window, and — new — checks for updates on every launch via native dialogs. Nova's JS still knows nothing about any of this. |
| **`src-tauri/tauri.conf.json`** | The real configuration file. Window title, size, dark theme, the CSP allowlist for Nova's `fetch()` calls, and the updater's public key + where it checks for new releases. |
| **`src-tauri/Cargo.toml`** | Rust's equivalent of `package.json` — the Tauri dependency plus the three update-related plugins (updater, dialog, process). |
| **`src-tauri/build.rs`** | Boilerplate every Tauri app needs; wires the config into the compiled binary. You'll never need to touch it. |
| **`src-tauri/capabilities/default.json`** | Tauri's permission system. Stays minimal — the updater runs entirely in Rust and is never invoked from JS, so no new frontend permissions were needed for it. |
| **`src-tauri/icons/`** | The app icon in every size/format Windows and the installer need (see below). |
| **`package.json`** | Just two npm scripts: `npm run dev` (live preview) and `npm run build` (produce the installer). |
| **`.github/workflows/build-windows.yml`** | A cloud robot that builds a quick test installer on every push — see below. |
| **`.github/workflows/release.yml`** | Builds, **signs**, and publishes a real versioned release whenever you push a tag — this is what actually produces the update manifest. |

---

## The app icon

I generated a real icon from Nova's own brand colors (its cyan accent and dark
background, with a stylized "N") rather than leaving a placeholder gray
square. It's already been generated in every format the Windows build needs:

- `icons/icon.png` — 1024×1024 master
- `icons/32x32.png`, `icons/128x128.png`, `icons/128x128@2x.png` — standard sizes
- `icons/icon.ico` — the multi-resolution Windows icon (16–256px bundled in one file)

**If you want a custom-designed icon later:** replace `icons/icon.png` with
your own 1024×1024 artwork, then run:
```
npx @tauri-apps/cli icon icons/icon.png
```
This regenerates every size/format automatically from your new master image.

---

## How to run it

You'll need **Node.js** (you already have this) and the **Rust toolchain**
(you don't, based on this environment — install it from
[rustup.rs](https://rustup.rs), it's a five-minute one-time setup on Windows).

```bash
# from the nova-desktop/ folder
npm install        # installs the Tauri CLI
npm run dev         # opens Nova in a live desktop window (auto-reloads on file changes)
```

## How to build the Windows installer

**Option A — build it yourself (needs Rust installed on Windows):**
```bash
npm install
npm run build
```
When it finishes, your installer is at:
```
src-tauri/target/release/bundle/nsis/Nova_1.0.0_x64-setup.exe
src-tauri/target/release/bundle/msi/Nova_1.0.0_x64_en-US.msi
```
Either file is a real, double-click-to-install Windows installer.

**Option B — let GitHub build it for you (no Rust install needed):**
1. Push this folder to a GitHub repository.
2. Go to the **Actions** tab → run **"Build Windows Installer"** (or just push to `main`).
3. When the run finishes, open it and download the **`nova-windows-installer`**
   artifact — that's your `.exe`/`.msi`, built on a real Windows machine in
   the cloud, no local setup required.

I built this workflow file for exactly this reason: you can get a genuine
Windows installer without installing anything beyond a GitHub account.

> **Honesty note:** I can't actually run `cargo`/Rust or produce a compiled
> `.exe` from inside this sandboxed environment — there's no Rust toolchain
> and no network access here. Every file in this project is complete,
> internally consistent, and validated (JSON/YAML syntax-checked, the icon
> file structurally verified, the frontend's JS re-checked for syntax
> errors) — but the actual `cargo build` / installer-bundling step needs to
> run on your machine or via the GitHub Actions workflow above.

---

## Does browser storage still work?

**Yes — and it actually works *better* than before.** Tauri serves your
frontend from a real origin (`tauri://localhost` on Windows/Linux,
`https://tauri.localhost`), not a bare `file://` path. `localStorage`,
`fetch()`, and everything else Nova relies on all behave exactly like a
normal website — none of the quirks that some browsers apply to local
files. Your existing `nova_brain` / `nova_memory` data will carry over
automatically the first time you open the desktop app on the same machine
you used the browser version on, **as long as you're opening it in the same
underlying browser engine** (see the honest caveat below).

**Important caveat:** `localStorage` is scoped per-origin *and per-app*.
Tauri's webview is a separate "browser" from Chrome/Edge/Firefox, so it
does **not** automatically see data you saved while running Nova as a
plain HTML file in your regular browser. The fix is simple — before
switching to the desktop app for good, use Nova's own **Export** button
(Topics tab) to download a `nova_backup.json` file from the browser
version, then use **Import** inside the new desktop app to load it in.
One click each way, and nothing is lost.

---

## When you outgrow localStorage: migrating to SQLite

`localStorage` is fine for Nova today, but it has real ceilings worth
knowing about:

- **Size**: most browsers cap it around 5–10MB per origin. A GIGA-dump
  brain with a few thousand facts, plus the neural network's data, plus
  conversation history, can realistically approach that ceiling.
- **No querying**: every read/write means loading and re-serializing the
  *entire* JSON blob — fine at hundreds of facts, noticeably slow at
  tens of thousands.
- **No structure**: you can't ask "give me every fact taught in the last
  week" without scanning everything in JavaScript.

If Nova's brain keeps growing (mega/giga dumps, dozens of topics, a large
neural network), SQLite is the right next step. Here's the concrete path,
designed to **never lose existing data**:

### 1. Add the SQL plugin (Rust side)
```toml
# src-tauri/Cargo.toml — add this dependency
tauri-plugin-sql = { version = "2", features = ["sqlite"] }
```
```rust
// src-tauri/src/main.rs — register the plugin
tauri::Builder::default()
    .plugin(tauri_plugin_sql::Builder::default().build())
    .run(tauri::generate_context!())
    .expect("error while running Nova desktop application");
```

### 2. A schema that mirrors Nova's existing data shape
```sql
CREATE TABLE topics (
    name TEXT PRIMARY KEY
);

CREATE TABLE facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL REFERENCES topics(name),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    UNIQUE(topic, question)
);
CREATE INDEX idx_facts_topic ON facts(topic);

CREATE TABLE memory_kv (
    key TEXT PRIMARY KEY,        -- "name", "xp", "sessions", "personality", etc.
    value TEXT NOT NULL          -- JSON-encoded value, same as today
);
```
This is a direct, lossless translation of the two existing `localStorage`
keys: every `brain[topic][question] = answer` becomes one row in `facts`;
everything in `memory` becomes rows in `memory_kv`. No data model changes,
no lost facts.

### 3. A one-time, automatic migration on first launch
Add a small migration step that runs once: on startup, check if SQLite has
any rows yet; if not, read the existing `nova_brain` / `nova_memory` from
`localStorage` (still fully accessible in the webview) and insert them into
the new tables via `tauri-plugin-sql`'s JS API:
```js
import Database from '@tauri-apps/plugin-sql';
const db = await Database.load('sqlite:nova.db');

const alreadyMigrated = await db.select("SELECT 1 FROM memory_kv WHERE key='migrated'");
if (!alreadyMigrated.length) {
  const brain = JSON.parse(localStorage.getItem('nova_brain') || '{}');
  for (const topic in brain) {
    await db.execute("INSERT OR IGNORE INTO topics(name) VALUES (?)", [topic]);
    for (const q in brain[topic]) {
      await db.execute(
        "INSERT OR IGNORE INTO facts(topic, question, answer) VALUES (?,?,?)",
        [topic, q, brain[topic][q]]
      );
    }
  }
  await db.execute("INSERT INTO memory_kv(key,value) VALUES ('migrated','true')");
}
```
Nova's `localStorage` copy is left in place untouched as a safety net —
nothing is deleted, so if anything about the migration looks wrong you can
always roll back to the pre-migration state.

### 4. Swap the storage functions, keep everything else
Nova's code already funnels every read/write through two functions —
`store.get(key)` / `store.set(key, value)`. That's exactly the seam to cut
along: reimplement those two functions to read/write SQLite instead of
`localStorage`, and the other ~2,500 lines of Nova (the agents, the neural
network, the memory map, the chat logic) don't need to change at all —
they only ever talk to `store`, never to `localStorage` directly.

This is a genuinely small, contained change specifically *because* of how
Nova was already written — the storage layer is already isolated behind
one clean interface.

---

## Auto-updates — so you never have to manually reinstall

Nova now checks for a new version every time it starts, using Tauri's
official updater. If one's available, you get a native "Update available —
install now?" dialog; say yes, and it downloads, installs, and offers to
restart — all without you touching an installer file again.

This is implemented **entirely in Rust** (`src-tauri/src/main.rs`).
Nova's `index.html` was not touched — it has no idea an updater exists.

### One-time setup before you can ship your first auto-updating release

**1. Generate a signing keypair.** Updates are cryptographically signed so
the app can verify a downloaded update actually came from you, not
someone else. From the `nova-desktop/` folder:
```bash
npm run tauri signer generate -- -w ~/.tauri/nova-update.key
```
This prints a **public key** and writes a **private key** file. Keep the
private key secret — never commit it.

> I generated everything else in this project, but I deliberately did
> **not** fabricate this keypair myself. Signing keys are a real security
> boundary — generating one requires the actual `tauri` CLI tool
> (which needs Rust/Node installed) so it comes out in the exact format
> the updater expects. This is the one manual step in the whole setup.

**2. Paste the public key** into `src-tauri/tauri.conf.json`:
```json
"plugins": { "updater": { "pubkey": "<paste the public key here>" } }
```

**3. Add the private key as a GitHub secret** (repo → Settings → Secrets
and variables → Actions):
- `TAURI_SIGNING_PRIVATE_KEY` — the contents of the private key file
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` — only if you set a password when generating it

**4. Point the updater at your real repo.** Replace the placeholder in
`tauri.conf.json`'s `plugins.updater.endpoints` with your actual GitHub
username/repo:
```
https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/releases/latest/download/latest.json
```

### Shipping an update, from then on

1. Bump the version number in **both** `src-tauri/tauri.conf.json` and
   `package.json` (e.g. `1.0.0` → `1.0.1`) — keep them matching.
2. Commit, then tag and push:
   ```bash
   git tag v1.0.1
   git push origin v1.0.1
   ```
3. `.github/workflows/release.yml` builds, signs, and publishes everything
   automatically — including the `latest.json` manifest the updater reads.
4. Anyone with an older Nova gets offered the update next time they open it.

There are now two GitHub Actions workflows, doing two different jobs:
- **`build-windows.yml`** — quick test builds on every push to `main` (no
  signing, no release, just "does it compile").
- **`release.yml`** — the real, signed, auto-update-producing release,
  triggered only by pushing a version tag like `v1.0.1`.

---



Per your requirements, I kept modifications to zero:
- **No HTML/CSS/JS rewritten.** `src/index.html` is byte-identical to what
  you had.
- **No new frontend build step, no bundler, no framework.** Tauri simply
  points its webview at your existing file.
- **Every existing feature works unmodified**: the 12-agent council, the
  neural network (up to the 4.7M-parameter MAXIMUM mode), Deep Study
  across all 9 sources, the memory map, voice, quizzes, XP/levels — none of
  it needed to know it's running in Tauri instead of a browser tab.
