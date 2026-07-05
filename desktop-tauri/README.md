# LoraHub Desktop

Tauri desktop shell for the existing LoraHub WebUI. This folder is isolated on
purpose: the browser WebUI keeps using `web/` and the API server unchanged.

## Run

```powershell
cd desktop-tauri
npm install
npm run tauri:dev
```

The desktop shell opens `http://127.0.0.1:18765`. If nothing is listening, it
starts:

```powershell
lorahub service start --foreground --host 127.0.0.1 --port 18765
```

Set another port with `LORAHUB_DESKTOP_PORT`.

## Build

```powershell
cd desktop-tauri
npm run tauri:build
```

The generated installer/exe is under `src-tauri/target/release/bundle/`.
