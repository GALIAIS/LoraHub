use std::{
    error::Error,
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

struct BackendProcess(Mutex<Option<Child>>);

pub fn run() {
    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let port = desktop_port()?;
            let url = format!("http://127.0.0.1:{port}");

            if !is_healthy(port) {
                let child = start_backend(port)?;
                *app.state::<BackendProcess>().0.lock().unwrap() = Some(child);
                wait_until_healthy(port, Duration::from_secs(30))?;
            }

            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.parse()?))
                .title("LoraHub")
                .inner_size(1280.0, 820.0)
                .min_inner_size(960.0, 640.0)
                .build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                if let Some(state) = window.try_state::<BackendProcess>() {
                    if let Ok(mut child) = state.0.lock() {
                        if let Some(mut child) = child.take() {
                            let _ = child.kill();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run LoraHub desktop");
}

fn desktop_port() -> Result<u16, Box<dyn Error>> {
    let raw = std::env::var("LORAHUB_DESKTOP_PORT").unwrap_or_else(|_| "18765".to_string());
    let port = raw.parse::<u16>()?;
    if port == 0 {
        return Err("LORAHUB_DESKTOP_PORT must be between 1 and 65535".into());
    }
    Ok(port)
}

fn project_dir() -> PathBuf {
    std::env::var_os("LORAHUB_PROJECT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn lorahub_cli(project: &std::path::Path) -> PathBuf {
    let local = if cfg!(windows) {
        project.join(".venv/Scripts/lorahub.exe")
    } else {
        project.join(".venv/bin/lorahub")
    };
    if local.is_file() {
        local
    } else {
        PathBuf::from("lorahub")
    }
}

fn start_backend(port: u16) -> Result<Child, Box<dyn Error>> {
    let project = project_dir();
    let mut cmd = Command::new(lorahub_cli(&project));
    let port = port.to_string();
    cmd.current_dir(project)
        .args(["service", "start", "--foreground", "--host", "127.0.0.1", "--port", &port])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }

    Ok(cmd.spawn()?)
}

fn wait_until_healthy(port: u16, timeout: Duration) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if is_healthy(port) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    Err(format!("LoraHub API did not answer on port {port}").into())
}

fn is_healthy(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
    let request = format!("GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    stream.read_to_string(&mut response).is_ok() && response.starts_with("HTTP/1.") && response.contains(" 200 ")
}
