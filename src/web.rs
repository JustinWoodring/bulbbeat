//! `--web` daemon: a tiny HTTP server exposing the playback pipeline.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

use crate::playback::{self, PlaybackConfig};

const AUDIO_EXTS: [&str; 6] = ["mp3", "flac", "wav", "ogg", "m4a", "aac"];
const MAX_BODY: usize = 64 * 1024;

#[derive(Clone, Default)]
struct WebConfig {
    no_restore: bool,
    lead_ms: Option<u64>,
}

/// One playback plus the task supervising it.
struct Active {
    file: String,
    handle: playback::PlaybackHandle,
    supervisor: JoinHandle<()>,
}

impl Active {
    fn playing(&self) -> bool {
        !self.handle.is_canceling() && !self.handle.done()
    }
}

struct ServerState {
    active: Mutex<Option<Active>>,
    cfg: WebConfig,
}

/// List audio files in `dir` (flat, no hidden files).
fn list_audio(dir: &Path) -> Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry.with_context(|| format!("entry in {}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| AUDIO_EXTS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false);
        if !ext_ok {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        out.push((name.to_string(), size));
    }
    out.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
    Ok(out)
}

/// Reject anything that is not a plain audio file directly in `dir`.
fn resolve_requested(dir: &Path, name: &str) -> Result<PathBuf> {
    if name.is_empty()
        || name.len() > 255
        || name.starts_with('.')
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
    {
        bail!("invalid file name");
    }
    let path = dir.join(name);
    if !path.is_file() {
        bail!("no such audio file: {name}");
    }
    Ok(path)
}

async fn stop_active(state: &ServerState) {
    let Some(active) = state.active.lock().take() else {
        return;
    };
    active.handle.cancel_now();
    let _ = active.supervisor.await;
}

async fn handle_play(state: &ServerState, body: &[u8]) -> Result<Value> {
    let req: Value =
        serde_json::from_slice(body).map_err(|e| anyhow!("invalid JSON body: {e}"))?;
    let name = req
        .get("file")
        .and_then(Value::as_str)
        .context("missing \"file\" field")?
        .to_string();
    let dir = std::env::current_dir()?;
    let path = resolve_requested(&dir, &name)?;

    // Replace whatever is playing.
    stop_active(state).await;

    let bulbs = Vec::new(); // web mode always auto-discovers
    let cfg = PlaybackConfig {
        no_restore: state.cfg.no_restore,
        lead_ms: state.cfg.lead_ms,
        creds: None,
    };
    let handle = playback::start(&path, bulbs, &cfg).await?;
    tracing::info!(file = %handle.file, "web playback started");
    let duration_ms = handle.duration_ms;
    let supervisor = tokio::spawn(playback::supervise(handle.clone()));
    let file_display = handle.file.clone();
    *state.active.lock() = Some(Active { file: file_display, handle, supervisor });
    Ok(json!({"ok": true, "file": name, "duration_ms": duration_ms}))
}

pub async fn serve(port: u16, no_restore: bool, lead_ms: Option<u64>) -> Result<()> {
    let state = Arc::new(ServerState {
        active: Mutex::new(None),
        cfg: WebConfig { no_restore, lead_ms },
    });
    let listener = TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("bind 0.0.0.0:{port}"))?;
    let dir = std::env::current_dir()?;
    tracing::info!("web UI on http://0.0.0.0:{port} — serving {}", dir.display());

    loop {
        let (stream, _peer) = listener.accept().await?;
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_conn(stream, state).await {
                tracing::debug!("web connection error: {e:#}");
            }
        });
    }
}

async fn serve_conn(mut stream: TcpStream, state: Arc<ServerState>) -> Result<()> {
    let _ = stream.set_nodelay(true);
    let (method, path, body) = read_request(&mut stream).await?;
    let (status, ctype, body) = route(&state, &method, &path, &body).await;
    write_response(&mut stream, status, ctype, &body).await
}

async fn route(state: &ServerState, method: &str, path: &str, body: &[u8]) -> (u16, &'static str, Vec<u8>) {
    let ok = |v: Value| (200, "application/json", v.to_string().into_bytes());
    match (method, path) {
        ("GET", "/") => (200, "text/html; charset=utf-8", INDEX_HTML.as_bytes().to_vec()),
        ("GET", "/api/files") => match list_audio(&std::env::current_dir().unwrap_or_default()) {
            Ok(files) => ok(json!({
                "dir": std::env::current_dir().map(|d| d.display().to_string()).unwrap_or_default(),
                "files": files.iter().map(|(n, s)| json!({"name": n, "size": s})).collect::<Vec<_>>(),
            })),
            Err(e) => (500, "application/json", json!({"error": format!("{e:#}")}).to_string().into_bytes()),
        },
        ("GET", "/api/status") => {
            let mut guard = state.active.lock();
            // Reap finished supervisors.
            if let Some(a) = guard.as_ref() {
                if a.supervisor.is_finished() {
                    let _ = guard.take();
                }
            }
            match guard.as_ref() {
                Some(a) => ok(json!({
                    "playing": a.playing(),
                    "file": a.file,
                    "elapsed_ms": a.handle.elapsed_ms(),
                    "duration_ms": a.handle.duration_ms,
                    "underruns": a.handle.underruns(),
                })),
                None => ok(json!({"playing": false, "file": null, "elapsed_ms": 0, "duration_ms": 0, "underruns": 0})),
            }
        }
        ("POST", "/api/play") => match handle_play(state, body).await {
            Ok(v) => ok(v),
            Err(e) => (400, "application/json", json!({"error": format!("{e:#}")}).to_string().into_bytes()),
        },
        ("POST", "/api/stop") => {
            stop_active(state).await;
            ok(json!({"ok": true}))
        }
        (m, p) => (
            if p == "/" || p == "/api/status" { 405 } else { 404 },
            "application/json",
            json!({"error": format!("no route for {m} {p}")}).to_string().into_bytes(),
        ),
    }
}

async fn read_request(stream: &mut TcpStream) -> Result<(String, String, Vec<u8>)> {
    let deadline = Duration::from_secs(10);
    let read = tokio::time::timeout(deadline, async {
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        let mut chunk = [0u8; 4096];
        let header_end = loop {
            if let Some(p) = find(&buf, b"\r\n\r\n") {
                break p;
            }
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                bail!("client closed before request headers");
            }
            buf.extend_from_slice(&chunk[..n]);
            if buf.len() > MAX_BODY {
                bail!("request too large");
            }
        };
        let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
        let mut lines = head.split("\r\n");
        let request_line = lines.next().context("empty request")?;
        let mut parts = request_line.split_whitespace();
        let method = parts.next().context("no method")?.to_string();
        let raw_path = parts.next().context("no path")?.to_string();
        let path = raw_path.split('?').next().unwrap_or("").to_string();
        let content_length = lines
            .find_map(|l| {
                let (n, v) = l.split_once(':')?;
                if n.trim().eq_ignore_ascii_case("content-length") {
                    v.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);
        if content_length > MAX_BODY {
            bail!("body too large");
        }
        let mut body = buf[header_end + 4..].to_vec();
        while body.len() < content_length {
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                bail!("client closed inside body");
            }
            body.extend_from_slice(&chunk[..n]);
        }
        body.truncate(content_length);
        Ok((method, path, body))
    })
    .await
    .with_context(|| "request timeout")??;
    Ok(read)
}

async fn write_response(
    stream: &mut TcpStream,
    status: u16,
    ctype: &str,
    body: &[u8],
) -> Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bulbbeat</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body {
    min-height: 100vh; padding: 2.5rem 1rem;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: radial-gradient(1200px 800px at 15% -10%, #26314f 0%, #12141d 45%, #0b0c11 100%);
    color: #e8eaf2; display: flex; flex-direction: column; align-items: center; gap: 1.4rem;
  }
  h1 { font-size: 1.35rem; font-weight: 650; letter-spacing: .02em; }
  h1 span { background: linear-gradient(90deg, #ff5f6d, #ffc371, #7ef29d, #6fb1ff); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .dir { color: #8b90a3; font-size: .8rem; }
  .card {
    width: min(680px, 94vw); background: rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.08); border-radius: 16px;
    backdrop-filter: blur(8px); padding: 1rem; box-shadow: 0 10px 30px rgba(0,0,0,.35);
  }
  .now { display: none; flex-direction: column; gap: .6rem; }
  .now.on { display: flex; }
  .now .file { font-weight: 600; }
  .now .sub { color: #8b90a3; font-size: .8rem; }
  .bar { height: 8px; border-radius: 99px; background: rgba(255,255,255,.09); overflow: hidden; }
  .bar > div { height: 100%; width: 0%; border-radius: 99px;
    background: linear-gradient(90deg, #ff5f6d, #ffc371, #7ef29d, #6fb1ff); transition: width .4s ease; }
  .times { display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; font-size: .75rem; color: #8b90a3; }
  button {
    font: inherit; cursor: pointer; border: 0; border-radius: 10px; padding: .45rem .9rem;
    background: rgba(255,255,255,.08); color: #e8eaf2; transition: background .15s;
  }
  button:hover { background: rgba(255,255,255,.16); }
  button.primary { background: linear-gradient(135deg, #ff5f6d, #b06ab3); font-weight: 600; }
  button.primary:hover { filter: brightness(1.15); }
  button.stop { background: rgba(255,95,109,.25); }
  ul { list-style: none; display: flex; flex-direction: column; gap: .3rem; }
  li { display: flex; align-items: center; gap: .6rem; padding: .5rem .6rem; border-radius: 10px; }
  li:hover { background: rgba(255,255,255,.05); }
  li .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  li .size { color: #8b90a3; font-size: .75rem; font-variant-numeric: tabular-nums; }
  .empty { color: #8b90a3; text-align: center; padding: 1rem 0; }
  .spin { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.25);
    border-top-color: #fff; border-radius: 50%; animation: r 0.8s linear infinite; vertical-align: -2px; }
  @keyframes r { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <h1>bulb<span>beat</span></h1>
  <div class="dir" id="dir"></div>

  <div class="card now" id="now">
    <div class="file" id="np-file"></div>
    <div class="bar"><div id="np-bar"></div></div>
    <div class="times"><span id="np-el">0:00</span><span id="np-dur">0:00</span></div>
    <div style="display:flex; gap:.5rem;">
      <button class="stop" id="stop">Stop</button>
      <span class="sub" id="np-sub"></span>
    </div>
  </div>

  <div class="card">
    <ul id="files"></ul>
    <div class="empty" id="empty" style="display:none">No audio files in this folder.</div>
  </div>

<script>
const fmt = ms => { const s = Math.floor(ms/1000); return Math.floor(s/60) + ":" + String(s%60).padStart(2, "0"); };
let busy = null;

async function play(name, btn) {
  if (busy) return;
  busy = name; btn.innerHTML = '<span class="spin"></span>';
  try {
    const r = await fetch("/api/play", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({file: name})});
    const j = await r.json();
    if (!r.ok) alert(j.error || "playback failed");
  } catch (e) { alert(e); }
  btn.textContent = "▶"; busy = null;
}

async function stop() { await fetch("/api/stop", {method: "POST"}); }

async function refresh() {
  const [filesR, statusR] = await Promise.all([fetch("/api/files"), fetch("/api/status")]);
  const files = await filesR.json();
  const st = await statusR.json();
  document.getElementById("dir").textContent = files.dir;

  const now = document.getElementById("now");
  now.classList.toggle("on", !!st.file);
  if (st.file) {
    document.getElementById("np-file").textContent = st.file;
    document.getElementById("np-el").textContent = fmt(st.elapsed_ms);
    document.getElementById("np-dur").textContent = fmt(st.duration_ms);
    document.getElementById("np-bar").style.width =
      (st.duration_ms ? Math.min(100, 100 * st.elapsed_ms / st.duration_ms) : 0) + "%";
    document.getElementById("np-sub").textContent = st.playing ? "" : "finished";
  }

  const ul = document.getElementById("files");
  ul.innerHTML = "";
  document.getElementById("empty").style.display = files.files.length ? "none" : "block";
  for (const f of files.files) {
    const li = document.createElement("li");
    const name = document.createElement("span"); name.className = "name"; name.textContent = f.name;
    const size = document.createElement("span"); size.className = "size";
    size.textContent = (f.size / 1048576).toFixed(1) + " MB";
    const btn = document.createElement("button"); btn.className = "primary"; btn.textContent = "▶";
    btn.onclick = () => play(f.name, btn);
    li.append(name, size, btn); ul.append(li);
  }
}
document.getElementById("stop").onclick = stop;
refresh();
setInterval(refresh, 500);
</script>
</body>
</html>"#;
