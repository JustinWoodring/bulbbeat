//! Playback orchestration shared by the CLI and the web daemon.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::audio::{self, PlaybackShared};
use kasa_rs::{Creds, bulb::Caps};

/// Options controlling a playback run.
#[derive(Clone, Default)]
pub struct PlaybackConfig {
    pub no_restore: bool,
    pub lead_ms: Option<u64>,
    pub creds: Option<Creds>,
}

struct HandleInner {
    cancel: Arc<AtomicBool>,
    shared: Arc<PlaybackShared>,
    stream: Mutex<Option<cpal::Stream>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

/// A started playback: audio is live, bulb tasks are running.
/// Cheap to clone; every clone sees the same underlying state.
#[derive(Clone)]
pub struct PlaybackHandle {
    pub file: String,
    pub duration_ms: u64,
    inner: Arc<HandleInner>,
}

impl PlaybackHandle {
    pub fn cancel_now(&self) {
        self.inner.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_canceling(&self) -> bool {
        self.inner.cancel.load(Ordering::Relaxed)
    }

    pub fn done(&self) -> bool {
        self.inner.shared.done.load(Ordering::Relaxed)
    }

    /// Milliseconds of audio actually played so far.
    pub fn elapsed_ms(&self) -> u64 {
        self.inner
            .shared
            .t0()
            .map(|t0| t0.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    pub fn underruns(&self) -> u64 {
        self.inner.shared.underruns.load(Ordering::Relaxed)
    }

    /// Resolve when audio drained or cancellation was requested.
    pub async fn finished(&self) {
        loop {
            if self.is_canceling() || self.done() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Stop audio and wait for bulb tasks (they restore on cancel).
    /// Idempotent.
    pub async fn shutdown(&self) {
        self.cancel_now();
        drop(self.inner.stream.lock().take());
        let tasks: Vec<JoinHandle<()>> = self.inner.tasks.lock().drain(..).collect();
        for task in tasks {
            let _ = task.await;
        }
    }
}

/// With no explicit bulbs, discover every bulb on the LAN and filter to
/// devices that can actually light up.
pub async fn resolve_bulbs(bulbs: Vec<String>) -> Result<Vec<String>> {
    if !bulbs.is_empty() {
        return Ok(bulbs);
    }
    let devices = kasa_rs::discover::discover_devices(crate::DISCOVER_TIMEOUT_SECS).await?;
    if devices.is_empty() {
        bail!("no Kasa devices answered discovery — pass --bulb <ip> or check the network");
    }
    let (bulbs, not_bulbs): (Vec<_>, Vec<_>) = devices.into_iter().partition(|d| d.is_bulb());
    for d in not_bulbs {
        tracing::warn!("{} ({}) is neither color nor dimmable — skipping", d.alias, d.model);
    }
    if bulbs.is_empty() {
        bail!("discovered Kasa devices but none of them are bulbs");
    }
    let ips: Vec<String> = bulbs.iter().map(|d| d.ip.clone()).collect();
    tracing::info!("discovered {} bulb(s): {}", ips.len(), ips.join(", "));
    Ok(ips)
}

/// Decode, analyze, open the default output, and drive every bulb.
/// Audio is already playing when this returns (zero usable bulbs → error).
pub async fn start(path: &Path, bulbs: Vec<String>, cfg: &PlaybackConfig) -> Result<PlaybackHandle> {
    let decoded = audio::decode(path)?;
    let duration_ms = (decoded.samples.len() / decoded.channels.max(1)) as u64 * 1000
        / decoded.sample_rate as u64;

    let steps = Arc::new(crate::analyze::analyze(
        &crate::mono_mixdown(&decoded),
        decoded.sample_rate,
    ));
    let flashes = steps.iter().filter(|s| s.transition_ms < 100).count();
    let per_s = if duration_ms > 0 {
        steps.len() * 1000 / duration_ms as usize
    } else {
        steps.len()
    };
    tracing::info!(
        "analyzed {}:{} — {} light commands ({} beat flashes), avg {}/s",
        duration_ms / 60_000,
        (duration_ms / 1000) % 60,
        steps.len(),
        flashes,
        per_s
    );

    let bulbs = resolve_bulbs(bulbs).await?;
    let (stream, shared) = audio::open_output(&decoded)?;

    let cancel = Arc::new(AtomicBool::new(false));
    let (ready_tx, mut ready_rx) = mpsc::channel::<Result<Caps>>(bulbs.len());
    let mut tasks = Vec::with_capacity(bulbs.len());
    for host in &bulbs {
        let task = tokio::spawn(crate::scheduler::run_bulb(
            host.clone(),
            cfg.creds.clone(),
            steps.clone(),
            shared.clone(),
            cfg.lead_ms.map(Duration::from_millis),
            !cfg.no_restore,
            cancel.clone(),
            ready_tx.clone(),
        ));
        tasks.push(task);
    }
    drop(ready_tx);

    let mut usable = 0usize;
    for _ in 0..bulbs.len() {
        match ready_rx.recv().await {
            Some(Ok(_caps)) => usable += 1,
            Some(Err(e)) => tracing::warn!("bulb unavailable: {e:#}"),
            None => break,
        }
    }
    if usable == 0 {
        bail!("no usable bulbs out of {} target(s); not playing", bulbs.len());
    }

    use cpal::traits::StreamTrait;
    stream.play().context("start playback")?;

    Ok(PlaybackHandle {
        file: path.display().to_string(),
        duration_ms,
        inner: Arc::new(HandleInner {
            cancel,
            shared,
            stream: Mutex::new(Some(stream)),
            tasks: Mutex::new(tasks),
        }),
    })
}

/// Run a handle to completion, then stop audio and reap bulb tasks.
/// Cancellation is external: set `cancel_now()` (or let the track drain).
pub async fn supervise(handle: PlaybackHandle) {
    handle.finished().await;
    if handle.done() && !handle.is_canceling() {
        // Grace so trailing bulb steps (sent ahead of audio) land.
        tokio::time::sleep(Duration::from_millis(1000)).await;
    } else {
        tracing::info!("playback stopped; restoring bulbs");
    }
    handle.shutdown().await;
}
