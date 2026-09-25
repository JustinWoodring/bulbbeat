//! Per-bulb scheduling: send each light step ahead of its audio time.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::mpsc;

use crate::analyze::LightStep;
use crate::audio::PlaybackShared;
use kasa_rs::{Creds, bulb::{Bulb, Caps}};

const LEAD_BASE_MS: u64 = 25;
const T0_POLL_MS: u64 = 10;
/// Ramp tail: final segment's device-side ramp finishes before restore.
const RESTORE_TAIL_MS: u64 = 1000;

/// Drive one bulb for the duration of a playback.
///
/// Sends `Result<Caps>` on `ready_tx` exactly once: `Ok` as soon as the bulb
/// joined (before playback starts), `Err` when it could not be used at all.
pub async fn run_bulb(
    host: String,
    creds: Option<Creds>,
    steps: Arc<Vec<LightStep>>,
    shared: Arc<PlaybackShared>,
    lead_override: Option<Duration>,
    restore: bool,
    cancel: Arc<AtomicBool>,
    ready_tx: mpsc::Sender<Result<Caps>>,
) {
    let bulb = Bulb::connect(&host, creds.as_ref()).await;
    let mut bulb = match bulb {
        Ok(b) => b,
        Err(e) => {
            let _ = ready_tx.send(Err(anyhow::anyhow!("{e:#}"))).await;
            return;
        }
    };
    let caps = bulb.caps.clone();
    let lead = match lead_override {
        Some(l) => l,
        None => match bulb.rtt_probe().await {
            Ok(rtt) => rtt.mul_f32(0.5) + Duration::from_millis(LEAD_BASE_MS),
            Err(e) => {
                let _ = ready_tx.send(Err(anyhow::anyhow!("{e:#}"))).await;
                return;
            }
        },
    };
    tracing::info!(host, alias = %caps.alias, color = caps.is_color, dimmable = caps.is_dimmable,
        lead_ms = lead.as_millis() as u64, "bulb ready");
    // Report usable before playback starts; from here on failures are
    // logged, never fatal.
    let _ = ready_tx.send(Ok(caps.clone())).await;

    if let Err(e) = run_playback(&host, &mut bulb, steps, shared, lead, restore, &cancel).await {
        tracing::warn!(host, "playback scheduling ended with error: {e:#}");
    }
}

async fn run_playback(
    host: &str,
    bulb: &mut Bulb,
    steps: Arc<Vec<LightStep>>,
    shared: Arc<PlaybackShared>,
    lead: Duration,
    restore: bool,
    cancel: &Arc<AtomicBool>,
) -> Result<()> {

    // Wait for the audio clock origin.
    let t0 = loop {
        if let Some(t0) = shared.t0() {
            break t0;
        }
        if cancel.load(Ordering::Relaxed) {
            // Audio never started; just put things back.
            if restore {
                let _ = bulb.restore().await;
            }
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(T0_POLL_MS)).await;
    };

    for step in steps.iter() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let target = t0 + Duration::from_millis(step.start_ms) - lead;
        let now = Instant::now();
        if target > now {
            tokio::time::sleep_until(target.into()).await;
        }
        let skew = Instant::now().duration_since(target);
        tracing::debug!(host, step = step.start_ms, skew_ms = skew.as_millis() as i64,
            hue = step.hue, sat = step.sat, bri = step.bri, "send");
        // A stale color is worse than a missed one: skip the step on error;
        // send_json reconnects lazily on the next call.
        let result = if bulb.caps.is_color {
            bulb.set_hsv(step.hue, step.sat, step.bri, step.transition_ms).await
        } else {
            bulb.set_brightness(step.bri, step.transition_ms).await
        };
        if let Err(e) = result {
            tracing::warn!(host, step = step.start_ms, "set_light failed, skipping: {e:#}");
        }
    }

    if restore {
        if cancel.load(Ordering::Relaxed) {
            if let Err(e) = bulb.restore().await {
                tracing::warn!(host, "restore failed: {e:#}");
            }
        } else {
            tokio::time::sleep(Duration::from_millis(RESTORE_TAIL_MS)).await;
            if let Err(e) = bulb.restore().await {
                tracing::warn!(host, "restore failed: {e:#}");
            }
        }
    }
    Ok(())
}
