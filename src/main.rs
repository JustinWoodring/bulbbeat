mod analyze;
mod audio;

mod playback;
mod scheduler;
mod web;


use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use crate::audio::Decoded;
use kasa_rs::Creds;
use crate::playback::PlaybackConfig;

/// How long to wait for discovery answers when no --bulb is given.
const DISCOVER_TIMEOUT_SECS: u64 = 3;

#[derive(Parser)]
#[command(name = "bulbbeat", about = "Music-synced Kasa bulb lighting", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Serve a web UI on --port instead of running a one-shot command
    #[arg(long)]
    web: bool,
    /// Port for the web UI
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[derive(Subcommand)]
enum Command {
    /// Discover Kasa devices on the local network
    Discover {
        /// How long to listen for responses
        #[arg(long, default_value_t = 3)]
        timeout_secs: u64,
    },
    /// Play an audio file with synced lighting
    Play {
        /// Audio file (mp3, flac, wav, ogg, m4a)
        audio: String,
        /// Bulb IP address (repeat for multiple bulbs)
        #[arg(long = "bulb")]
        bulbs: Vec<String>,
        /// Kasa account email (for KLAP devices; optional)
        #[arg(long)]
        email: Option<String>,
        /// Kasa account password (for KLAP devices; optional)
        #[arg(long)]
        password: Option<String>,
        /// Analyze and print the light plan without bulbs or audio output
        #[arg(long)]
        dry_run: bool,
        /// Do not restore bulb state at exit
        #[arg(long)]
        no_restore: bool,
        /// Override the RTT-probe-based lead time (debug)
        #[arg(long)]
        lead_ms: Option<u64>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    match cli.command {
        Some(Command::Discover { timeout_secs }) => {
            let devices = kasa_rs::discover::discover_devices(timeout_secs).await?;
            if devices.is_empty() {
                tracing::warn!("no Kasa devices answered within {}s", timeout_secs);
                return Ok(());
            }
            for d in &devices {
                println!(
                    "{}\t{}\t{}\tcolor={}\tdimmable={}",
                    d.ip,
                    d.model,
                    d.alias,
                    if d.is_color { "yes" } else { "no" },
                    if d.is_dimmable { "yes" } else { "no" }
                );
            }
            Ok(())
        }
        Some(Command::Play {
            audio,
            bulbs,
            email,
            password,
            dry_run,
            no_restore,
            lead_ms,
        }) => play(audio, bulbs, email, password, dry_run, no_restore, lead_ms).await,
        None if cli.web => web::serve(cli.port, false, None).await,
        None => bail!("nothing to do: pass --web or a subcommand"),
    }
}

/// Mono mixdown for analysis: average all channels.
fn mono_mixdown(decoded: &Decoded) -> Vec<f32> {
    match decoded.channels {
        0 | 1 => decoded.samples.clone(),
        n => decoded
            .samples
            .chunks(n)
            .map(|frame| frame.iter().sum::<f32>() / n as f32)
            .collect(),
    }
}

async fn play(
    audio: String,
    bulbs: Vec<String>,
    email: Option<String>,
    password: Option<String>,
    dry_run: bool,
    no_restore: bool,
    lead_ms: Option<u64>,
) -> Result<()> {
    let path = std::path::PathBuf::from(&audio);
    let decoded = audio::decode(&path)?;

    if dry_run {
        let steps = analyze::analyze(&mono_mixdown(&decoded), decoded.sample_rate);
        println!("start_ms\thue\tsat\tbri\ttransition_ms");
        for s in &steps {
            println!("{}\t{}\t{}\t{}\t{}", s.start_ms, s.hue, s.sat, s.bri, s.transition_ms);
        }
        return Ok(());
    }

    let creds = match (&email, &password) {
        (Some(e), Some(p)) => Some(Creds {
            email: e.clone(),
            password: p.clone(),
        }),
        (None, None) => None,
        _ => bail!("--email and --password must be given together"),
    };

    let handle = playback::start(&path, bulbs, &PlaybackConfig {
        no_restore,
        lead_ms,
        creds,
    })
    .await?;

    tokio::select! {
        _ = handle.finished() => tracing::info!("playback finished"),
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("interrupted; stopping audio and restoring bulbs");
            handle.cancel_now();
        }
    }
    handle.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_mixdown_averages_channels() {
        let d = Decoded {
            samples: vec![0.2, 0.4, -0.2, 0.2],
            channels: 2,
            sample_rate: 8000,
        };
        let mono = mono_mixdown(&d);
        assert_eq!(mono, vec![0.3, 0.0]);
    }
}
