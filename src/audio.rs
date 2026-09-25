//! Audio decoding (symphonia) and timed playback (cpal).
//!
//! The sync contract with the scheduler is [`PlaybackShared::t0`]: the instant
//! the first output callback writes frames — the audio clock origin.

use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use parking_lot::Mutex as PlMutex;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::formats::probe::Hint;

/// Fully decoded, interleaved audio.
pub struct Decoded {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub sample_rate: u32,
}

/// Decode a local audio file to interleaved f32 at the native rate.
pub fn decode(path: &Path) -> Result<Decoded> {
    let src = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .with_context(|| format!("probe {}", path.display()))?;

    let track = format
        .default_track(TrackType::Audio)
        .context("no audio track")?;
    let track_id = track.id;
    let CodecParameters::Audio(params) = track
        .codec_params
        .as_ref()
        .context("audio track has no codec parameters")?
    else {
        bail!("default track is not an audio track");
    };
    if params.codec == CODEC_ID_NULL_AUDIO {
        bail!("audio track has no decodable codec");
    }
    let sample_rate = params.sample_rate.context("track missing sample_rate")?;
    let channels = params
        .channels
        .as_ref()
        .map(|c| c.count())
        .filter(|&c| c > 0)
        .context("track missing channel count")?;
    let params: AudioCodecParameters = params.clone();

    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(
        &params,
        &AudioDecoderOptions::default(),
    )?;

    let mut samples = Vec::new();
    let mut scratch = Vec::new();
    loop {
        let Some(packet) = format.next_packet()? else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded: GenericAudioBufferRef<'_> = decoder
            .decode(&packet)
            .with_context(|| format!("decode packet in {}", path.display()))?;
        scratch.clear();
        decoded.copy_to_vec_interleaved(&mut scratch);
        samples.extend_from_slice(&scratch);
    }
    if samples.is_empty() {
        bail!("no audio decoded from {}", path.display());
    }
    Ok(Decoded { samples, channels, sample_rate })
}

/// State shared between the scheduler and the audio callback.
pub struct PlaybackShared {
    buf: PlMutex<VecDeque<f32>>,
    /// Total interleaved frames (stereo) scheduled for playback.
    total: usize,
    /// Frames consumed so far.
    consumed: PlMutex<usize>,
    /// First-callback instant: the audio clock origin (`t0`).
    t0: PlMutex<Option<Instant>>,
    pub done: AtomicBool,
    pub underruns: AtomicU64,
}

impl PlaybackShared {
    fn new(total: usize) -> Self {
        Self {
            buf: PlMutex::new(VecDeque::with_capacity(total)),
            total,
            consumed: PlMutex::new(0),
            t0: PlMutex::new(None),
            done: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
        }
    }

    /// The audio clock origin, set by the first non-silent callback.
    pub fn t0(&self) -> Option<Instant> {
        *self.t0.lock()
    }

    fn pull(&self, frames: usize, channels: usize, out: &mut [f32]) {
        let mut q = match self.buf.try_lock() {
            Some(q) => q,
            None => {
                // Never block the callback.
                out.fill(0.0);
                return;
            }
        };
        let mut consumed = self.consumed.lock();
        let want = frames * channels;
        if q.is_empty() && self.done.load(Ordering::Relaxed) {
            out.fill(0.0);
            return;
        }
        let n = want.min(q.len());
        if q.len() < want && !self.done.load(Ordering::Relaxed) {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
        for slot in out.iter_mut().take(n) {
            *slot = q.pop_front().unwrap_or(0.0);
        }
        for slot in out.iter_mut().skip(n) {
            *slot = 0.0;
        }
        *consumed += n / channels.max(1);
        if self.t0.lock().is_none() {
            *self.t0.lock() = Some(Instant::now());
        }
        if *consumed >= self.total {
            self.done.store(true, Ordering::Relaxed);
        }
    }
}

/// Open the default output device and return a paused stream plus shared
/// state with the whole file prefilled.
pub fn open_output(decoded: &Decoded) -> Result<(cpal::Stream, Arc<PlaybackShared>)> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no default output device")?;

    let mut config = device
        .supported_output_configs()
        .context("query output configs")?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .collect::<Vec<_>>();
    if config.is_empty() {
        let available: Vec<String> = device
            .supported_output_configs()
            .map(|it| it.map(|c| format!("{:?}", c.sample_format())).collect())
            .unwrap_or_default();
        bail!("no F32 output config on device; available formats: {available:?}");
    }
    // Prefer a range that can host the file's rate; else the first range.
    let range = match config.iter().find(|c| {
        c.min_sample_rate() <= decoded.sample_rate && decoded.sample_rate <= c.max_sample_rate()
    }) {
        Some(r) => r.clone(),
        None => config.swap_remove(0),
    };
    // Some virtual devices report a max rate near u32::MAX — never request
    // that; clamp the file's rate into the range with a 192 kHz sanity cap.
    let lo = range.min_sample_rate();
    let hi = range.max_sample_rate().min(192_000).max(lo);
    let rate = decoded.sample_rate.clamp(lo, hi);
    let cfg: cpal::StreamConfig = range.with_sample_rate(rate).into();

    // Mixdown to stereo interleaved.
    let stereo = mixdown_stereo(&decoded.samples, decoded.channels);

    // Resample if needed.
    let stereo = if cfg.sample_rate != decoded.sample_rate {
        resample_linear(&stereo, decoded.sample_rate, cfg.sample_rate)
    } else {
        stereo
    };
    let total_frames = stereo.len() / 2;
    if total_frames == 0 {
        bail!("decoded audio is empty after mixdown");
    }

    let shared = Arc::new(PlaybackShared::new(total_frames));
    shared.buf.lock().extend(stereo);

    let dev_channels = cfg.channels as usize;
    let cb_shared = shared.clone();
    let stream = device
        .build_output_stream(
            cfg,
            move |out: &mut [f32], _cb| {
                let frames = out.len() / dev_channels.max(1);
                let mut tmp = vec![0.0f32; frames * 2];
                cb_shared.pull(frames, 2, &mut tmp);
                match dev_channels {
                    1 => {
                        for (i, slot) in out.iter_mut().enumerate() {
                            let l = tmp.get(2 * i).copied().unwrap_or(0.0);
                            let r = tmp.get(2 * i + 1).copied().unwrap_or(0.0);
                            *slot = (l + r) * 0.5;
                        }
                    }
                    2 => out.copy_from_slice(&tmp),
                    n => {
                        for (i, frame) in out.chunks_mut(n).enumerate() {
                            frame[0] = tmp.get(2 * i).copied().unwrap_or(0.0);
                            frame[1] = tmp.get(2 * i + 1).copied().unwrap_or(0.0);
                            for extra in frame.iter_mut().skip(2) {
                                *extra = 0.0;
                            }
                        }
                    }
                }
            },
            |err| tracing::error!(%err, "audio output stream error"),
            None,
        )
        .context("build output stream")?;
    Ok((stream, shared))
}

fn mixdown_stereo(samples: &[f32], channels: usize) -> Vec<f32> {
    match channels {
        0 => Vec::new(),
        1 => samples.iter().flat_map(|&s| [s, s]).collect(),
        2 => samples.to_vec(),
        n => samples
            .chunks(n)
            .flat_map(|frame| [frame[0], frame[1]])
            .collect(),
    }
}

/// Linear-interpolation resampler for interleaved stereo — quality is ample
/// for monitoring and it is exact at every output sample position.
fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let in_frames = input.len() / 2;
    let ratio = to as f64 / from as f64;
    let out_frames = ((in_frames as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_frames * 2);
    for i in 0..out_frames {
        let pos = i as f64 / ratio;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(in_frames - 1);
        let frac = (pos - i0 as f64) as f32;
        for ch in 0..2 {
            let a = input[i0 * 2 + ch];
            let b = input[i1 * 2 + ch];
            out.push(a + (b - a) * frac);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-rolled 44-byte PCM16 mono 8 kHz sine WAV, decoded through the
    /// symphonia path.
    #[test]
    fn decode_wav_sine() {
        let sr = 8000u32;
        let seconds = 1;
        let n = (sr * seconds) as usize;
        let mut data: Vec<u8> = Vec::new();
        for i in 0..n {
            let s = (0.5 * (std::f64::consts::TAU * 440.0 * i as f64 / sr as f64).sin()
                * 32767.0) as i16;
            data.extend_from_slice(&s.to_le_bytes());
        }
        let mut wav: Vec<u8> = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);

        let path = std::env::temp_dir().join(format!("bulbbeat-test-{}.wav", std::process::id()));
        std::fs::write(&path, &wav).unwrap();
        let decoded = decode(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(decoded.sample_rate, sr);
        assert_eq!(decoded.channels, 1);
        // Allow one packet of slack for encoder/decoder padding.
        assert!(
            (decoded.samples.len() as i64 - n as i64).abs() <= 2048,
            "expected ~{} samples, got {}",
            n,
            decoded.samples.len()
        );
    }

    #[test]
    fn mixdown_channels() {
        let mono = mixdown_stereo(&[0.5, 0.25], 1);
        assert_eq!(mono, vec![0.5, 0.5, 0.25, 0.25]);
        let stereo = mixdown_stereo(&[0.1, 0.9], 2);
        assert_eq!(stereo, vec![0.1, 0.9]);
        let six = mixdown_stereo(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 6);
        assert_eq!(six, vec![0.1, 0.2]);
    }

    #[test]
    fn resample_identity_and_upsample() {
        let x = vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.5];
        assert_eq!(resample_linear(&x, 44100, 44100), x);
        let up = resample_linear(&x, 8000, 16000);
        assert_eq!(up.len(), x.len() * 2);
        // First output sample equals first input sample.
        assert_eq!(up[0], 0.0);
        assert_eq!(up[1], 0.0);
    }

    #[test]
    fn playback_shared_pull_and_done() {
        let shared = PlaybackShared::new(2); // 2 stereo frames
        shared.buf.lock().extend([0.1f32, 0.2, 0.3, 0.4]);
        let mut out = [0f32; 4];
        shared.pull(2, 2, &mut out);
        assert_eq!(out, [0.1, 0.2, 0.3, 0.4]);
        assert!(shared.done.load(Ordering::Relaxed));
        assert!(shared.t0().is_some());
        assert_eq!(shared.underruns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn playback_shared_underrun_fills_zero() {
        let shared = PlaybackShared::new(10);
        let mut out = [0f32; 4];
        shared.pull(2, 2, &mut out);
        assert_eq!(out, [0.0; 4]);
        assert_eq!(shared.underruns.load(Ordering::Relaxed), 1);
        assert!(!shared.done.load(Ordering::Relaxed));
    }
}
