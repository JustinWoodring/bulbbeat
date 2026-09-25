//! Offline audio analysis and music→light mapping.
//!
//! All mapping constants live here and only here.
//!
//! Output model: a 1 Hz "base" step carries the color underlay (hue from
//! spectral centroid, saturation from bass share, slowly-varying brightness
//! from loudness). On top of that, spectral-flux onset detection schedules
//! short-attack flash + decay steps so beats stay visible despite the bulb's
//! ~1 s transition morph.

use rustfft::{FftPlanner, num_complex::Complex};

pub const SEGMENT_MS: u64 = 1000;

const MIN_LAST_SEGMENT_MS: u64 = 50;
const BASS_HZ: f32 = 250.0;
const SILENCE_RMS: f32 = 1e-5;

/// Spectral bands and the hue each "hotter than usual" band pulls toward.
/// Bass→red, low→amber, mid (vocals)→green, high (synths/cymbals)→azure,
/// air (atmosphere)→violet.
const BAND_EDGES: [(f32, f32); 5] =
    [(20.0, 150.0), (150.0, 400.0), (400.0, 2000.0), (2000.0, 8000.0), (8000.0, 20000.0)];
const HUE_ANCHORS_DEG: [f32; 5] = [0.0, 60.0, 130.0, 210.0, 270.0];
/// Total above-median share needed before hue is steered at all; below it
/// (a perfectly median mix) the previous hue carries.
const MIN_EXCESS_WEIGHT: f32 = 0.02;

/// Base underlay: brightness dynamic range and contrast exponent.
const BRI_MIN: f32 = 12.0;
const BRI_MAX: f32 = 88.0;
const LOUD_EXP: f32 = 1.4;
const BASE_TRANSITION_MS: u32 = 900;

/// Onset flash/decay shaping.
const STFT_WINDOW: usize = 1024;
const STFT_HOP: usize = 512;
const ONSET_RATIO: f32 = 1.6; // vs mean of previous frames
const ONSET_MIN_NORM: f32 = 0.3; // vs song-level 98th percentile flux
const ONSET_OVER_MEDIAN: f32 = 4.0; // vs song-level median flux (kills tones)
const MAX_ONSETS_PER_SEC: usize = 3;
const MIN_ONSET_GAP_MS: u32 = 120;
const FLASH_TRANSITION_MS: u32 = 40;
const FLASH_BASE_DELTA: f32 = 18.0;
const FLASH_NORM_DELTA: f32 = 34.0;
const DECAY_DELAY_MS: u32 = 230;
const DECAY_TRANSITION_MS: u32 = 380;

/// One light command at an arbitrary time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightStep {
    pub start_ms: u64,
    pub hue: u16,
    pub sat: u8,
    pub bri: u8,
    pub transition_ms: u32,
}

struct SegFeatures {
    rms: f32,
    bass_ratio: f32,
    /// Per-band power share (fractions summing to ~1).
    shares: [f32; 5],
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 * (1.0 - std::f32::consts::TAU * k as f32 / (n - 1).max(1) as f32).cos())
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

/// FFT-based per-segment features. `seg` is already in mono.
fn segment_features(seg: &[f32], sample_rate: u32) -> SegFeatures {
    let n = seg.len();
    let window = hann(n);
    let nfft = n.next_power_of_two();
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); nfft];
    for (i, (&s, &w)) in seg.iter().zip(&window).enumerate() {
        buf[i] = Complex::new(s * w, 0.0);
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    fft.process(&mut buf);

    let bin_hz = sample_rate as f32 / nfft as f32;
    let bins = &buf[1..nfft / 2];
    let mags: Vec<f32> = bins.iter().map(|c| c.norm()).collect();
    let mut band_p = [0.0f32; 5];
    for (i, &m) in mags.iter().enumerate() {
        let f = (i + 1) as f32 * bin_hz;
        for (b, (lo, hi)) in BAND_EDGES.iter().enumerate() {
            if f >= *lo && f < *hi {
                band_p[b] += m * m;
            }
        }
    }
    let band_total: f32 = band_p.iter().sum();
    let shares = if band_total > f32::EPSILON {
        band_p.map(|p| p / band_total)
    } else {
        [0.0; 5]
    };
    // Bass share measured in spectral energy (power), not magnitude: a square
    // wave's harmonics decay 1/k in amplitude but 1/k² in power, so the
    // energy fraction of the fundamental is what makes bass "vivid".
    let bass: f32 = mags
        .iter()
        .enumerate()
        .filter(|(i, _)| (i + 1) as f32 * bin_hz < BASS_HZ)
        .map(|(_, &m)| m * m)
        .sum();
    let total_power: f32 = mags.iter().map(|m| m * m).sum();
    let bass_ratio = if total_power > f32::EPSILON { bass / total_power } else { 0.0 };

    SegFeatures { rms: rms(seg), bass_ratio, shares }
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        len => {
            let idx = p / 100.0 * (len - 1) as f32;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            let frac = idx - lo as f32;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }
}

fn wrap_delta(d: f32) -> f32 {
    ((d + 180.0).rem_euclid(360.0)) - 180.0
}

/// Smooth hue toward `raw_target` by 60% of the wrapped distance.
fn smooth_hue(prev: f32, raw_target: f32) -> f32 {
    (prev + 0.6 * wrap_delta(raw_target - prev)).rem_euclid(360.0)
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Map a mono signal to light commands. Rate-independent: band edges are in
/// Hz, so any sample rate works.
pub fn analyze(mono: &[f32], sample_rate: u32) -> Vec<LightStep> {
    let seg_len = (sample_rate as usize * SEGMENT_MS as usize) / 1000;
    if seg_len == 0 || mono.is_empty() {
        return Vec::new();
    }
    let min_last = (sample_rate as usize * MIN_LAST_SEGMENT_MS as usize) / 1000;

    let mut segs: Vec<&[f32]> = Vec::new();
    let mut start = 0;
    while start < mono.len() {
        let end = (start + seg_len).min(mono.len());
        if end - start >= min_last {
            segs.push(&mono[start..end]);
        }
        start = end;
    }
    if segs.is_empty() {
        return Vec::new();
    }

    let features: Vec<SegFeatures> =
        segs.iter().map(|s| segment_features(s, sample_rate)).collect();
    let onsets = detect_onsets(mono, sample_rate);

    let mut rms_sorted: Vec<f32> = features.iter().map(|f| f.rms).collect();
    rms_sorted.sort_by(|a, b| a.total_cmp(b));
    let lo = percentile(&rms_sorted, 5.0);
    let hi = percentile(&rms_sorted, 95.0);

    // Per-band median share over the song: a band "pulls" the hue only when
    // hotter than its own typical presence.
    let mut median_shares = [0.0f32; 5];
    for b in 0..5 {
        let mut col: Vec<f32> = features.iter().map(|f| f.shares[b]).collect();
        col.sort_by(|a, b| a.total_cmp(b));
        median_shares[b] = percentile(&col, 50.0);
    }

    let mut steps: Vec<LightStep> = Vec::with_capacity(segs.len() * 2);
    let mut prev_hue: Option<f32> = None;
    let mut prev_sat: Option<u8> = None;
    let mut onset_iter = onsets.iter().peekable();

    for (seg_i, f) in features.iter().enumerate() {
        let seg_start_ms = seg_i as u64 * SEGMENT_MS;
        let loud = if hi <= lo { 0.5 } else { clamp01((f.rms - lo) / (hi - lo)) };

        if f.rms < SILENCE_RMS || f.shares.iter().all(|&s| s <= f32::EPSILON) {
            // Digital silence: carry previous hue/sat, dim but never off.
            steps.push(LightStep {
                start_ms: seg_start_ms,
                hue: prev_hue.unwrap_or(0.0).round().clamp(0.0, 360.0) as u16,
                sat: prev_sat.unwrap_or(55),
                bri: 5,
                transition_ms: BASE_TRANSITION_MS,
            });
            continue;
        }

        // Hue: circular blend of band anchors weighted by above-median
        // presence. A balanced mix carries the previous hue.
        let mut weights = [0.0f32; 5];
        for (b, &share) in f.shares.iter().enumerate() {
            weights[b] = (share - median_shares[b]).max(0.0);
        }
        let wsum: f32 = weights.iter().sum();
        let hue_raw = if wsum < MIN_EXCESS_WEIGHT {
            let dominant = f
                .shares
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(b, _)| b)
                .unwrap_or(0);
            HUE_ANCHORS_DEG[dominant]
        } else {
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            for (b, &w) in weights.iter().enumerate() {
                let rad = HUE_ANCHORS_DEG[b].to_radians();
                sx += w * rad.cos();
                sy += w * rad.sin();
            }
            sy.atan2(sx).to_degrees().rem_euclid(360.0)
        };
        let hue = match prev_hue {
            Some(prev) => smooth_hue(prev, hue_raw),
            None => hue_raw.rem_euclid(360.0),
        };
        let hue = hue.round().clamp(0.0, 360.0) as u16;

        let sat = (55.0 + 45.0 * f.bass_ratio).round().clamp(0.0, 100.0) as u8;
        let base_bri = (BRI_MIN + (BRI_MAX - BRI_MIN) * loud.powf(LOUD_EXP))
            .round()
            .clamp(0.0, 100.0) as u8;

        steps.push(LightStep {
            start_ms: seg_start_ms,
            hue,
            sat,
            bri: base_bri,
            transition_ms: BASE_TRANSITION_MS,
        });
        prev_hue = Some(hue as f32);
        prev_sat = Some(sat);

        // Onset flashes on top of the base underlay: take this second's
        // onsets, strongest first, up to MAX_ONSETS_PER_SEC.
        let seg_end_ms = seg_start_ms + SEGMENT_MS;
        let mut in_second: Vec<&Onset> = Vec::new();
        while let Some(o) = onset_iter.peek() {
            if o.ms >= seg_end_ms {
                break;
            }
            if o.ms >= seg_start_ms {
                in_second.push(o);
            }
            onset_iter.next();
        }
        in_second.sort_by(|a, b| b.norm.total_cmp(&a.norm));
        in_second.truncate(MAX_ONSETS_PER_SEC);
        in_second.sort_by(|a, b| a.ms.cmp(&b.ms));

        for onset in in_second {
            let offset_ms = onset.ms - seg_start_ms;
            let flash = FLASH_BASE_DELTA + FLASH_NORM_DELTA * onset.norm;
            let bri = (base_bri as f32 + flash).round().clamp(0.0, 100.0) as u8;
            steps.push(LightStep {
                start_ms: onset.ms,
                hue,
                sat,
                bri,
                transition_ms: FLASH_TRANSITION_MS,
            });
            // Decay back to base shortly after; if the flash is too close
            // to the next second, its base step acts as the decay.
            let decay_start = offset_ms + DECAY_DELAY_MS as u64;
            if decay_start + 80 < SEGMENT_MS {
                steps.push(LightStep {
                    start_ms: seg_start_ms + decay_start as u64,
                    hue,
                    sat,
                    bri: base_bri,
                    transition_ms: DECAY_TRANSITION_MS,
                });
            }
        }
    }

    steps.sort_by_key(|s| s.start_ms);
    steps
}

struct Onset {
    /// Global millisecond position in the song.
    ms: u64,
    /// Normalized strength (0..1 vs song-level p98 flux).
    norm: f32,
}

/// Detect onsets over the whole song with one continuous STFT: frame k
/// starts at `k * STFT_HOP` samples. A continuous history avoids the
/// window fill-in spike that a per-second restart would poison every
/// segment boundary with.
fn detect_onsets(mono: &[f32], sample_rate: u32) -> Vec<Onset> {
    if mono.len() < STFT_WINDOW * 2 {
        return Vec::new();
    }
    let frames = (mono.len() - STFT_WINDOW) / STFT_HOP + 1;
    let window = hann(STFT_WINDOW);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(STFT_WINDOW);
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); STFT_WINDOW];
    let mut prev_mags: Option<Vec<f32>> = None;
    let mut flux: Vec<f32> = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * STFT_HOP;
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(mono[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        let mags: Vec<f32> = buf[..STFT_WINDOW / 2].iter().map(|c| c.norm()).collect();
        let cur = match &prev_mags {
            None => 0.0,
            Some(prev) => mags
                .iter()
                .zip(prev.iter())
                .map(|(&c, &p)| (c - p).max(0.0))
                .sum(),
        };
        flux.push(cur);
        prev_mags = Some(mags);
    }

    let mut sorted_flux = flux[8..].to_vec();
    sorted_flux.sort_by(|a, b| a.total_cmp(b));
    let flux_p98 = percentile(&sorted_flux, 98.0);
    let flux_p50 = percentile(&sorted_flux, 50.0);
    if flux_p98 <= f32::EPSILON {
        return Vec::new();
    }

    let mut onsets: Vec<Onset> = Vec::new();
    for k in 8..flux.len().saturating_sub(2) {
        let history = &flux[k - 8..k];
        let mean: f32 = history.iter().sum::<f32>() / history.len() as f32;
        // Strict on the left, tolerant on the right: a plateau of equal
        // values must not fire per frame.
        let is_peak = flux[k] > flux[k - 1]
            && flux[k] >= flux[k + 1]
            && flux[k] >= flux[k + 2];
        // The median gate rejects stationary tones whose flux "crests" are
        // just the level itself; percussive material has a near-zero median.
        if is_peak
            && flux[k] > ONSET_RATIO * mean
            && flux[k] > ONSET_OVER_MEDIAN * flux_p50
            && flux[k] / flux_p98 >= ONSET_MIN_NORM
        {
            onsets.push(Onset {
                ms: (k * STFT_HOP * 1000 / sample_rate as usize) as u64,
                norm: clamp01(flux[k] / flux_p98),
            });
        }
    }
    // Enforce minimum gap between flashes.
    let mut spaced: Vec<Onset> = Vec::new();
    for o in onsets {
        if spaced.last().is_none_or(|last| o.ms.saturating_sub(last.ms) >= MIN_ONSET_GAP_MS as u64) {
            spaced.push(o);
        }
    }
    spaced
}


#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 8000;

    fn sine(freq: f32, seconds: f32, amp: f32) -> Vec<f32> {
        let n = (SR as f32 * seconds) as usize;
        (0..n)
            .map(|i| (amp * (std::f32::consts::TAU * freq * i as f32 / SR as f32).sin()))
            .collect()
    }

    fn square(freq: f32, seconds: f32, amp: f32) -> Vec<f32> {
        let n = (SR as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                if ((i as f32 * freq / SR as f32) % 1.0) < 0.5 {
                    amp
                } else {
                    -amp
                }
            })
            .collect()
    }

    fn base_steps(steps: &[LightStep]) -> Vec<&LightStep> {
        steps.iter().filter(|s| s.transition_ms == BASE_TRANSITION_MS).collect()
    }

    #[test]
    fn silence_gives_dim_floor() {
        let steps = analyze(&vec![0.0f32; SR as usize], SR);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].bri, 5);
        assert_eq!(steps[0].start_ms, 0);
    }

    #[test]
    fn mid_sine_is_greenish() {
        let signal = sine(440.0, 1.0, 0.5);
        let steps = analyze(&signal, SR);
        assert_eq!(steps.len(), 1, "steady tone should not flash");
        let base = base_steps(&steps)[0];
        assert!((60..=140).contains(&base.hue), "hue {} not in [60,140]", base.hue);
    }

    #[test]
    fn bass_tone_is_red() {
        let signal = sine(80.0, 1.0, 0.5);
        let steps = analyze(&signal, SR);
        let base = base_steps(&steps)[0];
        assert!(base.hue <= 45, "hue {} should be red", base.hue);
    }

    #[test]
    fn high_tone_is_azure() {
        let signal = sine(3000.0, 1.0, 0.5);
        let steps = analyze(&signal, SR);
        let base = base_steps(&steps)[0];
        assert!((180..=240).contains(&base.hue), "hue {} should be azure", base.hue);
    }

    #[test]
    fn bass_square_is_vivid() {
        let signal = square(110.0, 1.0, 0.9);
        let steps = analyze(&signal, SR);
        let base = base_steps(&steps)[0];
        assert!(base.sat >= 80, "sat {} < 80", base.sat);
    }

    #[test]
    fn louder_segment_is_brighter() {
        let mut signal = sine(440.0, 1.0, 0.1);
        signal.extend(sine(440.0, 1.0, 0.4)); // 4× amplitude → 4× RMS
        let steps = analyze(&signal, SR);
        let base = base_steps(&steps);
        assert_eq!(base.len(), 2);
        assert!(
            base[1].bri > base[0].bri,
            "bri[1] {} not > bri[0] {}",
            base[1].bri,
            base[0].bri
        );
    }

    #[test]
    fn trailing_tiny_segment_is_dropped() {
        // 2.0 s of audio plus a 20 ms tail (< 50 ms) → exactly 2 base steps.
        let mut signal = sine(440.0, 2.0, 0.3);
        signal.extend(vec![0.1f32; SR as usize / 50]);
        let steps = analyze(&signal, SR);
        let base = base_steps(&steps);
        assert_eq!(base.len(), 2);
        assert_eq!(base[1].start_ms, 1000);
    }

    #[test]
    fn hue_smoothing_moves_toward_target() {
        // 3000 Hz tone for 1 s then 120 Hz for 1 s: hue[0] near blue,
        // hue[1] moved toward red but must not overshoot to full red.
        let mut signal = sine(3000.0, 1.0, 0.5);
        signal.extend(sine(120.0, 1.0, 0.5));
        let steps = analyze(&signal, SR);
        let base = base_steps(&steps);
        assert_eq!(base.len(), 2, "steady tones should not flash: {steps:?}");
        assert!(base[0].hue > 200, "hue[0] {} should be near blue", base[0].hue);
        let dist = |h: u16| (h as i32 - 360).abs().min(h as i32);
        assert!(
            dist(base[1].hue) < dist(base[0].hue),
            "hue[1] {} not closer to red than hue[0] {}",
            base[1].hue,
            base[0].hue
        );
        assert!(base[1].hue > 50, "hue[1] {} should not have fully reached red", base[1].hue);
    }

    #[test]
    fn hue_smoothing_math() {
        assert_eq!(wrap_delta(10.0), 10.0);
        assert_eq!(wrap_delta(190.0), -170.0);
        assert_eq!(wrap_delta(-190.0), 170.0);
        // 60% of the wrapped distance, shortest way around.
        assert_eq!(smooth_hue(240.0, 0.0), 312.0);
        assert_eq!(smooth_hue(0.0, 360.0), 0.0);
        assert_eq!(smooth_hue(100.0, 200.0), 160.0);
        assert_eq!(smooth_hue(300.0, 20.0), 348.0); // wraps upward through 360
    }

    /// Impulse clicks must produce visible flashes; a steady tone must not.
    #[test]
    fn clicks_flash_but_steady_tone_does_not() {
        let mut clicks = vec![0.0f32; SR as usize * 2];
        for i in (0..clicks.len()).step_by(SR as usize / 4) {
            for j in 0..64.min(clicks.len() - i) {
                clicks[i + j] = 0.9 * (1.0 - j as f32 / 64.0);
            }
        }
        let steps = analyze(&clicks, SR);
        let flashes: Vec<&LightStep> =
            steps.iter().filter(|s| s.transition_ms == FLASH_TRANSITION_MS).collect();
        assert!(flashes.len() >= 3, "expected ≥3 flashes, got {}: {steps:?}", flashes.len());
        for f in &flashes {
            assert!(f.bri >= 30, "flash bri {} too dim", f.bri);
            assert!(f.transition_ms < 100, "flash must be a fast attack");
        }

        let steps = analyze(&sine(440.0, 2.0, 0.5), SR);
        assert!(
            steps.iter().all(|s| s.transition_ms == BASE_TRANSITION_MS),
            "steady tone must not flash: {steps:?}"
        );
    }

    /// Command-rate guard: even a click every 100 ms must stay under the cap.
    #[test]
    fn command_rate_is_capped() {
        let mut clicks = vec![0.0f32; SR as usize * 3];
        for i in (0..clicks.len()).step_by(SR as usize / 10) {
            for j in 0..32.min(clicks.len() - i) {
                clicks[i + j] = 0.9;
            }
        }
        let steps = analyze(&clicks, SR);
        for sec in 0..3u64 {
            let in_sec = steps.iter().filter(|s| {
                s.start_ms >= sec * 1000 && s.start_ms < (sec + 1) * 1000
            }).count();
            assert!(
                in_sec <= 1 + 2 * MAX_ONSETS_PER_SEC,
                "second {sec} emitted {in_sec} commands"
            );
        }
    }
}
