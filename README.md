<div align="center">

# bulbbeat

**Music-synced TP-Link Kasa bulb lighting**

[![Crates.io](https://img.shields.io/crates/v/bulbbeat?logo=rust)](https://crates.io/crates/bulbbeat)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange?logo=rust)](https://www.rust-lang.org)

Plays an audio file on your default output device and drives every Kasa bulb
on your LAN in sync — bass drops go red, vocals go green, atmospheric synths
go violet, and beats land as short brightness flashes.

Powered by [`kasa-rs`](https://github.com/JustinWoodring/kasa-rs).

</div>

## Why it works

Kasa bulbs only accept light updates about once per second over WiFi, so
naive beat-sync looks like slow breathing. bulbbeat inverts the problem:

1. the whole track is **decoded and analyzed offline**,
2. every light command is scheduled to arrive **one RTT/2 ahead** of the
   audio it describes,
3. the bulb's own `transition_period` ramp morphs between commands.

Result: continuous color motion with beat transients punched through via
short 40 ms attacks.

## Installation

```sh
cargo install bulbbeat
```

or from source:

```sh
git clone https://github.com/JustinWoodring/bulbbeat
cd bulbbeat && cargo build --release
```

## Usage

```
bulbbeat discover [--timeout-secs <3>]
bulbbeat play <AUDIO> [--bulb <IP>]... [--email <E>] [--password <P>]
              [--dry-run] [--no-restore] [--lead-ms <N>]
bulbbeat --web [--port <8080>]
```

### Discover

```sh
$ bulbbeat discover
192.168.1.83    KL130(US)    Office Lamp     color=yes  dimmable=yes
192.168.1.84    KL130(US)    Bedroom Lamp    color=yes  dimmable=yes
```

### Play

```sh
bulbbeat play ~/Music/song.mp3
```

No `--bulb`? Every bulb that answers discovery is driven simultaneously.
Pin specific devices with `--bulb <ip>` (repeatable). The bulb's pre-play
state is restored on exit — Ctrl-C included.

| Flag | Effect |
|---|---|
| `--bulb <ip>` | Target a specific bulb (repeatable; skips discovery) |
| `--dry-run` | Print the light plan as TSV — no bulbs, no audio |
| `--no-restore` | Leave bulbs in the last light state at exit |
| `--lead-ms <n>` | Override the RTT-probed send lead (debug) |
| `--email` / `--password` | Kasa cloud credentials for KLAP devices (defaults and blank credentials are tried automatically) |

### Web UI

```sh
cd ~/Music && bulbbeat --web
```

Serves a dark, dependency-free single-page UI on port 8080 (all interfaces):
every audio file in the current directory, one click to play, a live
progress bar, and a stop button — restore runs on stop or natural end.
Open it from your phone, your laptop, anywhere on the LAN.

```
GET  /api/files   audio files in the working directory
GET  /api/status  playback state (elapsed, duration, underruns)
POST /api/play    {"file": "song.mp3"}
POST /api/stop    stop playback and restore bulbs
```

## How it works

1. **Decode** — the whole file is decoded to interleaved f32 (symphonia) and
   prefilled into the playback buffer before the stream starts.
2. **Analyze** — per 1 s segment: FFT (Hann window). Each band's power share
   (bass 20–150, low 150–400, mid 400–2000, high 2k–8k, air 8k–20k Hz) is
   compared to its song-level median; bands hotter than usual pull the hue
   around color anchors:

   | Band | Range | Anchor |
   |---|---|---|
   | Bass | 20–150 Hz | 🔴 red 0° |
   | Low | 150–400 Hz | 🟠 amber 60° |
   | Mid (vocals) | 400–2000 Hz | 🟢 green 130° |
   | High (synths) | 2k–8k Hz | 🔵 azure 210° |
   | Air (atmosphere) | 8k–20k Hz | 🟣 violet 270° |

   Sub-250 Hz energy share → saturation; loudness (5th/95th percentile RMS)
   → base brightness. Digital silence dims to 5%.
3. **Beats** — a continuous spectral-flux onset detector (STFT, ~12 ms hop)
   schedules short-attack **flash + decay** commands on transients — up to
   3 flashes/s, ~40 ms attack, ~380 ms decay.
4. **Schedule** — one task per bulb. Lead time = median of 5 sysinfo RTTs / 2
   + 25 ms. Each command is sent at `t0 + start_ms − lead`, where `t0` is
   the instant the first audio callback writes frames. Errors skip a step
   (a stale color is worse than a missed one); the transport reconnects
   lazily.
5. **Protocol** — legacy XOR TCP:9999 and KLAP HTTP/AES (v1+v2 handshakes)
   are auto-detected per bulb via [`kasa-rs`](https://crates.io/crates/kasa-rs).
   Light commands use `transition_light_state`.

## Requirements

- A Kasa color or dimmable bulb on the same LAN (KL130, LB130, KL110, …)
- An audio output device (mp3, flac, wav, ogg, m4a input formats)
- Rust 1.82+ to build

## License

MIT — see [LICENSE](LICENSE).
