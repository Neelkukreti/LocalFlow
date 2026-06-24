# LocalFlow

**Fully local, offline voice dictation for Windows — a self-hosted take on Wispr Flow.**

Hold a key, speak, release. Your speech is transcribed on-device with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), optionally polished
by a local [Ollama](https://ollama.com) model, and pasted wherever your cursor
is. **No audio or text ever leaves your machine.** No account, no subscription,
no cloud.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/platform-Windows-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

```
HOLD Right-Ctrl ─▶ record mic ─▶ faster-whisper ─▶ (optional) LLM cleanup ─▶ paste at cursor
```

## Features

- 🎙️ **Hold-to-talk dictation** anywhere — works in any app, pastes at your cursor
- 🧠 **AI cleanup** (optional) — a local LLM fixes punctuation/casing and removes "um"s
- ⚡ **GPU-accelerated** — `faster-whisper` on CUDA (auto-falls back to CPU)
- 🔒 **100% offline** — no account, no cloud, nothing leaves your machine
- 🖥️ **System tray app** — runs in the background; icon turns green while listening, amber while thinking
- 🔊 **Audio cues** — subtle start/stop/done tones
- 📖 **Personal dictionary** — bias transcription toward your names & jargon
- ⌨️ **Voice commands** — say *"new line"* / *"new paragraph"* for real breaks
- 📝 **History** — every dictation logged to `history.jsonl`

### vs. Wispr Flow

| | Wispr Flow | LocalFlow |
|---|:---:|:---:|
| Runs fully offline | ❌ (cloud) | ✅ |
| Free / no subscription | ❌ | ✅ |
| Hold-to-talk anywhere | ✅ | ✅ |
| AI formatting | ✅ | ✅ (local LLM) |
| Personal dictionary | ✅ | ✅ |
| Voice commands | ✅ | ✅ (basic) |
| Tray app + feedback | ✅ | ✅ |
| Your data stays private | ⚠️ | ✅ |

---

## Download (easiest)

Grab the latest `.exe` from the [**Releases**](https://github.com/Neelkukreti/LocalFlow/releases) page:

| File | For | Notes |
|---|---|---|
| **`LocalFlow-cpu.exe`** | Any Windows PC | Works everywhere, no GPU needed. Uses a smaller Whisper model. |
| **`LocalFlow-gpu.exe`** | NVIDIA GPU (CUDA) | Fast, uses `large-v3`. Larger download (bundles CUDA libs). |

Double-click to run. On first launch it writes a `config.toml` next to the exe
and downloads the Whisper model (one time). Then **hold Right-Ctrl, speak, and
release** — the text appears at your cursor.

> **Windows SmartScreen** may warn about an unsigned exe — click *More info →
> Run anyway*. The source is right here if you'd rather build it yourself.

> **If keypresses aren't detected**, right-click the exe → *Run as
> administrator*. Global keyboard hooks sometimes need it.

---

## Optional: AI cleanup

LocalFlow works great as-is (raw transcription). For Wispr-Flow-style polish
(punctuation, removing "um"s, fixing casing) it can pipe the transcript through
a **local** LLM via Ollama — and it auto-detects this, so it's zero-config:

1. Install [Ollama](https://ollama.com).
2. Pull a model: `ollama pull qwen2.5:14b` (or set a smaller one in `config.toml`).

That's it. Next launch, LocalFlow sees Ollama running and enables cleanup
automatically. No Ollama? It silently stays in raw-transcription mode.

---

## Run from source

```powershell
git clone https://github.com/Neelkukreti/LocalFlow.git
cd LocalFlow
python -m venv .venv

# CPU:
.venv\Scripts\python -m pip install -r requirements.txt
# or GPU (NVIDIA, CUDA 12):
.venv\Scripts\python -m pip install -r requirements-gpu.txt

.venv\Scripts\python localflow.py
```

## Build your own exe

```powershell
# install CPU or GPU requirements first (determines the build type), then:
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python -m PyInstaller --noconfirm --clean localflow.spec
# -> dist\LocalFlow.exe
```

Or just push a tag — the [GitHub Actions workflow](.github/workflows/release.yml)
builds both CPU and GPU exes and attaches them to the release.

---

## Configuration

Everything lives in `config.toml` (created on first run). Sensible `auto`
defaults mean you usually don't need to touch it.

| Setting | What it does |
|---|---|
| `hotkey.key` | Push-to-talk key (`right ctrl`, `f9`, `ctrl+space`, …) |
| `whisper.model` | `auto`, or `tiny`/`base`/`small`/`medium`/`large-v3`/`distil-large-v3` |
| `whisper.device` | `auto` / `cuda` / `cpu` |
| `whisper.language` | `en` for speed, or `auto` to detect |
| `cleanup.enabled` | `auto` (on if Ollama present) / `true` / `false` |
| `cleanup.model` | Any pulled Ollama model |
| `dictionary.terms` | List of names/jargon to bias transcription toward |
| `commands.enabled` | Interpret spoken "new line" / "new paragraph" |
| `feedback.tray` | System tray icon (else console mode) |
| `feedback.sounds` | Start/stop/done audio cues |
| `history.enabled` | Log dictations to `history.jsonl` |
| `output.method` | `paste` (fast) or `type` (char-by-char) |

## How it works

1. **Record** — `sounddevice` captures 16 kHz mono audio while the hotkey is held.
2. **Transcribe** — `faster-whisper` turns audio into text, with built-in voice
   activity detection to trim silence. Runs on CUDA if available, else CPU.
3. **Clean** *(optional)* — the raw transcript goes to a local Ollama model with
   a tight prompt: fix punctuation/casing, drop fillers, change nothing else.
4. **Inject** — the result is pasted at your cursor (your clipboard is restored).

## Troubleshooting

| Problem | Fix |
|---|---|
| No text appears | Run as administrator (keyboard hook). |
| Cleanup never turns on | Make sure `ollama serve` is running and the model is pulled. |
| GPU exe falls back to CPU | Update your NVIDIA driver; LocalFlow auto-falls back if CUDA can't init. |
| Transcription slow on CPU | Use a smaller model: set `whisper.model = "base"` or `"tiny"`. |

## License

MIT — see [LICENSE](LICENSE).

---

*Built with [Claude Code](https://claude.com/claude-code).*
