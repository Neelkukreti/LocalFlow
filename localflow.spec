# PyInstaller spec for LocalFlow.
#
# Builds a single-file LocalFlow.exe. The same spec produces either a CPU or a
# GPU build depending on which requirements are installed in the build env:
#   - requirements.txt      -> no NVIDIA libs present -> CPU exe (~small)
#   - requirements-gpu.txt  -> NVIDIA libs bundled    -> GPU exe (~large)
#
# Whisper model weights are NOT bundled; they download to the user cache on
# first run, keeping the exe smaller.

import importlib.util
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = [], [], []

# Heavy packages that need their data files / dynamic libs collected.
for pkg in ["faster_whisper", "ctranslate2", "av", "onnxruntime", "sounddevice", "tokenizers"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# NVIDIA CUDA runtime libs - only present in a GPU build environment.
for pkg in ["nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"]:
    if importlib.util.find_spec(pkg) is not None:
        binaries += collect_dynamic_libs(pkg)

block_cipher = None

a = Analysis(
    ["localflow.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="LocalFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
