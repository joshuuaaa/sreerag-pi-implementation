#!/usr/bin/env bash
# scripts/setup_pi.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-shot setup script for Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit).
# Run as the 'pi' user (NOT root).  Needs sudo for apt and systemd steps.
#
# Usage:
#   chmod +x scripts/setup_pi.sh
#   ./scripts/setup_pi.sh
#
# What it does:
#   1. System packages (ALSA, PortAudio, I2C tools, build essentials)
#   2. Python virtual environment in ~/crisis-assistant/.venv
#   3. Python dependencies from requirements.txt
#   4. Enable I2C interface for LCD
#   5. Install systemd service for auto-start
#   6. Whisper.cpp download reminder
#   7. Model presence check
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$APP_DIR/.venv"
SERVICE_SRC="$APP_DIR/systemd/crisis-assistant.service"
SERVICE_DST="/etc/systemd/system/crisis-assistant.service"
PI_USER="${SUDO_USER:-pi}"

echo "════════════════════════════════════════════════════════════"
echo "  Crisis Assistant – Raspberry Pi 5 Setup"
echo "  App dir: $APP_DIR"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "▶ [1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    portaudio19-dev libportaudio2 \
    libasound2-dev alsa-utils \
    i2c-tools libi2c-dev \
    libopenblas-dev libatlas-base-dev \
    cmake build-essential git \
    ffmpeg \
    --no-install-recommends

echo "✅ System packages installed"

# ── 2. Enable I2C ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [2/7] Enabling I2C interface..."
if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null && \
   ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
    if [ -f /boot/firmware/config.txt ]; then
        echo "dtparam=i2c_arm=on" | sudo tee -a /boot/firmware/config.txt > /dev/null
    else
        echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt > /dev/null
    fi
    echo "✅ I2C enabled (will take effect after reboot)"
else
    echo "✅ I2C already enabled"
fi

# Add user to i2c group
sudo usermod -a -G i2c "$PI_USER" 2>/dev/null || true

# ── 3. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "▶ [3/7] Creating Python virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q
echo "✅ Virtual environment ready"

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "▶ [4/7] Installing Python dependencies (this takes ~15 min on Pi 5)..."

# llama-cpp-python: compile with OpenBLAS for 2–3× CPU speedup
echo "  Building llama-cpp-python with OpenBLAS..."
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python==0.2.82 --no-binary llama-cpp-python -q

# Remaining requirements (excluding llama line already installed)
grep -v "^llama-cpp-python" "$APP_DIR/requirements.txt" | \
    grep -v "^#" | grep -v "^$" | \
    pip install -r /dev/stdin -q

echo "✅ Python dependencies installed"

# ── 5. Whisper.cpp ────────────────────────────────────────────────────────────
echo ""
echo "▶ [5/7] Whisper.cpp setup..."
WHISPER_DIR="$APP_DIR/models/whisper/whisper-cpp"
if [ ! -d "$WHISPER_DIR" ]; then
    echo "  Cloning whisper.cpp..."
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
    echo "  Building whisper.cpp main binary..."
    make -C "$WHISPER_DIR" main -j4
    echo "  Downloading tiny.en model (~75 MB)..."
    bash "$WHISPER_DIR/models/download-ggml-model.sh" tiny.en
    mkdir -p "$APP_DIR/models/whisper"
    cp "$WHISPER_DIR/models/ggml-tiny.en.bin" "$APP_DIR/models/whisper/" 2>/dev/null || true
    echo "✅ Whisper.cpp built and model downloaded"
else
    echo "✅ Whisper.cpp already present"
fi

# ── 6. Systemd service ────────────────────────────────────────────────────────
echo ""
echo "▶ [6/7] Installing systemd service..."
if [ -f "$SERVICE_SRC" ]; then
    # Patch user and paths in service file
    sed "s|__PI_USER__|$PI_USER|g; s|__APP_DIR__|$APP_DIR|g; s|__VENV_DIR__|$VENV_DIR|g" \
        "$SERVICE_SRC" | sudo tee "$SERVICE_DST" > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable crisis-assistant.service
    echo "✅ Service installed and enabled"
    echo "   Start now: sudo systemctl start crisis-assistant"
else
    echo "⚠️  No systemd/crisis-assistant.service found – skipping"
fi

# ── 7. Model presence check ───────────────────────────────────────────────────
echo ""
echo "▶ [7/7] Model file check..."
LLM_MODEL="$APP_DIR/models/llm/Phi-3-mini-4k-instruct-q4.gguf"
PIPER_MODEL="$APP_DIR/models/piper/en_US-lessac-medium.onnx"
RAG_INDEX="$APP_DIR/data/index/faiss.index"

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  ✅ $1 ($SIZE)"
    else
        echo "  ❌ MISSING: $1"
        echo "     → $2"
    fi
}

check_file "$LLM_MODEL"     "Download Phi-3 Mini GGUF from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
check_file "$PIPER_MODEL"   "Download lessac-medium from https://github.com/rhasspy/piper/releases"
check_file "$RAG_INDEX"     "Run: python scripts/build_rag_index.py (on laptop), then rsync data/index/ to Pi"

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Activate venv  :  source $VENV_DIR/bin/activate"
echo "  Run demo mode  :  CRISIS_CONFIG=configs/dev.yaml python main.py"
echo "  Run production :  python main.py"
echo "  Service control:  sudo systemctl start|stop|status crisis-assistant"
echo ""
echo "  ⚠️  Reboot required to activate I2C and group changes."
echo "════════════════════════════════════════════════════════════"
