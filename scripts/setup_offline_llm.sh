#!/usr/bin/env bash
set -euo pipefail

MODEL_CHOICE="${1:-phi3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/llm"
mkdir -p "${MODEL_DIR}"

case "${MODEL_CHOICE}" in
  phi3|phi-3|phi3-mini)
    FILE_NAME="Phi-3-mini-4k-instruct-q4.gguf"
    URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    ;;
  tinyllama|tiny)
    FILE_NAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ;;
  *)
    echo "Unknown model: ${MODEL_CHOICE}"
    echo "Usage: $0 [phi3|tinyllama]"
    exit 1
    ;;
esac

TARGET_PATH="${MODEL_DIR}/${FILE_NAME}"

echo "Downloading ${FILE_NAME}..."
wget -c -O "${TARGET_PATH}" "${URL}"

echo "Model ready: ${TARGET_PATH}"
echo "Update configs/pi_production.yaml -> llm.model_path to this path if needed."
