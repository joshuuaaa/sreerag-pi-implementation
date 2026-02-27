# Crisis Assistant

Offline, voice-activated emergency guidance device for Raspberry Pi 5.

## Hardware

| Component | Detail |
|---|---|
| Platform | Raspberry Pi 5 (8 GB RAM, 64 GB SD) |
| Button | GPIO 17 – hold-to-talk, triple-press reset, 5 s long-hold shutdown |
| Display | I²C 16×2 LCD at 0x27 |
| Mic | USB microphone |
| Speaker | 3.5 mm audio jack |

## AI Pipeline

```
Button hold → Record (PyAudio)
→ Whisper.cpp (STT, ~1-2 s)
→ SituationAnalyzer (keyword NER)
→ DecisionEngine (YAML protocol trees)
→ RAGEngine (FAISS + bge-small-en-v1.5)
→ LLMEngine (Phi-3 Mini Q4 GGUF, ~3-8 s)
→ Piper TTS (~500 ms)
→ ALSA playback
```

## Quick Start

### 1 – Laptop: build the RAG index

```bash
pip install faiss-cpu sentence-transformers
python scripts/build_rag_index.py   # reads data/manuals/*.txt
# outputs: data/index/faiss.index + documents.pkl
rsync -avz data/index/ pi@<PI_IP>:~/crisis-assistant/data/index/
```

### 1b – Download embedding model once for offline RAG runtime

```bash
mkdir -p models/embeddings
python - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
model.save("models/embeddings/bge-small-en-v1.5")
print("saved to models/embeddings/bge-small-en-v1.5")
PY

# copy to Pi (one-time)
rsync -avz models/embeddings/ pi@<PI_IP>:~/crisis-assistant/models/embeddings/
```

### 2 – Pi: one-shot setup

```bash
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh   # installs deps, builds whisper.cpp, enables I2C
sudo reboot
```

### 3 – Pi: run

```bash
# Hardware mode (GPIO button required)
python main.py

# Demo / dev mode (text input, no hardware needed)
CRISIS_CONFIG=configs/dev.yaml python main.py

# Force text-only mode (skip GPIO/LCD/mic/speaker init)
CRISIS_CONFIG=configs/dev.yaml CRISIS_TEXT_ONLY=1 python main.py

# Fully offline run (no Hugging Face network calls)
HF_HUB_OFFLINE=1 CRISIS_CONFIG=configs/dev.yaml CRISIS_TEXT_ONLY=1 python main.py
```

### 4 – Run tests

```bash
python scripts/test_system.py
```

## Project Layout

```
main.py                      Entry point
configs/
  pi_production.yaml         Pi 5 absolute paths
  dev.yaml                   Laptop / CI relative paths
src/
  orchestrator.py            Turn-level pipeline coordinator
  analyzer/situation_analyzer.py  Keyword-based NER
  decision/engine.py         YAML decision-tree navigator
  rag/engine.py              FAISS vector search (keyword fallback)
  llm/engine.py              Phi-3 Mini chat (llama-cpp-python)
  audio/{stt,tts}.py         Whisper.cpp + Piper subprocess wrappers
  hardware/{button,lcd,audio}.py  GPIO / I2C / PyAudio
  session/manager.py         In-memory conversation state
  prompt/styles.py           System prompt templates
decision_trees/
  bleeding.yaml  cpr.yaml  burns.yaml  fracture.yaml  shock.yaml
data/
  manuals/       Source .txt documents for RAG index build
  index/         Built FAISS index (generated – not committed)
scripts/
  build_rag_index.py  Run on laptop to generate FAISS index
  setup_pi.sh         Pi 5 provisioning script
  test_system.py      Integration tests (no hardware required)
systemd/
  crisis-assistant.service   Auto-start systemd service template
models/
  llm/Phi-3-mini-4k-instruct-q4.gguf
  piper/en_US-lessac-medium.onnx
  whisper/ggml-tiny.en.bin
```

## Button Gestures

| Gesture | Action |
|---|---|
| Hold | Start recording |
| Release | Process and respond |
| Triple press | Reset conversation |
| Long hold | No power action |
