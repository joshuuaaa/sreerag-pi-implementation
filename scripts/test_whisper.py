from faster_whisper import WhisperModel
import os

model_path = os.path.expanduser("~/crisis-assistant/models/whisper")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    download_root=model_path
)

segments, _ = model.transcribe("data/test.wav")

for seg in segments:
    print(seg.text)
