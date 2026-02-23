import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.llm.engine import LLMEngine


def main() -> None:
    config = load_config("configs/pi_production.yaml")
    llm_cfg = config.get("llm", {})

    model_path = llm_cfg.get("model_path", "")
    print(f"Model path: {model_path}")
    print(f"Model exists: {Path(model_path).exists()}")

    engine = LLMEngine(llm_cfg)
    prompt = "You are a concise assistant. Reply with exactly: READY"
    response = engine.generate(prompt, max_tokens=8)

    print(json.dumps({"response": response}, indent=2))


if __name__ == "__main__":
    main()
