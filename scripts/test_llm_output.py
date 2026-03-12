#!/usr/bin/env python3
"""
scripts/test_llm_output.py
──────────────────────────
Live LLM output test – no mic, no speaker, no GPIO.
Drives the full orchestrator with typed text and prints each response
so you can visually verify there is no roleplay / === leakage.

Usage:
    python scripts/test_llm_output.py              # auto scenario
    python scripts/test_llm_output.py --chat       # interactive REPL
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

SEP  = "─" * 60
DSEP = "═" * 60

SCENARIOS = [
    # (label, turns)
    (
        "Burn scenario",
        [
            "I have a bad burn on my arm from a hot pan",
            "Yes I can feel it, it hurts a lot",
            "I'm running cold water on it now",
            "The skin looks red and blistering",
        ],
    ),
    (
        "Bleeding scenario",
        [
            "There is heavy bleeding from a deep cut on his leg",
            "Yes he is conscious and talking to me",
            "I'm pressing a cloth on it",
            "The cloth is soaked through with blood",
        ],
    ),
]


def check_response(text: str) -> list[str]:
    """Return a list of warning strings for any roleplay artefacts found."""
    warnings = []
    lower = text.lower()
    if "===" in text:
        warnings.append("Contains === separator")
    if "response:" in lower:
        warnings.append("Contains 'Response:'")
    if "reply:" in lower:
        warnings.append("Contains 'Reply:'")
    for label in ("user:", "human:", "person:", "patient:", "support:"):
        if label in lower:
            warnings.append(f"Contains '{label}'")
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    if len(sentences) > 5:
        warnings.append(f"Suspiciously long ({len(sentences)} sentences) – possible multi-turn drift")
    return warnings


def run_scenario(orch, label: str, turns: list[str]):
    print()
    print(DSEP)
    print(f"  SCENARIO: {label}")
    print(DSEP)

    orch.reset_session()
    orch.start_session()

    all_ok = True
    for i, msg in enumerate(turns, 1):
        print(f"\n  Turn {i}")
        print(f"  {SEP}")
        print(f"  YOU : {msg}")

        result = orch.process_message(msg)
        reply  = result.get("response", "")

        print(f"  ASST: {reply}")
        print(f"  state={result.get('state','')}  lcd={result.get('lcd_display','')}")

        warnings = check_response(reply)
        if warnings:
            all_ok = False
            for w in warnings:
                print(f"  ⚠️  WARNING: {w}")
        else:
            print("  ✅ No roleplay artefacts detected")

    return all_ok


def interactive_chat(orch):
    print()
    print(DSEP)
    print("  INTERACTIVE MODE  (type 'quit' to exit, 'reset' for new session)")
    print(DSEP)

    orch.start_session()

    while True:
        try:
            msg = input("\n  YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not msg:
            continue
        if msg.lower() == "quit":
            break
        if msg.lower() == "reset":
            orch.reset_session()
            orch.start_session()
            print("  [Session reset]")
            continue

        result  = orch.process_message(msg)
        reply   = result.get("response", "")
        print(f"\n  ASST: {reply}")
        print(f"  (state={result.get('state','')}  lcd={result.get('lcd_display','')})")

        warnings = check_response(reply)
        for w in warnings:
            print(f"  ⚠️  {w}")


def main():
    parser = argparse.ArgumentParser(description="Live LLM output test – no audio hardware")
    parser.add_argument("--chat", action="store_true", help="Interactive REPL instead of auto scenarios")
    parser.add_argument("--config", default="configs/dev.yaml", help="Config file to use")
    args = parser.parse_args()

    # ── load config and build orchestrator ────────────────────────────────────
    from src.utils import load_config
    from src.orchestrator import IntelligentOrchestrator

    print()
    print(DSEP)
    print("  Crisis Assistant – LLM Output Test (no audio)")
    print(DSEP)

    cfg = load_config(args.config)
    if not cfg:
        print(f"  ERROR: could not load config '{args.config}'")
        sys.exit(1)

    print(f"  Config  : {args.config}")
    print(f"  LLM     : {cfg.get('llm', {}).get('model_path', 'not set')}")
    print(f"  max_tok : {cfg.get('llm', {}).get('max_tokens', '?')}")
    print()

    orch = IntelligentOrchestrator(cfg)

    if orch.llm.llm is None:
        print("  ⚠️  LLM model not loaded – responses will be fallback text only.")
        print("     (model file missing or llama-cpp-python not installed)")
        print()

    # ── run ───────────────────────────────────────────────────────────────────
    if args.chat:
        interactive_chat(orch)
    else:
        results = []
        for label, turns in SCENARIOS:
            ok = run_scenario(orch, label, turns)
            results.append((label, ok))

        print()
        print(DSEP)
        print("  SUMMARY")
        print(SEP)
        for label, ok in results:
            icon = "✅" if ok else "❌"
            print(f"  {icon}  {label}")
        print(DSEP)
        sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
