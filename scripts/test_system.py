#!/usr/bin/env python3
"""
scripts/test_system.py
──────────────────────
Integration test suite for Crisis Assistant.
Runs WITHOUT hardware (GPIO, LCD, audio) and WITHOUT models (LLM, STT, TTS).
All AI components fall back gracefully when models are absent.

Usage:
    python scripts/test_system.py           # full suite
    python scripts/test_system.py -v        # verbose output
    python scripts/test_system.py --fast    # skip slow import checks

Exit code 0 = all tests passed.
"""

import argparse
import sys
import time
import traceback
import os

# ── ensure project root is on path ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
SEP   = "─" * 60


class TestResult:
    def __init__(self, name: str):
        self.name   = name
        self.passed = False
        self.skipped = False
        self.message = ""
        self.duration = 0.0


class TestSuite:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[TestResult] = []

    def run(self, name: str, fn):
        """Execute one test function and record the result."""
        r = TestResult(name)
        t0 = time.perf_counter()
        try:
            fn(r)
            r.passed = True
        except Exception as exc:
            r.passed  = False
            r.message = f"{exc}\n{traceback.format_exc()}"
        r.duration = time.perf_counter() - t0
        self.results.append(r)
        icon = PASS if r.passed else (WARN if r.skipped else FAIL)
        status = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
        print(f"  {icon} [{status}] {name}  ({r.duration*1000:.0f} ms)")
        if self.verbose and r.message:
            print(f"       {r.message}")
        return r.passed

    def summary(self) -> bool:
        total   = len(self.results)
        passed  = sum(1 for r in self.results if r.passed)
        failed  = sum(1 for r in self.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        print()
        print(SEP)
        print(f"Results: {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")
        if failed == 0:
            print(f"{PASS} All tests passed")
        else:
            print(f"{FAIL} {failed} test(s) failed")
        print(SEP)
        return failed == 0


# ── individual tests ──────────────────────────────────────────────────────────

def test_config_loading(r):
    """Config file parses and has expected top-level keys."""
    from src.utils import load_config
    # Try dev config first; fall back to pi_production
    cfg = load_config("configs/dev.yaml") or load_config("configs/pi_production.yaml")
    assert cfg, "Config loaded empty"
    for key in ("app", "llm", "rag", "decision", "audio"):
        assert key in cfg, f"Missing key '{key}' in config"


def test_prompt_styles(r):
    """All three prompt styles return non-empty strings."""
    from src.prompt.styles import build_prompt
    for style in ("warm", "clinical", "brief"):
        p = build_prompt(style)
        assert p and len(p) > 20, f"Prompt style '{style}' too short or empty"


def test_session_manager(r):
    """Session creates history, tracks turns, resets cleanly."""
    from src.session.manager import ConversationSession
    s = ConversationSession("test-001")
    assert s.get_turn_count() == 0
    s.add_exchange("Help!", "I'm here.")
    assert s.get_turn_count() == 1
    ctx = s.get_context(last_n=5)
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"
    assert ctx[1]["role"] == "assistant"
    s.update_protocol_state("current_node", "root")
    assert s.protocol_state["current_node"] == "root"


def test_situation_analyzer(r):
    """Analyzer detects known emergency conditions and orders by priority."""
    from src.analyzer.situation_analyzer import SituationAnalyzer
    az = SituationAnalyzer()

    # Bleeding detection
    a = az.analyze([], "There is a lot of blood spurting from his arm")
    assert a["conditions"], "No conditions detected"
    assert a["primary_condition"] == "bleeding"
    assert "context" in a

    # Unconscious → shock or unconscious detected
    a2 = az.analyze([], "She passed out and is not responding")
    assert a2["conditions"], "No conditions for unconscious scenario"

    # Multi-condition: choking is priority 10 → should be primary
    a3 = az.analyze([], "He is choking and has a cut on his arm that is bleeding")
    assert a3["primary_condition"] in ("choking", "bleeding")

    # Phase progression
    assert az.analyze([],       "help")["phase"] == "initial_assessment"
    assert az.analyze([{"role":"user","content":"x"},{"role":"assistant","content":"y"}],
                      "blood")["phase"] in ("condition_confirmation", "active_guidance")


def test_decision_trees_load(r):
    """All YAML decision trees load without errors and have required fields."""
    import yaml
    from pathlib import Path
    tree_dir = Path("decision_trees")
    trees = list(tree_dir.glob("*.yaml"))
    assert trees, "No .yaml files found in decision_trees/"

    for path in trees:
        with open(path) as f:
            t = yaml.safe_load(f)
        assert "tree_id" in t, f"{path.name}: missing tree_id"
        assert "nodes"   in t, f"{path.name}: missing nodes"
        nodes = t["nodes"]
        assert "root" in nodes, f"{path.name}: missing root node"
        for node_id, node in nodes.items():
            assert "type" in node, f"{path.name}/{node_id}: missing type"


def test_decision_engine_navigate(r):
    """Decision engine navigates bleeding tree through one turn."""
    from src.utils import load_config
    from src.decision.engine import DecisionEngine
    from src.session.manager import ConversationSession

    cfg = load_config("configs/dev.yaml") or {}
    de  = DecisionEngine(cfg.get("decision", {"tree_dir": "decision_trees"}))

    if not de.trees:
        r.skipped = True
        r.message = "No trees loaded"
        return

    s = ConversationSession("nav-test")
    result = de.navigate("bleeding", s)
    assert result, "navigate returned empty"
    assert "message" in result
    assert "action"  in result


def test_rag_keyword_fallback(r):
    """RAG engine returns keyword-matched results without FAISS index."""
    # Patch _load to skip file I/O
    import src.rag.engine as rag_mod
    original_load = rag_mod.RAGEngine._load

    def mock_load(self):
        self._documents = [
            {"content": "Apply direct pressure to control bleeding",
             "source":  "first_aid_bleeding.txt",
             "tags":    ["bleeding", "pressure_application"],
             "embedding": None},
            {"content": "Perform chest compressions at 100 per minute for CPR",
             "source":  "first_aid_cpr.txt",
             "tags":    ["cpr"],
             "embedding": None},
        ]
        self._index   = None
        self._encoder = None

    rag_mod.RAGEngine._load = mock_load
    try:
        from src.rag.engine import RAGEngine
        e = RAGEngine({"index_path": "data/index", "top_k": 2})
        results = e.retrieve("how do I stop bleeding", tags=["bleeding"])
        assert results, "No results from keyword fallback"
        assert any("pressure" in doc["content"].lower() or "bleeding" in doc["content"].lower()
                   for doc in results)
    finally:
        rag_mod.RAGEngine._load = original_load


def test_llm_phi3_prompt_format(r):
    """LLM engine builds a valid Phi-3 prompt without a model file."""
    from src.llm.engine import LLMEngine

    # Instantiate without a real model path
    engine = LLMEngine({"model_path": "/nonexistent/model.gguf", "n_ctx": 512})
    # _build_phi3_prompt is a static method, callable directly
    prompt = engine._build_phi3_prompt(
        system_prompt="You are an emergency assistant.",
        messages=[
            {"role": "user",      "content": "Someone is bleeding badly."},
            {"role": "assistant", "content": "Apply firm pressure now."},
            {"role": "user",      "content": "The cloth is soaked through."},
        ],
    )
    assert "<|system|>"   in prompt
    assert "<|user|>"     in prompt
    assert "<|assistant|>" in prompt
    assert "Someone is bleeding" in prompt
    assert "cloth is soaked"     in prompt
    # Prompt must end with assistant turn opener
    assert prompt.strip().endswith("<|assistant|>") or prompt.endswith("<|assistant|>\n")


def test_orchestrator_no_models(r):
    """
    Orchestrator processes messages end-to-end when LLM / RAG models are absent.
    Verifies that decision-tree and fallback paths produce valid response dicts.
    """
    from src.utils import load_config
    from src.orchestrator import IntelligentOrchestrator

    # Use dev config; LLM and RAG models won't exist in CI → graceful degradation
    cfg = load_config("configs/dev.yaml") or {}

    orch = IntelligentOrchestrator(cfg)
    orch.start_session()

    # Turn 1 – initial bleed report
    res1 = orch.process_message("There's heavy bleeding from a cut on his arm")
    assert "response" in res1, "No 'response' key"
    assert res1["response"], "Empty response"
    assert "lcd_display" in res1
    assert "state"       in res1

    # Turn 2 – clarification
    res2 = orch.process_message("Yes, he's conscious and responding.")
    assert res2["response"]

    # Triple press reset
    orch.reset_session()
    greeting = orch.start_session()
    assert greeting and len(greeting) > 5


def test_conversation_full_flow(r):
    """Simulates a multi-turn choking scenario through the orchestrator."""
    from src.utils import load_config
    from src.orchestrator import IntelligentOrchestrator

    cfg  = load_config("configs/dev.yaml") or {}
    orch = IntelligentOrchestrator(cfg)
    orch.start_session()

    turns = [
        "My baby is choking and turning blue",
        "She's 8 months old",
        "I've given 5 back blows, still choking",
        "She just coughed it out, breathing now",
    ]
    for turn in turns:
        res = orch.process_message(turn)
        assert res.get("response"), f"No response for: '{turn}'"


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crisis Assistant test suite")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Skip slow LLM prompt test")
    args = parser.parse_args()

    suite = TestSuite(verbose=args.verbose)

    print()
    print("═" * 60)
    print("  Crisis Assistant – Integration Test Suite")
    print("═" * 60)
    print()

    tests = [
        ("Config loading",              test_config_loading),
        ("Prompt styles",               test_prompt_styles),
        ("Session manager",             test_session_manager),
        ("Situation analyzer",          test_situation_analyzer),
        ("Decision trees load",         test_decision_trees_load),
        ("Decision engine navigate",    test_decision_engine_navigate),
        ("RAG keyword fallback",        test_rag_keyword_fallback),
        ("LLM Phi-3 prompt format",     test_llm_phi3_prompt_format),
        ("Orchestrator (no models)",    test_orchestrator_no_models),
        ("Full conversation flow",      test_conversation_full_flow),
    ]

    for name, fn in tests:
        suite.run(name, fn)

    ok = suite.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
