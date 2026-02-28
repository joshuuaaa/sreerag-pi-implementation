#!/usr/bin/env python3
"""
scripts/build_rag_index.py
──────────────────────────
Build phase for the Crisis Assistant RAG system.
Run this ONCE on your laptop (or any machine with internet access) to:

  1. Read all .txt files from  data/manuals/
  2. Chunk them into overlapping segments
  3. Generate embeddings with BAAI/bge-small-en-v1.5
  4. Build a FAISS flat-L2 index
  5. Save  data/index/faiss.index  +  data/index/documents.pkl

Then rsync / copy the data/index/ folder to the Raspberry Pi (one-time transfer,
~5 MB index + ~130 MB embedding model).

Usage:
    pip install faiss-cpu sentence-transformers
    python scripts/build_rag_index.py [--input data/manuals] [--output data/index]
"""

import argparse
import os
import pickle
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_rag_index")


# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[str]:
    """
    Split *text* into overlapping word-level chunks.

    Args:
        text:        Raw document text.
        chunk_size:  Target chunk size in words.
        overlap:     Number of words shared between adjacent chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def infer_tags(filename: str, content: str) -> List[str]:
    """
    Derive metadata tags from the filename and a content keyword scan.

    These tags are used by the runtime RAG engine when the orchestrator passes
    ``rag_tags`` from a decision-tree node to pre-filter candidates.
    """
    tags = []
    stem = Path(filename).stem.lower()

    # Filename-based tags
    tag_map = {
        "bleed":    "bleeding",
        "cpr":      "cpr",
        "burn":     "burns",
        "fractur":  "fracture",
        "shock":    "shock",
        "choking":  "choking",
        "wound":    "wound_care",
        "first":    "first_aid",
        "cardiac":  "cardiac",
        "hypotherm": "hypothermia",
        "frost":     "hypothermia",
        "heat":      "heat_illness",
        "dehydrat":  "dehydration",
        "smoke":     "smoke_inhalation",
        "inhal":     "smoke_inhalation",
        "seiz":      "seizure",
        "head":      "head_injury",
        "drown":     "drowning",
        "chest":     "chest_injury",
        "amput":     "amputation",
        "snake":     "snake_bite",
        "poison":    "poisoning",
        "postpart":  "postpartum_haemorrhage",
        "haemorrh":  "postpartum_haemorrhage",
        "spinal":    "spinal_injury",
        "airway":    "airway_blockage",
        "pelvi":     "pelvic_injury",
    }
    for key, tag in tag_map.items():
        if key in stem:
            tags.append(tag)

    # Content-based tags (lightweight keyword scan)
    content_lower = content.lower()
    content_tags = {
        "tourniquet":           "severe_bleeding",
        "direct pressure":      "pressure_application",
        "cpr":                  "cpr",
        "chest compression":    "cpr",
        "heimlich":             "choking",
        "airway":               "airway",
        "burn":                 "burns",
        "blister":              "burns",
        "fracture":             "fracture",
        "splint":               "fracture",
        "shock":                "shock",
        "unconscious":          "unconscious",
        "elevation":            "elevation",
        "hypothermia":          "hypothermia",
        "shivering":            "hypothermia",
        "frostbite":            "hypothermia",
        "heat stroke":          "heat_illness",
        "heatstroke":           "heat_illness",
        "heat exhaustion":      "heat_illness",
        "dehydration":          "dehydration",
        "oral rehydration":     "dehydration",
        "smoke":                "smoke_inhalation",
        "carbon monoxide":      "smoke_inhalation",
        "seizure":              "seizure",
        "convulsion":           "seizure",
        "head injury":          "head_injury",
        "skull":                "head_injury",
        "concussion":           "head_injury",
        "drowning":             "drowning",
        "drowned":              "drowning",
        "sucking chest":        "chest_injury",
        "pneumothorax":         "chest_injury",
        "rib fracture":         "chest_injury",
        "amputation":           "amputation",
        "amputated":            "amputation",
        "severed limb":         "amputation",
        "snake bite":           "snake_bite",
        "snakebite":            "snake_bite",
        "venom":                "snake_bite",
        "poisoning":            "poisoning",
        "ingested":             "poisoning",
        "carbon monoxide":      "smoke_inhalation",
        "postpartum":           "postpartum_haemorrhage",
        "uterine massage":      "postpartum_haemorrhage",
        "spinal injury":        "spinal_injury",
        "spine":                "spinal_injury",
        "jaw thrust":           "spinal_injury",
        "pelvic":               "pelvic_injury",
        "airway blockage":      "airway_blockage",
        "choking":              "airway_blockage",
        "electrical":           "electrical_injury",
    }
    for keyword, tag in content_tags.items():
        if keyword in content_lower and tag not in tags:
            tags.append(tag)

    return tags or ["general"]


# ── index builder ─────────────────────────────────────────────────────────────

def build_index(input_dir: str, output_dir: str, model_name: str, chunk_size: int, overlap: int):
    """
    Main build routine.

    Reads all ``*.txt`` files from *input_dir*, chunks them, embeds them,
    builds a FAISS flat-L2 index, and writes artefacts to *output_dir*.
    """
    try:
        import faiss
    except ImportError:
        logger.error("faiss-cpu is not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers is not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    # ── 1. Discover source files ──────────────────────────────────────────────
    txt_files = list(Path(input_dir).glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in '%s'", input_dir)
        sys.exit(1)

    logger.info("Found %d source files in '%s'", len(txt_files), input_dir)

    # ── 2. Chunk documents ────────────────────────────────────────────────────
    all_documents: List[Dict[str, Any]] = []

    for txt_path in sorted(txt_files):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("Skipping empty file: %s", txt_path.name)
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        tags   = infer_tags(txt_path.name, text)

        logger.info(
            "  %-40s → %3d chunks  tags=%s",
            txt_path.name, len(chunks), tags,
        )

        for i, chunk in enumerate(chunks):
            all_documents.append({
                "content":    chunk,
                "source":     txt_path.name,
                "chunk_id":   i,
                "tags":       tags,
                "embedding":  None,  # filled below
            })

    logger.info("Total chunks: %d", len(all_documents))

    # ── 3. Generate embeddings ────────────────────────────────────────────────
    logger.info("Loading embedding model: %s  (this downloads ~130 MB on first run)", model_name)
    encoder = SentenceTransformer(model_name)

    texts = [doc["content"] for doc in all_documents]
    logger.info("Encoding %d chunks…", len(texts))

    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    ).astype(np.float32)

    # Attach embeddings for the subset search path in the engine
    for doc, emb in zip(all_documents, embeddings):
        doc["embedding"] = emb

    # ── 4. Build FAISS index ──────────────────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    logger.info(
        "FAISS index built: %d vectors × %d dimensions",
        index.ntotal, dim,
    )

    # ── 5. Save artefacts ─────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    index_path = os.path.join(output_dir, "faiss.index")
    docs_path  = os.path.join(output_dir, "documents.pkl")

    faiss.write_index(index, index_path)
    logger.info("Saved FAISS index → %s (%.1f KB)", index_path, os.path.getsize(index_path) / 1024)

    with open(docs_path, "wb") as f:
        pickle.dump(all_documents, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved documents   → %s (%.1f KB)", docs_path, os.path.getsize(docs_path) / 1024)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RAG INDEX BUILD COMPLETE")
    print("="*60)
    print(f"  Source files : {len(txt_files)}")
    print(f"  Total chunks : {len(all_documents)}")
    print(f"  Embedding dim: {dim}")
    print(f"  Index file   : {index_path}")
    print(f"  Docs file    : {docs_path}")
    print("\n  Next step – copy to Pi:")
    print(f"  rsync -avz {output_dir}/ pi@<PI_IP>:/home/pi/crisis-assistant/data/index/")
    print("="*60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS RAG index for Crisis Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",  default="data/manuals",
        help="Directory containing .txt medical source documents",
    )
    parser.add_argument(
        "--output", default="data/index",
        help="Output directory for faiss.index and documents.pkl",
    )
    parser.add_argument(
        "--model",  default="BAAI/bge-small-en-v1.5",
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=300,
        help="Target chunk size in words",
    )
    parser.add_argument(
        "--overlap", type=int, default=50,
        help="Word overlap between adjacent chunks",
    )
    args = parser.parse_args()

    build_index(
        input_dir  = args.input,
        output_dir = args.output,
        model_name = args.model,
        chunk_size = args.chunk_size,
        overlap    = args.overlap,
    )


if __name__ == "__main__":
    main()
