#!/usr/bin/env python3
"""Tests for retrieval-mode fallback behavior."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from ore_rag_assistant import Chunk, build_lexical_payload, select_retrieval


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=0,
            text="Ore algebra differential operator over QQ(x)",
            source="generated/symbols.jsonl",
            source_type="generated",
            qualname="OreAlgebra",
        ),
        Chunk(
            chunk_id=1,
            text="Unrelated content",
            source="generated/symbols.jsonl",
            source_type="generated",
            qualname="Other",
        ),
    ]


def _payload_with_dense(chunks: list[Chunk]) -> dict:
    return {
        "lexical": build_lexical_payload(chunks),
        "dense": {
            "model": "fake-model",
            "metric": "inner_product",
            "faiss_index_file": "fake.faiss",
            "embeddings_file": "fake.npy",
        },
    }


class TestSelectRetrievalFallback(unittest.TestCase):
    def test_auto_mode_falls_back_to_lexical_when_dense_runtime_is_missing(self) -> None:
        chunks = _chunks()
        payload = _payload_with_dense(chunks)
        with mock.patch(
            "ore_rag_assistant.hybrid_search",
            side_effect=RuntimeError(
                "Dense retrieval requires sentence-transformers + numpy. "
                "Install with: pip install sentence-transformers numpy"
            ),
        ):
            mode_used, results = select_retrieval(
                index_payload=payload,
                chunks=chunks,
                query="differential operator",
                k=1,
                mode="auto",
                index_path=Path("."),
                hybrid_alpha=0.7,
                source_priority="flat",
                symbols_ratio=0.75,
                max_pdf_extras=2,
            )
        self.assertEqual(mode_used, "lexical")
        self.assertEqual([r.chunk_id for r in results], [0])

    def test_hybrid_mode_falls_back_to_lexical_when_dense_runtime_is_missing(self) -> None:
        chunks = _chunks()
        payload = _payload_with_dense(chunks)
        with mock.patch(
            "ore_rag_assistant.hybrid_search",
            side_effect=RuntimeError(
                "Dense retrieval requires FAISS for vector-store lookup. "
                "Install with: pip install faiss-cpu"
            ),
        ):
            mode_used, results = select_retrieval(
                index_payload=payload,
                chunks=chunks,
                query="ore algebra",
                k=1,
                mode="hybrid",
                index_path=Path("."),
                hybrid_alpha=0.7,
                source_priority="flat",
                symbols_ratio=0.75,
                max_pdf_extras=2,
            )
        self.assertEqual(mode_used, "lexical")
        self.assertEqual([r.chunk_id for r in results], [0])

    def test_dense_mode_keeps_error_when_dense_runtime_is_missing(self) -> None:
        chunks = _chunks()
        payload = _payload_with_dense(chunks)
        with mock.patch(
            "ore_rag_assistant.dense_search",
            side_effect=RuntimeError(
                "Dense retrieval requires sentence-transformers + numpy. "
                "Install with: pip install sentence-transformers numpy"
            ),
        ):
            with self.assertRaises(RuntimeError):
                select_retrieval(
                    index_payload=payload,
                    chunks=chunks,
                    query="ore algebra",
                    k=1,
                    mode="dense",
                    index_path=Path("."),
                    hybrid_alpha=0.7,
                    source_priority="flat",
                    symbols_ratio=0.75,
                    max_pdf_extras=2,
                )


if __name__ == "__main__":
    unittest.main()
