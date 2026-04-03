#!/usr/bin/env python3
"""Focused tests for streamlit_chat_app helpers."""

from __future__ import annotations

import sys
import types
import unittest

from llm_service import ContextItem

streamlit_stub = types.ModuleType("streamlit")


def _cache_data(*_args: object, **_kwargs: object):
    def decorator(func):
        return func

    return decorator


streamlit_stub.cache_data = _cache_data
sys.modules.setdefault("streamlit", streamlit_stub)

from streamlit_chat_app import _order_contexts_cited_first


def _contexts() -> list[ContextItem]:
    return [
        ContextItem(
            context_id="ctx_1",
            source_type="generated",
            title="First",
            location="generated/symbols.jsonl:1",
            text="First context",
            score=0.9,
        ),
        ContextItem(
            context_id="ctx_2",
            source_type="generated",
            title="Second",
            location="generated/symbols.jsonl:2",
            text="Second context",
            score=0.8,
        ),
        ContextItem(
            context_id="ctx_3",
            source_type="generated",
            title="Third",
            location="generated/symbols.jsonl:3",
            text="Third context",
            score=0.7,
        ),
        ContextItem(
            context_id="ctx_4",
            source_type="generated",
            title="Fourth",
            location="generated/symbols.jsonl:4",
            text="Fourth context",
            score=0.6,
        ),
    ]


class TestOrderContextsCitedFirst(unittest.TestCase):
    def test_moves_cited_contexts_to_front_in_original_retrieval_order(self) -> None:
        ordered = _order_contexts_cited_first(_contexts(), ["ctx_3", "ctx_1"])
        self.assertEqual(
            [item.context_id for item in ordered],
            ["ctx_1", "ctx_3", "ctx_2", "ctx_4"],
        )

    def test_preserves_original_order_when_no_citations_are_present(self) -> None:
        ordered = _order_contexts_cited_first(_contexts(), [])
        self.assertEqual(
            [item.context_id for item in ordered],
            ["ctx_1", "ctx_2", "ctx_3", "ctx_4"],
        )

    def test_ignores_missing_citation_ids_safely(self) -> None:
        ordered = _order_contexts_cited_first(_contexts(), ["ctx_missing", "ctx_2"])
        self.assertEqual(
            [item.context_id for item in ordered],
            ["ctx_2", "ctx_1", "ctx_3", "ctx_4"],
        )


if __name__ == "__main__":
    unittest.main()
