"""Lexical coverage, question type and passage structure features."""

from __future__ import annotations

import math
import re

from .text import QUESTION_WORDS, norm_text, parse_doc_id, tokens


def content_features(query: str, doc_id: str, title: str, text: str) -> dict[str, float]:
    q_all = tokens(query, keep_stop=True)
    q_terms = [t for t in tokens(query) if t not in QUESTION_WORDS]
    title_terms = tokens(title)
    text_terms = tokens(text)
    title_set = set(title_terms)
    text_set = set(text_terms)
    base, paragraph = parse_doc_id(doc_id)
    q_base, _ = parse_doc_id(str(query))
    q_norm = norm_text(query)
    title_norm = norm_text(title)
    text_norm = norm_text(text)
    numbers_q = set(re.findall(r"\d+(?:[.,]\d+)?", q_norm))
    numbers_doc = set(re.findall(r"\d+(?:[.,]\d+)?", title_norm + " " + text_norm))
    rare_terms = sorted(q_terms, key=len, reverse=True)[:4]
    phrase = " ".join(q_terms[: min(5, len(q_terms))])

    def cov(target: set[str]) -> float:
        if not q_terms:
            return 0.0
        return sum(1 for t in q_terms if t in target) / len(q_terms)

    def rare_cov(target: set[str]) -> float:
        if not rare_terms:
            return 0.0
        return sum(1 for t in rare_terms if t in target) / len(rare_terms)

    first_positions = [text_norm.find(t) for t in q_terms if text_norm.find(t) >= 0]
    first_pos_score = 0.0 if not first_positions else 1.0 / (1.0 + min(first_positions) / 80.0)

    return {
        "q_len": float(len(q_terms)),
        "q_all_len": float(len(q_all)),
        "title_len_log": math.log1p(len(title_terms)),
        "text_len_log": math.log1p(len(text_terms)),
        "paragraph": float(paragraph),
        "paragraph_prior": 1.0 / (1.0 + paragraph / 8.0),
        "same_base_as_query_id": 1.0 if base == q_base else 0.0,
        "title_cov": cov(title_set),
        "text_cov": cov(text_set),
        "title_rare_cov": rare_cov(title_set),
        "text_rare_cov": rare_cov(text_set),
        "title_or_text_cov": cov(title_set | text_set),
        "query_phrase_in_text": 1.0 if phrase and phrase in text_norm else 0.0,
        "query_phrase_in_title": 1.0 if phrase and phrase in title_norm else 0.0,
        "full_query_in_text": 1.0 if q_norm and q_norm in text_norm else 0.0,
        "full_query_in_title": 1.0 if q_norm and q_norm in title_norm else 0.0,
        "number_overlap": float(len(numbers_q & numbers_doc)),
        "has_number_query": 1.0 if numbers_q else 0.0,
        "first_pos_score": first_pos_score,
        "is_definition_query": 1.0
        if any(t in q_all for t in ["definicion", "define", "significa", "concepto"])
        else 0.0,
        "is_count_query": 1.0
        if any(t in q_all for t in ["cuantos", "cuantas", "cuanto", "cuanta"])
        else 0.0,
        "is_where_query": 1.0 if "donde" in q_all else 0.0,
        "is_who_query": 1.0 if any(t in q_all for t in ["quien", "quienes"]) else 0.0,
        "is_when_query": 1.0 if "cuando" in q_all else 0.0,
    }
