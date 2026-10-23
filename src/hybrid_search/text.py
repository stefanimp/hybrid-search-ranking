"""Spanish text normalization and ranking-score transformations."""

from __future__ import annotations

import re
import unicodedata

import numpy as np

STOPWORDS = set(
    """
    de la el los las un una unos unas y o u que cual cuales cuanto cuantos cuanta cuantas
    como cuando donde quien quienes porque por para al del a en con sin sobre entre hasta
    desde durante es son era eran fue fueron ser estar esta estan estaba se su sus mi mis
    tu tus nuestro nuestra nuestros nuestras lo le les nos os hay existe existen tiene
    tienen tenia tuvieron hace hacen puede pueden podria podrian debe deben mas menos muy
    ya aun todavia tambien si no cada otro otra otros otras mismo misma principales
    principal importante importantes actual actualmente mundo dia dias ejemplo ejemplos
    tipo tipos forma formas manera cosa cosas
    """.split()
)
QUESTION_WORDS = set(
    "que cual cuales cuanto cuantos cuanta cuantas como cuando donde quien quienes porque".split()
)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn"
    )


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(str(text)).lower()).strip()


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    out = re.findall(r"[a-z0-9]+", norm_text(text))
    out = [t for t in out if len(t) >= 2]
    if keep_stop:
        return out
    return [t for t in out if t not in STOPWORDS]


def parse_doc_id(doc_id: str) -> tuple[str, int]:
    base, _, par = str(doc_id).partition("#")
    try:
        return base, int(par or 0)
    except ValueError:
        return base, 0


def norm01(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return arr
    span = float(arr.max() - arr.min())
    if span <= 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span


def reciprocal_rank(rank: int, k: int = 60) -> float:
    return 1.0 / (k + float(rank))
