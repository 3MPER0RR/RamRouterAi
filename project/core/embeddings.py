import numpy as np


MEMORY_WORDS = [
    "remember",
    "memory",
    "save",
    "note"
]

TOOL_WORDS = [
    "execute",
    "tool",
    "scan",
    "analyze",
    "parse"
]


def extract_features(text):

    tokens = text.lower().split()

    vec = np.array([

        len(text),

        len(tokens),

        int("?" in text),

        int("http" in text),

        int("```" in text),

        int(any(
            w in text.lower()
            for w in MEMORY_WORDS
        )),

        int(any(
            w in text.lower()
            for w in TOOL_WORDS
        ))

    ], dtype=float)

    vec /= (np.linalg.norm(vec) + 1e-8)

    return vec
