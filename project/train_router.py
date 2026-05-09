import json
from pathlib import Path

import numpy as np

from core.embeddings import extract_features

from core.qnn_core import (
    QNN,
    InMemoryStore,
    train,
)


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "routing_datasets.json"
)

CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "router_qnn.pkl"
)


def load_dataset():

    print(f"[INFO] Loading dataset: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = np.array([
        extract_features(
            item["text"]
        )
        for item in data
    ])

    y = np.array([
        item["score"]
        for item in data
    ])

    return X, y


def build_model():

    store = InMemoryStore()

    if CHECKPOINT_PATH.exists():

        try:

            store.load(str(CHECKPOINT_PATH))

            print(
                f"[INFO] Loaded checkpoint: "
                f"{CHECKPOINT_PATH}"
            )

        except Exception as exc:

            print(
                f"[WARN] Failed loading checkpoint: "
                f"{exc}"
            )

    else:

        print(
            "[INFO] No checkpoint found. "
            "Using fresh weights."
        )

    qnn = QNN(
        [7, 16, 8, 1],
        store,
        lr=0.003
    )

    return qnn


def main():

    print("=" * 52)
    print(" ROUTER QNN TRAINING ")
    print("=" * 52)

    X, y = load_dataset()

    print(
        f"[INFO] Dataset loaded: "
        f"{len(X)} samples"
    )

    qnn = build_model()

    train(
        qnn,
        X,
        y,
        epochs=3000,
        log_every=250,
        checkpoint_every=3000,
        checkpoint_path=str(CHECKPOINT_PATH),
    )

    print()
    print("[INFO] Training completed.")
    print(
        f"[INFO] Checkpoint saved to: "
        f"{CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()