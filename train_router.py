import json
import numpy as np

from core.embeddings import extract_features

from core.qnn_core import (
    QNN,
    InMemoryStore,
    train
)

# Load dataset
with open(
    "datasets/routing_dataset.json",
    "r"
) as f:

    data = json.load(f)

# Build arrays
X = np.array([
    extract_features(
        x["text"]
    )
    for x in data
])

y = np.array([
    x["score"]
    for x in data
])

# Create model
store = InMemoryStore()

qnn = QNN(
    [7, 16, 8, 1],
    store,
    lr=0.003
)

# Train
train(
    qnn,
    X,
    y,
    epochs=3000,
    log_every=250,
    checkpoint_every=3000,
    checkpoint_path="checkpoints/router_qnn.pkl"
)

print()
print("Training completed.")
