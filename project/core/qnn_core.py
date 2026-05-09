"""
QNN Core — Fase 1 (v2, gradienti verificati)
============================================
Rete neurale quantum-inspired con:
  - InMemoryStore: parameter server in-process, zero overhead di rete
  - Pesi complessi: W = amplitude * e^(i*phase)
  - Attivazione: modulus_activation = tanh(|z|) * z/|z|
  - Backward pass con gradienti di Wirtinger (verificati con gradient check)
  - Training loop con Adam optimizer su numeri complessi
  - Gradient check integrato per debugging

Dipendenze: solo NumPy
"""

import numpy as np
import time
import pickle
from typing import Any


# ─────────────────────────────────────────────────────
# INMEMORYSTORE
# ─────────────────────────────────────────────────────

class InMemoryStore:
    """
    Parameter server in-process ispirato a Redis.
    Tensori NumPy salvati per riferimento — latenza ~1ns, zero serializzazione.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._ttl:   dict[str, float] = {}

    def set(self, key: str, value: Any, ttl: float | None = None):
        self._store[key] = value
        if ttl is not None:
            self._ttl[key] = time.monotonic() + ttl

    def get(self, key: str, default=None) -> Any:
        if key in self._ttl and time.monotonic() > self._ttl[key]:
            self._store.pop(key, None)
            self._ttl.pop(key, None)
            return default
        return self._store.get(key, default)

    def delete(self, key: str):
        self._store.pop(key, None)
        self._ttl.pop(key, None)

    def hset(self, ns: str, field: str, value: Any):
        if ns not in self._store:
            self._store[ns] = {}
        self._store[ns][field] = value

    def hget(self, ns: str, field: str, default=None) -> Any:
        return self._store.get(ns, {}).get(field, default)

    def keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]

    def flush(self):
        self._store.clear()
        self._ttl.clear()

    def checkpoint(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"store": self._store, "ttl": self._ttl}, f)
        print(f"[Store] Checkpoint → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._store = data["store"]
        self._ttl   = data["ttl"]
        print(f"[Store] Caricato ← {path}")

    def __repr__(self):
        return f"InMemoryStore(keys={len(self._store)})"


# ─────────────────────────────────────────────────────
# ATTIVAZIONE — modulus_activation
#
# f(z) = tanh(|z|) * z / |z|
#
# Proprietà quantistica: preserva la fase (informazione
# direzionale nel piano complesso) e normalizza l'ampiezza.
#
# Derivate di Wirtinger (verificate con gradient check):
#   df/dz  = 0.5 * (tanh'(r)/r + tanh(r)/r) * |z| / r
#           = 0.5 * (dt + t/r)          [scalare reale]
#   df/dz* = 0.5 * (dt - t/r) * (z/r)² [complessa]
#
# Chain rule per backprop:
#   dL/dz* = conj(df/dz) * delta + (df/dz*) * conj(delta)
#   dove delta = dL/da*
# ─────────────────────────────────────────────────────

_EPS = 1e-8

def modulus_activation(z: np.ndarray) -> np.ndarray:
    r = np.abs(z) + _EPS
    return np.tanh(r) * z / r

def wirtinger_activation_backward(z: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Propaga il gradiente di Wirtinger attraverso modulus_activation.

    Args:
        z:     pre-attivazione complessa
        delta: dL/da* (gradiente rispetto al coniugato dell'attivazione)

    Returns:
        dL/dz* — gradiente rispetto al coniugato della pre-attivazione
    """
    r  = np.abs(z) + _EPS
    t  = np.tanh(r)
    dt = 1.0 - t * t

    df_dz      = 0.5 * (dt + t / r)           # reale
    df_dz_conj = 0.5 * (dt - t / r) * (z / r) ** 2  # complessa

    return np.conj(df_dz) * delta + df_dz_conj * np.conj(delta)


# ─────────────────────────────────────────────────────
# ADAM OPTIMIZER SU NUMERI COMPLESSI
#
# Adam tratta parte reale e immaginaria separatamente
# tramite la rappresentazione polare dei pesi.
# ─────────────────────────────────────────────────────

class AdamState:
    def __init__(self, shape, dtype=complex):
        self.m = np.zeros(shape, dtype=dtype)   # primo momento
        self.v = np.zeros(shape, dtype=float)    # secondo momento (magnitudine)
        self.t = 0

    def step(self, grad: np.ndarray, lr=0.001, b1=0.9, b2=0.999, eps=1e-8) -> np.ndarray:
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * (np.abs(grad) ** 2)
        m_hat = self.m / (1 - b1 ** self.t)
        v_hat = self.v / (1 - b2 ** self.t)
        return lr * m_hat / (np.sqrt(v_hat) + eps)


# ─────────────────────────────────────────────────────
# QNN CORE
# ─────────────────────────────────────────────────────

class QNN:
    """
    Quantum-inspired Neural Network.

    Pesi: W = A * e^(iφ), bias complessi.
    Attivazione: modulus_activation su tutti i layer nascosti.
    Output: |a_L| — misura quantistica → valore reale in [0, +∞).
    Ottimizzatore: Adam su gradiente di Wirtinger.

    Args:
        layer_sizes: es. [2, 16, 8, 1]
        store:       InMemoryStore condiviso
        lr:          learning rate Adam
    """

    def __init__(self, layer_sizes: list[int], store: InMemoryStore, lr: float = 0.001):
        self.layers  = layer_sizes
        self.store   = store
        self.lr      = lr
        self.n       = len(layer_sizes) - 1
        self._adam_W: list[AdamState] = []
        self._adam_b: list[AdamState] = []
        self._init_weights()

    # ── Inizializzazione ────────────────────────────────

    def _init_weights(self):
        for i in range(self.n):
            n_in, n_out = self.layers[i], self.layers[i + 1]
            # He/Glorot adattato per complessi
            scale = np.sqrt(2.0 / (n_in + n_out))
            W = (np.random.randn(n_out, n_in) + 1j * np.random.randn(n_out, n_in)) * scale
            b = np.zeros((n_out, 1), dtype=complex)
            self.store.hset(f"L{i}:W", "v", W)
            self.store.hset(f"L{i}:b", "v", b)
            self._adam_W.append(AdamState(W.shape))
            self._adam_b.append(AdamState(b.shape))
        n_params = sum(
            self.layers[i] * self.layers[i+1] + self.layers[i+1]
            for i in range(self.n)
        )
        print(f"[QNN] {self.layers} — {n_params} parametri complessi — lr={self.lr}")

    def _W(self, i): return self.store.hget(f"L{i}:W", "v")
    def _b(self, i): return self.store.hget(f"L{i}:b", "v")

    # ── Forward pass ────────────────────────────────────

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Restituisce (output_reale, cache).
        cache[i] = (a_in, z_out) per ogni layer i.
        """
        a = x.astype(complex).reshape(-1, 1)
        cache = [(a, None)]
        for i in range(self.n):
            z = self._W(i) @ a + self._b(i)
            a = modulus_activation(z)
            cache.append((a, z))
        return np.abs(cache[-1][0]).squeeze(), cache

    # ── Loss ────────────────────────────────────────────

    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    # ── Backward pass ───────────────────────────────────

    def backward(self, cache: list, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Backward pass con gradienti di Wirtinger.

        Step 1: dL/dout (reale, MSE)
        Step 2: dL/da_L* attraverso la misura |a_L|
        Step 3: per ogni layer in ordine inverso:
                - dL/dz* tramite wirtinger_activation_backward
                - dL/dW* = dL/dz* @ a_prev^H
                - dL/db* = sum(dL/dz*, axis=batch)
                - dL/da_prev* = W^H @ dL/dz*
        """
        y_pred = np.atleast_1d(y_pred)
        y_true = np.atleast_1d(y_true)

        grads_W, grads_b = [], []

        # dL/d(output reale)
        dL_dout = (2.0 / y_pred.size) * (y_pred - y_true)  # reale, shape (n_out,)
        dL_dout = dL_dout.reshape(-1, 1)

        # Attraverso la misura: d|a|/da* = a / (2|a|)
        a_last = cache[-1][0]
        delta = dL_dout * (a_last / (2.0 * (np.abs(a_last) + _EPS)))

        for i in reversed(range(self.n)):
            a_prev, _ = cache[i]
            a_curr, z_curr = cache[i + 1]
            batch = a_prev.shape[1] if a_prev.ndim > 1 else 1

            # Propaga attraverso l'attivazione (tutti i layer tranne l'ultimo
            # vengono già gestiti nel passo precedente, ma dobbiamo applicarlo
            # anche al layer corrente prima di calcolare i gradienti dei pesi)
            if z_curr is not None:
                delta = wirtinger_activation_backward(z_curr, delta)

            dW = (delta @ a_prev.conj().T) / batch
            db = delta.mean(axis=1, keepdims=True)

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            # Propaga a layer precedente
            delta = self._W(i).conj().T @ delta

        return grads_W, grads_b

    # ── Update Adam ─────────────────────────────────────

    def update(self, grads_W, grads_b):
        for i in range(self.n):
            W = self._W(i) - self._adam_W[i].step(grads_W[i], self.lr)
            b = self._b(i)  - self._adam_b[i].step(grads_b[i], self.lr)
            self.store.hset(f"L{i}:W", "v", W)
            self.store.hset(f"L{i}:b", "v", b)

    # ── Train step ──────────────────────────────────────

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred, cache = self.forward(x)
        loss = self.mse(y_pred, np.atleast_1d(y))
        gW, gb = self.backward(cache, y_pred, np.atleast_1d(y))
        self.update(gW, gb)
        return loss

    # ── Gradient check ──────────────────────────────────

    def gradient_check(self, x: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
        """
        Confronta i gradienti analitici con quelli numerici (finite differences).
        Restituisce l'errore relativo massimo. < 1e-5 = corretto.
        """
        x = np.atleast_1d(x).astype(complex)
        y = np.atleast_1d(y).astype(float)

        y_pred, cache = self.forward(x)
        gW_anal, gb_anal = self.backward(cache, y_pred, y)

        errors = []

        def loss_fn():
            out, _ = self.forward(x)
            return self.mse(out, y)

        for i in range(self.n):
            W = self._W(i)
            dW_num = np.zeros_like(W)
            for r in range(W.shape[0]):
                for c in range(W.shape[1]):
                    # Perturbazione reale
                    W[r, c] += eps
                    self.store.hset(f"L{i}:W", "v", W)
                    lp = loss_fn()
                    W[r, c] -= 2 * eps
                    self.store.hset(f"L{i}:W", "v", W)
                    lm = loss_fn()
                    W[r, c] += eps
                    dL_dx = (lp - lm) / (2 * eps)

                    # Perturbazione immaginaria
                    W[r, c] += 1j * eps
                    self.store.hset(f"L{i}:W", "v", W)
                    lp = loss_fn()
                    W[r, c] -= 2j * eps
                    self.store.hset(f"L{i}:W", "v", W)
                    lm = loss_fn()
                    W[r, c] += 1j * eps
                    dL_dy = (lp - lm) / (2 * eps)

                    dW_num[r, c] = 0.5 * (dL_dx + 1j * dL_dy)

            self.store.hset(f"L{i}:W", "v", W)
            err = np.max(np.abs(gW_anal[i] - dW_num) / (np.abs(dW_num) + 1e-10))
            errors.append(float(err))

        max_err = max(errors)
        status = "✓ CORRETTO" if max_err < 1e-4 else "✗ ERRORE"
        print(f"[GradCheck] Errore relativo max: {max_err:.2e}  {status}")
        return max_err


# ─────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────

def train(
    qnn: QNN,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 2000,
    checkpoint_every: int = 1000,
    checkpoint_path: str = "qnn_v2.pkl",
    log_every: int = 200,
) -> list[float]:

    history = []
    t0 = time.monotonic()
    n = X.shape[0]

    print(f"\n{'─'*52}")
    print(f"  Training QNN {qnn.layers}")
    print(f"  {n} campioni | {epochs} epoche | lr={qnn.lr}")
    print(f"{'─'*52}")

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        epoch_loss = sum(qnn.step(X[i], y[i:i+1]) for i in idx) / n
        history.append(epoch_loss)

        if epoch % log_every == 0 or epoch == 1:
            t = time.monotonic() - t0
            print(f"  Epoca {epoch:>5} | Loss {epoch_loss:.6f} | {t:.1f}s")

        if epoch % checkpoint_every == 0:
            qnn.store.checkpoint(checkpoint_path)

    t_tot = time.monotonic() - t0
    pct = (1 - history[-1] / history[0]) * 100 if history[0] > 0 else 0
    print(f"{'─'*52}")
    print(f"  Completato in {t_tot:.1f}s")
    print(f"  Loss: {history[0]:.6f} → {history[-1]:.6f}  ({pct:+.1f}%)")
    print(f"{'─'*52}\n")
    return history


def evaluate(qnn: QNN, X: np.ndarray, y: np.ndarray) -> dict:
    preds, losses = [], []
    for xi, yi in zip(X, y):
        out, _ = qnn.forward(xi)
        out = float(np.atleast_1d(out)[0])
        preds.append(float(out > 0.5))
        losses.append(QNN.mse(np.array([out]), np.array([float(yi)])))
    acc = float(np.mean(np.array(preds) == y))
    return {"accuracy": acc, "loss": float(np.mean(losses))}


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 52)
    print("  QNN CORE v2 — Wirtinger verificati")
    print("=" * 52)

    # ── 0. Gradient check prima di tutto ─────────────
    print("\n[0] Gradient check")
    store_gc = InMemoryStore()
    qnn_gc = QNN([2, 6, 4, 1], store_gc, lr=0.001)
    x_gc = np.array([0.4, -0.7])
    y_gc = np.array([1.0])
    qnn_gc.gradient_check(x_gc, y_gc)

    # ── 1. XOR ───────────────────────────────────────
    print("\n[1] XOR classico")
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor = np.array([0.0, 1.0, 1.0, 0.0])

    store_xor = InMemoryStore()
    qnn_xor = QNN([2, 16, 8, 1], store_xor, lr=0.005)
    train(qnn_xor, X_xor, y_xor, epochs=3000, log_every=500, checkpoint_every=3000)
    m = evaluate(qnn_xor, X_xor, y_xor)
    print(f"  XOR — Accuracy: {m['accuracy']*100:.0f}%  Loss: {m['loss']:.6f}")

    # Predizioni dettagliate
    print("  Predizioni:")
    for xi, yi in zip(X_xor, y_xor):
        out, _ = qnn_xor.forward(xi)
        out_val = float(np.atleast_1d(out)[0])
        pred = int(out_val > 0.5)
        mark = "✓" if pred == int(yi) else "✗"
        print(f"    {xi} → {out_val:.4f} (pred={pred}, target={int(yi)}) {mark}")

    # ── 2. Dataset circolare ──────────────────────────
    print("\n[2] Classificazione circolare (300 campioni)")
    rng = np.random.default_rng(0)
    X_c = rng.uniform(-2, 2, (300, 2))
    y_c = (X_c[:,0]**2 + X_c[:,1]**2 > 1.0).astype(float)

    split = 240
    X_tr, X_te = X_c[:split], X_c[split:]
    y_tr, y_te = y_c[:split], y_c[split:]

    store_c = InMemoryStore()
    qnn_c = QNN([2, 32, 16, 1], store_c, lr=0.003)
    train(qnn_c, X_tr, y_tr, epochs=2000, log_every=400, checkpoint_every=2000)

    tr_m = evaluate(qnn_c, X_tr, y_tr)
    te_m = evaluate(qnn_c, X_te, y_te)
    print(f"  Train — Accuracy: {tr_m['accuracy']*100:.1f}%  Loss: {tr_m['loss']:.6f}")
    print(f"  Test  — Accuracy: {te_m['accuracy']*100:.1f}%  Loss: {te_m['loss']:.6f}")

    print("\n✓ Fase 1 v2 completata.")
    print("  Prossimo: Fase 2 — Memoria episodica per sessioni di pentesting.")
