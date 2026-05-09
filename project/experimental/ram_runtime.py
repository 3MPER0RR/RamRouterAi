import os
import sys
import time
import asyncio
from collections import deque
from pathlib import Path

import httpx
import psutil

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# PATH + ENV
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

print(f"[ENV] {env_path}")
print(f"[API URL] {os.getenv('LLM_API_URL')}")
print(f"[API KEY] {'SET' if os.getenv('LLM_API_KEY') else 'MISSING'}")

from core.qnn_core import QNN, InMemoryStore
from core.runtime import Runtime


# ─────────────────────────────────────────────
# RAM CACHE
# ─────────────────────────────────────────────

class RAMCache:

    def __init__(self, max_items=256, ttl=300):
        self.max_items = max_items
        self.ttl = ttl
        self.cache = {}
        self.order = deque()

    def put(self, key, value):
        now = time.time()

        if key in self.cache:
            try:
                self.order.remove(key)
            except ValueError:
                pass

        self.cache[key] = {
            "value": value,
            "timestamp": now
        }

        self.order.append(key)

        while len(self.order) > self.max_items:
            old = self.order.popleft()
            self.cache.pop(old, None)

    def get(self, key):
        item = self.cache.get(key)

        if not item:
            return None

        if time.time() - item["timestamp"] > self.ttl:
            self.cache.pop(key, None)
            try:
                self.order.remove(key)
            except ValueError:
                pass
            return None

        return item["value"]


# ─────────────────────────────────────────────
# RAM RUNTIME
# ─────────────────────────────────────────────

class RAMRuntime(Runtime):

    def __init__(self, qnn):

        super().__init__(qnn)

        self.cache = RAMCache(max_items=512, ttl=600)

        self.offline_mode = False

        self.api_url = os.getenv("LLM_API_URL")
        self.api_key = os.getenv("LLM_API_KEY")

    # ── SYSTEM INFO ─────────────────────────────

    def memory_usage(self):
        p = psutil.Process(os.getpid())
        return round(p.memory_info().rss / 1024 / 1024, 2)

    # ── LOCAL LOGIC ─────────────────────────────

    async def local_reasoning(self, text):

        t = text.lower()

        if "hello" in t:
            return "Hello from RAM runtime."

        if "memory" in t:
            return f"RAM usage: {self.memory_usage()} MB"

        if "status" in t:
            return "system ok"

        return None

    # ── LLM API CALL ────────────────────────────

    async def llm_call(self, text: str):

        if not self.api_url:
            return "[API ERROR] missing url"

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )

        if r.status_code != 200:
            print("[API DEBUG RAW RESPONSE]")
            print(r.text)
            return f"[API ERROR] {r.status_code}"

        data = r.json()

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return str(data)

    # ── MAIN PIPELINE ───────────────────────────

    async def process(self, text):

        # CACHE HIT
        cached = self.cache.get(text)
        if cached is not None:
            print("[CACHE HIT]")
            return cached

        # LOCAL FAST PATH
        local = await self.local_reasoning(text)
        if local is not None:
            self.cache.put(text, local)
            return local

        # CORE QNN ROUTING
        response = await super().process(text)

        # API OVERRIDE (se risposta sembra LLM-heavy)
        if isinstance(response, str) and len(response) > 200:
            try:
                api_resp = await self.llm_call(text)
                self.cache.put(text, api_resp)
                return api_resp
            except Exception as exc:
                print(f"[API FAIL] {exc}")

        self.cache.put(text, response)
        return response


# ─────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────

def build_qnn():

    store = InMemoryStore()

    ckpt = BASE_DIR / "checkpoints/router_qnn.pkl"

    if ckpt.exists():
        try:
            store.load(str(ckpt))
            print(f"[INFO] Loaded checkpoint: {ckpt}")
        except Exception as e:
            print(f"[WARN] checkpoint load failed: {e}")

    return QNN([7, 16, 8, 1], store, lr=0.003)


# ─────────────────────────────────────────────
# LOOP
# ─────────────────────────────────────────────

async def interactive(runtime):

    print("\nRAM Runtime active\n")

    while True:

        try:
            text = input(">>> ").strip()

            if text.lower() in ["exit", "quit"]:
                break

            out = await runtime.process(text)

            print("\n" + str(out) + "\n")

        except KeyboardInterrupt:
            break


async def main():

    qnn = build_qnn()
    runtime = RAMRuntime(qnn)

    await interactive(runtime)


if __name__ == "__main__":
    asyncio.run(main())