import asyncio
from pathlib import Path

from dotenv import load_dotenv

from core.qnn_core import QNN, InMemoryStore
from core.runtime import Runtime


load_dotenv()


def build_qnn() -> QNN:
    store = InMemoryStore()

    checkpoint_path = Path("checkpoints/router_qnn.pkl")
    if checkpoint_path.exists():
        try:
            store.load(str(checkpoint_path))
            print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load checkpoint: {exc}")
    else:
        print("[WARN] No checkpoint found, using fresh weights.")

    qnn = QNN(
        [7, 16, 8, 1],
        store,
        lr=0.003
    )
    return qnn


async def main():
    qnn = build_qnn()
    runtime = Runtime(qnn)

    print("\nQNN runtime ready.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            text = input(">>> ").strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break

            response = await runtime.process(text)
            print()
            print(response)
            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except EOFError:
            print("\nExiting.")
            break
        except Exception as exc:
            print(f"[ERROR] {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())