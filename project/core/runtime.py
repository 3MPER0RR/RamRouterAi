from dotenv import load_dotenv

load_dotenv()

import os
import httpx
import asyncio

from core.embeddings import extract_features
from core.routing import Router
from core.memory import EpisodicMemory


class OpenAICompatibleClient:

    def __init__(self):

        self.api_url = os.getenv(
            "LLM_API_URL"
        )

        self.api_key = os.getenv(
            "LLM_API_KEY"
        )

        self.model = os.getenv(
            "LLM_MODEL"
        )

    async def chat(self, prompt):

        headers = {
            "Authorization":
            f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        async with httpx.AsyncClient() as client:

            r = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )

        data = r.json()

        return data["choices"][0]["message"]["content"]


class Runtime:

    def __init__(self, qnn):

        self.qnn = qnn

        self.memory = EpisodicMemory()

        self.client = OpenAICompatibleClient()

    async def process(self, text):

        features = extract_features(text)

        score, _ = self.qnn.forward(features)

        score = float(score)

        route = Router.decide(score)

        print(f"[ROUTE] {route} | score={score:.3f}")

        if route == "local":

            response = "Local lightweight response."

        elif route == "memory":

            mem = self.memory.retrieve(features)

            response = str(mem)

        elif route == "tool":

            response = "Tool execution placeholder."

        else:

            response = await self.client.chat(text)

        self.memory.add(
            text=text,
            embedding=features,
            response=response,
            score=score
        )

        return response
