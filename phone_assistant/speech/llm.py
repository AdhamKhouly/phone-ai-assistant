"""Local language model through Ollama's streaming chat API on localhost. No cloud calls."""
from __future__ import annotations

import json
from typing import AsyncIterator

import aiohttp

OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaLLM:
    def __init__(self, model: str, *, num_ctx: int = 4096, num_predict: int = 160, temperature: float = 0.4,
                 keep_alive: str = "30m", url: str = OLLAMA_URL) -> None:
        self.model, self.name, self.url = model, model, url
        self.keep_alive = keep_alive
        self.options = {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120, sock_read=60))
        return self._session

    async def warm(self) -> None:
        """Load the model into memory so the first reply doesn't pay the cold-start cost."""
        async for _ in self.stream([{"role": "user", "content": "Say OK."}], num_predict=1):
            pass

    async def stream(self, messages: list[dict], num_predict: int | None = None) -> AsyncIterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True, "think": False,
                "keep_alive": self.keep_alive,
                "options": dict(self.options, **({"num_predict": num_predict} if num_predict else {}))}
        sess = await self._sess()
        async with sess.post(self.url + "/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                if not line.strip():
                    continue
                piece = json.loads(line)
                text = piece.get("message", {}).get("content", "")
                if text:
                    yield text
                if piece.get("done"):
                    return

    async def unload(self) -> None:
        sess = await self._sess()
        async with sess.post(self.url + "/api/generate", json={"model": self.model, "keep_alive": 0}) as r:
            await r.read()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
