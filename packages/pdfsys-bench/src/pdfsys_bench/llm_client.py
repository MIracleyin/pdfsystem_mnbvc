"""OpenAI-compatible LLM client for eval / annotation tooling.

Reads endpoint + credentials from environment (typically loaded from a
repo-root ``.env`` via python-dotenv) and wraps the official ``openai``
SDK so any OpenAI-style provider works unchanged.

Two call shapes:

* :meth:`LlmClient.chat` — text-in, text-out
* :meth:`LlmClient.chat_with_image` — vision input (PIL image, bytes,
  or a path) for PDF-page annotation

Run as a script to smoke-test the endpoint::

    uv run python -m pdfsys_bench.llm_client
    uv run python -m pdfsys_bench.llm_client --image path/to/page.png
"""

from __future__ import annotations

import argparse
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(slots=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str = _DEFAULT_MODEL

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "LlmConfig":
        """Load config from environment, optionally hydrating from a .env file.

        Search order for .env: ``dotenv_path`` arg, then walks up from CWD
        looking for a ``.env`` (matches what most tooling expects).
        """
        try:
            from dotenv import load_dotenv  # noqa: PLC0415
        except ImportError:  # pragma: no cover — dotenv listed as a dep
            load_dotenv = None  # type: ignore[assignment]

        if load_dotenv is not None:
            if dotenv_path is not None:
                load_dotenv(dotenv_path, override=False)
            else:
                load_dotenv(override=False)  # walks up to find .env

        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL not set (check .env)")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (check .env)")

        return cls(base_url=base_url, api_key=api_key, model=model)


class LlmClient:
    """Thin wrapper around the OpenAI SDK pointed at any OpenAI-compatible host."""

    def __init__(self, config: LlmConfig | None = None) -> None:
        from openai import OpenAI  # noqa: PLC0415

        self.config = config or LlmConfig.from_env()
        self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    # ---------------------------------------------------------------- public

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Single-turn text completion. Returns the assistant message content.

        Note: some hosted models on this endpoint (e.g. MiMo) are reasoning
        models that consume tokens in a hidden ``reasoning_content`` stream
        before producing visible ``content``. Setting ``max_tokens`` too low
        will return ``""``. Leave it ``None`` (server default) unless you
        know the model's budget.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_with_image(
        self,
        prompt: str,
        image: PILImage | bytes | str | Path,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        image_format: str = "PNG",
    ) -> str:
        """Vision call. ``image`` may be a PIL image, raw bytes, or a file path."""
        data_url = _to_data_url(image, image_format=image_format)

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

        resp = self._client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        """Returns model IDs exposed by the endpoint (used in smoke tests)."""
        return [m.id for m in self._client.models.list().data]


# ---------------------------------------------------------------- helpers

def _to_data_url(
    image: PILImage | bytes | str | Path,
    *,
    image_format: str = "PNG",
) -> str:
    if isinstance(image, (str, Path)):
        path = Path(image)
        raw = path.read_bytes()
        mime = _guess_mime(path.suffix, default=f"image/{image_format.lower()}")
    elif isinstance(image, bytes):
        raw = image
        mime = f"image/{image_format.lower()}"
    else:
        # PIL.Image — encode in memory.
        import io  # noqa: PLC0415

        buf = io.BytesIO()
        image.save(buf, format=image_format)
        raw = buf.getvalue()
        mime = f"image/{image_format.lower()}"

    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _guess_mime(suffix: str, *, default: str) -> str:
    s = suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(s, default)


# ---------------------------------------------------------------- smoke test

def _smoke(image_path: str | None) -> None:
    cfg = LlmConfig.from_env()
    print(f"[smoke] base_url = {cfg.base_url}")
    print(f"[smoke] model    = {cfg.model}")

    client = LlmClient(cfg)

    try:
        models = client.list_models()
        print(f"[smoke] /models OK — {len(models)} models")
        for m in models[:5]:
            print(f"        - {m}")
    except Exception as e:  # noqa: BLE001 — smoke output should not raise
        print(f"[smoke] /models failed: {type(e).__name__}: {e}")

    # Reasoning models (MiMo, R1-style) burn tokens on hidden chain-of-thought
    # first — leave max_tokens at the server default for the smoke ping.
    if image_path:
        resp = client._client.chat.completions.create(  # noqa: SLF001 — diagnostic only
            model=client.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this PDF page layout in one sentence."},
                    {"type": "image_url", "image_url": {"url": _to_data_url(Path(image_path))}},
                ],
            }],
            temperature=0.0,
        )
    else:
        resp = client._client.chat.completions.create(  # noqa: SLF001
            model=client.config.model,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            temperature=0.0,
        )

    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    finish = resp.choices[0].finish_reason
    print(f"[smoke] finish_reason = {finish}")
    if content:
        print(f"[smoke] content: {content!r}")
    elif reasoning:
        print("[smoke] content empty — reasoning model only emitted CoT:")
        print(reasoning[:300] + ("..." if len(reasoning) > 300 else ""))
    else:
        print("[smoke] empty response")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the OpenAI-compatible LLM endpoint")
    parser.add_argument("--image", help="Optional path to an image for a vision call")
    args = parser.parse_args()
    _smoke(args.image)


if __name__ == "__main__":
    _main()
