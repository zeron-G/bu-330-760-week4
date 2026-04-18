"""Codex OAuth helpers for running PydanticAI through ChatGPT's Codex backend."""

from __future__ import annotations

from dataclasses import replace
import base64
import json
import time
from pathlib import Path
from typing import cast

import httpx
from openai import AsyncOpenAI
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, check_allow_model_requests
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_REFRESH_URL = "https://auth.openai.com/oauth/token"


class CodexOAuthManager:
    """Loads and refreshes ChatGPT Codex OAuth tokens from ``~/.codex/auth.json``."""

    def __init__(self, auth_path: Path = CODEX_AUTH_PATH):
        self.auth_path = auth_path
        self._auth_data: dict = {}
        self._last_loaded = 0.0

    @staticmethod
    def is_available(auth_path: Path = CODEX_AUTH_PATH) -> bool:
        if not auth_path.exists():
            return False
        try:
            data = json.loads(auth_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        tokens = data.get("tokens", {})
        return data.get("auth_mode") == "chatgpt" and bool(tokens.get("access_token"))

    def _load_auth_data(self, force: bool = False) -> dict:
        if not force and self._auth_data and time.time() - self._last_loaded < 60:
            return self._auth_data
        if not self.auth_path.exists():
            raise RuntimeError(f"Codex auth file not found: {self.auth_path}")

        data = json.loads(self.auth_path.read_text(encoding="utf-8"))
        if data.get("auth_mode") != "chatgpt":
            raise RuntimeError("Codex auth is not configured for ChatGPT OAuth.")

        self._auth_data = data
        self._last_loaded = time.time()
        return data

    def _load_tokens(self, force: bool = False) -> dict:
        return self._load_auth_data(force=force).get("tokens", {})

    def get_account_id(self) -> str:
        account_id = self._load_tokens().get("account_id", "")
        if not account_id:
            raise RuntimeError("Codex OAuth account_id is missing.")
        return account_id

    @staticmethod
    def _token_expiring_soon(access_token: str) -> bool:
        if not access_token:
            return True
        try:
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            return time.time() > claims.get("exp", 0) - 300
        except Exception:
            return False

    async def refresh_access_token(self) -> str:
        tokens = self._load_tokens(force=True)
        refresh_token = tokens.get("refresh_token", "")
        if not refresh_token:
            raise RuntimeError("Codex OAuth refresh token is missing.")

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                CODEX_REFRESH_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CODEX_CLIENT_ID,
                    "scope": "openid profile email offline_access",
                },
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Codex token refresh failed: {response.status_code} {response.text[:300]}"
            )

        refreshed = response.json()
        auth_data = self._load_auth_data(force=True)
        auth_tokens = auth_data.setdefault("tokens", {})
        auth_tokens["access_token"] = refreshed["access_token"]
        auth_tokens["id_token"] = refreshed.get("id_token", auth_tokens.get("id_token", ""))
        if refreshed.get("refresh_token"):
            auth_tokens["refresh_token"] = refreshed["refresh_token"]
        auth_data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        self.auth_path.write_text(json.dumps(auth_data, indent=2), encoding="utf-8")

        self._auth_data = auth_data
        self._last_loaded = time.time()
        return auth_tokens["access_token"]

    async def get_access_token(self) -> str:
        access_token = self._load_tokens().get("access_token", "")
        if self._token_expiring_soon(access_token):
            return await self.refresh_access_token()
        if not access_token:
            raise RuntimeError("Codex OAuth access token is missing.")
        return access_token


class CodexResponsesModel(OpenAIResponsesModel):
    """OpenAI Responses model that adapts Codex's stream-only backend to normal requests."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        if not self._get_instructions(messages, model_request_parameters):
            model_request_parameters = replace(
                model_request_parameters,
                instruction_parts=[InstructionPart(content="You are a helpful assistant.")],
            )
        resolved_settings = cast(OpenAIResponsesModelSettings, model_settings or {})
        response = await self._responses_create(
            messages,
            True,
            resolved_settings,
            model_request_parameters,
        )

        async with response:
            streamed = await self._process_streamed_response(
                response,
                resolved_settings,
                model_request_parameters,
            )
            async for _ in streamed:
                pass
            return streamed.get()


def build_codex_model(model_name: str) -> CodexResponsesModel:
    """Create a PydanticAI model backed by ChatGPT Codex OAuth."""

    manager = CodexOAuthManager()
    client = AsyncOpenAI(
        api_key=manager.get_access_token,
        base_url=CODEX_BASE_URL,
        default_headers={"ChatGPT-Account-ID": manager.get_account_id()},
    )
    provider = OpenAIProvider(openai_client=client)
    return CodexResponsesModel(
        model_name,
        provider=provider,
        settings={"openai_store": False},
    )
