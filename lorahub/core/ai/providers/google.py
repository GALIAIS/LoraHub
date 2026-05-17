"""Google Gemini via the OpenAI-compatible endpoint.

Google ships a chat-completions shim at
``https://generativelanguage.googleapis.com/v1beta/openai/`` that
accepts standard OpenAI-style requests. Using it lets us reuse the
OpenAIChatProvider plumbing instead of writing a separate Gemini
client.
"""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ModelInfo,
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="google",
    name="Google Gemini",
    homepage="https://ai.google.dev",
    docs_url="https://ai.google.dev/gemini-api/docs/openai",
    auth_help="Create a key at https://aistudio.google.com/app/apikey",
    default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    default_model="gemini-2.5-flash",
    models=(
        ModelInfo("gemini-2.5-pro", "Gemini 2.5 Pro", vision=True, context=2_000_000),
        ModelInfo("gemini-2.5-flash", "Gemini 2.5 Flash", vision=True, context=1_000_000),
        ModelInfo("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", vision=True, context=1_000_000),
        ModelInfo("gemini-2.0-flash", "Gemini 2.0 Flash", vision=True, context=1_000_000),
    ),
)
register(DESCRIPTOR)


class GoogleProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
