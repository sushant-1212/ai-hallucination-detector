"""
llm_provider.py
----------------
Single place that decides which LLM + embedding backend to use
(Gemini, OpenAI, or Grok) based on the LLM_PROVIDER env var.

Keeping this in one file means the rest of the app (claim_extractor,
retriever, evaluator) never has to know or care which provider is active.

Note on Grok: xAI's API is OpenAI-compatible for chat, but xAI does not
currently offer an embeddings endpoint. When LLM_PROVIDER=grok, embeddings
fall back to a free local model (sentence-transformers) so no second API
key is required.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


def get_chat_model(temperature: float = 0.0):
    """Return a LangChain chat model for the configured provider."""
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from your .env file.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=api_key)

    if PROVIDER == "grok":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is missing from your .env file.")
        model_name = os.getenv("GROK_MODEL", "grok-2-latest")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    # default: gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from your .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
        google_api_key=api_key,
    )


def get_embeddings():
    """Return a LangChain embeddings object for the configured provider."""
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from your .env file.")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    if PROVIDER == "grok":
        # xAI has no embeddings endpoint yet - use a free local model instead.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # default: gemini
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )


def provider_name() -> str:
    return PROVIDER
