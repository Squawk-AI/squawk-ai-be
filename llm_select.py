# llm_select.py
from __future__ import annotations

# 1) Load environment early
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(usecwd=True))

import os
import time
import numpy as np
from typing import List, Tuple, Type, Dict, Optional, Literal

# Chat LLMs (keep as you had them)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Your config flag: set "openai" or "huggingface"
from config import EMBEDDING_MODEL_PROVIDER

# ---------------- LLM selection (unchanged) ----------------
model_options: List[Tuple[str, Type[BaseChatModel]]] = [
    ("gpt-4o", ChatOpenAI),
    ("gpt-3.5-turbo-0125", ChatOpenAI),
    ("claude-3-opus-20240229", ChatAnthropic),
    ("claude-3-5-sonnet-20241022", ChatAnthropic),
    ("gpt-5", ChatOpenAI),
]

node_model_index: Dict[str, int] = {
    "summarize_issue": 0,
    "generate_question": 0,
    "build_fault_tree": 0,
    "update_tree": 0,
}

def get_llm_for_node(node_name: str) -> BaseChatModel:
    idx = node_model_index.get(node_name, 0)
    model_name, provider_cls = model_options[idx]
    # Default
    if idx == 4:
        return provider_cls(model=model_name)

    # Tuned decoding for generate_question to encourage specific, non-repetitive questions
    if node_name == "generate_question" and provider_cls is ChatOpenAI:
        return provider_cls(
            model=model_name,
            temperature=0.3,
            top_p=0.9,
            presence_penalty=0.3,
        )

    # Other nodes
    return provider_cls(model=model_name, temperature=0)

# ---------------- Embedding clients (no LangChain) ----------------
class HFEmbeddings:
    """
    Hugging Face Inference API embeddings via huggingface_hub.InferenceClient.
    - Adds BGE 'query:' / 'passage:' prefixes
    - L2 normalizes for cosine similarity
    - Retries transient errors (429/5xx)
    """
    def __init__(
        self,
        model: str = "BAAI/bge-base-en-v1.5",   # 768-d
        token: Optional[str] = None,
        timeout: int = 60,
        normalize: bool = True,
        max_retries: int = 4,
        backoff_base: float = 0.8,
        batch_size: int = 32,
    ):
        from huggingface_hub import InferenceClient  # local import keeps deps optional
        from huggingface_hub.utils import HfHubHTTPError

        self._HfHubHTTPError = HfHubHTTPError
        token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or pass token=...")

        self.client = InferenceClient(model=model, token=token, timeout=timeout)
        self.model = model
        self.normalize = normalize
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.batch_size = batch_size
        self.dim = 768 if "bge-base" in model else (1024 if "bge-m3" in model else None)

    @staticmethod
    def _l2(v: List[float]) -> List[float]:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        return (a / n).tolist() if n > 0 else a.tolist()

    def _call(self, text: str) -> List[float]:
        delay = self.backoff_base
        for i in range(self.max_retries):
            try:
                return self.client.feature_extraction(text)
            except self._HfHubHTTPError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                transient = code in (429, 500, 502, 503, 504)
                if transient and i < self.max_retries - 1:
                    time.sleep(delay); delay *= 1.8; continue
                body = ""
                try: body = e.response.text[:300]
                except Exception: pass
                raise RuntimeError(f"Hugging Face error {code}: {body}") from e
            except Exception as e:
                if i < self.max_retries - 1:
                    time.sleep(delay); delay *= 1.8; continue
                raise

    @staticmethod
    def _q(text: str) -> str: return f"query: {text}"
    @staticmethod
    def _p(text: str) -> str: return f"passage: {text}"

    def embed_query(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise TypeError("embed_query expects str")
        vec = self._call(self._q(text))
        return self._l2(vec) if self.normalize else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not isinstance(texts, list) or (texts and not isinstance(texts[0], str)):
            raise TypeError("embed_documents expects List[str]")
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i+self.batch_size]
            for t in chunk:
                vec = self._call(self._p(t))
                out.append(self._l2(vec) if self.normalize else vec)
            if len(chunk) >= self.batch_size:
                time.sleep(0.01)  # gentle pacing
        return out

class OpenAIEmbeddingsSimple:
    """
    OpenAI embeddings via openai Python SDK (no LangChain).
    - Defaults to text-embedding-3-small (1536-d)
    - L2 normalizes for cosine similarity
    """
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        timeout: int = 60,
        normalize: bool = True,
        batch_size: int = 128,
    ):
        from openai import OpenAI
        self._OpenAI = OpenAI
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or pass api_key=...")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.normalize = normalize
        self.batch_size = batch_size
        self.dim = 1536 if "small" in model else (3072 if "large" in model else None)

    @staticmethod
    def _l2(v: List[float]) -> List[float]:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        return (a / n).tolist() if n > 0 else a.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise TypeError("embed_query expects str")
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = resp.data[0].embedding
        return self._l2(vec) if self.normalize else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not isinstance(texts, list) or (texts and not isinstance(texts[0], str)):
            raise TypeError("embed_documents expects List[str]")
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i+self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            for d in resp.data:
                vec = d.embedding
                out.append(self._l2(vec) if self.normalize else vec)
        return out

# ---------------- Select provider ----------------
if EMBEDDING_MODEL_PROVIDER == "openai":
    _OPENAI_MODEL = "text-embedding-3-small"
    embeddings = OpenAIEmbeddingsSimple(model=_OPENAI_MODEL)
    embedding_model_name = _OPENAI_MODEL

elif EMBEDDING_MODEL_PROVIDER == "huggingface":
    _HF_MODEL = "BAAI/bge-base-en-v1.5"  # use "BAAI/bge-m3" for 1024-d multilingual
    embeddings = HFEmbeddings(model=_HF_MODEL)
    embedding_model_name = _HF_MODEL

else:
    raise ValueError(f"Unsupported EMBEDDING_MODEL_PROVIDER: {EMBEDDING_MODEL_PROVIDER}")

# ---------------- Vector table name helper ----------------
TABLE_NAME = (
    f"aircraft_manual_{embedding_model_name.replace('/', '_').replace('-', '_').replace('.', '_')}"
    .lower()
)
