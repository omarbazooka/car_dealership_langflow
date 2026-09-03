from __future__ import annotations

import os
from pathlib import Path

from lfx.custom import Component
from lfx.io import IntInput, MessageTextInput, Output, SecretStrInput
from lfx.schema import Data

from car_dealership_core import DEFAULT_DB, lexical_knowledge_search, sync_knowledge_folder


class DealershipKnowledgeRAG(Component):
    display_name = "Dealership Knowledge RAG"
    description = "RAG over dealership policies/FAQ. Uses Chroma + Gemini embeddings when API key is available; deterministic lexical fallback is only for local smoke tests."
    icon = "database-zap"
    name = "DealershipKnowledgeRAG"

    inputs = [
        MessageTextInput(name="query", display_name="Query", tool_mode=True, required=True),
        IntInput(name="top_k", display_name="Top K", value=4, tool_mode=True, required=False),
        MessageTextInput(name="knowledge_dir", display_name="Knowledge Directory", value="/app/knowledge", advanced=True),
        MessageTextInput(name="chroma_dir", display_name="Chroma Directory", value="/data/chroma", advanced=True),
        MessageTextInput(name="db_path", display_name="DB Path", value=DEFAULT_DB, advanced=True),
        SecretStrInput(name="google_api_key", display_name="Google API Key", required=False, advanced=True),
    ]
    outputs = [Output(display_name="Retrieved Context", name="context", method="retrieve")]

    def _key(self) -> str:
        raw = getattr(self, "google_api_key", None)
        if raw:
            try:
                return raw.get_secret_value()
            except AttributeError:
                return str(raw)
        return os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

    def _vector_retrieve(self, key: str, top_k: int):
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        knowledge_dir = Path(self.knowledge_dir)
        chroma_dir = Path(self.chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=key)
        store = Chroma(
            collection_name="car_dealership_knowledge",
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )

        # Idempotent small-demo indexing. Hash-ish IDs are stable by source/chunk index.
        existing_payload = store.get(include=["metadatas"])
        existing = set(existing_payload.get("ids", []))
        docs, ids = [], []
        for p in sorted([*knowledge_dir.glob("*.md"), *knowledge_dir.glob("*.txt")]):
            text = p.read_text(encoding="utf-8")
            # Paragraph-oriented chunking keeps policies readable without another dependency.
            chunks = []
            current = []
            size = 0
            for para in [x.strip() for x in text.split("\n\n") if x.strip()]:
                if size + len(para) > 1200 and current:
                    chunks.append("\n\n".join(current))
                    current, size = [], 0
                current.append(para)
                size += len(para)
            if current:
                chunks.append("\n\n".join(current))
            for i, chunk in enumerate(chunks):
                doc_id = f"{p.stem}-{i}"
                if doc_id not in existing:
                    ids.append(doc_id)
                    docs.append(Document(page_content=chunk, metadata={"source": p.name}))
        if docs:
            store.add_documents(docs, ids=ids)

        results = store.similarity_search_with_relevance_scores(self.query, k=top_k)
        return [
            {"content": doc.page_content, "source": doc.metadata.get("source"), "score": float(score)}
            for doc, score in results
        ]

    def retrieve(self) -> Data:
        top_k = max(1, min(int(self.top_k or 4), 8))
        sync_knowledge_folder(self.db_path or DEFAULT_DB, self.knowledge_dir)
        key = self._key()
        mode = "lexical-smoke-fallback"
        error = None
        results = []
        if key:
            try:
                results = self._vector_retrieve(key, top_k)
                mode = "chroma+gemini-embedding-001"
            except Exception as exc:  # tool returns controlled failure context instead of crashing agent
                error = f"Vector retrieval failed: {type(exc).__name__}: {exc}"
        if not results:
            results = lexical_knowledge_search(self.db_path or DEFAULT_DB, self.query, top_k)
        payload = {"mode": mode, "query": self.query, "results": results}
        if error:
            payload["warning"] = error
        self.status = payload
        return Data(data=payload)
