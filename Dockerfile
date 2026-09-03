FROM langflowai/langflow-all:1.12.0

USER root
RUN uv pip install --python /app/.venv/bin/python \
    "langchain-google-genai>=4.1,<5" \
    "langchain-chroma>=0.2,<1" \
    "chromadb>=1,<2" \
    "kagglehub>=0.3,<1"
USER 1000
