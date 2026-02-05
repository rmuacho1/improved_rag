First attempt at building a RAG and serving it to an Ollama model:
1. Uses Qwen3-embeddings:0.6B for embeddings
2. ChromaDB for vector database
3. Uses BGE-v2-m3 for Reranker (outperforms Qwen3 reranker models imo)
4. Final model is Qwen3:1.7B - lightweight and super-fast, no noticeable difference compared to 4B and 8B models

The data used was Wikipedia articles focusing on WW2 themes (Operation Overlord, Potsdam Declaration).

Uses 3.8GB VRAM, extremely fast and consumer-friendly.

Enjoy!
