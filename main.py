import torch
import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from ollama import chat
import os
from rich.console import Console
from rich.markdown import Markdown

console = Console()

# Load embedding and reranker model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", model_kwargs={"torch_dtype": torch.bfloat16}, tokenizer_kwargs={"padding_side": "left"})
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, model_kwargs={"torch_dtype": torch.bfloat16}, tokenizer_kwargs={"padding_side": "left"})

os.system('cls')

client = chromadb.PersistentClient(path="./database1")
collection = client.get_collection(name="mega")

# Loop
while True:

    try:
        # Create and embed query
        query = input("You: ")
        print("")
        query_vector = model.encode(query, prompt_name="query").tolist()

        # Calculate similarity
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=30
        )

        chunks = results['documents'][0]

        pairs = [[query, chunk] for chunk in chunks]
        rerank = rerank_model.predict(pairs, batch_size=1)

        reranked_results = sorted(
            zip(rerank, chunks), 
            key=lambda x: x[0], 
            reverse=True
        )

        top_context = [content for score,content in reranked_results[:3]]

        context= "\n\n".join(top_context)

        prompt = f"""Use the following wikipedia article to answer the user's question. If an answer is found, ONLY use markdown to format it.
        If the information is not in the excerpts, say "I don't know".
        
        Context:
        {context} 
        
        Question:
        {query}
        """

        response = chat(
            model="qwen3:1.7b",
            messages=[
                {"role": "system", "content": "You are a military historian specializing in WW2 logistics."},
                {"role": "user", "content": prompt}],
            options={
                "num_ctx": 2048,
                "temperature": 0.5,
                },
            stream=False
        )

        md = Markdown(response['message']['content'])
        console.print("Ollama: ", md)

        print("")

    except KeyboardInterrupt:
        break