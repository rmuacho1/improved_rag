from datasets import load_dataset
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

with open("data/mega.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Load Splitter and Database
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n====================","\n=","\n==", "\n===", "\n====", "\n", " ", ""]
)

client = chromadb.PersistentClient(path="./database1")
collection = client.get_or_create_collection(name="mega")

chunks = splitter.split_text(full_text)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device, model_kwargs={"torch_dtype": torch.bfloat16})

all_embeddings = model.encode(chunks, show_progress_bar=True).tolist()

for i,chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[all_embeddings[i]]
    )