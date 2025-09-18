import os
import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Ajustes de paralelismo
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# Cargar texto y dividir en chunks
# =========================
with open("juegos_hambre.txt", "r", encoding="utf-8") as f:
    texto = f.read()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(texto)
print(f"✅ {len(chunks)} fragmentos generados")

# =========================
# Crear PersistentClient ChromaDB
# =========================
chroma_client = chromadb.PersistentClient(path="db_juegos")
collection = chroma_client.get_or_create_collection("juegos_hambre")

# Insertar los chunks (solo si no están ya)
if collection.count() == 0:
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"doc_{i}"],
            metadatas=[{"source": "juegos_hambre.txt", "chunk": i}]
        )
    print(f"✅ Insertados {len(chunks)} fragmentos en ChromaDB")
else:
    print(f"⚡ ChromaDB ya tiene {collection.count()} fragmentos")

# =========================
# Cargar modelo libre: GPT4All
# =========================
model_name = "nomic-ai/gpt4all-j"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if device=="mps" else None)

# =========================
# Función RAG
# =========================
def query_rag(pregunta, n_results=3):
    # 1️⃣ Recuperar documentos de ChromaDB
    results = collection.query(query_texts=[pregunta], n_results=n_results)
    retrieved_docs = results["documents"][0]
    
    # 2️⃣ Crear prompt con contexto
    context = "\n".join(retrieved_docs)
    prompt = f"Responde SOLO basándote en este contexto:\n{context}\n\nPregunta: {pregunta}"
    
    # 3️⃣ Tokenizar y generar respuesta
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return respuesta

# =========================
# Ejemplo de consulta
# =========================
if __name__ == "__main__":
    pregunta = "¿Quién se ofreció como voluntaria en lugar de su hermana?"
    print("Pregunta:", pregunta)
    print("Respuesta:", query_rag(pregunta))