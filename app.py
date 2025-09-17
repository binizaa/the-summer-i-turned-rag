from rich import print
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

local_llm = ChatOllama(model="mistral")

# RAG
def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )
    result = chain.invoke("What is the use of Text Splitting?")
    print(result)

# 1. Character Text Splitting
print("#### Character Text Splitting ####")

text = "El autor, también guionista y director de cine, ha sido más conocido por la novela ‘Las ventajas de ser un marginado’, todo un éxito entre el público adolescente, pero en este compendio de horror y fantasía desempolva a referentes como Stephen King."

# Manual Splitting
chunks = []
chunk_size = 35 # Characters
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
print(documents)

# Automatic Text Splitting
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size = 35, chunk_overlap=3, separator='', strip_whitespace=False)
documents = text_splitter.create_documents([text])
print(documents)  

# 2. Recursive Character Text Splitting
print("#### Recursive Character Text Splitting ####")

from langchain.text_splitter import RecursiveCharacterTextSplitter
with open('content.txt', 'r', encoding='utf-8') as file:
    text = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 450, chunk_overlap=0) # ["\n\n", "\n", " ", ""] 65,450
print(text_splitter.create_documents([text])) 

# 3. Document Specific Splitting
print("#### Document Specific Splitting ####")

# Document Specific Splitting - Markdown
from langchain.text_splitter import MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size = 40, chunk_overlap=0)
markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""
print(splitter.create_documents([markdown_text]))

# Document Specific Splitting - Python
from langchain.text_splitter import PythonCodeTextSplitter
python_text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

for i in range(10):
    print (i)
"""
python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)
print(python_splitter.create_documents([python_text]))

# Document Specific Splitting - Javascript
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
javascript_text = """
// Function is called, the return value will end up in x
let x = myFunction(4, 3);

function myFunction(a, b) {
// Function returns the product of a and b
  return a * b;
}
"""
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=65, chunk_overlap=0
)
print(js_splitter.create_documents([javascript_text]))

# 4. Semantic Chunking with a Local Model
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
print("#### Semantic Chunking with Local Embeddings ####")

try:
    # Use OllamaEmbeddings to connect to your locally running 'nomic-embed-text' model.
    local_embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    print("OllamaEmbeddings initialized successfully.")

    # Initialize the SemanticChunker with the local embedding model.
    text_splitter = SemanticChunker(
        local_embeddings_model, 
        breakpoint_threshold_type="percentile",
        # breakpoint_threshold_amount=97.0
    )

    documents = text_splitter.create_documents([text])
    
    print("\n[bold green]Semantic Chunks generated:[/bold green]")
    for i, doc in enumerate(documents):
        print(f"[bold blue]Chunk {i+1}:[/bold blue]\n{doc.page_content}\n---")

except Exception as e:
    print(f"[bold red]Error during semantic chunking:[/bold red] {e}")
    print("[bold red]Please ensure Ollama is running and 'nomic-embed-text' is pulled.[/bold red]")

# 5. Agentic Chunking
print("#### Proposition-Based Chunking ####")

# https://arxiv.org/pdf/2312.06648.pdf
import torch
from rich import print
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from typing import List, Optional
from agentic_chunker import AgenticChunker

# --- Configuración del LLM Local ---
try:
    local_llm = ChatOllama(model="llama3", temperature=0.1) 
    print("[bold green]LLM Local (ChatOllama) inicializado.[/bold green]")
except Exception as e:
    print(f"[bold red]Error al inicializar ChatOllama:[/bold red] {e}")
    exit()

# --- Definición del Schema para Extraer Ideas ---
class Ideas(BaseModel):
    ideas: List[str]

# --- Adaptación de la Extracción de Proposiciones ---
# Creamos un prompt más directo para que el LLM local identifique "ideas clave".
prompt_template_ideas = """
Eres un experto en análisis de texto. Tu tarea es extraer las ideas principales o proposiciones clave de un párrafo dado.
Responde solo con una lista de estas ideas, donde cada idea sea una oración o frase corta y clara.
Formato de salida esperado:
Solo lista las ideas separadas por saltos de línea. Por ejemplo:
Idea principal 1
Idea principal 2
Otra idea importante

Párrafo:
{text}

Ideas principales:
"""

extraction_prompt = ChatPromptTemplate.from_template(prompt_template_ideas)

def get_ideas_from_paragraph(paragraph_text: str) -> List[str]:
    """
    Utiliza el LLM local para extraer ideas clave de un párrafo.
    """
    try:
        idea_chain = extraction_prompt | local_llm | StrOutputParser()
        
        raw_output = idea_chain.invoke({"text": paragraph_text})
        
        # Dividimos la salida en una lista de ideas
        ideas = [idea.strip() for idea in raw_output.split('\n') if idea.strip()]
        
        if not ideas:
            return ["No se pudieron extraer ideas claras de este párrafo."]
        
        return ideas
    except Exception as e:
        print(f"[bold red]Error al extraer ideas:[/bold red] {e}")
        return ["Error al procesar el párrafo."] 

# --- Preparación del Texto y Extracción de Ideas ---
# Asumiendo que 'text' ya está cargado y contiene el contenido a procesar.
# Si 'text' no está definido, cárgalo desde un archivo o fuente.
# Ejemplo:
# with open('tu_libro.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# Si no tienes 'text' cargado, usa un placeholder para probar el código:
if 'text' not in locals():
    text = """
Artificial Intelligence is transforming industries. It enables automation and data analysis at unprecedented scales.
Machine learning, a subset of AI, allows systems to learn from data without explicit programming.
Deep learning, a further specialization, uses neural networks with many layers to model complex patterns.
These advancements are driving innovation in healthcare, finance, and transportation.
The ethical implications and societal impact of AI are subjects of ongoing discussion and research.
Natural Language Processing (NLP) enables computers to understand and process human language.
Computer Vision allows machines to "see" and interpret visual information.
Robotics combines AI with mechanical engineering to create intelligent machines.
The future of AI promises further integration into our daily lives, with potential for significant advancements.
"""
    print("[bold yellow]Usando texto de ejemplo para demostración.[/bold yellow]")


print("#### Extracción de Ideas (con LLM local) ####")
paragraphs = text.split("\n\n")
text_propositions = [] # Renombramos a 'ideas_principales' para mayor claridad
print(f"[bold blue]Procesando {len(paragraphs)} párrafos...[/bold blue]")

# Limitamos a unos pocos párrafos para la demostración rápida
for i, para in enumerate(paragraphs[:5]):
    print(f"Procesando párrafo {i+1}/{min(len(paragraphs), 5)}...")
    if para.strip(): # Evita procesar párrafos vacíos
        propositions = get_ideas_from_paragraph(para)
        text_propositions.extend(propositions)
    else:
        print(f"Saltando párrafo vacío {i+1}.")

print(f"\n[bold green]Se extrajeron {len(text_propositions)} ideas principales.[/bold green]")
print("[bold blue]Primeras 10 ideas extraídas:[/bold blue]")
print(text_propositions[:10])

