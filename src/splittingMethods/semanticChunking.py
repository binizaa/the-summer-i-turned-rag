# 4. Semantic Chunking
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from rich import print

def semanticChunking(text):
    """
    Genera semantic chunks a partir de un texto usando embeddings locales.
    Retorna una lista de strings (chunks).
    """
    try:
        local_embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        text_splitter = SemanticChunker(
            local_embeddings_model, 
            breakpoint_threshold_type="percentile",
        )
        documents = text_splitter.create_documents([text])
        return [doc.page_content for doc in documents]

    except Exception as e:
        print(f"[bold red]Error durante semantic chunking:[/bold red] {e}")
        return []
