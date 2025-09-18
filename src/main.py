from splittingMethods.agenticChunking import agentChunking
from splittingMethods.semanticChunking import semanticChunking
from rich import print

with open('content.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print("#### [bold blue]Semantic Chunking with Local Embeddings[/bold blue] ####")
chunks = semanticChunking(text)

print(f"\n[bold green]Se generaron {len(chunks)} Semantic Chunks:[/bold green]\n")
for i, chunk in enumerate(chunks, 1):
    print(f"[bold cyan]Chunk {i}:[/bold cyan]")
    print(chunk)  
    print("---")

print(agentChunking(text))