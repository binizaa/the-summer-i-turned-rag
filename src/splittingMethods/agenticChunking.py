from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from typing import List
from rich import print

def agentChunking(text: str) -> List[str]:
    """
    Extrae las proposiciones o ideas principales de un texto dado usando un LLM local.
    Además imprime el proceso en consola con logs bonitos.
    """
    try:
        local_llm = ChatOllama(model="llama3", temperature=0.1)
        print("[bold green]LLM Local (ChatOllama) inicializado.[/bold green]")
    except Exception as e:
        print(f"[bold red]Error al inicializar ChatOllama:[/bold red] {e}")
        return ["Error al cargar el modelo LLM."]

    # Prompt: sin prefijos
    prompt_template_ideas = """
    Eres un experto en análisis de texto. Tu tarea es extraer las ideas principales o proposiciones clave de un párrafo dado.
    Responde solo con una lista de estas ideas, donde cada idea sea una oración o frase corta y clara.
    No incluyas ningún prefijo como "Idea principal", "1.", o guiones. Solo lista las ideas.

    Párrafo:
    {text}

    Ideas principales:
    """
    extraction_prompt = ChatPromptTemplate.from_template(prompt_template_ideas)

    def extract_ideas_from_paragraph(paragraph_text: str) -> List[str]:
        """Función interna para extraer ideas de un solo párrafo."""
        try:
            idea_chain = extraction_prompt | local_llm | StrOutputParser()
            raw_output = idea_chain.invoke({"text": paragraph_text})

            ideas = [idea.strip() for idea in raw_output.split('\n') if idea.strip()]
            if not ideas:
                return ["No se pudieron extraer ideas claras."]
            return ideas

        except Exception as e:
            print(f"[bold red]Error al extraer ideas:[/bold red] {e}")
            return ["Error al procesar el párrafo."]

    paragraphs = text.split("\n\n")
    all_propositions = []

    print("#### [bold blue]Extracción de Ideas (con LLM local)[/bold blue] ####")
    print(f"[bold cyan]Procesando {len(paragraphs)} párrafos...[/bold cyan]")

    for i, para in enumerate(paragraphs, 1):
        if para.strip():
            print(f"[yellow]Procesando párrafo {i}/{len(paragraphs)}...[/yellow]")
            propositions = extract_ideas_from_paragraph(para)
            all_propositions.extend(propositions)
        else:
            print(f"[dim]Saltando párrafo vacío {i}.[/dim]")

    print(f"\n[bold green]Se extrajeron {len(all_propositions)} ideas principales.[/bold green]")

    return all_propositions
