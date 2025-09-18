from langchain.docstore.document import Document
from langchain.text_splitter import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    Language
)
from typing import List

def splitMarkdownText(markdown_text: str, chunk_size: int = 40, chunk_overlap: int = 0) -> List[Document]:
    """
    Divide un texto en formato Markdown en fragmentos.
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([markdown_text])

def splitPythonCode(python_text: str, chunk_size: int = 100, chunk_overlap: int = 0) -> List[Document]:
    """
    Divide código Python en fragmentos, respetando la sintaxis.
    """
    splitter = PythonCodeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([python_text])

def splitJavascriptCode(javascript_text: str, chunk_size: int = 65, chunk_overlap: int = 0) -> List[Document]:
    """
    Divide código JavaScript en fragmentos, respetando la sintaxis.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.create_documents([javascript_text])

if __name__ == "__main__":
    # Datos de ejemplo
    markdown_text = """
    # Fun in California

    ## Driving

    Try driving on the 1 down to San Diego

    ### Food

    Make sure to eat a burrito while you're there

    ## Hiking

    Go to Yosemite
    """

    python_text = """
    class Person:
      def __init__(self, name, age):
        self.name = name
        self.age = age

    p1 = Person("John", 36)

    for i in range(10):
        print (i)
    """

    javascript_text = """
    // Function is called, the return value will end up in x
    let x = myFunction(4, 3);

    function myFunction(a, b) {
    // Function returns the product of a and b
      return a * b;
    }
    """

    # Llamada a las funciones y visualización de resultados
    print("#### Markdown Splitting ####")
    markdown_chunks = split_markdown_text(markdown_text)
    print(markdown_chunks)

    print("\n#### Python Code Splitting ####")
    python_chunks = split_python_code(python_text)
    print(python_chunks)

    print("\n#### JavaScript Code Splitting ####")
    javascript_chunks = split_javascript_code(javascript_text)
    print(javascript_chunks)