from langchain.text_splitter import CharacterTextSplitter

# 1. Character Text Splitting
def CharacterTextSplitter(chunk_size, chunk_overlap, text):

    print("#### Character Text Splitting ####")

    text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, separator='', strip_whitespace=False)
    documents = text_splitter.create_documents([text])
    print(documents)

# 2. Recursive Character Text Splitting
def RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, text):

    print("#### Recursive Character Text Splitting ####")
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap) # ["\n\n", "\n", " ", ""] 65,450
    print(text_splitter.create_documents([text])) 