import os
import atexit
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings,  AzureChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

AZURE_AI_SEARCH_SERVICE_NAME = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
AZURE_AI_SEARCH_INDEX_NAME = "ai-day"
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_EMBEDDINGS_MODEL = "text-embedding-3-large"
AZURE_OPENAI_MODEL = "gpt-4o-mini"


embeddings = AzureOpenAIEmbeddings(
    model= AZURE_EMBEDDINGS_MODEL,
    azure_endpoint= AZURE_OPENAI_ENDPOINT,
    openai_api_key= AZURE_OPENAI_API_KEY,
)

vector_store: AzureSearch = AzureSearch(
    embedding_function=embeddings.embed_query,
    azure_search_endpoint= AZURE_AI_SEARCH_SERVICE_NAME,
    azure_search_key= AZURE_AI_SEARCH_API_KEY,
    index_name= AZURE_AI_SEARCH_INDEX_NAME,
)

@atexit.register
def cleanup_vector_store():
    try:
        del vector_store
    except Exception:
        pass

def load_documents_from_directory(directory_path: str):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Diretório '{directory_path}' não encontrado. Caminho atual: {os.getcwd()}")

    loader = DirectoryLoader(directory_path, glob="*.docx", show_progress=True)
    docs = loader.load()
    vector_store.add_documents(documents=docs)
    print("Documentos processados e adicionados ao vetor.")
      
def load_pdf_from_directory(directory_path: str):
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import CharacterTextSplitter

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Diretório '{directory_path}' não encontrado. Caminho atual: {os.getcwd()}")

    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    vector_store.add_documents(documents=docs)
    print("PDFs processados e adicionados ao vetor.")

def chat_on_files():
  azure_endpoint=AZURE_OPENAI_ENDPOINT
  api_key=AZURE_OPENAI_API_KEY
  openai_api_version="2024-02-15-preview"

  llm = AzureChatOpenAI(
            temperature=0.7,
            azure_deployment="gpt-4o",
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=openai_api_version,
            max_tokens=4096,
            max_retries=0)

  qa = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vector_store.as_retriever(),
      return_source_documents=True
  )

  query = f"""
    Preciso de um candidato ideal para uma vaga de Presidente da empresa , preciso do nome e uma justificativa do porquê o candidato é ideal."""

  result = qa({"query": query})
  
  print("Resposta: ", result["result"])

#load_documents_from_directory("C:\\temp\\Rag\\ARQUIVOS")
#load_pdf_from_directory("C:\\temp\\Rag\\ARQUIVOS")

chat_on_files()