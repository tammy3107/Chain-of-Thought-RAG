from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
import os
import numpy as np
import streamlit as st

local_path = os.listdir("papers/")
local_path = ["papers/"+i for i in local_path]
print(local_path)
data = []

for f in local_path:
  loader = UnstructuredPDFLoader(f)
  data.append(loader.load())
data = np.asarray(data)
data = list(data.flatten())


from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma


# Split and chunk
embedding_model = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)
text_splitter = SemanticChunker(embedding_model,breakpoint_threshold_type = "gradient" )
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
    collection_name="local-rag",
    persist_directory="./chroma_db"
)
