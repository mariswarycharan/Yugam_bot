import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings,HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
loader = DirectoryLoader('data/', glob="**/*.txt", show_progress=True, loader_cls=TextLoader,loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(separators=['\n'],chunk_size=1, chunk_overlap=1)
texts = text_splitter.split_documents(documents)    
print(len(texts))
vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/yugam_vector_store_new_n")
print("Vector DB Successfully Created!")