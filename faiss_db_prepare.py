from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings,HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import FAISS


text = ''
loader = DirectoryLoader('data/', glob="**/*.txt", show_progress=True, loader_cls=TextLoader,loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()

text += documents[0].page_content + "\n\n"
text += documents[1].page_content 

text = "how are you?"

text_splitter = RecursiveCharacterTextSplitter(separators=['\n'],chunk_size=1, chunk_overlap=1)
chunks = text_splitter.split_text(text)

print('No of documents (chunks) ==> ',len(chunks))

embeddings = HuggingFaceBgeEmbeddings(
model_name="BAAI/bge-base-en-v1.5", encode_kwargs={"normalize_embeddings": True},)
   
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
print('vector_store started.....')
vector_store.save_local("source_data/emtry_db")
print('vector_store ended.....')

