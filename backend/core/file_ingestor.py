from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from backend.config import Config

class FileIngestor:
    def __init__(self):
        # self.embed_model = get_embed_model()  # Defined in models/embed.py
        self.embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

    def ingest(self, file_path: str):
        """Load, split, and store documents in vector DB."""
        # Load document
        loader = UnstructuredLoader(file_path)
        documents = loader.load()
        
        # Split text
        splits = self.text_splitter.split_documents(documents)
        
        # Create and save vector store
        vectorstore = FAISS.from_documents(
            splits,
            self.embed_model
        )
        vectorstore.save_local(Config.VECTOR_DB_DIR)
        # documents = SimpleDirectoryReader(file_path).load_data()
        # index = VectorStoreIndex.from_documents(documents, embed_model=self.embed_model)
        # index.storage_context.persist(persist_dir="./storage/vector_db")