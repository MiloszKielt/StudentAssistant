from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import List, Optional, Set, Dict, Type

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader, UnstructuredMarkdownLoader, \
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from core.validation_methods import validate_string


@dataclass
class VectorStoreProvider:
    """
    Class responsible for providing retriever for vectorstore. It automatically detects changes to documents and rebuilds itself.
    Supports only TXT and PDF files for documents.

    Attributes:
        embedding_model (Embeddings): model used for text tokenization and embedding
        chunk_size (int): size of document chunk. Default: 1000
        chunk_overlap (int): number of chunk overlaps. Default: 200
        documents_path (Path): Path to documents directory. Default: Path("documents").
    """
    embedding_model: Embeddings
    k: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 200
    documents_path: Path = Path("../storage/uploads")
    __retriever: Optional[VectorStoreRetriever] = field(default=None, init=False)
    __cached_documents: Dict[Path, float] = field(default_factory=dict, init=False)
    __loaders: Dict[str, Type] = field(init=False, default_factory=lambda: {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".html": BSHTMLLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
    })

    def __post_init__(self):
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("k must be over 0!")

        if not isinstance(self.chunk_size, int) or self.chunk_size <= 200:
            raise ValueError("Chunk size must be at least 200!")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap mustn't be bigger or equal the size of chunk!")

        if not isinstance(self.documents_path, Path):
            if not validate_string(self.documents_path):
                raise ValueError("Path to documents must be a Path object!")
            else:
                self.documents_path = Path(self.documents_path)

        self.documents_path.mkdir(exist_ok=True)
        self.__documents_changed_check()

    @property
    def documents_files(self) -> Set[Path]:
        return set(self.documents_path.glob("*.*"))

    def __documents_changed_check(self) -> bool:
        current = {path: path.stat().st_mtime for path in self.documents_files}
        if current != self.__cached_documents:
            self.__cached_documents = current
            self.__retriever = None
            return True
        return False

    def __validate_document(self, path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0

    def __load_and_split_document(self, path: Path) -> List[Document]:
        if not self.__validate_document(path):
            raise FileNotFoundError(f"Invalid document file: {path}")

        suffix = path.suffix
        loader = self.__loaders.get(path.suffix)
        if not loader:
            raise ValueError(f"Unsupported file type: {suffix}")

        documents = loader(str(path)).load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        return text_splitter.split_documents(documents)

    def __load_and_split_documents(self) -> List[Document]:
        if self.__documents_changed_check() or not hasattr(self, '__chunks'):
            if len(self.documents_files) == 0:
                raise ValueError("There must be at least one document in documents folder!")
            self.__chunks = list(chain.from_iterable(self.__load_and_split_document(doc) for doc in self.__cached_documents.keys()))
        return self.__chunks

    @property
    def retriever(self) -> VectorStoreRetriever:
        if not self.__retriever or self.__documents_changed_check():
            chunks = self.__load_and_split_documents()
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            self.__retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
        return self.__retriever
