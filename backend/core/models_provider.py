import os
from abc import ABC
from typing import Optional

from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .validation_methods import validate_string

OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"


class LLMFactory(ABC):
    @staticmethod
    def __validateTemperature(temperature: float):
        return isinstance(temperature, (float, int)) and (temperature >= 0 and temperature <= 1)
    
    @staticmethod
    def openai(model: Optional[str] = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
        """Provides an OpenAI LLM model according to given parameters.

        Args:
            model (str, optional): model for LLM. Defaults to "gpt-4o-mini".
            temperature (float, optional): base temperature of the LLM. Defaults to 0.

        Raises:
            ValueError: No OPENAI_API_KEY provided in .env!
            ValueError: Model name must be a nonempty string!
            ValueError: Temperature must be in range [0, 1]!

        Returns:
            ChatOpenAI: OpenAI LLM model
        """
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please provide OPENAI_API_KEY to the .env file!")
        
        if not validate_string(model):
            raise ValueError("Model name must be a nonempty string!")
        
        if not LLMFactory.__validateTemperature(temperature):
            raise ValueError("Temperature must be in range [0, 1]!")
        
        return ChatOpenAI(api_key=openai_api_key, model=model, temperature=temperature)
    
    @staticmethod
    def ollama(model: str = "llama3.2", temperature: float = 0) -> ChatOllama:
        """Provides an Ollama LLM model according to the given parameters.

        Args:
            model (str, optional): model for LLM. Defaults to "llama3.2".
            temperature (float, optional): base temperature of the LLM. Defaults to 0.

        Raises:
            ValueError: Model name must be a nonempty string!
            ValueError: Temperature must be in range [0, 1]!

        Returns:
            ChatOllama: Ollama LLM model
        """
        if not validate_string(model):
            raise ValueError("Model must be provided as a nonempty string!")
        
        if not LLMFactory.__validateTemperature(temperature):
            raise ValueError("Temperature must be in range [0, 1]!")
        
        return ChatOllama(model=model, temperature=temperature)
    
class EmbeddingFactory(ABC):
    @staticmethod
    def openai() -> OpenAIEmbeddings:
        """Provides OpenAI embedding model

        Raises:
            ValueError: Please provide OPENAI_API_KEY to the .env file!

        Returns:
            OpenAIEmbeddings: OpenAI embedding model
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please provide OPENAI_API_KEY to the .env file!")
        
        return OpenAIEmbeddings(api_key=openai_api_key, model=OPENAI_EMBEDDING_MODEL)
    
    @staticmethod
    def huggingface(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = False,
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        encode_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None
    ) -> HuggingFaceEmbeddings:
        """Provides a HuggingFace embedding model according to given parameters.

        Args:
            model_name (str, optional): Name or path of the HuggingFace model. 
                Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            normalize (bool, optional): Whether to normalize embeddings. Defaults to False.
            device (str, optional): Device to run the model on (e.g., "cpu", "cuda"). 
                Defaults to "cpu".
            cache_folder (str, optional): Path to cache folder for the model. 
                Defaults to None.
            encode_kwargs (dict, optional): Additional kwargs for encoding. 
                Defaults to None (will use normalize_embeddings).
            model_kwargs (dict, optional): Additional kwargs for model initialization. 
                Defaults to None (will use device).

        Raises:
            ValueError: Model name must be a nonempty string!
            ValueError: Device must be a nonempty string!
            ValueError: Cache folder path must be a string if provided!

        Returns:
            HuggingFaceEmbeddings: HuggingFace embedding model
        """
        if not validate_string(model_name):
            raise ValueError("Model name must be a nonempty string!")
        
        if not validate_string(device) or device != 'cpu':
            raise ValueError("Device must be a nonempty string!")
        
        if cache_folder is not None and not isinstance(cache_folder, str):
            raise ValueError("Cache folder path must be a string if provided!")

        if model_kwargs is None:
            model_kwargs = {'device': device}
        if encode_kwargs is None:
            encode_kwargs = {'normalize_embeddings': normalize}

        kwargs = {
            'model_name': model_name,
            'model_kwargs': model_kwargs,
            'encode_kwargs': encode_kwargs
        }
        
        if cache_folder:
            kwargs['cache_folder'] = cache_folder

        return HuggingFaceEmbeddings(**kwargs)