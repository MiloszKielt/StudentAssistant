from backend.core.agents.assistant_agent import AssistantAgent
from backend.core.models_provider import LLMFactory, EmbeddingFactory
from dotenv import load_dotenv

load_dotenv()

assistant = AssistantAgent(LLMFactory.openai(), EmbeddingFactory.openai())


print(type(assistant.invoke("What is the capital of poland")))