from langchain_core.language_models.chat_models import BaseChatModel


def validate_string(string: str):
    return isinstance(string, str) and len(string.strip()) > 0

def validate_llm(llm: BaseChatModel):
    return isinstance(llm, BaseChatModel)