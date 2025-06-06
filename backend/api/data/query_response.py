from pydantic import BaseModel

class QueryResponse(BaseModel):
    """Model representing the response to a query message.
    This model is used to encapsulate the answer provided by the assistant agent in response to a user's query.

    Attributes:
        answer (str): The answer to the user's query, which may include information retrieved from various sources or generated by the assistant agent.
    """
    answer: str
