from pydantic import BaseModel

class QueryMessage(BaseModel):
    """Model representing a query message sent by the user.
    This model is used to encapsulate the query string that the user wants to ask.

    Attributes:
        query (str): The question or query string that the user wants to ask. 
    """
    query: str