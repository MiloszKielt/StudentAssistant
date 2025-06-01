# StudentAssistant
This application is a multi-agent assistant aimed towards students mainly as a question answering tool, but also as a exam-style Q&A generator. In the future it might have more functionalities.

## Instalation
This application's dependencies can be installed in two ways:
#### UV (recommended)
```
uv sync
```
This is the recommended approach since [UV](https://docs.astral.sh/uv/) handles all virtual environment creation and resolves all libraries very fast.
#### Pip
```
pip install -r requirements.txt
```

#### Running the program
After installing required dependencies to run the entire program you need to run its 3 seperate elements using these commands (from the base folder of the project):

```
python -m backend.mcp.mcp_server
uvicorn backend.api.api:app --port 8000
streamlit run frontend/app.py
```


## Features