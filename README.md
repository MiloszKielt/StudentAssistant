# Overview

This application is a multi-agent assistant aimed towards students mainly as a question answering tool, but also as a exam-style Q&A generator. In the future it might have more functionalities.

# Setting up the project

## Installation

```bash
# Clone repository
git clone https://github.com/MiloszKielt/StudentAssistant.git
cd study-assistant

# Install dependencies
uv sync # recommended
#### OR
pip install -r requirements.txt

# Set environment variables
cp .env.example .env  # Edit with your API keys
```

## Configuration

Edit `/backend/config.py` and `.env` for:

- API keys (OpenAI, Tavily, etc.)
- File upload paths
- Model parameters

# Running the project

In order to run the project, you can use either the ready docker-compose file or run the separate instances (API, MCP server and streamlit app) separately

## Docker

To run the project through docker you simply need to execute:

```bash
docker-compose up --build
```

## Python

To run the project from terminal commands, you will need to execute them in separate terminal windows. They should be executed from the root directory of the repository

MCP

```bash
python -m backend.mcp.mcp_server
#### OR
uv run -m backend.mcp.mcp_server # when using uv
```

Backend API

```bash
uvicorn backend.api.api:app --port 8000
#### OR
uv run uvicorn backend.api.api:app --port 8000 # when using uv
```

Streamlit app

```bash
streamlit run frontend/app.py
#### OR
uv run streamlit run frontend/app.py # when using uv
```

# System architecture

## Connections architecture
![connections between different parts of architecture](https://github.com/user-attachments/assets/30beba9c-ad50-42fa-8cc4-70b6c79f8f1a)


## Folder structure

```
StudentAssistant
├── .gitignore
├── .python-version
├── README.md
├── docker-compose.yml
├── dockerfile
├── pyproject.toml
├── requirements.txt
│
├── backend/
│   ├── config.py                # Main configuration
│   │
│   ├── api/
│   │   ├── api.py               # FastAPI endpoints
│   │   ├── mcp_client.py        # MCP service connector
│   │   │
│   │   ├── agents/
│   │   │   ├── RAG/
│   │   │   │   ├── rag_agent.py       # RAG processor
│   │   │   │   └── vector_store.py    # FAISS/Chroma integration
│   │   │   │
│   │   │   └── assistant/
│   │   │       ├── assistant_agent.py  # Primary interface
│   │   │       ├── decision_agent.py   # Routing logic
│   │   │       ├── summarize_agent.py  # Content condensation
│   │   │       └── task_planner.py     # Workflow orchestration
│   │   │
│   │   └── data/
│   │       ├── query_message.py  # Request Pydantic model
│   │       └── query_response.py # Response Pydantic model
│   │
│   ├── core/
│   │   ├── agents/
│   │   │   └── base_agent.py    # Abstract agent class
│   │   │
│   │   ├── models_provider.py   # LLM initialization
│   │   └── validation_methods.py# Data validators
│   │
│   └── mcp/
│       ├── agents/
│       │   ├── exam_question_agent.py  # Test generator
│       │   └── web_search_agent.py     # Web augmentation tool
│       │
│       └── mcp_server.py        # Model Context Protocol server
│
├── frontend/
│   └── app.py                   # Streamlit UI
│
└── storage/
    ├── uploads/                 # User-uploaded documents
    └── vector_db/               # Generated embeddings (FAISS/Chroma)
```

## API Endpoints

The backend API exposes two accessible endpoints:

`/upload`  - EP used for uploading files to the local RAG database.

`/query` - EP used to converse with the models and ask questions.

# Usage examples

## Through Streamlit application
![streamlit UI](https://github.com/user-attachments/assets/03f889d3-b953-4c05-a34c-76130071b615)

## Through Swagger UI
![swagger UI](https://github.com/user-attachments/assets/61f59250-c6d5-4e9d-a03a-b59df89956a4)

