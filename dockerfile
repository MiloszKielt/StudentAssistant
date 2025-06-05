FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first, then install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install uv streamlit uvicorn supervisor

# Now copy the rest of the code
COPY . .

EXPOSE 8000 8501

# Create supervisord config
RUN echo "[supervisord]\nnodaemon=true\n" \
         "[program:mcp_server]\ncommand=python -m backend.mcp.mcp_server\n" \
         "[program:api]\ncommand=uvicorn backend.api.api:app --port 8000\n" \
         "[program:frontend]\ncommand=streamlit run frontend/app.py\n" \
         > /etc/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisord.conf"]