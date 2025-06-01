FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install uv streamlit uvicorn supervisor

# Expose ports for API (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Create supervisord config
RUN echo "[supervisord]\nnodaemon=true\n" \
         "[program:mcp_server]\ncommand=python -m backend.mcp.mcp_server\n" \
         "[program:api]\ncommand=uvicorn backend.api.api:app --port 8000\n" \
         "[program:frontend]\ncommand=streamlit run frontend/app.py\n" \
         > /etc/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisord.conf"]