import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_PORT = os.getenv("API_PORT", "8000")
API_HOST = os.getenv("API_HOST", "localhost")

st.title("Student's Assistant")
st.write("This is a simple web application to interact with the Assistant.")
st.write("You can upload your own decuments that will later get processed via RAG to best answer your question.")
st.write("You can ask questions and get answers from the Assistant. If the assistant won't find anything relevant to your question in the uploaded documents, it will try to answer your question using the internet search.")

st.subheader("Upload your files here:")
uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
submit_files_button = st.button("Submit Files")
if submit_files_button:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write("File name:", uploaded_file.name)
            st.write("File type:", uploaded_file.type)
            st.write("File size:", uploaded_file.size)
            try:
                response = requests.post(
                    f"http://{API_HOST}:{API_PORT}/upload", 
                    files={"file": uploaded_file}
                )
                # return response.json()
                st.success(response)
            except ConnectionRefusedError as e:
                st.error("Connection error with backend")
    else:
        st.warning("Please upload at least one file.")

st.subheader("Ask a question:")
question = st.text_input("Enter your question here:")
submit_question_button = st.button("Submit Question")
if submit_question_button:
    if question:
        # Send the question to the backend API
        response = requests.post(f"http://{API_HOST}:{API_PORT}/query", json={"query": question})
        if response.status_code == 200:
            answer = response.json().get("answer")
            st.write("Answer:", answer)
        else:
            st.error("Error: " + response.text)
    else:
        st.warning("Please enter a question.")