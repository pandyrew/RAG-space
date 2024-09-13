# RAG-space

RAG-space is a Retrieval-Augmented Generation (RAG) model that combines document retrieval and language generation to provide informative responses based on a custom knowledge base.

## Project Description

This project implements a RAG model using Python, FastAPI, LangChain, Llama 2 (via Ollama), FAISS, and SentenceTransformers. It retrieves relevant information from a local knowledge base and generates responses using the Llama 2 language model.

For more detailed technical information about the project, please refer to the `info.txt` file.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/RAG-space.git
   cd RAG-space
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Ollama:
   Follow the instructions at [Ollama's official website](https://ollama.ai/download) to install Ollama for your operating system.

4. Pull the Llama 2 model using Ollama:
   ```
   ollama pull llama2
   ```

## Running the Server

1. Start the Ollama service (if not already running):
   ```
   ollama serve
   ```

2. Run the FastAPI server:
   ```
   python app.py
   ```

The server will start on `http://localhost:8000`.

## Getting a Response

To get a response from the RAG model, send a POST request to the `/generate` endpoint:

curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"text": "What color is Mars?"}'

This will return a JSON response containing the generated answer and the retrieved documents used for context.
