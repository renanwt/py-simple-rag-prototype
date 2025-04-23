# Simple RAG Prototype

## 1. Instructions

### Objective

This project demonstrates a basic Retrieval-Augmented Generation (RAG) system. It accepts a user's natural language question via a command-line interface (CLI), retrieves relevant information from a small local dataset using vector similarity search, and then uses an GoogleAI language model (LLM) to generate a concise answer based *only* on the retrieved context.

### Features

*   **CLI Input:** Simple command-line interface to ask questions.
*   **Data Loading:** Loads data from a CSV file (`data/tax_data.csv`).
*   **Vector Retrieval:**
    *   Uses `sentence-transformers` (`all-MiniLM-L6-v2` model by default) to create embeddings for the data and the user query.
    *   Performs cosine similarity search to find the most relevant text snippets.
*   **LLM Integration:**
    *   Uses the Google AI API (specifically `gemini-1.5-flash` by default) to generate answers.
    *   Prompts the LLM to answer strictly based on the retrieved context.

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/renanwt/simple-rag-prototype.git
    cd simple-rag-prototype
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `sentence-transformers` might download the embedding model on first run, requiring an internet connection.*

4.  **Set Up Google API Key:**
    *   Rename the `.env.example` file to `.env`.
    *   Open the `.env` file and replace `YOUR_API_KEY_HERE` with your actual GoogleAI API key:
      ```dotenv
      GOOGLE_API_KEY=AIzaSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      ```

### Running the Script

Execute the main Python script from your terminal:

```bash
python main.py
```

## 2. Short Summary of Approach

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions based on provided local data sources. Here's the approach:

1.  **Data Loading:** Reads text content from multiple CSV files specified in the configuration (`data/tax_data.csv`, `data/city_services.csv`).
2.  **Embedding:** Uses the `sentence-transformers` library (`all-MiniLM-L6-v2` model) to convert all loaded text snippets into semantic vector embeddings. This happens once at the start.
3.  **Retrieval:** When a user asks a question:
    *   The question is also converted into an embedding vector using the same model.
    *   Cosine similarity is calculated between the question vector and all text snippet vectors.
    *   The text snippets with the highest similarity scores (above a threshold) are selected as the relevant `context`.
4.  **Memory:** A basic conversational memory stores the last few question-answer pairs from the ongoing session.
5.  **Generation:**
    *   A prompt is constructed containing the retrieved `context` and the recent `conversation history`.
    *   This prompt instructs the Google Gemini LLM (`gemini-1.5-flash` accessed via `langchain-google-genai`) to generate an answer *strictly* based on the provided information.
6.  **Interaction:** The system runs as a command-line interface (CLI), handling user input and displaying the LLM's response in a loop.

## 3. What I'd Do Next With More Time

Given more time, I would focus on improvements like:

    *   I'd adapt the logic to make it a deployable API service (using FastAPI) to allow integration with other applications.
    
    *   Implement **document chunking** to handle larger source documents more effectively by breaking them into smaller, more manageable pieces for embedding and retrieval.

    *   Integrate a dedicated **vector database** (e.g., FAISS, ChromaDB, Pinecone) for more efficient storage, indexing, and retrieval, especially as data volume grows.

    *   I'd create an AWS CDK (Cloud Development Kit) package to define and automate the deployment of the necessary cloud infrastructure (e.g., API Gateway, Lambda, possibly a managed vector using RDS).