# Simple RAG Prototype

## Objective

This project demonstrates a basic Retrieval-Augmented Generation (RAG) system. It accepts a user's natural language question via a command-line interface (CLI), retrieves relevant information from a small local dataset using vector similarity search, and then uses an GoogleAI language model (LLM) to generate a concise answer based *only* on the retrieved context.

## Features

*   **CLI Input:** Simple command-line interface to ask questions.
*   **Data Loading:** Loads data from a CSV file (`data/tax_data.csv`).
*   **Vector Retrieval:**
    *   Uses `sentence-transformers` (`all-MiniLM-L6-v2` model by default) to create embeddings for the data and the user query.
    *   Performs cosine similarity search to find the most relevant text snippets.
*   **LLM Integration:**
    *   Uses the Google AI API (specifically `gemini-1.5-flash` by default) to generate answers.
    *   Prompts the LLM to answer strictly based on the retrieved context.

## Setup

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

## Running the Script

Execute the main Python script from your terminal:

```bash
python main.py