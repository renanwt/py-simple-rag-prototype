import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

DATA_FILEPATH = "data/tax_data.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"
TOP_K = 2


def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV File"""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        # Ensure 'text column exists
        if 'text' not in df.columns:
            raise ValueError("CSV must contain a 'text' column.")
        # Handle potential missing values in 'text' column if necessary
        df['text'] = df['text'].fillna('')
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def create_embeddings(texts: list[str], model: SentenceTransformer):
    """Creates embeddings for a list of texts using the specified model."""
    print(f"Creating embeddings using {EMBEDDING_MODEL}...")
    start_time = time.time()
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    end_time = time.time()
    print(f"Embeddings created in {end_time - start_time:.2f} seconds")
    return embeddings

def find_relevant_context(query: str, corpus_embeddings, corpus: list[str], model: SentenceTransformer, top_k: int) -> str:
    """Finds the most relevant text snippets based on semantic similarity"""
    if not query:
        return ""
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    top_results = np.argpartition(-cos_scores.cpu().numpy(), range(top_k))[0:top_k]

    similarity_threshold = 0.7

    relevant_indices = [idx for idx in top_results if cos_scores[idx] > similarity_threshold]

    if not relevant_indices:
        print("No sufficiently relevant context found")
        return ""
    
    context = "\n --- \n".join([corpus[idx] for idx in relevant_indices])
    # print(f"\n--- Retrieved Context (Top {len(relevant_indices)} Snippets) ---")
    # print(context)
    # print("--- End ---")
    return context

def ask_llm(context: str, question: str, model_name: str, api_key: str) -> str:
    """Asks the LLM a question based on the provided context."""
    if not context:
        return "I couldn't find relevant information in the provided data to answer your question"
    
    prompt = f"""Answer the following question based only on the provided context. If the context does not contain the answer, say "I cannot answer this question based on the provided context. "
                Context:
                    {context}
                Question:
                    {question}
                Answer:"""
    print("\n--- Sending Prompt to LLM ---")
    # print(prompt)
    # print("--- End Prompt ---")

    try:
        chat = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            google_api_key=api_key
        )

        response = chat.invoke([
            SystemMessage(content="You are a helpful assistant that answers questions strictly based on the context provided."),
            HumanMessage(content=prompt)
        ])

        answer = response.content.strip()
        return answer
    except Exception as e:
        print(f"Error interacting with GoogleAI API: {e}")
        return "Sorry, an error was found trying to generate an answer."

def main():
    """Main function to run the RAG CLI."""

    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found. Please set it in the .env file.")
        exit(1)
    
    dataframe = load_data(DATA_FILEPATH)
    corpus = dataframe['text'].tolist()

    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading sentence transformer model '{EMBEDDING_MODEL}': {e}")
        print("Ensure you have internet connectivity for the first download.")
        exit(1)

    corpus_embeddings = create_embeddings(corpus, embedding_model)

    print("\n--- Simple RAG CLI ---")
    print("Ask a question about the loaded data. Type 'quit' or 'exit' to stop.")

    while True:
        user_question = input("\nYour Question: ").strip()
        if user_question.lower() in ['quit', 'exit']:
            break
        if not user_question:
            continue

        # 1. Retrieval
        retrieved_context = find_relevant_context(
            user_question,
            corpus_embeddings,
            corpus,
            embedding_model,
            top_k=TOP_K
        )

        # 2. Generation
        llm_answer = ask_llm(
            retrieved_context,
            user_question,
            LLM_MODEL,
            API_KEY
        )

        print("\nLLM Answer:")
        print(llm_answer)

if __name__ == "__main__":
    main()