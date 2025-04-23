import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from typing import List, Tuple

# --- Configuration ---
DATA_FILEPATHS = ["data/tax_data.csv", "data/city_services.csv"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.1
MAX_HISTORY_TURNS = 4


def load_data(filepaths: List[str]) -> pd.DataFrame:
    """Loads data from a list of CSV files."""
    all_dfs = []
    print(f"Loading data from: {', '.join(filepaths)}")
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath)
            if 'text' not in df.columns:
                print(f"Warning: CSV '{filepath}' is missing 'text' column. Skipping.")
                continue
            df['text'] = df['text'].fillna('')
            df['source_file'] = os.path.basename(filepath)
            all_dfs.append(df)
            print(f" - Loaded {len(df)} rows from {filepath}")
        except FileNotFoundError:
            print(f"Error: Data file not found at {filepath}. Skipping.")
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}. Skipping.")

    if not all_dfs:
        print("Error: No valid data loaded. Exiting.")
        exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(combined_df)}")
    return combined_df

def create_embeddings(texts: list[str], model: SentenceTransformer):
    """Creates embeddings for a list of texts using the specified model."""
    print(f"Creating embeddings for {len(texts)} text snippets using {EMBEDDING_MODEL}...")
    start_time = time.time()
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    end_time = time.time()
    print(f"Embeddings created in {end_time - start_time:.2f} seconds")
    return embeddings

def find_relevant_context(query: str, corpus_embeddings, corpus: list[str], model: SentenceTransformer, top_k: int, threshold: float) -> str:
    """Finds the most relevant text snippets based on semantic similarity"""
    if not query:
        return ""
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu().numpy()

    top_indices = np.argsort(cos_scores)[-top_k:][::-1]

    relevant_indices = [idx for idx in top_indices if cos_scores[idx] >= threshold]

    if not relevant_indices:
        print(f"No context found above similarity threshold {threshold}")
        return ""

    context_snippets = [corpus[idx] for idx in relevant_indices]
    context = "\n---\n".join(context_snippets)

    # print(f"\n--- Retrieved Context (Top {len(relevant_indices)} Snippets, Threshold={threshold}) ---")
    # print("...") 
    # print("--- End Context ---")
    return context

def format_history(history: List[Tuple[str, str]]) -> str:
    """Formats conversation history for the prompt."""
    if not history:
        return ""
    formatted = "\nPrevious conversation turns:\n"
    for q, a in history:
        formatted += f"User: {q}\nAssistant: {a}\n"
    return formatted.strip()

def ask_llm(context: str, question: str, model_name: str, api_key: str, history: List[Tuple[str, str]] | None = None) -> str:
    """Asks the LLM a question based on the provided context and optional history."""

    history_string = format_history(history) if history else ""

    if not context and not history_string:
         return "I have no context or conversation history to answer this question."
    elif not context:
        prompt = f"""{history_string}

                Answer the following question based on the previous conversation turns. If the previous turns do not contain the answer, say "I cannot answer this question based on the previous conversation or the documents I have access to."

                Question:
                {question}

            Answer:"""
    else:
        prompt = f"""{history_string}

                Answer the following question based *only* on the provided context below. If the context does not contain the answer, say "I cannot answer this question based on the provided context."

                Context:
                {context}

                Question:
                {question}

            Answer:"""

    print("\n--- Sending Prompt to LLM ---")
    # print(prompt) # Uncomment to see the full prompt being sent
    # print("...")
    # print("--- End Prompt ---")

    try:
        chat = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=api_key
        )

        response = chat.invoke([
            SystemMessage(content="You are a helpful assistant. Answer questions concisely based *only* on the provided context or previous conversation turns, as instructed in the prompt. Do not add information not present in the provided text."),
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

    dataframe = load_data(DATA_FILEPATHS)
    corpus = dataframe['text'].tolist()

    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading sentence transformer model '{EMBEDDING_MODEL}': {e}")
        print("Ensure you have internet connectivity for the first download.")
        exit(1)

    corpus_embeddings = create_embeddings(corpus, embedding_model)

    print("\n--- Simple RAG CLI ---")
    print("Ask a question about local taxes or city services. Type 'quit' or 'exit' to stop.")

    conversation_history: List[Tuple[str, str]] = []

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
            top_k=TOP_K,
            threshold=SIMILARITY_THRESHOLD
        )

        # 2. Generation
        history_for_prompt = conversation_history[-MAX_HISTORY_TURNS:]

        llm_answer = ask_llm(
            retrieved_context,
            user_question,
            LLM_MODEL,
            API_KEY,
            history=history_for_prompt
        )

        print("\nLLM Answer:")
        print(llm_answer)

        if not llm_answer.startswith("Sorry, an error was found"):
             conversation_history.append((user_question, llm_answer))


if __name__ == "__main__":
    main()
