import os
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def store_audio_embeddings(texts):
    """
    Store the audio embeddings in the PostgreSQL vector store.
    """
    try:
        embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model="models/embedding-001"
        )

        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator
        )

        vectorstore.add_texts(texts)
        print("Data successfully stored in the vectorstore table.")
    except Exception as e:
        print(f"Error storing embeddings in PGVector: {str(e)}")
