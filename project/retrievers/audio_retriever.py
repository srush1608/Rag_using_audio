from langchain_community.vectorstores import PGVector
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")

def retrieve_audio_documents(query: str) -> list:
    try:
        embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="models/embedding-001"
        )

        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator
        )

        retrieved_docs = vectorstore.similarity_search(query, k=5)
        if not retrieved_docs:
            print("No relevant documents found.")
            return None
        
        return retrieved_docs

    except Exception as e:
        print(f"Error in document retrieval: {str(e)}")
        return None

def generate_audio_response(query: str) -> str:
    try:
        retrieved_docs = retrieve_audio_documents(query)
        print("----------------------------------------------")

        print(retrieved_docs)
        print("----------------------------------------------")
        if not retrieved_docs:
            return "Sorry, no relevant information found in the audio content."

        context = " ".join([doc.page_content for doc in retrieved_docs])

        chatgroq = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-8b-8192",
            temperature=0.0,
            max_retries=2
        )

        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""Query: {query}
            Based on the following audio content, generate a detailed response:
            {context}
            """
        )

        llm_chain = prompt_template | chatgroq

        response = llm_chain.invoke({"query": query, "context": context})

        print(f"Generated response: {response}")
        return response

    except Exception as e:
        print(f"Error in response generation: {str(e)}")
        return "An error occurred while processing your query."
