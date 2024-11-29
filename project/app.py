from loaders.audio_loaders import load_and_transcribe_audio
from processors.audio_processor import chunk_audio_text
from embeddings.audio_embedding import store_audio_embeddings
from retrievers.audio_retriever import generate_audio_response
from langchain_core.documents.base import Document

def main():
    file_path ="C:/Users/Coditas-Admin/Desktop/rag_audio/project/audio_files/1 Comparison Of Vernacular And Refined Speech.mp3"
    
    print("Loading audio file...")
    transcript = load_and_transcribe_audio(file_path)
    if transcript is None:
        print("Error: No transcript generated.")
        return
    
    print("Processing transcript into chunks...")
    if isinstance(transcript, list):
        transcript = " ".join([t.page_content for t in transcript])
    chunks = chunk_audio_text([transcript]) 
    print(len(chunks))

    print("Storing embeddings...")
    store_audio_embeddings(chunks)

    query = input("Enter your query: ")
    print("Generating response...")

    response = generate_audio_response(query)
    
    print("Generated Response: ", response)

if __name__ == "__main__":
    main()
