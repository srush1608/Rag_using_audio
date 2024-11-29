from langchain.text_splitter import CharacterTextSplitter

def chunk_audio_text(documents: list, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split the transcribed audio text into smaller chunks for embedding.
    """
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(" ".join(documents))  
    print(f"Generated {len(chunks)} chunks from audio documents.")
    return chunks
