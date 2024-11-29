from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
import os
from dotenv import load_dotenv

load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
audio_file_path = "C:/Users/Coditas-Admin/Desktop/rag_audio/project/audio_files/1 Comparison Of Vernacular And Refined Speech.mp3"
    

def load_and_transcribe_audio(file_path):
    """
    Load and transcribe an audio file using AssemblyAI.
    """
    if not os.path.exists(file_path):
        print(f"Error: The audio file '{file_path}' does not exist.")
        return None

    try:
        print(f"Loading audio file: {file_path}")
        loader = AssemblyAIAudioTranscriptLoader(
            file_path=file_path, api_key=ASSEMBLYAI_API_KEY
        )

        transcript = loader.load()
        if transcript:
            print("Transcript generated successfully.")
        else:
            print("Transcript generation failed.")
        return transcript
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None
