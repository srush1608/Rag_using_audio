from langchain.prompts import PromptTemplate

def get_audio_prompt():
    """
    Returns the prompt template for audio query responses.
    """
    return PromptTemplate(
        input_variables=["query"],
        template="Query: {query}\nGenerate a detailed response based on the provided audio content."
    )
