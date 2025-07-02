import os
from dotenv import load_dotenv
import ollama

load_dotenv()

class LLM:
    def get_response(self, text):
        raise NotImplementedError("Subclasses should implement this method")

class LocalLLM(LLM):
    def __init__(self, model='llama2'):
        self.model = model
        try:
            ollama.list()
        except Exception:
            raise ConnectionError("Ollama server not running. Please start it with 'ollama serve'")

    def get_response(self, text):
        response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': text}])
        return response['message']['content']

def get_llm(llm_type="local", model='llama2'):
    if llm_type == "local":
        return LocalLLM(model=model)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
