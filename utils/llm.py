"""
LLM module with memory integration.

Supports:
- Local LLM via Ollama
- Multi-layer memory context
- Conversation history
"""
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import ollama

load_dotenv()


class LLM:
    """Base LLM class."""
    
    def get_response(self, text: str, context: str = None, history: List[Dict] = None) -> str:
        raise NotImplementedError("Subclasses should implement this method")


class LocalLLM(LLM):
    """Local LLM using Ollama."""
    
    SYSTEM_PROMPT = """You are a helpful voice assistant. You have access to conversation history and memory.

Guidelines:
- Be concise but helpful (this is voice, not text)
- Remember and reference past conversations when relevant
- Learn user preferences and adapt
- If the user refers to something from the past, try to recall it
- Be natural and conversational

{context}"""
    
    def __init__(self, model: str = 'llama2'):
        """
        Initialize local LLM.
        
        Args:
            model: Ollama model name (e.g., 'llama2', 'llama3', 'mistral')
        """
        self.model = model
        try:
            ollama.list()
        except Exception:
            raise ConnectionError(
                "Ollama server not running. Please start it with 'ollama serve'"
            )
    
    def get_response(
        self,
        text: str,
        context: str = None,
        history: List[Dict[str, str]] = None
    ) -> str:
        """
        Get response from LLM with context and history.
        
        Args:
            text: Current user message
            context: Additional context (from memory search, user profile, etc.)
            history: List of previous messages [{'role': 'user/assistant', 'content': '...'}]
            
        Returns:
            Assistant's response
        """
        # Build messages list
        messages = []
        
        # System prompt with context
        system_content = self.SYSTEM_PROMPT.format(
            context=f"\n{context}" if context else ""
        )
        messages.append({'role': 'system', 'content': system_content})
        
        # Add conversation history
        if history:
            for msg in history[-10:]:  # Last 10 messages for context window
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Add current message
        messages.append({'role': 'user', 'content': text})
        
        # Get response
        response = ollama.chat(model=self.model, messages=messages)
        return response['message']['content']
    
    def summarize_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Use LLM to summarize a conversation segment.
        
        Args:
            messages: List of messages to summarize
            
        Returns:
            Dict with 'summary', 'topics', and 'user_preferences'
        """
        conversation = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in messages
        ])
        
        prompt = f"""Summarize this conversation segment. Extract:
1. A brief summary (1-2 sentences)
2. Main topics discussed (list of keywords)
3. Any user preferences or personal information learned

Conversation:
{conversation}

Respond in this exact format:
SUMMARY: <your summary>
TOPICS: <comma-separated topics>
USER_INFO: <comma-separated user preferences/info, or "none">"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Parse response
        result = {
            'summary': '',
            'topics': [],
            'user_preferences': []
        }
        
        for line in response['message']['content'].split('\n'):
            line = line.strip()
            if line.startswith('SUMMARY:'):
                result['summary'] = line[8:].strip()
            elif line.startswith('TOPICS:'):
                topics = line[7:].strip()
                result['topics'] = [t.strip() for t in topics.split(',') if t.strip()]
            elif line.startswith('USER_INFO:'):
                info = line[10:].strip()
                if info.lower() != 'none':
                    result['user_preferences'] = [i.strip() for i in info.split(',') if i.strip()]
        
        return result
    
    def extract_user_info(self, text: str) -> Dict[str, str]:
        """
        Extract user information from a message.
        
        Args:
            text: User message
            
        Returns:
            Dict of extracted info (e.g., {'name': 'John', 'location': 'New York'})
        """
        prompt = f"""Extract any personal information from this message.
Look for: name, location, age, occupation, interests, preferences.

Message: "{text}"

If found, respond with key: value pairs, one per line.
If nothing found, respond with "NONE"."""
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        result = {}
        content = response['message']['content'].strip()
        
        if content.upper() != 'NONE':
            for line in content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip().lower()] = value.strip()
        
        return result


def get_llm(llm_type: str = "local", model: str = 'llama2') -> LLM:
    """
    Get an LLM instance.
    
    Args:
        llm_type: Type of LLM ('local' for Ollama)
        model: Model name
        
    Returns:
        LLM instance
    """
    if llm_type == "local":
        return LocalLLM(model=model)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
