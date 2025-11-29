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
    
    SYSTEM_PROMPT = """You are a helpful voice assistant with memory of past conversations.

IMPORTANT RULES:
- Keep responses SHORT: 1-2 sentences maximum (this is voice, not text)
- NO emojis, NO filler words like "Ah, great!", "I see!", etc.
- Be direct and helpful
- Use memory context below to answer questions about past conversations
- If asked about past conversations, reference the context directly

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
        
        # Format context (only include past context, not recent messages which are in history)
        context_text = ""
        if context:
            # Filter out "Recent Conversation" section - that's already in history
            context_lines = context.split('\n')
            filtered_lines = []
            skip_section = False
            for line in context_lines:
                if "=== Recent Conversation ===" in line:
                    skip_section = True
                    continue
                elif line.startswith("===") and skip_section:
                    skip_section = False
                    filtered_lines.append(line)
                elif not skip_section:
                    filtered_lines.append(line)
            context_text = "\n".join(filtered_lines).strip()
        
        # System prompt with context
        system_content = self.SYSTEM_PROMPT.format(
            context=f"\n\nMemory Context:\n{context_text}" if context_text else ""
        )
        messages.append({'role': 'system', 'content': system_content})
        
        # Add conversation history (last 8 messages to leave room for context)
        if history:
            for msg in history[-8:]:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Add current message - be direct
        messages.append({'role': 'user', 'content': text})
        
        # Get response
        response = ollama.chat(model=self.model, messages=messages)
        response_text = response['message']['content'].strip()
        
        # Post-process: remove verbose patterns and emojis
        verbose_patterns = [
            "Ah, ", "I see! ", "Great! ", "Of course! ", "Well, ", 
            "You know, ", "Actually, ", "Basically, ", "So, ",
            "In our previous conversation, ", "As we discussed, ",
            "As I mentioned, ", "Let me tell you, "
        ]
        for pattern in verbose_patterns:
            if response_text.startswith(pattern):
                response_text = response_text[len(pattern):].strip()
        
        # Remove emojis
        import re
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        response_text = emoji_pattern.sub('', response_text)
        
        # Remove excessive punctuation
        response_text = response_text.replace('...', '.')
        response_text = response_text.replace('!!', '!')
        response_text = response_text.replace('??', '?')
        
        # Limit length (150 characters for voice - roughly 2 sentences)
        if len(response_text) > 150:
            # Try to cut at sentence boundary
            sentences = response_text.split('. ')
            shortened = []
            total_len = 0
            for sent in sentences[:2]:  # Max 2 sentences
                sent = sent.strip()
                if not sent:
                    continue
                if total_len + len(sent) > 150:
                    break
                shortened.append(sent)
                total_len += len(sent) + 2
            if shortened:
                response_text = '. '.join(shortened)
                if not response_text.endswith(('.', '!', '?')):
                    response_text += '.'
            else:
                # Fallback: cut at word boundary
                words = response_text.split()
                response_text = ' '.join(words[:25])  # ~25 words max
                if not response_text.endswith(('.', '!', '?')):
                    response_text += '...'
        
        return response_text.strip()
    
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
