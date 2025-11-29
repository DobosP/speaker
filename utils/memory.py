"""
Multi-layer memory system with PostgreSQL and vector search.

Architecture:
- Layer 1: Recent messages (short-term, last N messages)
- Layer 2: Conversation summaries (medium-term, condensed history)
- Layer 3: Vector embeddings (long-term, semantic search via pgvector)

This allows the assistant to:
1. Remember recent conversation context
2. Know user preferences and patterns from summaries
3. Search and recall specific past topics semantically
"""
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Database
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass  
class ConversationSummary:
    """Summary of a conversation segment."""
    summary: str
    topics: List[str]
    user_preferences: List[str]
    start_time: datetime
    end_time: datetime
    message_count: int


class MemoryManager:
    """
    Multi-layer memory manager with PostgreSQL backend.
    
    Layers:
    1. Recent messages - Full conversation context (last N messages)
    2. Summaries - Condensed conversation history
    3. Vector store - Semantic search over all messages
    """
    
    # SQL for creating tables
    CREATE_TABLES_SQL = """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Messages table (all conversations)
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(64) NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        embedding vector(384)  -- for all-MiniLM-L6-v2
    );
    
    -- Summaries table (condensed history)
    CREATE TABLE IF NOT EXISTS summaries (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(64),
        summary TEXT NOT NULL,
        topics TEXT[],
        user_preferences TEXT[],
        start_time TIMESTAMPTZ,
        end_time TIMESTAMPTZ,
        message_count INT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        embedding vector(384)
    );
    
    -- User profile (learned preferences)
    CREATE TABLE IF NOT EXISTS user_profile (
        id SERIAL PRIMARY KEY,
        key VARCHAR(255) UNIQUE NOT NULL,
        value TEXT NOT NULL,
        confidence FLOAT DEFAULT 1.0,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Indexes for fast lookup
    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
    
    -- Vector indexes for semantic search (using IVFFlat for speed)
    CREATE INDEX IF NOT EXISTS idx_messages_embedding ON messages 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_summaries_embedding ON summaries 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """
    
    def __init__(
        self,
        db_url: str = None,
        session_id: str = None,
        max_recent_messages: int = 20,
        max_context_tokens: int = 2000,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the memory manager.
        
        Args:
            db_url: PostgreSQL connection URL (or use DATABASE_URL env var)
            session_id: Unique session identifier (auto-generated if None)
            max_recent_messages: Max messages to keep in short-term memory
            max_context_tokens: Approx max tokens before summarizing
            embedding_model: Sentence transformer model for embeddings
        """
        self.db_url = db_url or os.getenv('DATABASE_URL', 'postgresql://localhost/voice_assistant')
        self.session_id = session_id or self._generate_session_id()
        self.max_recent_messages = max_recent_messages
        self.max_context_tokens = max_context_tokens
        
        # In-memory recent messages (Layer 1)
        self.recent_messages: List[Message] = []
        
        # Database connection
        self.conn = None
        self._db_available = False
        
        # Embedding model
        self.embedder = None
        self._embeddings_available = False
        
        # Initialize
        self._init_database()
        self._init_embeddings(embedding_model)
        self._load_recent_messages()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def _init_database(self):
        """Initialize database connection and tables."""
        if not POSTGRES_AVAILABLE:
            print("⚠️  psycopg2 not available. Memory will be in-memory only.")
            print("   Install with: pip install psycopg2-binary")
            return
        
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = True
            
            # Create tables
            with self.conn.cursor() as cur:
                cur.execute(self.CREATE_TABLES_SQL)
            
            self._db_available = True
            print(f"✅ Database connected (session: {self.session_id[:8]}...)")
        except Exception as e:
            print(f"⚠️  Database connection failed: {e}")
            print("   Memory will be in-memory only for this session.")
    
    def _init_embeddings(self, model_name: str):
        """Initialize the embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            print("⚠️  sentence-transformers not available. Semantic search disabled.")
            print("   Install with: pip install sentence-transformers")
            return
        
        try:
            print(f"🔄 Loading embedding model: {model_name}...")
            self.embedder = SentenceTransformer(model_name)
            self._embeddings_available = True
            print("✅ Embedding model loaded!")
        except Exception as e:
            print(f"⚠️  Failed to load embedding model: {e}")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text."""
        if not self._embeddings_available or not self.embedder:
            return None
        return self.embedder.encode(text, convert_to_numpy=True)
    
    def _load_recent_messages(self):
        """Load recent messages from database."""
        if not self._db_available:
            return
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT role, content, timestamp 
                    FROM messages 
                    WHERE session_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (self.session_id, self.max_recent_messages))
                
                rows = cur.fetchall()
                self.recent_messages = [
                    Message(
                        role=row['role'],
                        content=row['content'],
                        timestamp=row['timestamp']
                    )
                    for row in reversed(rows)  # Oldest first
                ]
        except Exception as e:
            print(f"⚠️  Failed to load recent messages: {e}")
    
    def add_message(self, role: str, content: str) -> Message:
        """
        Add a message to memory.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            The created Message object
        """
        message = Message(role=role, content=content)
        
        # Add to recent messages (Layer 1)
        self.recent_messages.append(message)
        
        # Trim if too many
        if len(self.recent_messages) > self.max_recent_messages:
            self.recent_messages = self.recent_messages[-self.max_recent_messages:]
        
        # Save to database with embedding
        if self._db_available:
            embedding = self._get_embedding(content)
            self._save_message_to_db(message, embedding)
        
        # Check if we need to summarize
        self._check_and_summarize()
        
        return message
    
    def _save_message_to_db(self, message: Message, embedding: Optional[np.ndarray]):
        """Save message to database."""
        try:
            with self.conn.cursor() as cur:
                if embedding is not None:
                    cur.execute("""
                        INSERT INTO messages (session_id, role, content, timestamp, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        self.session_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        embedding.tolist()
                    ))
                else:
                    cur.execute("""
                        INSERT INTO messages (session_id, role, content, timestamp)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        self.session_id,
                        message.role,
                        message.content,
                        message.timestamp
                    ))
        except Exception as e:
            print(f"⚠️  Failed to save message: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (words * 1.3)."""
        return int(len(text.split()) * 1.3)
    
    def _check_and_summarize(self):
        """Check if context is too long and create summary if needed."""
        total_tokens = sum(
            self._estimate_tokens(m.content) 
            for m in self.recent_messages
        )
        
        if total_tokens > self.max_context_tokens and len(self.recent_messages) > 10:
            # Summarize older messages
            self._create_summary()
    
    def _create_summary(self):
        """Create a summary of older messages."""
        if len(self.recent_messages) < 10:
            return
        
        # Take older half of messages for summarization
        messages_to_summarize = self.recent_messages[:len(self.recent_messages)//2]
        
        # Create summary text (will be improved with LLM)
        conversation_text = "\n".join([
            f"{m.role}: {m.content}" 
            for m in messages_to_summarize
        ])
        
        # Extract key info (basic version - can be enhanced with LLM)
        topics = self._extract_topics(conversation_text)
        
        summary = ConversationSummary(
            summary=f"Conversation with {len(messages_to_summarize)} messages about: {', '.join(topics[:5])}",
            topics=topics,
            user_preferences=[],
            start_time=messages_to_summarize[0].timestamp,
            end_time=messages_to_summarize[-1].timestamp,
            message_count=len(messages_to_summarize)
        )
        
        # Save summary to database
        if self._db_available:
            self._save_summary_to_db(summary)
        
        # Remove summarized messages from recent
        self.recent_messages = self.recent_messages[len(messages_to_summarize):]
        
        print(f"📝 Created summary of {summary.message_count} messages")
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text (simple keyword extraction)."""
        # Simple approach - can be enhanced with NLP
        words = text.lower().split()
        # Filter common words and get unique
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'as', 'by', 'about', 'what', 'how', 'when', 'where', 'why', 'who', 'this', 'that', 'these', 'those', 'can', 'could', 'would', 'should', 'will', 'have', 'has', 'had', 'do', 'does', 'did', 'be', 'been', 'being', 'user:', 'assistant:'}
        filtered = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Count frequency
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        
        # Return top topics
        sorted_topics = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_topics[:10]]
    
    def _save_summary_to_db(self, summary: ConversationSummary):
        """Save summary to database."""
        try:
            embedding = self._get_embedding(summary.summary)
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO summaries 
                    (session_id, summary, topics, user_preferences, start_time, end_time, message_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.session_id,
                    summary.summary,
                    summary.topics,
                    summary.user_preferences,
                    summary.start_time,
                    summary.end_time,
                    summary.message_count,
                    embedding.tolist() if embedding is not None else None
                ))
        except Exception as e:
            print(f"⚠️  Failed to save summary: {e}")
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory using semantic similarity.
        
        Args:
            query: Search query
            limit: Max results to return
            
        Returns:
            List of relevant messages/summaries
        """
        if not self._db_available or not self._embeddings_available:
            return []
        
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []
        
        results = []
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Search messages
                cur.execute("""
                    SELECT role, content, timestamp, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM messages 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding.tolist(), query_embedding.tolist(), limit))
                
                for row in cur.fetchall():
                    results.append({
                        'type': 'message',
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'similarity': row['similarity']
                    })
                
                # Search summaries
                cur.execute("""
                    SELECT summary, topics, start_time, end_time,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM summaries 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding.tolist(), query_embedding.tolist(), limit))
                
                for row in cur.fetchall():
                    results.append({
                        'type': 'summary',
                        'content': row['summary'],
                        'topics': row['topics'],
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'similarity': row['similarity']
                    })
        except Exception as e:
            print(f"⚠️  Search failed: {e}")
        
        # Sort by similarity
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results[:limit]
    
    def get_context_for_llm(self, current_query: str = None) -> str:
        """
        Get formatted context string for the LLM.
        
        Includes:
        1. Relevant past context (from vector search)
        2. Recent conversation history
        3. User profile/preferences
        
        Args:
            current_query: The current user query (for relevance search)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # 1. Search for relevant past context (only highly relevant)
        if current_query and self._embeddings_available and self._db_available:
            relevant = self.search_memory(current_query, limit=5)
            if relevant:
                # Only include items with high similarity (>0.6)
                high_relevance = [item for item in relevant if item.get('similarity', 0) > 0.6]
                if high_relevance:
                    context_parts.append("=== Past Conversations ===")
                    for item in high_relevance[:3]:  # Max 3 items
                        if item['type'] == 'message':
                            # Format: "User said: ..." or "Assistant said: ..."
                            role_label = "User" if item['role'] == 'user' else "Assistant"
                            content = item['content'][:150]  # Shorter for voice
                            context_parts.append(f"{role_label}: {content}")
                        elif item['type'] == 'summary':
                            context_parts.append(f"Summary: {item['content'][:150]}")
                    context_parts.append("")
        
        # 2. Recent conversation (only if we need older context beyond what's in history)
        # Note: Recent messages are passed separately as history, so we skip this
        # to avoid duplication
        
        # 3. User profile (if available)
        profile = self.get_user_profile()
        if profile:
            context_parts.append("=== User Profile ===")
            for key, value in profile.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get recent messages as list of dicts for LLM."""
        return [
            {'role': m.role, 'content': m.content}
            for m in self.recent_messages
        ]
    
    def update_user_profile(self, key: str, value: str, confidence: float = 1.0):
        """
        Update a user profile entry.
        
        Args:
            key: Profile key (e.g., 'preferred_name', 'interests')
            value: Profile value
            confidence: How confident we are (0-1)
        """
        if not self._db_available:
            return
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_profile (key, value, confidence, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        confidence = EXCLUDED.confidence,
                        updated_at = NOW()
                """, (key, value, confidence))
        except Exception as e:
            print(f"⚠️  Failed to update profile: {e}")
    
    def get_user_profile(self) -> Dict[str, str]:
        """Get all user profile entries."""
        if not self._db_available:
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT key, value FROM user_profile ORDER BY confidence DESC")
                return {row['key']: row['value'] for row in cur.fetchall()}
        except Exception as e:
            print(f"⚠️  Failed to get profile: {e}")
            return {}
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about stored conversations."""
        stats = {
            'session_id': self.session_id,
            'recent_messages': len(self.recent_messages),
            'db_available': self._db_available,
            'embeddings_available': self._embeddings_available,
        }
        
        if self._db_available:
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT COUNT(*) as count FROM messages")
                    stats['total_messages'] = cur.fetchone()['count']
                    
                    cur.execute("SELECT COUNT(*) as count FROM summaries")
                    stats['total_summaries'] = cur.fetchone()['count']
                    
                    cur.execute("SELECT COUNT(*) as count FROM user_profile")
                    stats['profile_entries'] = cur.fetchone()['count']
            except:
                pass
        
        return stats
    
    def clear_session(self):
        """Clear current session's messages (but keep summaries)."""
        self.recent_messages = []
        
        if self._db_available:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM messages WHERE session_id = %s",
                        (self.session_id,)
                    )
            except Exception as e:
                print(f"⚠️  Failed to clear session: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Convenience function
def create_memory_manager(**kwargs) -> MemoryManager:
    """Create a memory manager instance."""
    return MemoryManager(**kwargs)


# Test
if __name__ == "__main__":
    print("Testing Memory Manager...")
    print(f"PostgreSQL available: {POSTGRES_AVAILABLE}")
    print(f"Embeddings available: {EMBEDDINGS_AVAILABLE}")
    
    # Test without database
    memory = MemoryManager(db_url="postgresql://localhost/nonexistent")
    
    # Add some test messages
    memory.add_message("user", "Hello, my name is John")
    memory.add_message("assistant", "Nice to meet you, John!")
    memory.add_message("user", "I like programming in Python")
    
    print(f"\nRecent messages: {len(memory.recent_messages)}")
    print(f"\nContext for LLM:")
    print(memory.get_context_for_llm("What's my name?"))
    
    print(f"\nStats: {memory.get_conversation_stats()}")

