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
from typing import Callable, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import re
import time

from utils.memory_config import MemoryWriterConfig, config_from_dict
from utils.memory_writer import MemoryWriter, is_junk_stt_text

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

    MIGRATE_MESSAGES_SQL = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'messages' AND column_name = 'raw_text'
        ) THEN
            ALTER TABLE messages ADD COLUMN raw_text TEXT;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'messages' AND column_name = 'cleaned_text'
        ) THEN
            ALTER TABLE messages ADD COLUMN cleaned_text TEXT;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'messages' AND column_name = 'source'
        ) THEN
            ALTER TABLE messages ADD COLUMN source VARCHAR(32) DEFAULT 'user_final';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'messages' AND column_name = 'confidence'
        ) THEN
            ALTER TABLE messages ADD COLUMN confidence FLOAT DEFAULT 1.0;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'messages' AND column_name = 'saved_at'
        ) THEN
            ALTER TABLE messages ADD COLUMN saved_at TIMESTAMPTZ;
        END IF;
    END $$;
    """

    CREATE_TABLES_BASIC_SQL = """
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(64) NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS summaries (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(64),
        summary TEXT NOT NULL,
        topics TEXT[],
        user_preferences TEXT[],
        start_time TIMESTAMPTZ,
        end_time TIMESTAMPTZ,
        message_count INT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS user_profile (
        id SERIAL PRIMARY KEY,
        key VARCHAR(255) UNIQUE NOT NULL,
        value TEXT NOT NULL,
        confidence FLOAT DEFAULT 1.0,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
    """
    
    def __init__(
        self,
        db_url: str = None,
        session_id: str = None,
        max_recent_messages: int = 20,
        max_context_tokens: int = 2000,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool = True,
        smart_save: bool = True,
        persist_roles: tuple[str, ...] = ("user",),
        flush_interval_sec: float = 240.0,
        min_user_words: int = 3,
        memory_config: Optional[Dict[str, Any]] = None,
        memory_writer_config: Optional[MemoryWriterConfig] = None,
        text_cleaner: Callable[[str, str], Optional[str]] | None = None,
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
        self.db_url = db_url or os.getenv('DATABASE_URL', 'postgresql:///voice_assistant')
        self.session_id = session_id or self._generate_session_id()
        self.max_recent_messages = max_recent_messages
        self.max_context_tokens = max_context_tokens
        self.enable_embeddings = enable_embeddings
        self.smart_save = smart_save
        self.persist_roles = tuple(persist_roles)
        self.min_user_words = max(1, int(min_user_words))
        self._writer_config = memory_writer_config or config_from_dict(
            memory_config
        )
        if flush_interval_sec is not None:
            self._writer_config.save_interval_sec = max(
                30.0, float(flush_interval_sec)
            )
        self._text_cleaner = text_cleaner
        self._writer: Optional[MemoryWriter] = None
        self._last_assistant_text = ""
        
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
        if self.enable_embeddings:
            self._init_embeddings(embedding_model)
        else:
            print("Memory embeddings: disabled (faster smart-save mode).")
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
            
            # Create tables + migrate smart-save columns
            with self.conn.cursor() as cur:
                try:
                    cur.execute(self.CREATE_TABLES_SQL)
                except Exception as exc:
                    if self.enable_embeddings:
                        raise
                    print(
                        "⚠️  pgvector unavailable; using text-only memory "
                        f"schema ({exc})"
                    )
                    cur.execute(self.CREATE_TABLES_BASIC_SQL)
                cur.execute(self.MIGRATE_MESSAGES_SQL)
            
            self._db_available = True
            if self.smart_save and self._writer_config.enabled:
                self._writer = MemoryWriter(
                    config=self._writer_config,
                    persist_fn=self._persist_user_message,
                    text_cleaner=self._text_cleaner,
                )
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
                    WHERE session_id = %s AND role = 'user'
                    ORDER BY COALESCE(saved_at, timestamp) DESC 
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
    
    def add_message(self, role: str, content: str, *, persist: bool = True) -> Optional[Message]:
        """
        Add a message to in-session memory.

        User speech is queued for debounced PostgreSQL persistence; assistant
        replies stay in RAM for conversation context only.
        """
        cleaned = self._basic_clean_text(content)
        if not cleaned:
            return None
        if role == "user" and self._is_obvious_junk(cleaned):
            return None

        message = Message(role=role, content=cleaned)
        self.recent_messages.append(message)
        if len(self.recent_messages) > self.max_recent_messages:
            self.recent_messages = self.recent_messages[-self.max_recent_messages:]

        if role == "assistant":
            self._last_assistant_text = cleaned
        elif (
            persist
            and role == "user"
            and self._db_available
            and role in self.persist_roles
        ):
            self._queue_user_for_persist(cleaned, raw_text=content.strip())

        self._check_and_summarize()
        return message

    def queue_user_utterance(
        self,
        text: str,
        *,
        source: str = "user_final",
        confidence: float = 1.0,
    ) -> bool:
        """Queue user speech; also updates short-term history when substantive."""
        msg = self.add_message("user", text, persist=False)
        if msg is None:
            return False
        return self._queue_user_for_persist(
            msg.content,
            raw_text=text.strip(),
            source=source,
            confidence=confidence,
        )

    def _queue_user_for_persist(
        self,
        cleaned: str,
        *,
        raw_text: str,
        source: str = "user_final",
        confidence: float = 1.0,
    ) -> bool:
        if not self._db_available or "user" not in self.persist_roles:
            return False
        if self.smart_save and self._writer:
            return self._writer.enqueue(
                raw_text,
                source=source,
                confidence=confidence,
                last_assistant_text=self._last_assistant_text,
            )
        if not self._is_user_memory_worthy(cleaned):
            return False
        message = Message(role="user", content=cleaned)
        embedding = self._get_embedding(cleaned)
        self._save_message_to_db(
            message,
            embedding,
            raw_text=raw_text,
            cleaned_text=cleaned,
            source=source,
            confidence=confidence,
        )
        return True

    def set_text_cleaner(
        self, cleaner: Callable[[str, str], Optional[str]] | None
    ) -> None:
        """Optional hook: overrides built-in Ollama cleanup when set."""
        self._text_cleaner = cleaner
        if self._writer:
            self._writer.set_text_cleaner(cleaner)

    def _persist_user_message(
        self,
        *,
        raw_text: str,
        cleaned_text: str,
        source: str,
        confidence: float,
        captured_at: datetime,
        reason: str = "",
    ) -> None:
        if not self._is_user_memory_worthy(cleaned_text):
            return
        message = Message(role="user", content=cleaned_text, timestamp=captured_at)
        embedding = self._get_embedding(cleaned_text)
        self._save_message_to_db(
            message,
            embedding,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            source=source,
            confidence=confidence,
        )

    def _is_obvious_junk(self, text: str) -> bool:
        return is_junk_stt_text(text)

    @staticmethod
    def _basic_clean_text(content: str) -> str:
        text = (content or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = text.replace(">>", "").replace("<<", "")
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        text = text.strip(" -")
        return text.strip()

    def _is_user_memory_worthy(self, text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if len(normalized.split()) < self.min_user_words:
            return False
        junk_markers = (
            "thank you thank you",
            "thanks for watching",
            "subscribe",
            "hario",
            "blank audio",
            "birds chirping",
            "music",
        )
        if any(marker in normalized for marker in junk_markers):
            return False
        filler_only = {
            "thank you very much",
            "i think ill be right back",
            "i am saying something",
            "im saying something",
            "i was saying something right",
        }
        if normalized in filler_only:
            return False
        unique_words = set(normalized.split())
        if len(unique_words) <= 2 and len(normalized.split()) >= 5:
            return False
        return True

    def flush_pending(self) -> int:
        """Force flush buffered user utterances to PostgreSQL."""
        if self._writer:
            saved = self._writer.flush(force=True)
            if saved:
                print(f"Memory: flushed {saved} smart-saved message(s) to Postgres")
            return saved
        return 0
    
    def _save_message_to_db(
        self,
        message: Message,
        embedding: Optional[np.ndarray],
        *,
        raw_text: Optional[str] = None,
        cleaned_text: Optional[str] = None,
        source: str = "user_final",
        confidence: float = 1.0,
    ):
        """Save message to database."""
        saved_at = datetime.now()
        content = cleaned_text or message.content
        raw = raw_text if raw_text is not None else message.content
        try:
            with self.conn.cursor() as cur:
                if embedding is not None:
                    cur.execute("""
                        INSERT INTO messages (
                            session_id, role, content, timestamp, embedding,
                            raw_text, cleaned_text, source, confidence, saved_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        self.session_id,
                        message.role,
                        content,
                        message.timestamp,
                        embedding.tolist(),
                        raw,
                        content,
                        source,
                        confidence,
                        saved_at,
                    ))
                else:
                    cur.execute("""
                        INSERT INTO messages (
                            session_id, role, content, timestamp,
                            raw_text, cleaned_text, source, confidence, saved_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        self.session_id,
                        message.role,
                        content,
                        message.timestamp,
                        raw,
                        content,
                        source,
                        confidence,
                        saved_at,
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
                if embedding is not None:
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
                        embedding.tolist(),
                    ))
                else:
                    cur.execute("""
                        INSERT INTO summaries
                        (session_id, summary, topics, user_preferences, start_time, end_time, message_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        self.session_id,
                        summary.summary,
                        summary.topics,
                        summary.user_preferences,
                        summary.start_time,
                        summary.end_time,
                        summary.message_count,
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
                      AND role = 'user'
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
            'pending_db_messages': (
                self._writer.pending_count if self._writer else 0
            ),
            'db_available': self._db_available,
            'embeddings_available': self._embeddings_available,
            'smart_save': self.smart_save,
            'persist_roles': self.persist_roles,
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
        """Flush pending user memory and close database connection."""
        try:
            if self._writer:
                saved = self._writer.close()
                if saved:
                    print(f"Memory: flushed {saved} user utterance(s) on shutdown")
            else:
                self.flush_pending()
        except Exception as e:
            print(f"⚠️  Failed to flush pending memory: {e}")
        self._writer = None
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
