# UPDATED MAY 7, 2025
# Added archive.org integration as a fallback for 403/271 errors :)
# Also added functionality to prioritize certain domains and content
# More info on github: https://github.com/atineiatte/deep-research-at-home
# Make sure you use my gemma3 system prompt too

import logging
import json
import math
import time
import asyncio
import re
import random
import numpy as np
import aiohttp
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Callable, Awaitable, Optional, Any, Union, Set, Tuple
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from open_webui.models.users import User

name = "Deep Research at Home"


def setup_logger():
    # Get the root logger and clear all handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Get our specific logger
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # Create new handler with no timestamp
    handler = logging.StreamHandler()
    handler.set_name(name)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add handler and prevent propagation
    logger.addHandler(handler)
    logger.propagate = False

    # Also configure root logger to prevent double timestamps
    root_handler = logging.StreamHandler()
    root_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    root_handler.setFormatter(root_formatter)
    root_logger.addHandler(root_handler)
    root_logger.setLevel(logging.DEBUG)

    return logger


logger = setup_logger()


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls"""

    def __init__(self, max_size=10000000):
        self.cache = {}
        self.access_order = []  # Track access order for LRU
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        self.url_token_counts = {}  # Track token counts for URLs

    def _generate_key(self, text_key):
        """Generate a more reliable cache key"""
        import hashlib

        # Use SHA256 for better collision resistance
        text_sample = text_key[:2000] if len(text_key) > 2000 else text_key
        return hashlib.sha256(text_sample.encode("utf-8")).hexdigest()[:16]

    def get(self, text_key):
        """Get embedding from cache using text as key"""
        key = self._generate_key(text_key)
        result = self.cache.get(key)
        if result is not None:
            self.hit_count += 1
            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return result
        else:
            self.miss_count += 1
            return None

    def set(self, text_key, embedding):
        """Store embedding in cache"""
        key = self._generate_key(text_key)

        # If key already exists, just update and move to end
        if key in self.cache:
            self.cache[key] = embedding
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return

        # Add new entry
        self.cache[key] = embedding
        self.access_order.append(key)

        # LRU eviction if cache is too large
        while len(self.cache) > self.max_size:
            # Remove least recently used item
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]

    def stats(self):
        """Return cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
        }


class TransformationCache:
    """Simple cache for transformed embeddings to avoid redundant transformations"""

    def __init__(self, max_size=2500000):
        self.cache = {}
        self.access_order = []  # Track access order for LRU
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def _generate_key(self, text, transform_id):
        """Generate a more reliable cache key"""
        import hashlib

        text_sample = text[:2000] if len(text) > 2000 else text
        combined = f"{text_sample}_{str(transform_id)}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    def get(self, text, transform_id):
        """Get transformed embedding from cache"""
        key = self._generate_key(text, transform_id)
        result = self.cache.get(key)
        if result is not None:
            self.hit_count += 1
            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return result
        else:
            self.miss_count += 1
            return None

    def set(self, text, transform_id, transformed_embedding):
        """Store transformed embedding in cache"""
        key = self._generate_key(text, transform_id)

        # If key already exists, just update and move to end
        if key in self.cache:
            self.cache[key] = transformed_embedding
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return

        # Add new entry
        self.cache[key] = transformed_embedding
        self.access_order.append(key)

        # LRU eviction if cache is too large
        while len(self.cache) > self.max_size:
            # Remove least recently used item
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]

    def stats(self):
        """Return cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
        }


class ResearchStateManager:
    """Manages research state per conversation to ensure proper isolation"""

    def __init__(self):
        self.conversation_states = {}

    def get_state(self, conversation_id):
        """Get state for a specific conversation, creating if needed"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = {
                "research_completed": False,
                "prev_comprehensive_summary": "",
                "waiting_for_outline_feedback": False,
                "outline_feedback_data": None,
                "research_state": None,
                "follow_up_mode": False,
                "user_preferences": {"pdv": None, "strength": 0.0, "impact": 0.0},
                "research_dimensions": None,
                "research_trajectory": None,
                "pdv_alignment_history": [],
                "gap_coverage_history": [],
                "semantic_transformations": None,
                "section_synthesized_content": {},
                "section_citations": {},
                "url_selected_count": {},
                "url_considered_count": {},
                "url_token_counts": {},
                "master_source_table": {},
                "global_citation_map": {},
                "verified_citations": [],
                "flagged_citations": [],
                "citation_fixes": [],
                "memory_stats": {
                    "results_tokens": 0,
                    "section_tokens": {},
                    "synthesis_tokens": 0,
                    "total_tokens": 0,
                },
                "results_history": [],
                "search_history": [],
                "active_outline": [],
                "cycle_summaries": [],
                "completed_topics": set(),
                "irrelevant_topics": set(),
            }
        return self.conversation_states[conversation_id]

    def update_state(self, conversation_id, key, value):
        """Update a specific state value for a conversation"""
        state = self.get_state(conversation_id)
        state[key] = value

    def reset_state(self, conversation_id):
        """Reset the state for a specific conversation"""
        if conversation_id in self.conversation_states:
            del self.conversation_states[conversation_id]


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __current_event_call__: Callable[[dict], Awaitable[Any]]
    __user__: User
    __model__: str
    __request__: Any

    class Valves(BaseModel):
        ENABLED: bool = Field(
            default=True,
            description="Enable Deep Research pipe",
        )
        RESEARCH_MODEL: str = Field(
            default="gemma3:12b",
            description="Model for generating research queries and synthesizing results",
        )
        SYNTHESIS_MODEL: str = Field(
            default="gemma3:27b",
            description="Optional separate model for final synthesis (leave empty to use RESEARCH_MODEL)",
        )
        EMBEDDING_MODEL: str = Field(
            default="granite-embedding:30m",
            description="Model for semantic comparison of content",
        )
        QUALITY_FILTER_MODEL: str = Field(
            default="gemma3:4b",
            description="Model used for filtering irrelevant search results",
        )

        QUALITY_FILTER_ENABLED: bool = Field(
            default=True,
            description="Whether to use quality filtering for search results",
        )

        QUALITY_SIMILARITY_THRESHOLD: float = Field(
            default=0.60,
            description="Similarity threshold below which quality filtering is applied",
            ge=0.0,
            le=1.0,
        )
        MAX_CYCLES: int = Field(
            default=15,
            description="Maximum number of research cycles before terminating",
            ge=3,
            le=50,
        )
        MIN_CYCLES: int = Field(
            default=10,
            description="Minimum number of research cycles to perform",
            ge=1,
            le=10,
        )
        EXPORT_RESEARCH_DATA: bool = Field(
            default=True,
            description="Enable exporting of complete research data including results, queries, and timestamps",
        )
        SEARCH_RESULTS_PER_QUERY: int = Field(
            default=3,
            description="Base number of search results to use per query",
            ge=1,
            le=10,
        )
        EXTRA_RESULTS_PER_QUERY: int = Field(
            default=3,
            description="Maximum extra results to add when repeat URLs are detected",
            ge=0,
            le=5,
        )
        SUCCESSFUL_RESULTS_PER_QUERY: int = Field(
            default=1,
            description="Number of successful results to keep per query",
            ge=1,
            le=5,
        )
        CHUNK_LEVEL: int = Field(
            default=2,
            description="Level of chunking (1=phrase, 2=sentence, 3=paragraph, 4+=multi-paragraph)",
            ge=1,
            le=10,
        )
        COMPRESSION_LEVEL: int = Field(
            default=4,
            description="Level of compression (1=minimal, 10=maximum)",
            ge=1,
            le=10,
        )
        LOCAL_INFLUENCE_RADIUS: int = Field(
            default=3,
            description="Number of chunks before and after to consider for local similarity",
            ge=0,
            le=5,
        )
        QUERY_WEIGHT: float = Field(
            default=0.5,
            description="Weight to give query similarity vs document relevance (0.0-1.0)",
            ge=0.0,
            le=1.0,
        )
        FOLLOWUP_WEIGHT: float = Field(
            default=0.5,
            description="Weight to give followup query vs previous comprehensive summary (0.0-1.0)",
            ge=0.0,
            le=1.0,
        )
        TEMPERATURE: float = Field(
            default=0.7, description="Temperature for generation", ge=0.0, le=2.0
        )
        SYNTHESIS_TEMPERATURE: float = Field(
            default=0.6, description="Temperature for final synthesis", ge=0.0, le=2.0
        )
        OLLAMA_URL: str = Field(
            default="http://localhost:11434", description="URL for Ollama API"
        )
        SEARCH_URL: str = Field(
            default="http://192.168.1.1:8888/search?q=",
            description="URL for web search API",
        )
        MAX_FAILED_RESULTS: int = Field(
            default=6,
            description="Maximum number of failed results before abandoning a query",
            ge=1,
            le=10,
        )
        EXTRACT_CONTENT_ONLY: bool = Field(
            default=True,
            description="Extract only text content from HTML, removing scripts, styles, etc.",
        )
        PDF_MAX_PAGES: int = Field(
            default=25,
            description="Maximum number of pages to extract from a PDF",
            ge=5,
            le=500,
        )
        HANDLE_PDFS: bool = Field(
            default=True,
            description="Enable processing of PDF files",
        )
        RELEVANCY_SNIPPET_LENGTH: int = Field(
            default=2000,
            description="Number of characters to use when comparing extra results for relevance",
            ge=100,
            le=5000,
        )
        DOMAIN_PRIORITY: str = Field(
            default="",
            description="Comma or space-separated list of domain keywords to prioritize (e.g., '.gov, .edu, epa'). Leave empty to disable domain prioritization.",
        )

        CONTENT_PRIORITY: str = Field(
            default="",
            description="Comma or space-separated list of content keywords to prioritize (e.g., 'pfas, spatial, groundwater'). Leave empty to disable content prioritization.",
        )
        DOMAIN_MULTIPLIER: float = Field(
            default=1.3,
            description="Multiplier for results from priority domains (1.0 = no change, 2.0 = double score)",
            ge=1.0,
            le=3.0,
        )
        KEYWORD_MULTIPLIER_PER_MATCH: float = Field(
            default=1.1,
            description="Multiplier applied per matched content keyword (1.1 = 10% increase per keyword)",
            ge=1.0,
            le=1.5,
        )
        MAX_KEYWORD_MULTIPLIER: float = Field(
            default=2.0,
            description="Maximum total multiplier from content keywords",
            ge=1.0,
            le=3.0,
        )
        INTERACTIVE_RESEARCH: bool = Field(
            default=True,
            description="Enable user interaction during research",
        )
        USER_PREFERENCE_THROUGHOUT: bool = Field(
            default=True,
            description="Use user removal preferences throughout research cycles",
        )
        SEMANTIC_TRANSFORMATION_STRENGTH: float = Field(
            default=0.7,
            description="Strength of semantic transformations for directing research (0.0-1.0)",
            ge=0.0,
            le=1.0,
        )
        TRAJECTORY_MOMENTUM: float = Field(
            default=0.6,
            description="Weight given to previous research trajectory (0.0-1.0)",
            ge=0.0,
            le=1.0,
        )
        GAP_EXPLORATION_WEIGHT: float = Field(
            default=0.4,
            description="Weight given to exploring research gaps (0.0-1.0)",
            ge=0.0,
            le=1.0,
        )
        STEPPED_SYNTHESIS_COMPRESSION: bool = Field(
            default=True,
            description="Enable tiered compression for older vs newer research results",
        )
        MAX_RESULT_TOKENS: int = Field(
            default=4000,
            description="Maximum tokens per result for synthesis",
            ge=1000,
            le=8000,
        )
        COMPRESSION_SETPOINT: int = Field(
            default=4000,
            description="Length at which semantic compression of results engages",
            ge=300,
            le=8000,
        )
        REPEATS_BEFORE_EXPANSION: int = Field(
            default=3,
            description="Number of times a result must be repeated before adding extra results",
            ge=1,
            le=10,
        )
        REPEAT_WINDOW_FACTOR: float = Field(
            default=0.95,
            description="Control the sliding window factor for repeat contents",
            ge=0.0,
            le=1.0,
        )
        VERIFY_CITATIONS: bool = Field(
            default=True,
            description="Enable verification of citations against sources",
        )
        THREAD_WORKERS: int = Field(
            default=2,
            description="Number of worker threads for parallel processing",
            ge=1,
            le=2,
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

        # Use state manager to isolate conversation states
        self.state_manager = ResearchStateManager()
        self.conversation_id = None  # Will be set during pipe method

        # Shared resources (not conversation-specific)
        self.embedding_cache = EmbeddingCache(max_size=10000000)
        self.transformation_cache = TransformationCache(max_size=2500000)
        self.vocabulary_cache = None
        self.vocabulary_embeddings = None
        self.is_pdf_content = False

        # Initialize research date and executor
        self.research_date = datetime.now().strftime("%Y-%m-%d")
        self.trajectory_accumulator = None
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.valves.THREAD_WORKERS
        )

    def __del__(self):
        """Cleanup resources when the object is destroyed"""
        self.cleanup_resources()

    def cleanup_resources(self):
        """Clean up ThreadPoolExecutor and other resources"""
        if hasattr(self, "executor") and self.executor:
            try:
                self.executor.shutdown(wait=True)
                logger.info("ThreadPoolExecutor cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error during executor cleanup: {e}")
                # Force shutdown if graceful shutdown fails
                try:
                    self.executor.shutdown(wait=False)
                except:
                    pass

    def _flatten_outline_topics(self, outline: List[Dict]) -> List[str]:
        """
        Recursively flattens a nested research outline into a single list of topic strings.
        Handles any level of nesting in the outline structure.
        """
        logger.debug(f"Flattening outline with {len(outline)} top-level items")
        all_topics = []
        for i, item in enumerate(outline):
            logger.debug(
                f"Processing outline item {i+1}: {type(item)} - {str(item)[:200]}..."
            )
            if isinstance(item, dict):
                # Add the main topic string
                if "topic" in item and isinstance(item["topic"], str):
                    all_topics.append(item["topic"])
                    logger.debug(f"Added main topic: {item['topic']}")

                # Recursively process subtopics
                if "subtopics" in item and isinstance(item["subtopics"], list):
                    logger.debug(f"Processing {len(item['subtopics'])} subtopics")
                    for j, subtopic in enumerate(item["subtopics"]):
                        if isinstance(subtopic, str):
                            all_topics.append(subtopic)
                            logger.debug(f"Added string subtopic {j+1}: {subtopic}")
                        elif isinstance(subtopic, dict):
                            if "topic" in subtopic:
                                all_topics.append(subtopic["topic"])
                                logger.debug(
                                    f"Added dict subtopic {j+1}: {subtopic['topic']}"
                                )
                            # Recursively process nested subtopics
                            if "subtopics" in subtopic:
                                nested_topics = self._flatten_outline_topics([subtopic])
                                all_topics.extend(nested_topics)
                                logger.debug(
                                    f"Added {len(nested_topics)} nested topics from subtopic {j+1}"
                                )
            elif isinstance(
                item, str
            ):  # Handle cases where a subtopic list might just contain strings
                all_topics.append(item)
                logger.debug(f"Added string item: {item}")

        logger.debug(f"Flattening complete: {len(all_topics)} total topics")
        return all_topics

    async def initialize_research_state(
        self,
        user_message,
        research_outline,
        all_topics,
        outline_embedding,
        initial_results=None,
    ):
        """Initialize or reset research state consistently across interactive and non-interactive modes"""
        logger.debug("★★★ ENTERING initialize_research_state ★★★")
        logger.debug(
            f"Parameters: user_message={user_message[:100]}..., research_outline={len(research_outline)} items, all_topics={len(all_topics)} items"
        )

        try:
            state = self.get_state()
            logger.debug("Got state in initialize_research_state")
        except Exception as e:
            logger.error(
                f"★★★ ERROR GETTING STATE IN initialize_research_state: {e} ★★★",
                exc_info=True,
            )
            raise

        # Core research state
        self.update_state(
            "research_state",
            {
                "research_outline": research_outline,
                "all_topics": all_topics,
                "outline_embedding": outline_embedding,
                "user_message": user_message,
            },
        )

        # Initialize memory statistics with proper structure
        memory_stats = state.get("memory_stats", {})
        if not memory_stats or not isinstance(memory_stats, dict):
            memory_stats = {
                "results_tokens": 0,
                "section_tokens": {},
                "synthesis_tokens": 0,
                "total_tokens": 0,
            }
        self.update_state("memory_stats", memory_stats)

        # Update results_tokens if we have initial results
        if initial_results:
            results_tokens = 0
            for result in initial_results:
                # Get or calculate tokens for this result
                tokens = result.get("tokens", 0)
                if tokens == 0 and "content" in result:
                    tokens = await self.count_tokens(result["content"])
                    result["tokens"] = tokens
                results_tokens += tokens

            # Update memory stats with token count
            memory_stats["results_tokens"] = results_tokens
            self.update_state("memory_stats", memory_stats)

        # Initialize tracking variables
        self.update_state("topic_usage_counts", state.get("topic_usage_counts", {}))
        self.update_state("completed_topics", state.get("completed_topics", set()))
        self.update_state("irrelevant_topics", state.get("irrelevant_topics", set()))
        self.update_state("active_outline", all_topics.copy())
        self.update_state("cycle_summaries", state.get("cycle_summaries", []))

        # Results tracking
        results_history = state.get("results_history", [])
        if initial_results:
            results_history.extend(initial_results)
        self.update_state("results_history", results_history)

        # Search history
        search_history = state.get("search_history", [])
        self.update_state("search_history", search_history)

        # Initialize dimension tracking
        try:
            logger.debug("About to call initialize_research_dimensions")
            await self.initialize_research_dimensions(all_topics, user_message)
            logger.debug("★★★ Returned from initialize_research_dimensions ★★★")

            # Get fresh state after initialization
            state = self.get_state()
            logger.debug("Got fresh state after research dimensions initialization")

            research_dimensions = state.get("research_dimensions")
            logger.debug(
                f"Research dimensions from state: {research_dimensions is not None}"
            )

            if research_dimensions:
                logger.debug(
                    f"Research dimensions keys: {list(research_dimensions.keys())}"
                )
                logger.debug(
                    f"Coverage length: {len(research_dimensions.get('coverage', []))}"
                )
                self.update_state(
                    "latest_dimension_coverage", research_dimensions["coverage"].copy()
                )
                logger.debug("Updated latest_dimension_coverage")
            else:
                logger.debug("No research_dimensions found in state")

            # Source tracking
            logger.debug("Updating source tracking state")
            self.update_state(
                "master_source_table", state.get("master_source_table", {})
            )
            self.update_state("url_selected_count", state.get("url_selected_count", {}))
            self.update_state("url_token_counts", state.get("url_token_counts", {}))
            logger.debug("Source tracking state updated")
        except Exception as e:
            logger.error(f"★★★ ERROR AFTER RESEARCH DIMENSIONS: {e} ★★★", exc_info=True)
            # Continue with default values if there's an error
            self.update_state("master_source_table", {})
            self.update_state("url_selected_count", {})
            self.update_state("url_token_counts", {})

        # Trajectory accumulator reset
        logger.debug("Resetting trajectory accumulator")
        self.trajectory_accumulator = None
        logger.debug("Trajectory accumulator reset complete")

        try:
            logger.debug(
                f"About to log research state initialization with {len(all_topics)} topics"
            )
            logger.info(
                f"Research state initialized with {len(all_topics)} topics and {len(results_history)} initial results"
            )
            logger.debug("Research state initialization logged successfully")
            logger.debug("★★★ EXITING initialize_research_state SUCCESSFULLY ★★★")
        except Exception as e:
            logger.error(f"★★★ ERROR LOGGING RESEARCH STATE: {e} ★★★", exc_info=True)
            logger.debug("★★★ EXITING initialize_research_state WITH ERROR ★★★")

    async def update_token_counts(self, new_results=None):
        """Centralized function to update token counts consistently"""
        logger.debug("★★★ ENTERING update_token_counts ★★★")
        logger.debug(f"new_results: {new_results is not None}")

        try:
            state = self.get_state()
            logger.debug("Got state in update_token_counts")

            memory_stats = state.get(
                "memory_stats",
                {
                    "results_tokens": 0,
                    "section_tokens": {},
                    "synthesis_tokens": 0,
                    "total_tokens": 0,
                },
            )
            logger.debug(f"Got memory_stats: {memory_stats.keys()}")
        except Exception as e:
            logger.error(
                f"★★★ ERROR GETTING STATE/MEMORY_STATS IN update_token_counts: {e} ★★★",
                exc_info=True,
            )
            raise

        # Update results tokens if new results provided
        if new_results:
            for result in new_results:
                tokens = result.get("tokens", 0)
                if tokens == 0 and "content" in result:
                    tokens = await self.count_tokens(result["content"])
                    result["tokens"] = tokens
                memory_stats["results_tokens"] += tokens

        # If no results tokens but we have results history, recalculate
        results_history = state.get("results_history", [])
        if memory_stats["results_tokens"] == 0 and results_history:
            total_tokens = 0
            for result in results_history:
                tokens = result.get("tokens", 0)
                if tokens == 0 and "content" in result:
                    tokens = await self.count_tokens(result["content"])
                    result["tokens"] = tokens
                total_tokens += tokens
            memory_stats["results_tokens"] = total_tokens

        # Recalculate total tokens
        section_tokens_sum = sum(memory_stats.get("section_tokens", {}).values())
        memory_stats["total_tokens"] = (
            memory_stats["results_tokens"]
            + section_tokens_sum
            + memory_stats.get("synthesis_tokens", 0)
        )

        # Update state
        try:
            logger.debug("About to update memory_stats in state")
            self.update_state("memory_stats", memory_stats)
            logger.debug("Memory_stats updated in state successfully")
            logger.debug("★★★ EXITING update_token_counts SUCCESSFULLY ★★★")
            return memory_stats
        except Exception as e:
            logger.error(f"★★★ ERROR UPDATING MEMORY_STATS: {e} ★★★", exc_info=True)
            logger.debug("★★★ EXITING update_token_counts WITH ERROR ★★★")
            raise

    def get_state(self):
        """Get the current conversation state"""
        if not self.conversation_id:
            # Generate a temporary ID if we don't have one yet
            self.conversation_id = f"temp_{hash(str(self.__user__.id))}"
        return self.state_manager.get_state(self.conversation_id)

    def update_state(self, key, value):
        """Update a specific state value"""
        self.state_manager.update_state(self.conversation_id, key, value)

    def reset_state(self):
        """Reset the state for the current conversation"""
        if self.conversation_id:
            self.state_manager.reset_state(self.conversation_id)
            self.trajectory_accumulator = None
            self.is_pdf_content = False
            logger.info(f"Full state reset for conversation: {self.conversation_id}")

    async def emit_user_message(self, message: str, message_type: str = "info"):
        """Send a formatted message to the user through the UI"""
        if hasattr(self, "__current_event_emitter__"):
            icon = (
                "ℹ️"
                if message_type == "info"
                else "⚠️" if message_type == "warning" else "❌"
            )
            await self.__current_event_emitter__(
                {"type": "message", "content": f"{icon} {message}"}
            )
        logger.info(f"User message ({message_type}): {message}")

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using Ollama API with robust fallback"""
        if not text:
            return 0

        # Skip Ollama API call since it's returning 404 - use fallback directly
        logger.debug(
            "Skipping Ollama tokenize API (known to return 404), using fallback estimation"
        )

        # Fallback to simple estimation if API call fails
        words = text.split()
        estimated_tokens = int(len(words) * 1.3)  # Approximate token count using words
        logger.debug(
            f"Token count via fallback: {estimated_tokens} for {len(text)} chars"
        )
        return estimated_tokens

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text string using the configured embedding model with caching"""
        if not text or not text.strip():
            return None

        text = text[:2000]
        text = text.replace(":", " - ")

        # Check cache first
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding

    # Add robust JSON parsing with fallback
    async def _simple_json_repair(self, json_str: str) -> Optional[str]:
        """Attempt simple JSON repairs for common issues"""
        try:
            # Clean the JSON string first
            cleaned = json_str.strip()
            logger.debug(f"Starting JSON repair on: {cleaned[:200]}...")

            # Remove any trailing commas before closing brackets/braces
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

            # Fix missing quotes around keys (simple cases) - but be more careful
            # Only fix unquoted keys that are clearly identifiers
            cleaned = re.sub(r"(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', cleaned)

            # Fix single quotes to double quotes
            cleaned = cleaned.replace("'", '"')

            # Handle common JSON truncation issues
            # If the JSON ends abruptly, try to close it properly
            if cleaned.count("{") > cleaned.count("}"):
                # Count unclosed braces and brackets
                open_braces = cleaned.count("{")
                close_braces = cleaned.count("}")
                open_brackets = cleaned.count("[")
                close_brackets = cleaned.count("]")

                # Add missing closing brackets first, then braces
                if open_brackets > close_brackets:
                    cleaned += "]" * (open_brackets - close_brackets)
                if open_braces > close_braces:
                    cleaned += "}" * (open_braces - close_braces)

            # Remove any trailing incomplete content after the last complete brace
            # Find the last complete JSON object
            brace_count = 0
            last_complete_pos = -1
            for i, char in enumerate(cleaned):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        last_complete_pos = i

            if last_complete_pos > 0:
                cleaned = cleaned[: last_complete_pos + 1]

            logger.debug(f"After repair: {cleaned[:200]}...")

            # Test if the repair worked
            json.loads(cleaned)
            logger.debug("JSON repair successful!")
            return cleaned

        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
            return None

    async def parse_json_with_fallback(
        self, json_str: str, user_message: str, retry_count: int = 0
    ) -> Tuple[List[dict], List[str]]:
        """Attempts to parse JSON with retries and fallback mechanism"""
        # Clean the JSON string first
        cleaned_json = json_str.strip()

        # Try to extract JSON from markdown code blocks if present
        if "```json" in cleaned_json:
            start = cleaned_json.find("```json") + 7
            end = cleaned_json.find("```", start)
            if end != -1:
                cleaned_json = cleaned_json[start:end].strip()
        elif "```" in cleaned_json:
            start = cleaned_json.find("```") + 3
            end = cleaned_json.find("```", start)
            if end != -1:
                cleaned_json = cleaned_json[start:end].strip()

        try:
            outline_data = json.loads(cleaned_json)
            research_outline = outline_data.get("outline", [])
            # Validate outline structure
            if not research_outline or not isinstance(research_outline, list):
                raise ValueError("Invalid outline structure - empty or not a list")
            # Use recursive flattening to extract all topics at any depth
            all_topics = self._flatten_outline_topics(research_outline)
            if not all_topics:
                raise ValueError("No valid topics found in outline")
            logger.info(f"Successfully flattened outline into {len(all_topics)} topics")
            return research_outline, all_topics
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed (attempt {retry_count + 1}): {e}")
            logger.debug(f"Failed JSON content (first 500 chars): {cleaned_json[:500]}")
            logger.debug(f"Failed JSON content (last 500 chars): {cleaned_json[-500:]}")

            # Prevent infinite recursion
            if retry_count >= 2:
                logger.error("Maximum retry attempts reached, using fallback outline")
                logger.error(f"Final failed JSON: {cleaned_json}")
                await self.emit_user_message(
                    "I encountered an issue with the outline structure, but don't worry!\n"
                    "I'll use a reliable alternative structure to continue the research.\n"
                    "Your research process will continue smoothly.",
                    "warning",
                )
                return await self.create_fallback_outline(user_message)

            await self.emit_user_message(
                "The research outline structure needs some adjustments.\nI'll fix this automatically.\n",
                "info",
            )

            try:
                # Try simple JSON repair first
                repaired_json = await self._simple_json_repair(cleaned_json)
                if repaired_json:
                    return await self.parse_json_with_fallback(
                        repaired_json, user_message, retry_count + 1
                    )

                # If simple repair fails, try LLM repair
                repair_messages = [
                    {
                        "role": "system",
                        "content": "You are a JSON repair specialist. Fix the JSON structure while preserving content. Return ONLY valid JSON, no explanations.",
                    },
                    {
                        "role": "user",
                        "content": f"Fix this broken JSON and return only the corrected JSON:\n{cleaned_json[:1000]}",
                    },  # Limit length
                ]

                # Prepare form data for the API call
                form_data = {
                    "model": self.valves.QUALITY_FILTER_MODEL,
                    "messages": repair_messages,
                    "stream": False,
                    "temperature": 0.3,
                }

                repair_response = await generate_chat_completions(
                    self.__request__,
                    form_data,
                    user=self.__user__,
                )
                logger.debug(f"LLM repair response: {repair_response}")

                if not repair_response or "choices" not in repair_response:
                    raise ValueError("Failed to get repair response")

                repaired_json_str = repair_response["choices"][0]["message"][
                    "content"
                ].strip()
                logger.debug(
                    f"LLM repaired JSON (first 500 chars): {repaired_json_str[:500]}"
                )
                logger.debug(
                    f"LLM repaired JSON (last 500 chars): {repaired_json_str[-500:]}"
                )

                # Try parsing the repaired JSON (with incremented retry count)
                return await self.parse_json_with_fallback(
                    repaired_json_str, user_message, retry_count + 1
                )

            except Exception as repair_err:
                logger.error(f"JSON repair failed: {repair_err}", exc_info=True)
                # Use the fallback outline when repair fails
                await self.emit_user_message(
                    "I encountered an issue with the outline structure, but don't worry!\n"
                    "I'll use a reliable alternative structure to continue the research.\n"
                    "Your research process will continue smoothly.",
                    "warning",
                )
                return await self.create_fallback_outline(user_message)

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text string using the configured embedding model with caching"""
        if not text or not text.strip():
            return None

        text = text[:2000]
        text = text.replace(":", " - ")

        # Check cache first
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding

        # If not in cache, get from API
        try:
            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                payload = {
                    "model": self.valves.EMBEDDING_MODEL,
                    "input": text,
                }

                async with session.post(
                    f"{self.valves.OLLAMA_URL}/api/embed", json=payload, timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Handle both old and new API response formats
                        if "embedding" in result:
                            embedding = result.get("embedding", [])
                            if embedding:
                                # Cache the result
                                self.embedding_cache.set(text, embedding)
                                return embedding
                        elif "embeddings" in result and result["embeddings"]:
                            # New format with batch response (we only sent one text, so take first)
                            embedding = result["embeddings"][0]
                            if embedding:
                                # Cache the result
                                self.embedding_cache.set(text, embedding)
                                return embedding
                    else:
                        logger.warning(
                            f"Embedding request failed with status {response.status}"
                        )

            return None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    async def get_transformed_embedding(
        self, text: str, transformation=None
    ) -> Optional[List[float]]:
        """Get embedding with optional transformation applied, using caching for efficiency"""
        if not text or not text.strip():
            return None

        # If no transformation needed, just return regular embedding
        if transformation is None:
            return await self.get_embedding(text)

        # Check transformation cache first - simple lookup
        transform_id = (
            transformation.get("id", str(hash(str(transformation))))
            if isinstance(transformation, dict)
            else transformation
        )
        cached_transformed = self.transformation_cache.get(text, transform_id)
        if cached_transformed is not None:
            return cached_transformed

        # If not in transformation cache, get base embedding
        base_embedding = await self.get_embedding(text)
        if not base_embedding:
            return None

        # Apply transformation
        transformed = await self.apply_semantic_transformation(
            base_embedding, transformation
        )

        # Cache the transformed result only if successful
        if transformed:
            self.transformation_cache.set(text, transform_id, transformed)

        return transformed

    async def create_fallback_outline(self, query: str) -> Tuple[List[dict], List[str]]:
        """Creates a fallback outline when the LLM-generated outline fails"""
        logger.info("Creating fallback research outline structure")

        # Create a basic but flexible fallback outline
        fallback_outline = [
            {
                "topic": f"Overview of {query}",
                "subtopics": [
                    "Background and Context",
                    "Core Concepts",
                    "Current State",
                ],
            },
            {
                "topic": "Key Aspects and Analysis",
                "subtopics": [
                    "Major Components",
                    "Technical Details",
                    "Implementation Considerations",
                ],
            },
            {
                "topic": "Impact and Applications",
                "subtopics": [
                    "Use Cases",
                    "Benefits and Challenges",
                    "Future Directions",
                ],
            },
        ]

        # Extract all topics for further processing
        all_topics = []
        for section in fallback_outline:
            all_topics.append(section["topic"])
            all_topics.extend(section.get("subtopics", []))

        return fallback_outline, all_topics

    async def create_context_vocabulary(
        self, context_text: str, min_size: int = 1000
    ) -> List[str]:
        """Create a vocabulary from recent context when standard vocabulary is unavailable"""
        logger.info("Creating vocabulary from context as fallback")

        # Extract words from context
        words = re.findall(r"\b[a-zA-Z]{4,}\b", context_text.lower())

        # Get unique words
        unique_words = list(set(words))
        logger.info(f"Created context vocabulary with {len(unique_words)} words")

        return unique_words

    async def load_vocabulary(self):
        """Load the 10,000 word vocabulary for semantic analysis"""
        if self.vocabulary_cache is not None:
            return self.vocabulary_cache

        try:
            url = "https://www.mit.edu/~ecprice/wordlist.10000"
            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        text = await response.text()
                        self.vocabulary_cache = [
                            word.strip() for word in text.splitlines() if word.strip()
                        ]
                        logger.info(
                            f"Loaded {len(self.vocabulary_cache)} words vocabulary"
                        )
                        return self.vocabulary_cache
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")

            # Use context to create a vocabulary if standard one is unavailable
            # Get recent context from results history or any available text
            context_text = ""
            state = self.get_state()
            results_history = state.get("results_history", [])
            search_history = state.get("search_history", [])
            section_synthesized_content = state.get("section_synthesized_content", {})

            if results_history:
                # Use the last few results
                for result in results_history[-5:]:
                    context_text += result.get("content", "") + " "

            # Add any research queries
            if search_history:
                context_text += " ".join(search_history) + " "

            # Add any section content
            if section_synthesized_content:
                for content in list(section_synthesized_content.values())[:3]:
                    context_text += content + " "

            # If we still don't have enough context, just proceed with failure logging
            if len(context_text) < 5000:
                logger.error("Insufficient context for vocabulary creation")
                return None

            # Create vocabulary from context
            self.vocabulary_cache = await self.create_context_vocabulary(context_text)
            return self.vocabulary_cache

    async def load_prebuilt_vocabulary_embeddings(self):
        """Download and load pre-built vocabulary embeddings from GitHub"""
        try:
            import gzip
            import json
            import tempfile
            import os

            logger.info("Attempting to download pre-built vocabulary embeddings")
            url = "https://github.com/atineiatte/deep-research-at-home/raw/main/granite30m%20mit%2010k.gz"

            # Download the compressed file
            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_filename = temp_file.name
                            # Write the compressed data to the temporary file
                            temp_file.write(await response.read())

                        # Decompress and load the embeddings
                        try:
                            with gzip.open(temp_filename, "rt", encoding="utf-8") as f:
                                data = json.load(f)

                            # Clean up the temporary file
                            os.unlink(temp_filename)

                            # Convert the data to the expected format
                            self.vocabulary_cache = []
                            self.vocabulary_embeddings = {}

                            for word, embedding in data.items():
                                self.vocabulary_cache.append(word)
                                self.vocabulary_embeddings[word] = embedding

                            logger.info(
                                f"Successfully loaded {len(self.vocabulary_embeddings)} pre-built vocabulary embeddings"
                            )

                            # Store in state for persistence across calls
                            self.update_state(
                                "vocabulary_embeddings", self.vocabulary_embeddings
                            )

                            return self.vocabulary_embeddings
                        except Exception as e:
                            logger.error(
                                f"Error decompressing or parsing embeddings: {e}"
                            )
                            # Clean up the temporary file if it exists
                            if os.path.exists(temp_filename):
                                os.unlink(temp_filename)
                    else:
                        logger.warning(
                            f"Failed to download pre-built embeddings: HTTP {response.status}"
                        )

            # If we get here, something went wrong - fall back to original method
            logger.info("Falling back to on-demand vocabulary embedding generation")
            return await self.load_vocabulary_embeddings()

        except Exception as e:
            logger.error(f"Error downloading pre-built vocabulary embeddings: {e}")
            # Fall back to the original method
            logger.info("Falling back to on-demand vocabulary embedding generation")
            return await self.load_vocabulary_embeddings()

    async def load_vocabulary_embeddings(self):
        """Get embeddings for vocabulary words using existing batch processing or pre-built embeddings"""
        # If we already have the embeddings, return them
        if self.vocabulary_embeddings is not None:
            return self.vocabulary_embeddings

        # Check if we have them in state
        state = self.get_state()
        cached_embeddings = state.get("vocabulary_embeddings")
        if cached_embeddings:
            self.vocabulary_embeddings = cached_embeddings
            logger.info(
                f"Loaded {len(self.vocabulary_embeddings)} vocabulary embeddings from state"
            )
            return self.vocabulary_embeddings

        # Try to load pre-built embeddings first
        prebuilt_embeddings = await self.load_prebuilt_vocabulary_embeddings()
        if prebuilt_embeddings:
            return prebuilt_embeddings

        # If pre-built embeddings failed, load vocabulary and generate embeddings
        vocab = await self.load_vocabulary()
        if not vocab:
            logger.error("Failed to load vocabulary for embeddings")
            return {}

        self.vocabulary_embeddings = {}

        # Log the start of embedding process
        logger.info(f"Preloading embeddings for {len(vocab)} vocabulary words")

        # Process words sequentially
        for i, word in enumerate(vocab):
            if i % 100 == 0:  # Log progress every 100 words
                logger.info(f"Processing vocabulary word {i}/{len(vocab)}")

            # Get embedding for this word
            embedding = await self.get_embedding(word)
            if embedding:
                self.vocabulary_embeddings[word] = embedding

        logger.info(
            f"Generated embeddings for {len(self.vocabulary_embeddings)} vocabulary words"
        )

        # Store in state for persistence across calls
        self.update_state("vocabulary_embeddings", self.vocabulary_embeddings)

        return self.vocabulary_embeddings

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on the configured chunk level"""
        chunk_level = self.valves.CHUNK_LEVEL

        # If no chunking requested, return the whole text as a single chunk
        if chunk_level <= 0:
            return [text]

        # Level 1: Phrase-level chunking (split by commas, colons, semicolons)
        if chunk_level == 1:
            # Split by commas, colons, semicolons that are followed by a space
            # First split by newlines to maintain paragraph structure
            paragraphs = text.split("\n")

            # Then split each paragraph by phrases
            chunks = []
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue

                # Split paragraph into phrases
                paragraph_phrases = re.split(r"(?<=[,;:])\s+", paragraph)
                # Only add non-empty phrases
                for phrase in paragraph_phrases:
                    if phrase.strip():
                        chunks.append(phrase.strip())

            return chunks

        # Level 2: Sentence-level chunking (split by periods, exclamation, question marks)
        if chunk_level == 2:
            # Different handling for PDF vs regular content
            if self.is_pdf_content:
                # For PDFs: Don't remove newlines, handle sentences directly
                chunks = []
                # Split by sentences, preserving newlines
                sentences = re.split(r"(?<=[.!?])\s+", text)
                # Only add non-empty sentences
                for sentence in sentences:
                    if sentence.strip():
                        chunks.append(sentence.strip())
            else:
                # For regular content: First split by paragraphs
                paragraphs = text.split("\n")

                chunks = []
                for paragraph in paragraphs:
                    if not paragraph.strip():
                        continue

                    # Split paragraph into sentences
                    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                    # Only add non-empty sentences
                    for sentence in sentences:
                        if sentence.strip():
                            chunks.append(sentence.strip())

            return chunks

        # Level 3: Paragraph-level chunking
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        if chunk_level == 3:
            return paragraphs

        # Level 4-10: Multi-paragraph chunking (4=2 paragraphs, 5=3 paragraphs, etc.)
        chunks = []
        # Calculate how many paragraphs per chunk (chunk_level 4 = 2 paragraphs, 5 = 3 paragraphs, etc.)
        paragraphs_per_chunk = chunk_level - 2

        for i in range(0, len(paragraphs), paragraphs_per_chunk):
            chunk = "\n".join(paragraphs[i : i + paragraphs_per_chunk])
            chunks.append(chunk)

        return chunks

    async def compute_semantic_eigendecomposition(
        self, chunks, embeddings, cache_key=None
    ):
        """Perform semantic eigendecomposition on chunk embeddings with caching"""
        if not chunks or not embeddings or len(chunks) < 3:
            return None

        # Generate cache key if not provided
        if cache_key is None:
            # Create a stable cache key based on embeddings fingerprint
            embeddings_concat = np.concatenate(
                embeddings[: min(5, len(embeddings))], axis=0
            )
            fingerprint = np.mean(embeddings_concat, axis=0)
            cache_key = hash(str(fingerprint.round(2)))

        # Check cache first
        state = self.get_state()
        eigendecomposition_cache = state.get("eigendecomposition_cache", {})
        if cache_key in eigendecomposition_cache:
            return eigendecomposition_cache[cache_key]

        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings)

            # Check for invalid values
            if np.isnan(embeddings_array).any() or np.isinf(embeddings_array).any():
                logger.warning(
                    "Invalid values in embeddings, cannot perform eigendecomposition"
                )
                return None

            # Center the embeddings
            centered_embeddings = embeddings_array - np.mean(embeddings_array, axis=0)

            # Compute covariance matrix
            cov_matrix = np.cov(centered_embeddings, rowvar=False)

            # Perform eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Determine how many principal components to keep
            total_variance = np.sum(eigenvalues)
            if total_variance <= 0:
                logger.warning(
                    "Total variance is zero or negative, cannot continue with eigendecomposition"
                )
                return None

            explained_variance_ratio = eigenvalues / total_variance

            # Keep components that explain 80% of variance
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_components = np.argmax(cumulative_variance >= 0.8) + 1
            n_components = max(3, min(n_components, 10))  # At least 3, at most 10

            # Project embeddings onto principal components
            principal_components = eigenvectors[:, :n_components]
            projected_embeddings = np.dot(centered_embeddings, principal_components)

            result = {
                "eigenvalues": eigenvalues[:n_components].tolist(),
                "eigenvectors": principal_components.tolist(),
                "explained_variance": explained_variance_ratio[:n_components].tolist(),
                "projected_embeddings": projected_embeddings.tolist(),
                "n_components": n_components,
            }

            # Cache the result
            eigendecomposition_cache[cache_key] = result
            # Limit cache size
            if (
                len(eigendecomposition_cache) > 50
            ):  # Store up to 50 different decompositions
                oldest_key = next(iter(eigendecomposition_cache))
                del eigendecomposition_cache[oldest_key]
            self.update_state("eigendecomposition_cache", eigendecomposition_cache)

            return result
        except Exception as e:
            logger.error(f"Error in semantic eigendecomposition: {e}")
            return None

    async def create_semantic_transformation(
        self, semantic_eigendecomposition, pdv=None, trajectory=None, gap_vector=None
    ):
        """Create a semantic transformation matrix based on eigendecomposition and direction vectors"""
        if not semantic_eigendecomposition:
            return None

        # Generate a unique ID for this transformation
        state = self.get_state()
        transformation_id = f"transform_{str(hash(str(pdv)))[:8]}_{str(hash(str(trajectory)))[:8]}_{str(hash(str(gap_vector)))[:8]}"

        try:
            # Get principal components
            eigenvectors = np.array(semantic_eigendecomposition["eigenvectors"])
            eigenvalues = np.array(semantic_eigendecomposition["eigenvalues"])

            # Create initial transformation (identity)
            embedding_dim = eigenvectors.shape[0]
            transformation = np.eye(embedding_dim)

            # Get importance weights for each eigenvector
            variance_importance = eigenvalues / np.sum(eigenvalues)

            # Enhance dimensions based on eigenvalues (semantic importance)
            for i, importance in enumerate(variance_importance):
                eigenvector = eigenvectors[:, i]
                # Scale amplification by dimension importance
                amplification = 1.0 + importance * 2.0  # 1.0 to 3.0
                # Add outer product to emphasize this dimension
                transformation += (amplification - 1.0) * np.outer(
                    eigenvector, eigenvector
                )

            # Calculate weights for different direction vectors
            pdv_weight = (
                self.valves.SEMANTIC_TRANSFORMATION_STRENGTH
                * state.get("user_preferences", {}).get("impact", 0.0)
                if pdv is not None
                else 0.0
            )

            # Calculate trajectory weight
            trajectory_weight = (
                self.valves.TRAJECTORY_MOMENTUM if trajectory is not None else 0.0
            )

            # Calculate adaptive gap weight based on research progress
            gap_weight = 0.0
            if gap_vector is not None:
                # Get current cycle and max cycles for adaptive calculation
                current_cycle = len(state.get("cycle_summaries", [])) + 1
                max_cycles = self.valves.MAX_CYCLES
                fade_start_cycle = min(5, int(0.5 * max_cycles))

                # Get gap coverage history to analyze trend
                gap_coverage_history = state.get("gap_coverage_history", [])

                # Determine if gaps are still valuable for research direction
                if current_cycle <= fade_start_cycle:
                    # Early cycles: use full gap weight
                    gap_weight = self.valves.GAP_EXPLORATION_WEIGHT
                else:
                    # Calculate adaptive weight based on research progress
                    # Linear fade from full weight to zero
                    remaining_cycles = max_cycles - current_cycle
                    total_fade_cycles = max_cycles - fade_start_cycle
                    if total_fade_cycles > 0:  # Avoid division by zero
                        fade_ratio = remaining_cycles / total_fade_cycles
                        gap_weight = self.valves.GAP_EXPLORATION_WEIGHT * max(
                            0.0, fade_ratio
                        )
                    else:
                        gap_weight = 0.0

            # Normalize weights to sum to at most 0.8 (leaving some room for the eigendecomposition base)
            total_weight = pdv_weight + trajectory_weight + gap_weight
            if total_weight > 0.8:
                scale_factor = 0.8 / total_weight
                pdv_weight *= scale_factor
                trajectory_weight *= scale_factor
                gap_weight *= scale_factor

            # Apply PDV transformation
            if pdv is not None and pdv_weight > 0.1:
                pdv_array = np.array(pdv)
                norm = np.linalg.norm(pdv_array)
                if norm > 1e-10:
                    pdv_array = pdv_array / norm
                    transformation += pdv_weight * np.outer(pdv_array, pdv_array)

            # Apply trajectory transformation
            if trajectory is not None and trajectory_weight > 0.1:
                trajectory_array = np.array(trajectory)
                norm = np.linalg.norm(trajectory_array)
                if norm > 1e-10:
                    trajectory_array = trajectory_array / norm
                    transformation += trajectory_weight * np.outer(
                        trajectory_array, trajectory_array
                    )

            # Apply gap vector transformation
            if gap_vector is not None and gap_weight > 0.1:
                gap_array = np.array(gap_vector)
                norm = np.linalg.norm(gap_array)
                if norm > 1e-10:
                    gap_array = gap_array / norm
                    transformation += gap_weight * np.outer(gap_array, gap_array)

            return {
                "id": transformation_id,
                "matrix": transformation.tolist(),
                "dimension": embedding_dim,
                "pdv_weight": pdv_weight,
                "trajectory_weight": trajectory_weight,
                "gap_weight": gap_weight,
            }

        except Exception as e:
            logger.error(f"Error creating semantic transformation: {e}")
            return None

    async def apply_semantic_transformation(self, embedding, transformation):
        """Apply semantic transformation to an embedding"""
        if not transformation or not embedding:
            return embedding

        try:
            # Convert to numpy arrays
            embedding_array = np.array(embedding)

            # If transformation is an ID string, look up the transformation
            if isinstance(transformation, str):
                # In a real implementation, retrieve from cache/storage
                logger.warning(f"Transformation ID not found: {transformation}")
                return embedding

            # If it's a transformation object, get the matrix
            transform_matrix = np.array(transformation["matrix"])

            # Check for invalid values
            if (
                np.isnan(embedding_array).any()
                or np.isnan(transform_matrix).any()
                or np.isinf(embedding_array).any()
                or np.isinf(transform_matrix).any()
            ):
                logger.warning("Invalid values in embedding or transformation matrix")
                return embedding

            # Apply transformation
            transformed = np.dot(embedding_array, transform_matrix)

            # Check for valid result
            if np.isnan(transformed).any() or np.isinf(transformed).any():
                logger.warning("Transformation produced invalid values")
                return embedding

            # Normalize to unit vector
            norm = np.linalg.norm(transformed)
            if norm > 1e-10:  # Avoid division by near-zero
                transformed = transformed / norm
                return transformed.tolist()
            else:
                logger.warning("Transformation produced zero vector")
                return embedding
        except Exception as e:
            logger.error(f"Error applying semantic transformation: {e}")
            return embedding

    async def extract_token_window(
        self, content: str, start_token: int, window_size: int
    ) -> str:
        """Extract a window of tokens from content"""
        try:
            # Get a rough estimate of tokens per character in this content
            total_tokens = await self.count_tokens(content)
            chars_per_token = len(content) / max(1, total_tokens)

            # Approximate character positions
            start_char = int(start_token * chars_per_token)
            window_chars = int(window_size * chars_per_token)

            # Ensure we don't go out of bounds
            start_char = max(0, min(start_char, len(content) - 1))
            end_char = min(len(content), start_char + window_chars)

            # Extract the window
            window_content = content[start_char:end_char]

            # Ensure we have complete sentences
            # Find the first sentence boundary
            if start_char > 0:
                first_period = window_content.find(". ")
                if first_period > 0 and first_period < len(window_content) // 10:
                    window_content = window_content[first_period + 2 :]

            # Find the last sentence boundary
            last_period = window_content.rfind(". ")
            if last_period > 0 and last_period > len(window_content) * 0.9:
                window_content = window_content[: last_period + 1]

            return window_content

        except Exception as e:
            logger.error(f"Error extracting token window: {e}")
            # If error, return a portion of the content
            if len(content) > 0:
                # Calculate safe window
                safe_start = min(
                    len(content) - 1,
                    max(0, int(len(content) * (start_token / total_tokens))),
                )
                safe_end = min(len(content), safe_start + window_size)
                return content[safe_start:safe_end]
            return content

    async def clean_text_formatting(self, content: str) -> str:
        """Clean text formatting by merging short lines and handling repeated character patterns"""
        # Handle repeated character patterns first
        # Split into lines to process each line individually
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Check for repeated characters (5+ identical characters in a row)
            repeated_char_pattern = re.compile(
                r"((.)\2{4,})"
            )  # Same character 5+ times
            matches = list(repeated_char_pattern.finditer(line))

            if matches:
                # Process each match in reverse order to avoid index shifts
                for match in reversed(matches):
                    char_sequence = match.group(1)
                    char = match.group(2)
                    if len(char_sequence) >= 5:
                        # Keep first 2 and last 2 instances, replace middle with (...)
                        replacement = char * 2 + "(...)" + char * 2
                        start, end = match.span()
                        line = line[:start] + replacement + line[end:]

            # Check for repeated character patterns (like abc abc abc abc)
            # Look for patterns of 2-3 chars that repeat at least 3 times
            for pattern_length in range(2, 4):  # Check for 2-3 character patterns
                i = 0
                while (
                    i <= len(line) - pattern_length * 5
                ):  # Need at least 5 repetitions
                    pattern = line[i : i + pattern_length]

                    # Check if this is a repeating pattern
                    repetition_count = 0
                    for j in range(i, len(line) - pattern_length + 1, pattern_length):
                        if line[j : j + pattern_length] == pattern:
                            repetition_count += 1
                        else:
                            break

                    # If we found a repeated pattern
                    if repetition_count >= 5:
                        # Keep first 2 and last 2 repetitions, replace middle with (...)
                        replacement = pattern * 2 + "(...)" + pattern * 2
                        total_length = pattern_length * repetition_count
                        line = line[:i] + replacement + line[i + total_length :]

                    i += 1

            # Check for repeated patterns with ellipsis that are created by earlier processing
            ellipsis_pattern = re.compile(r"(\S\S\(\.\.\.\)\S\S\s+)(\1){2,}")
            ellipsis_matches = list(ellipsis_pattern.finditer(line))

            if ellipsis_matches:
                # Process each match in reverse order to avoid index shifts
                for match in reversed(ellipsis_matches):
                    # Replace multiple repetitions with just one instance
                    single_instance = match.group(1)
                    start, end = match.span()
                    line = line[:start] + single_instance + line[end:]

            cleaned_lines.append(line)

        # Now handle short lines processing with semantic awareness
        lines = cleaned_lines
        merged_lines = []
        short_line_group = []

        # Define better mixed case pattern (lowercase followed by uppercase in the same word)
        # This will match patterns like: "PsychologyFor", "MediaPsychology", etc.
        mixed_case_pattern = re.compile(r"[a-z][A-Z]")

        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            word_count = len(current_line.split())

            # Check if this is a short line (5 words or fewer)
            if word_count <= 5 and current_line:
                # Check if it's part of a numbered list
                is_numbered_item = False

                # Match various numbering patterns:
                # - "1. Item"
                # - "1) Item"
                # - "1: Item"
                # - "A. Item"
                # - "A) Item"
                # - "A: Item"
                # - "Item 1."
                # - "Item 1)"
                # - "Item 1:"
                number_patterns = [
                    r"^\d+[\.\)\:]",
                    r"^[A-Za-z][\.\)\:]",
                    r".*\d+[\.\)\:]$",
                ]

                # Check if line matches any numbered pattern
                for pattern in number_patterns:
                    if re.search(pattern, current_line):
                        is_numbered_item = True
                        break

                # Check if this is part of a sequence of numbered items
                if is_numbered_item and short_line_group:
                    # Look for sequential numbering
                    prev_number = None
                    curr_number = None

                    # Try to extract numbers from current and previous line
                    prev_line = short_line_group[-1]
                    prev_match = re.search(r"(\d+)[\.\)\:]", prev_line)
                    curr_match = re.search(r"(\d+)[\.\)\:]", current_line)

                    if prev_match and curr_match:
                        try:
                            prev_number = int(prev_match.group(1))
                            curr_number = int(curr_match.group(1))

                            # Check if sequential
                            if curr_number == prev_number + 1:
                                is_numbered_item = True
                            else:
                                is_numbered_item = False
                        except ValueError:
                            pass

                # If it's a numbered item in a sequence, treat as normal text
                if is_numbered_item:
                    # Add it as separate line
                    if short_line_group:
                        # Flush any existing short lines
                        for j, short_line in enumerate(short_line_group):
                            merged_lines.append(short_line)
                        short_line_group = []

                    # Add this numbered item
                    merged_lines.append(current_line)
                else:
                    # Add to current group of short lines
                    short_line_group.append(current_line)
            else:
                # Process any existing short line group before adding this line
                if short_line_group:
                    # Check if we have 5 or more short lines in a sequence
                    if len(short_line_group) >= 5:
                        # Count mixed case occurrences in the group
                        mixed_case_count = 0
                        total_lc_to_uc = 0

                        for line in short_line_group:
                            # Count individual lowercase-to-uppercase transitions
                            for j in range(1, len(line)):
                                if (
                                    j > 0
                                    and line[j - 1].islower()
                                    and line[j].isupper()
                                ):
                                    total_lc_to_uc += 1

                            # Also check if the line itself has the mixed case pattern
                            if mixed_case_pattern.search(line):
                                mixed_case_count += 1

                        # If many lines have mixed case patterns or there are many transitions,
                        # they're likely navigation/menu items
                        has_mixed_case = (
                            mixed_case_count >= len(short_line_group) * 0.3
                        ) or (total_lc_to_uc >= 3)

                        # Keep first two and last two, replace middle with note
                        if merged_lines:
                            # Combine first two with previous line if possible
                            for j in range(min(2, len(short_line_group))):
                                merged_lines[-1] += f". {short_line_group[j]}"

                            # Add note about removed headers
                            if has_mixed_case:
                                merged_lines.append("(Navigation menu removed)")
                            else:
                                merged_lines.append("(Headers removed)")

                            # Add last two as separate lines
                            last_idx = len(short_line_group) - 2
                            if (
                                last_idx >= 2
                            ):  # Ensure we have lines left after removing middle
                                merged_lines.append(short_line_group[last_idx])
                                merged_lines.append(short_line_group[last_idx + 1])
                        else:
                            # If no previous line, handle differently
                            for j in range(min(2, len(short_line_group))):
                                merged_lines.append(short_line_group[j])

                            # Add note about removed headers or menu
                            if has_mixed_case:
                                merged_lines.append("(Navigation menu removed)")
                            else:
                                merged_lines.append("(Headers removed)")

                            last_idx = len(short_line_group) - 2
                            if last_idx >= 2:
                                merged_lines.append(short_line_group[last_idx])
                                merged_lines.append(short_line_group[last_idx + 1])
                    else:
                        # For small groups, merge with previous line if possible
                        for j, short_line in enumerate(short_line_group):
                            if j == 0 and merged_lines:
                                # First short line gets merged with previous
                                merged_lines[-1] += f". {short_line}"
                            else:
                                # Subsequent lines added separately
                                merged_lines.append(short_line)

                    # Reset short line group
                    short_line_group = []

                # Add current non-short line
                if current_line:
                    merged_lines.append(current_line)

            i += 1

        # Handle any remaining short line group
        if short_line_group:
            if len(short_line_group) >= 5:
                # Count mixed case occurrences in the group
                mixed_case_count = 0
                total_lc_to_uc = 0

                for line in short_line_group:
                    # Count individual lowercase-to-uppercase transitions
                    for j in range(1, len(line)):
                        if j > 0 and line[j - 1].islower() and line[j].isupper():
                            total_lc_to_uc += 1

                    # Also check if the line itself has the mixed case pattern
                    if mixed_case_pattern.search(line):
                        mixed_case_count += 1

                # If many lines have mixed case patterns or there are many transitions,
                # they're likely navigation/menu items
                has_mixed_case = (mixed_case_count >= len(short_line_group) * 0.3) or (
                    total_lc_to_uc >= 3
                )

                # Keep first two and last two, replace middle with note
                if merged_lines:
                    # Combine first two with previous line if possible
                    for j in range(min(2, len(short_line_group))):
                        merged_lines[-1] += f". {short_line_group[j]}"

                    # Add note about removed headers
                    if has_mixed_case:
                        merged_lines.append("(Navigation menu removed)")
                    else:
                        merged_lines.append("(Headers removed)")

                    # Add last two as separate lines
                    last_idx = len(short_line_group) - 2
                    if last_idx >= 2:
                        merged_lines.append(short_line_group[last_idx])
                        merged_lines.append(short_line_group[last_idx + 1])
                else:
                    # If no previous line, handle differently
                    for j in range(min(2, len(short_line_group))):
                        merged_lines.append(short_line_group[j])

                    # Add appropriate removal note
                    if has_mixed_case:
                        merged_lines.append("(Navigation menu removed)")
                    else:
                        merged_lines.append("(Headers removed)")

                    last_idx = len(short_line_group) - 2
                    if last_idx >= 2:
                        merged_lines.append(short_line_group[last_idx])
                        merged_lines.append(short_line_group[last_idx + 1])
            else:
                # For small groups, merge with previous line if possible
                for j, short_line in enumerate(short_line_group):
                    if j == 0 and merged_lines:
                        # First short line gets merged with previous
                        merged_lines[-1] += f". {short_line}"
                    else:
                        # Subsequent lines added separately
                        merged_lines.append(short_line)

        return "\n".join(merged_lines)

    async def compress_content_with_local_similarity(
        self,
        content: str,
        query_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
        ratio: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Apply semantic compression with local similarity influence and token limiting"""
        # Skip compression for very short content
        if len(content) < 100:
            return content

        # Apply token limit if specified
        if max_tokens:
            content_tokens = await self.count_tokens(content)
            if content_tokens <= max_tokens:
                return content

            # If over limit, use token-based compression ratio
            if not ratio:
                ratio = max_tokens / content_tokens

        # Split content into chunks based on chunk_level
        chunks = self.chunk_text(content)

        # Skip compression if only one chunk
        if len(chunks) <= 1:
            return content

        # Get embeddings for chunks sequentially
        chunk_embeddings = []
        for chunk in chunks:
            embedding = await self.get_embedding(chunk)
            if embedding:
                chunk_embeddings.append(embedding)

        # Skip compression if not enough embeddings
        if len(chunk_embeddings) <= 1:
            return content

        # Define compression ratio if not provided
        if ratio is None:
            compress_ratios = {
                1: 0.9,  # 90% - minimal compression
                2: 0.8,  # 80%
                3: 0.7,  # 70%
                4: 0.6,  # 60%
                5: 0.5,  # 50% - moderate compression
                6: 0.4,  # 40%
                7: 0.3,  # 30%
                8: 0.2,  # 20%
                9: 0.15,  # 15%
                10: 0.1,  # 10% - maximum compression
            }
            level = self.valves.COMPRESSION_LEVEL
            ratio = compress_ratios.get(level, 0.5)

        # Calculate how many chunks to keep
        n_chunks = len(chunk_embeddings)
        n_keep = max(1, min(n_chunks - 1, int(n_chunks * ratio)))

        # Ensure we're compressing at least a little
        if n_keep >= n_chunks:
            n_keep = max(1, n_chunks - 1)

        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(chunk_embeddings)

            # Calculate document centroid
            document_centroid = np.mean(embeddings_array, axis=0)

            # Calculate local similarity for each chunk
            local_similarities = []
            local_radius = self.valves.LOCAL_INFLUENCE_RADIUS  # Get from valve

            for i in range(len(embeddings_array)):
                # Calculate similarity to adjacent chunks (local influence)
                local_sim = 0.0
                count = 0

                # Check previous chunks within radius
                for j in range(max(0, i - local_radius), i):
                    local_sim += cosine_similarity(
                        [embeddings_array[i]], [embeddings_array[j]]
                    )[0][0]
                    count += 1

                # Check next chunks within radius
                for j in range(i + 1, min(len(embeddings_array), i + local_radius + 1)):
                    local_sim += cosine_similarity(
                        [embeddings_array[i]], [embeddings_array[j]]
                    )[0][0]
                    count += 1

                if count > 0:
                    local_sim /= count

                local_similarities.append(local_sim)

            # Calculate importance scores with all factors
            importance_scores = []
            state = self.get_state()
            user_preferences = state.get(
                "user_preferences", {"pdv": None, "strength": 0.0, "impact": 0.0}
            )

            for i, embedding in enumerate(embeddings_array):
                # Fix any NaN or Inf values
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    embedding = np.nan_to_num(
                        embedding, nan=0.0, posinf=1.0, neginf=-1.0
                    )

                # Calculate similarity to document centroid
                doc_similarity = cosine_similarity([embedding], [document_centroid])[0][
                    0
                ]

                # Calculate similarity to query
                query_similarity = cosine_similarity([embedding], [query_embedding])[0][
                    0
                ]

                # Calculate similarity to previous summary if provided
                summary_similarity = 0.0
                if summary_embedding is not None:
                    summary_similarity = cosine_similarity(
                        [embedding], [summary_embedding]
                    )[0][0]
                    # Blend query and summary similarity
                    query_similarity = (
                        query_similarity * self.valves.FOLLOWUP_WEIGHT
                    ) + (summary_similarity * (1.0 - self.valves.FOLLOWUP_WEIGHT))

                # Include local similarity influence
                local_influence = local_similarities[i]

                # Include preference direction vector if available
                pdv_alignment = 0.5  # Neutral default
                if (
                    self.valves.USER_PREFERENCE_THROUGHOUT
                    and user_preferences["pdv"] is not None
                ):
                    chunk_embedding_np = np.array(embedding)
                    pdv_np = np.array(user_preferences["pdv"])
                    alignment = np.dot(chunk_embedding_np, pdv_np)
                    pdv_alignment = (alignment + 1) / 2  # Normalize to 0-1

                    # Weight by preference strength
                    pdv_influence = min(0.3, user_preferences["strength"] / 10)
                else:
                    pdv_influence = 0.0

                # Weight the factors
                doc_weight = (
                    1.0 - self.valves.QUERY_WEIGHT
                ) * 0.4  # Some preference towards relevance towards query
                local_weight = (
                    1.0 - self.valves.QUERY_WEIGHT
                ) * 0.8  # More preference towards standout local chunks
                query_weight = self.valves.QUERY_WEIGHT * (1.0 - pdv_influence)

                final_score = (
                    (doc_similarity * doc_weight)
                    + (query_similarity * query_weight)
                    + (local_influence * local_weight)
                    + (pdv_alignment * pdv_influence)
                )

                importance_scores.append((i, final_score))

            # Sort chunks by importance (most important first)
            importance_scores.sort(key=lambda x: x[1], reverse=True)

            # Select the top n_keep most important chunks
            selected_indices = [x[0] for x in importance_scores[:n_keep]]

            # Sort indices to maintain original document order
            selected_indices.sort()

            # Get the selected chunks
            selected_chunks = [chunks[i] for i in selected_indices if i < len(chunks)]

            # Join compressed chunks back into text with proper formatting
            chunk_level = self.valves.CHUNK_LEVEL
            if chunk_level == 1:  # Phrase level
                compressed_content = " ".join(selected_chunks)
            elif chunk_level == 2:  # Sentence level
                processed_sentences = []
                for sentence in selected_chunks:
                    if not sentence.endswith((".", "!", "?", ":", ";")):
                        sentence += "."
                    processed_sentences.append(sentence)
                compressed_content = " ".join(processed_sentences)
            else:  # Paragraph levels
                compressed_content = "\n".join(selected_chunks)

            # Verify token count if max_tokens specified
            if max_tokens:
                final_tokens = await self.count_tokens(compressed_content)

                # If still over limit, apply additional compression
                if final_tokens > max_tokens:
                    # Calculate new ratio based on tokens
                    new_ratio = max_tokens / final_tokens
                    # Recursively compress with more aggressive ratio
                    compressed_content = (
                        await self.compress_content_with_local_similarity(
                            compressed_content,
                            query_embedding,
                            summary_embedding,
                            ratio=new_ratio,
                        )
                    )

            return compressed_content

        except Exception as e:
            logger.error(f"Error during compression with local similarity: {e}")

            # If max_tokens specified and error occurred, do basic truncation
            if max_tokens and content:
                # Estimate character position based on token limit
                content_tokens = await self.count_tokens(content)
                if content_tokens > max_tokens:
                    char_ratio = max_tokens / content_tokens
                    char_limit = int(len(content) * char_ratio)
                    return content[:char_limit]

            return content

    async def compress_content_with_eigendecomposition(
        self,
        content: str,
        query_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
        ratio: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Apply semantic compression using eigendecomposition with token limiting"""
        # Skip compression for very short content
        if len(content) < 200:
            return content

        # Apply token limit if specified
        if max_tokens:
            content_tokens = await self.count_tokens(content)
            if content_tokens <= max_tokens:
                return content

            # If over limit, use token-based compression ratio
            if not ratio:
                ratio = max_tokens / content_tokens

        # Split content into chunks based on chunk_level
        chunks = self.chunk_text(content)

        # Skip compression if only one chunk
        if len(chunks) <= 2:
            return content

        # Get embeddings for chunks sequentially
        chunk_embeddings = []
        for chunk in chunks:
            embedding = await self.get_embedding(chunk)
            if embedding:
                chunk_embeddings.append(embedding)

        # Skip compression if not enough embeddings
        if len(chunk_embeddings) <= 2:
            return content

        # Define compression ratio if not provided
        if ratio is None:
            compress_ratios = {
                1: 0.9,  # 90% - minimal compression
                2: 0.8,  # 80%
                3: 0.7,  # 70%
                4: 0.6,  # 60%
                5: 0.5,  # 50% - moderate compression
                6: 0.4,  # 40%
                7: 0.3,  # 30%
                8: 0.2,  # 20%
                9: 0.15,  # 15%
                10: 0.1,  # 10% - maximum compression
            }
            level = self.valves.COMPRESSION_LEVEL
            ratio = compress_ratios.get(level, 0.5)

        # Calculate how many chunks to keep
        n_chunks = len(chunks)
        n_keep = max(1, min(n_chunks - 1, int(n_chunks * ratio)))

        # Ensure we're compressing at least a little
        if n_keep >= n_chunks:
            n_keep = max(1, n_chunks - 1)

        try:
            # Perform semantic eigendecomposition
            eigendecomposition = await self.compute_semantic_eigendecomposition(
                chunks, chunk_embeddings
            )

            if eigendecomposition:
                # Calculate importance scores based on the eigendecomposition
                embeddings_array = np.array(chunk_embeddings)
                importance_scores = []

                # Create basic directions
                directions = {}
                if query_embedding:
                    directions["query"] = query_embedding
                if summary_embedding:
                    directions["summary"] = summary_embedding

                state = self.get_state()
                user_preferences = state.get(
                    "user_preferences", {"pdv": None, "strength": 0.0, "impact": 0.0}
                )
                if user_preferences["pdv"] is not None:
                    directions["pdv"] = user_preferences["pdv"]

                # --- FIX STARTS HERE ---
                # Correctly get semantic_transformations from the state
                semantic_transformations = state.get("semantic_transformations")
                # --- FIX ENDS HERE ---

                # Create transformation
                transformation = await self.create_semantic_transformation(
                    eigendecomposition,
                    pdv=(
                        user_preferences["pdv"]
                        if user_preferences["impact"] > 0.1
                        else None
                    ),
                )

                # Project chunks into the principal component space for better analysis
                projected_chunks = eigendecomposition["projected_embeddings"]
                eigenvectors = np.array(eigendecomposition["eigenvectors"])

                # Calculate local coherence using the eigenspace
                local_coherence = []
                local_radius = self.valves.LOCAL_INFLUENCE_RADIUS

                for i in range(len(projected_chunks)):
                    # Calculate similarity to adjacent chunks
                    local_sim = 0.0
                    count = 0

                    # Look at previous and next chunks within radius
                    for j in range(
                        max(0, i - local_radius),
                        min(len(projected_chunks), i + local_radius + 1),
                    ):
                        if i == j:
                            continue

                        # Use weighted similarity in eigenspace
                        sim = 0.0
                        for k in range(eigendecomposition["n_components"]):
                            # Weight by eigenvalue importance
                            weight = eigendecomposition["explained_variance"][k]
                            dim_sim = 1.0 - abs(
                                projected_chunks[i][k] - projected_chunks[j][k]
                            )
                            sim += weight * dim_sim

                        local_sim += sim
                        count += 1

                    if count > 0:
                        local_sim /= count
                    local_coherence.append(local_sim)

                # Calculate relevance to query using transformed embeddings
                if query_embedding:
                    try:
                        # Ensure we're getting transformed embeddings if a transformation is available
                        if semantic_transformations:
                            transformed_query = (
                                await self.apply_semantic_transformation(
                                    query_embedding, semantic_transformations
                                )
                            )
                            if transformed_query:
                                query_embedding = transformed_query

                        # Calculate similarities with transformed query in one operation
                        query_relevance = []
                        for chunk_embedding in chunk_embeddings:
                            if chunk_embedding:
                                # Get similarity to transformed query
                                similarity = cosine_similarity(
                                    [chunk_embedding], [query_embedding]
                                )[0][0]
                                query_relevance.append(similarity)
                            else:
                                query_relevance.append(
                                    0.5
                                )  # Default for missing embeddings
                    except Exception as e:
                        logger.warning(f"Error calculating query relevance: {e}")
                        query_relevance = [0.5] * len(projected_chunks)
                else:
                    # Default relevance if no query
                    query_relevance = [0.5] * len(projected_chunks)

                # Combine scores
                for i in range(len(chunks)):
                    if i >= len(local_coherence) or i >= len(query_relevance):
                        continue

                    # Weights for different factors
                    coherence_weight = 0.4
                    relevance_weight = 0.6

                    # Adjust based on user preferences
                    if (
                        user_preferences["pdv"] is not None
                        and user_preferences["impact"] > 0.1
                    ):
                        # Reduce other weights to make room for preference weight
                        pdv_weight = min(0.3, user_preferences["impact"])
                        coherence_weight *= 1.0 - pdv_weight
                        relevance_weight *= 1.0 - pdv_weight

                        # Calculate PDV alignment if available
                        if i < len(chunk_embeddings):
                            try:
                                chunk_embed = chunk_embeddings[i]
                                pdv_alignment = np.dot(
                                    chunk_embed, user_preferences["pdv"]
                                )
                                # Normalize to 0-1 range
                                pdv_alignment = (pdv_alignment + 1) / 2
                            except Exception as e:
                                logger.warning(f"Error calculating PDV alignment: {e}")
                                pdv_alignment = 0.5
                        else:
                            pdv_alignment = 0.5

                        final_score = (
                            (local_coherence[i] * coherence_weight)
                            + (query_relevance[i] * relevance_weight)
                            + (pdv_alignment * pdv_weight)
                        )
                    else:
                        final_score = (local_coherence[i] * coherence_weight) + (
                            query_relevance[i] * relevance_weight
                        )

                    importance_scores.append((i, final_score))

                # Sort chunks by importance
                importance_scores.sort(key=lambda x: x[1], reverse=True)

                # Select the top n_keep chunks
                selected_indices = [x[0] for x in importance_scores[:n_keep]]

                # Sort to maintain document order
                selected_indices.sort()

                # Get selected chunks
                selected_chunks = [
                    chunks[i] for i in selected_indices if i < len(chunks)
                ]

                # Join compressed chunks with proper formatting
                chunk_level = self.valves.CHUNK_LEVEL
                if chunk_level == 1:  # Phrase level
                    compressed_content = " ".join(selected_chunks)
                elif chunk_level == 2:  # Sentence level
                    processed_sentences = []
                    for sentence in selected_chunks:
                        if not sentence.endswith((".", "!", "?", ":", ";")):
                            sentence += "."
                        processed_sentences.append(sentence)
                    compressed_content = " ".join(processed_sentences)
                else:  # Paragraph levels
                    compressed_content = "\n".join(selected_chunks)

                # Verify token count if max_tokens specified
                if max_tokens:
                    final_tokens = await self.count_tokens(compressed_content)

                    # If still over limit, apply additional compression
                    if final_tokens > max_tokens:
                        # Calculate new ratio based on tokens
                        new_ratio = max_tokens / final_tokens
                        # Recursively compress with more aggressive ratio
                        compressed_content = (
                            await self.compress_content_with_eigendecomposition(
                                compressed_content,
                                query_embedding,
                                summary_embedding,
                                ratio=new_ratio,
                            )
                        )

                return compressed_content

            # Fallback if eigendecomposition fails
            logger.warning(
                "Eigendecomposition compression failed, using original method"
            )
            return await self.compress_content_with_local_similarity(
                content, query_embedding, summary_embedding, ratio, max_tokens
            )

        except Exception as e:
            logger.error(f"Error during compression with eigendecomposition: {e}")
            # Fall back to original compression method
            try:
                return await self.compress_content_with_local_similarity(
                    content, query_embedding, summary_embedding, ratio, max_tokens
                )
            except Exception as fallback_error:
                logger.error(f"Fallback compression also failed: {fallback_error}")

                # If max_tokens specified and all compression failed, do basic truncation
                if max_tokens and content:
                    # Estimate character position based on token limit
                    content_tokens = await self.count_tokens(content)
                    if content_tokens > max_tokens:
                        char_ratio = max_tokens / content_tokens
                        char_limit = int(len(content) * char_ratio)
                        return content[:char_limit]

                return content  # Return original content if both methods fail

    async def handle_repeated_content(
        self, content: str, url: str, query_embedding: List[float], repeat_count: int
    ) -> str:
        """Process repeated content with improved sliding window and adaptive shrinkage"""
        state = self.get_state()
        url_selected_count = state.get("url_selected_count", {})
        url_token_counts = state.get("url_token_counts", {})

        # Only consider URLs that were actually shown to the user
        selected_count = url_selected_count.get(url, 0)

        # If first occurrence, return unchanged
        if selected_count < 1:
            total_tokens = await self.count_tokens(content)
            url_token_counts[url] = total_tokens
            self.update_state("url_token_counts", url_token_counts)
            return content

        # Get total tokens for this URL
        total_tokens = url_token_counts.get(url, 0)
        if total_tokens == 0:
            # Count if not already done
            total_tokens = await self.count_tokens(content)
            url_token_counts[url] = total_tokens
            self.update_state("url_token_counts", url_token_counts)

        # Calculate max window size
        max_tokens = self.valves.MAX_RESULT_TOKENS
        window_factor = self.valves.REPEAT_WINDOW_FACTOR

        # For any repeated result, decide whether to apply sliding window or compression/centering
        if total_tokens > max_tokens:
            # Large content - apply sliding window logic
            # Calculate window position based on repeat count and content size
            window_start = int((repeat_count - 1) * window_factor * max_tokens)

            # Check if we've reached the end of the content
            if window_start >= total_tokens:
                # We've cycled through once, now start shrinking
                cycles_completed = window_start // total_tokens

                # Calculate shrinkage: keep 70% for each full cycle completed
                shrink_factor = 0.7**cycles_completed

                # Calculate new window size with shrinkage
                window_size = int(max_tokens * shrink_factor)
                window_size = max(200, window_size)  # Set minimum window size

                # Recalculate start position for new cycle with smaller window
                window_start = window_start % total_tokens

                logger.info(
                    f"Repeat URL {url} (count: {selected_count}): applying shrinkage after full cycle. "
                    f"Factor: {shrink_factor:.2f}, window size: {window_size} tokens"
                )
            else:
                # Still sliding through content, use full window size
                window_size = max_tokens
                logger.info(
                    f"Repeat URL {url} (count: {selected_count}): sliding window, "
                    f"starting at token {window_start}, window size {window_size}"
                )

            # Extract window of tokens from content
            window_content = await self.extract_token_window(
                content, window_start, window_size
            )

            return window_content
        else:
            # Content already fits within max tokens - apply compression/centering
            logger.info(
                f"Repeat URL {url} (count: {selected_count}): applying compression/centering for content already within token limit"
            )

            # Get content embedding to find most relevant section
            content_embedding = await self.get_embedding(content[:2000])
            if not content_embedding:
                return content

            # Calculate relevance to query to identify most relevant portion
            try:
                # Get text chunks and their embeddings
                chunks = self.chunk_text(content)
                if len(chunks) <= 3:  # Not enough chunks to do meaningful re-centering
                    return content

                # Get chunk embeddings sequentially
                chunk_embeddings = []
                relevance_scores = []
                for i, chunk in enumerate(chunks):
                    chunk_embedding = await self.get_embedding(chunk[:2000])
                    if chunk_embedding:
                        chunk_embeddings.append(chunk_embedding)
                        relevance = cosine_similarity(
                            [chunk_embedding], [query_embedding]
                        )[0][0]
                        relevance_scores.append((i, relevance))

                # Sort by relevance
                relevance_scores.sort(key=lambda x: x[1], reverse=True)

                # Get most relevant chunk index
                if relevance_scores:
                    most_relevant_idx = relevance_scores[0][0]

                    # Re-center the window around the most relevant chunk
                    start_idx = max(0, most_relevant_idx - len(chunks) // 4)
                    end_idx = min(len(chunks), most_relevant_idx + len(chunks) // 4 + 1)

                    # Combine chunks to form re-centered content
                    recentered_content = "\n".join(chunks[start_idx:end_idx])
                    return recentered_content

            except Exception as e:
                logger.error(f"Error re-centering window: {e}")

            # Fallback to original content if re-centering fails
            return content

    async def apply_stepped_compression(
        self,
        results_history: List[Dict],
        query_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Apply tiered compression to all research results based on age"""
        if not self.valves.STEPPED_SYNTHESIS_COMPRESSION or len(results_history) <= 2:
            return results_history

        # Make a copy to avoid modifying the original
        results = results_history.copy()

        # Divide results into first 50% (older) and second 50% (newer)
        mid_point = len(results) // 2
        older_results = results[:mid_point]
        newer_results = results[mid_point:]

        # Track token counts before and after compression
        total_tokens_before = 0
        total_tokens_after = 0

        # Define token limit for results
        max_tokens = self.valves.COMPRESSION_SETPOINT

        # Process older results with standard compression
        processed_older = []
        for result in older_results:
            content = result.get("content", "")
            url = result.get("url", "")

            # Count tokens in original content
            original_tokens = await self.count_tokens(content)
            total_tokens_before += original_tokens

            # Skip very short content
            if len(content) < 300:
                result["tokens"] = original_tokens
                processed_older.append(result)
                total_tokens_after += original_tokens
                continue

            # Apply standard compression
            compression_level = self.valves.COMPRESSION_LEVEL

            # Map compression level to ratio
            compress_ratios = {
                1: 0.9,  # 90% - minimal compression
                2: 0.8,  # 80%
                3: 0.7,  # 70%
                4: 0.6,  # 60%
                5: 0.5,  # 50% - moderate compression
                6: 0.4,  # 40%
                7: 0.3,  # 30%
                8: 0.2,  # 20%
                9: 0.15,  # 15%
                10: 0.1,  # 10% - maximum compression
            }
            ratio = compress_ratios.get(compression_level, 0.5)

            try:
                # Compress using eigendecomposition with token limit
                compressed = await self.compress_content_with_eigendecomposition(
                    content, query_embedding, summary_embedding, ratio, max_tokens
                )

                # Count tokens in compressed content
                compressed_tokens = await self.count_tokens(compressed)
                total_tokens_after += compressed_tokens

                # Create new result with compressed content
                new_result = result.copy()
                new_result["content"] = compressed
                new_result["tokens"] = compressed_tokens

                # Log the token reduction
                logger.info(
                    f"Standard compression (older result): {original_tokens} → {compressed_tokens} tokens "
                    f"({compressed_tokens/original_tokens:.1%} of original)"
                )

                processed_older.append(new_result)

            except Exception as e:
                logger.error(f"Error during standard compression: {e}")
                # Keep original if compression fails
                result["tokens"] = original_tokens
                processed_older.append(result)
                total_tokens_after += original_tokens

        # Process newer results with more compression
        processed_newer = []
        for result in newer_results:
            content = result.get("content", "")
            url = result.get("url", "")

            # Count tokens in original content
            original_tokens = await self.count_tokens(content)
            total_tokens_before += original_tokens

            # Skip very short content
            if len(content) < 300:
                result["tokens"] = original_tokens
                processed_newer.append(result)
                total_tokens_after += original_tokens
                continue

            # Apply one level higher compression for newer results
            compression_level = min(10, self.valves.COMPRESSION_LEVEL + 1)

            # Map compression level to ratio
            compress_ratios = {
                1: 0.9,  # 90% - minimal compression
                2: 0.8,  # 80%
                3: 0.7,  # 70%
                4: 0.6,  # 60%
                5: 0.5,  # 50% - moderate compression
                6: 0.4,  # 40%
                7: 0.3,  # 30%
                8: 0.2,  # 20%
                9: 0.15,  # 15%
                10: 0.1,  # 10% - maximum compression
            }
            ratio = compress_ratios.get(compression_level, 0.5)

            try:
                # Compress using eigendecomposition with token limit
                compressed = await self.compress_content_with_eigendecomposition(
                    content, query_embedding, summary_embedding, ratio, max_tokens
                )

                # Count tokens in compressed content
                compressed_tokens = await self.count_tokens(compressed)
                total_tokens_after += compressed_tokens

                # Create new result with compressed content
                new_result = result.copy()
                new_result["content"] = compressed
                new_result["tokens"] = compressed_tokens

                # Log the token reduction
                logger.info(
                    f"Higher compression (newer result): {original_tokens} → {compressed_tokens} tokens "
                    f"({compressed_tokens/original_tokens:.1%} of original)"
                )

                processed_newer.append(new_result)

            except Exception as e:
                logger.error(f"Error during higher compression: {e}")
                # Keep original if compression fails
                result["tokens"] = original_tokens
                processed_newer.append(result)
                total_tokens_after += original_tokens

        # Log the overall token reduction
        token_reduction = total_tokens_before - total_tokens_after
        if total_tokens_before > 0:
            percent_reduction = (token_reduction / total_tokens_before) * 100
            logger.info(
                f"Stepped compression total results: {total_tokens_before} → {total_tokens_after} tokens "
                f"(saved {token_reduction} tokens, {percent_reduction:.1f}% reduction)"
            )

        # Update memory statistics consistently
        await self.update_token_counts()

        # Combine and return all results in original order
        return processed_older + processed_newer

    async def calculate_research_trajectory(self, previous_queries, successful_results):
        """Calculate the research trajectory based on successful searches from recent cycles only"""
        if not previous_queries or not successful_results:
            return None

        # Check trajectory cache to avoid expensive recalculation
        state = self.get_state()
        trajectory_cache = state.get("trajectory_cache", {})

        # Use limited recent items to create cache key
        recent_query_key = hash(
            str(
                previous_queries[-3:]
                if len(previous_queries) >= 3
                else previous_queries
            )
        )
        recent_result_key = hash(
            str([r.get("url", "") for r in successful_results[-5:] if "url" in r])
        )
        cache_key = f"{recent_query_key}_{recent_result_key}"

        if cache_key in trajectory_cache:
            logger.info(f"Using cached trajectory for key: {cache_key}")
            return trajectory_cache[cache_key]

        # Use the trajectory accumulator if initialized
        if self.trajectory_accumulator is None:
            # Initialize with first sample embedding dimension
            sample_embedding = None
            for result in successful_results[:6]:
                content = result.get("content", "")[:2000]
                if content:
                    sample_embedding = await self.get_embedding(content)
                    if sample_embedding:
                        embedding_dim = len(sample_embedding)
                        self.trajectory_accumulator = TrajectoryAccumulator(
                            embedding_dim
                        )
                        break

            # If we couldn't get a sample, use default dimension
            if not sample_embedding:
                self.trajectory_accumulator = TrajectoryAccumulator(384)

        try:
            # Limit to last 5 cycles worth of data for efficiency
            max_history_cycles = 5
            queries_per_cycle = self.valves.SEARCH_RESULTS_PER_QUERY
            results_per_query = self.valves.SUCCESSFUL_RESULTS_PER_QUERY

            # Calculate maximum items to keep
            max_queries = max_history_cycles * queries_per_cycle
            max_results = max_queries * results_per_query

            # Take only the most recent queries and results
            recent_queries = (
                previous_queries[-max_queries:]
                if len(previous_queries) > max_queries
                else previous_queries
            )
            recent_results = (
                successful_results[-max_results:]
                if len(successful_results) > max_results
                else successful_results
            )

            logger.info(
                f"Calculating research trajectory with {len(recent_queries)} recent queries and {len(recent_results)} recent results"
            )

            # Get embeddings for queries sequentially
            query_embeddings = []
            for query in recent_queries:
                embedding = await self.get_embedding(query)
                if embedding:
                    query_embeddings.append(embedding)

            # Process results sequentially
            result_embeddings = []
            for result in recent_results:
                content = result.get("content", "")
                if not content:
                    continue
                embedding = await self.get_embedding(content[:2000])
                if embedding:
                    result_embeddings.append(embedding)

            if not query_embeddings or not result_embeddings:
                return None

            # Update trajectory accumulator with new cycle data
            self.trajectory_accumulator.add_cycle_data(
                query_embeddings, result_embeddings
            )

            # Get accumulated trajectory
            trajectory = self.trajectory_accumulator.get_trajectory()

            # If trajectory exists and we have PDV, calculate alignment to track for adaptive fade-out
            if trajectory:
                # Store the trajectory
                trajectory_cache[cache_key] = trajectory
                # Limit cache size
                if len(trajectory_cache) > 10:
                    oldest_key = next(iter(trajectory_cache))
                    del trajectory_cache[oldest_key]
                self.update_state("trajectory_cache", trajectory_cache)

                # Calculate PDV alignment if available
                pdv = state.get("user_preferences", {}).get("pdv")
                if pdv:
                    # Calculate alignment between trajectory and PDV
                    pdv_array = np.array(pdv)
                    trajectory_array = np.array(trajectory)
                    alignment = np.dot(trajectory_array, pdv_array)
                    # Normalize to 0-1 range
                    alignment = (alignment + 1) / 2

                    # Store in alignment history
                    pdv_alignment_history = state.get("pdv_alignment_history", [])
                    pdv_alignment_history.append(alignment)
                    # Keep only recent history
                    if len(pdv_alignment_history) > 5:
                        pdv_alignment_history = pdv_alignment_history[-5:]
                    self.update_state("pdv_alignment_history", pdv_alignment_history)

                    logger.info(f"PDV-Trajectory alignment: {alignment:.3f}")

            return trajectory

        except Exception as e:
            logger.error(f"Error calculating research trajectory: {e}")
            return None

    async def calculate_gap_vector(self):
        """Calculate a vector pointing toward research gaps"""
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if not research_dimensions:
            return None

        try:
            coverage = np.array(research_dimensions["coverage"])
            components = np.array(research_dimensions["eigenvectors"])

            # Get current cycle for adaptive calculations
            current_cycle = len(state.get("cycle_summaries", [])) + 1
            max_cycles = self.valves.MAX_CYCLES
            fade_start_cycle = min(5, int(0.5 * max_cycles))

            # Determine adaptive fade-out based on research progress
            fade_factor = 1.0
            if current_cycle > fade_start_cycle:
                # Linear fade from full influence to zero
                remaining_cycles = max_cycles - current_cycle
                total_fade_cycles = max_cycles - fade_start_cycle
                if total_fade_cycles > 0:
                    fade_factor = max(0.0, remaining_cycles / total_fade_cycles)
                else:
                    fade_factor = 0.0

            # Early exit if we've faded out completely
            if fade_factor <= 0.01:
                logger.info("Gap vector faded out completely, returning None")
                return None

            # Store gap coverage for tracking
            gap_coverage_history = state.get("gap_coverage_history", [])
            gap_coverage_history.append(np.mean(coverage).item())
            if len(gap_coverage_history) > 5:
                gap_coverage_history = gap_coverage_history[-5:]
            self.update_state("gap_coverage_history", gap_coverage_history)

            # Create a weighted sum of components based on coverage gaps
            gap_vector = np.zeros(components.shape[1])

            for i, cov in enumerate(coverage):
                # Calculate gap (1.0 - coverage)
                gap = 1.0 - cov

                # Only consider significant gaps
                if gap > 0.3:
                    # Ensure components is a numpy array
                    if isinstance(components, np.ndarray) and i < len(components):
                        # Weight by gap size - larger gaps have more influence
                        gap_vector += gap * components[i]
                    else:
                        logger.warning(f"Invalid components at index {i}")

            # Apply adaptive fade-out
            gap_vector *= fade_factor

            # Check for NaN or Inf values
            if np.isnan(gap_vector).any() or np.isinf(gap_vector).any():
                logger.warning("Invalid values in gap vector")
                return None

            # Normalize
            norm = np.linalg.norm(gap_vector)
            if norm > 1e-10:
                gap_vector = gap_vector / norm
                return gap_vector.tolist()
            else:
                logger.warning("Gap vector has zero norm")
                return None
        except Exception as e:
            logger.error(f"Error calculating gap vector: {e}")
            return None

    async def update_topic_usage_counts(self, used_topics):
        """Update usage counts for topics used in queries"""
        state = self.get_state()
        topic_usage_counts = state.get("topic_usage_counts", {})

        # Increment counter for each used topic
        for topic in used_topics:
            topic_usage_counts[topic] = topic_usage_counts.get(topic, 0) + 1

        # Store updated counts
        self.update_state("topic_usage_counts", topic_usage_counts)

    async def calculate_query_similarity(
        self,
        content_embedding: List[float],
        query_embedding: List[float],
        outline_embedding: Optional[List[float]] = None,
        summary_embedding: Optional[List[float]] = None,
    ) -> float:
        """Calculate similarity to query with optional context embeddings using caching"""

        # Get similarity cache
        state = self.get_state()
        similarity_cache = state.get("similarity_cache", {})

        # Generate cache keys for each embedding
        content_key = hash(str(np.array(content_embedding).round(2)))
        query_key = hash(str(np.array(query_embedding).round(2)))

        # First check if we have the full combined similarity cached
        combined_key = f"combined_{content_key}_{query_key}"
        if outline_embedding:
            outline_key = hash(str(np.array(outline_embedding).round(2)))
            combined_key += f"_{outline_key}"
        if summary_embedding:
            summary_key = hash(str(np.array(summary_embedding).round(2)))
            combined_key += f"_{summary_key}"

        if combined_key in similarity_cache:
            return similarity_cache[combined_key]

        # Convert to numpy arrays
        c_emb = np.array(content_embedding)
        q_emb = np.array(query_embedding)

        # Normalize embeddings
        c_emb = c_emb / np.linalg.norm(c_emb)
        q_emb = q_emb / np.linalg.norm(q_emb)

        # Check cache for base query similarity
        base_key = f"{content_key}_{query_key}"
        if base_key in similarity_cache:
            query_sim = similarity_cache[base_key]
        else:
            # Base query similarity using cosine similarity
            query_sim = np.dot(c_emb, q_emb)
            # Cache the result
            similarity_cache[base_key] = query_sim

        # Weight factors
        query_weight = 0.4  # Primary query importance
        outline_weight = 0.3  # Research outline importance
        summary_weight = 0.3  # Previous summary importance

        # If we have an outline embedding, include it
        outline_sim = 0.0
        if outline_embedding is not None:
            # Check cache for outline similarity
            outline_key = hash(str(np.array(outline_embedding).round(2)))
            outline_cache_key = f"{content_key}_{outline_key}"

            if outline_cache_key in similarity_cache:
                outline_sim = similarity_cache[outline_cache_key]
            else:
                o_emb = np.array(outline_embedding)
                o_emb = o_emb / np.linalg.norm(o_emb)
                outline_sim = np.dot(c_emb, o_emb)
                # Cache the result
                similarity_cache[outline_cache_key] = outline_sim
        else:
            # Redistribute weight
            query_weight += outline_weight
            outline_weight = 0.0

        # If we have a summary embedding (for follow-ups), include it
        summary_sim = 0.0
        if summary_embedding is not None:
            # Check cache for summary similarity
            summary_key = hash(str(np.array(summary_embedding).round(2)))
            summary_cache_key = f"{content_key}_{summary_key}"

            if summary_cache_key in similarity_cache:
                summary_sim = similarity_cache[summary_cache_key]
            else:
                s_emb = np.array(summary_embedding)
                s_emb = s_emb / np.linalg.norm(s_emb)
                summary_sim = np.dot(c_emb, s_emb)
                # Cache the result
                similarity_cache[summary_cache_key] = summary_sim
        else:
            # Redistribute weight
            query_weight += summary_weight
            summary_weight = 0.0

        # Weighted combination of similarities
        combined_sim = (
            (query_sim * query_weight)
            + (outline_sim * outline_weight)
            + (summary_sim * summary_weight)
        )

        # Cache the combined result
        similarity_cache[combined_key] = combined_sim

        # Limit cache size
        if len(similarity_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(similarity_cache.keys())[:200]
            for k in keys_to_remove:
                del similarity_cache[k]

        # Update similarity cache
        self.update_state("similarity_cache", similarity_cache)

        return combined_sim

    async def scale_token_limit_by_relevance(
        self,
        result: Dict,
        query_embedding: List[float],
        pdv: Optional[List[float]] = None,
    ) -> int:
        """Scale the token limit for a result based on its relevance to the query and PDV"""
        base_token_limit = self.valves.MAX_RESULT_TOKENS

        # Default to base if no similarity available
        if "similarity" not in result:
            return base_token_limit

        # Get the similarity score
        similarity = result.get("similarity", 0.5)

        # Calculate PDV alignment if available
        pdv_alignment = 0.5  # Neutral default
        if pdv is not None:
            try:
                # Get result content embedding
                content = result.get("content", "")
                content_embedding = await self.get_embedding(content[:2000])

                if content_embedding:
                    # Calculate alignment with PDV
                    alignment = np.dot(content_embedding, pdv)
                    pdv_alignment = (alignment + 1) / 2  # Normalize to 0-1
            except Exception as e:
                logger.error(f"Error calculating PDV alignment: {e}")

        # Combine similarity and PDV alignment
        combined_relevance = (similarity * 0.7) + (pdv_alignment * 0.3)

        # Scale between 0.5x and 1.5x of base limit
        scaling_factor = 0.5 + (combined_relevance * 1.0)  # Range: 0.5 to 1.5
        scaled_limit = int(base_token_limit * scaling_factor)

        # Cap at reasonable minimum and maximum
        min_limit = int(base_token_limit * 0.5)  # 50% of base
        max_limit = int(base_token_limit * 1.5)  # 150% of base

        scaled_limit = max(min_limit, min(max_limit, scaled_limit))

        logger.info(
            f"Scaled token limit for result: {scaled_limit} tokens "
            f"(similarity: {similarity:.2f}, scaling factor: {scaling_factor:.2f})"
        )

        return scaled_limit

    async def calculate_preference_impact(self, kept_items, removed_items, all_topics):
        """Calculate the impact of user preferences based on the proportion modified"""
        if not kept_items or not removed_items:
            return 0.0

        # Calculate impact based on proportion of items removed
        total_items = len(all_topics)
        if total_items == 0:
            return 0.0

        impact = len(removed_items) / total_items
        logger.info(
            f"User preference impact: {impact:.3f} ({len(removed_items)}/{total_items} items removed)"
        )
        return impact

    async def calculate_preference_direction_vector(
        self, kept_items: List[str], removed_items: List[str], all_topics: List[str]
    ) -> Dict:
        """Calculate the Preference Direction Vector based on kept and removed items"""
        if not kept_items or not removed_items:
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

        # Get embeddings for kept and removed items in parallel
        kept_embeddings = []
        removed_embeddings = []

        # Get embeddings for kept items sequentially
        kept_embeddings = []
        for item in kept_items:
            embedding = await self.get_embedding(item)
            if embedding:
                kept_embeddings.append(embedding)

        removed_embeddings = []
        for item in removed_items:
            embedding = await self.get_embedding(item)
            if embedding:
                removed_embeddings.append(embedding)

        if not kept_embeddings or not removed_embeddings:
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

        try:
            # Calculate mean vectors
            kept_mean = np.mean(kept_embeddings, axis=0)
            removed_mean = np.mean(removed_embeddings, axis=0)

            # Check for NaN or Inf values
            if (
                np.isnan(kept_mean).any()
                or np.isnan(removed_mean).any()
                or np.isinf(kept_mean).any()
                or np.isinf(removed_mean).any()
            ):
                logger.warning("Invalid values in kept or removed mean vectors")
                return {"pdv": None, "strength": 0.0, "impact": 0.0}

            # Calculate the preference direction vector
            pdv = kept_mean - removed_mean

            # Normalize the vector
            pdv_norm = np.linalg.norm(pdv)
            if pdv_norm < 1e-10:
                logger.warning("PDV has near-zero norm")
                return {"pdv": None, "strength": 0.0, "impact": 0.0}

            pdv = pdv / pdv_norm

            # Calculate preference strength (distance between centroids)
            strength = np.linalg.norm(kept_mean - removed_mean)

            # Calculate impact factor based on proportion of items removed
            impact = await self.calculate_preference_impact(
                kept_items, removed_items, all_topics
            )

            return {"pdv": pdv.tolist(), "strength": float(strength), "impact": impact}
        except Exception as e:
            logger.error(f"Error calculating PDV: {e}")
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

    async def translate_pdv_to_words(self, pdv):
        """Translate a Preference Direction Vector (PDV) into human-readable concepts using vocabulary embeddings"""
        if not pdv:
            return None

        # Ensure we have vocabulary embeddings
        if not self.vocabulary_embeddings:
            # If not loaded yet, load them now
            self.vocabulary_embeddings = await self.load_vocabulary_embeddings()

        if not self.vocabulary_embeddings:
            return None

        try:
            # Convert PDV to numpy array
            pdv_array = np.array(pdv)

            # Find vocabulary words that align with this direction
            word_alignments = []
            for word, embedding in self.vocabulary_embeddings.items():
                # Calculate alignment (dot product) with PDV
                alignment = np.dot(pdv_array, embedding)
                word_alignments.append((word, alignment))

            # Get top aligned words (highest dot product)
            top_words = sorted(word_alignments, key=lambda x: x[1], reverse=True)[:10]

            # Return as comma-separated string
            return ", ".join([word for word, _ in top_words])
        except Exception as e:
            logger.error(f"Error translating PDV to words: {e}")
            return None

    async def translate_dimensions_to_words(self, dimensions, coverage):
        """Translate research dimensions to human-readable concepts using vocabulary embeddings with caching"""
        # --- FIX STARTS HERE ---
        # Gracefully handle the case where dimensions might be None
        if not dimensions or not coverage or not dimensions.get("eigenvectors"):
            return []
        # --- FIX ENDS HERE ---

        # Get state for caching
        state = self.get_state()
        dimensions_cache = state.get("dimensions_translation_cache", {})

        # Create a unique cache key based on dimensions and coverage
        dim_hash = hash(str(dimensions.get("eigenvectors", [])[:3]))
        coverage_hash = hash(str(coverage))
        cache_key = f"dim_{dim_hash}_{coverage_hash}"

        # Check if we have a cached translation
        if cache_key in dimensions_cache:
            logger.info(f"Using cached dimension translation")
            return dimensions_cache[cache_key]

        dimension_labels = []

        # Ensure we have vocabulary embeddings
        if not self.vocabulary_embeddings:
            # If not loaded yet, load them now
            self.vocabulary_embeddings = await self.load_vocabulary_embeddings()

        if not self.vocabulary_embeddings:
            default_labels = [f"Dimension {i+1}" for i in range(len(coverage))]
            dimensions_cache[cache_key] = default_labels
            self.update_state("dimensions_translation_cache", dimensions_cache)
            return default_labels

        # Get eigenvectors which represent the dimensions
        eigenvectors = np.array(dimensions.get("eigenvectors", []))

        if len(eigenvectors) == 0 or len(eigenvectors) != len(coverage):
            default_labels = [f"Dimension {i+1}" for i in range(len(coverage))]
            dimensions_cache[cache_key] = default_labels
            self.update_state("dimensions_translation_cache", dimensions_cache)
            return default_labels

        try:
            # Process each dimension
            for i, eigen_vector in enumerate(eigenvectors):
                # Find vocabulary words that align with this dimension
                word_alignments = []
                for word, embedding in self.vocabulary_embeddings.items():
                    # Calculate alignment (dot product) with dimension vector
                    alignment = np.dot(eigen_vector, embedding)
                    word_alignments.append((word, alignment))

                # Get top positive aligned words
                top_words = sorted(word_alignments, key=lambda x: x[1], reverse=True)[
                    :3
                ]
                top_words_str = ", ".join([word for word, _ in top_words])

                # Create label
                cov_percentage = coverage[i]
                dimension_labels.append(
                    {
                        "dimension": i + 1,
                        "words": top_words_str,
                        "coverage": cov_percentage,
                    }
                )

            # Cache the translation
            dimensions_cache[cache_key] = dimension_labels
            self.update_state("dimensions_translation_cache", dimensions_cache)

            return dimension_labels
        except Exception as e:
            logger.error(f"Error translating dimensions to words: {e}")
            default_labels = [f"Dimension {i+1}" for i in range(len(coverage))]
            dimensions_cache[cache_key] = default_labels
            self.update_state("dimensions_translation_cache", dimensions_cache)
            return default_labels

    async def calculate_preference_alignment(self, content_embedding, pdv):
        """Calculate alignment between content and preference vector"""
        if not pdv or not content_embedding:
            return 0.5  # Neutral value if we can't calculate

        try:
            # Calculate dot product between vectors
            alignment = np.dot(content_embedding, pdv)

            # Normalize to 0-1 scale (dot product is between -1 and 1)
            normalized = (alignment + 1) / 2

            return normalized
        except Exception as e:
            logger.error(f"Error calculating preference alignment: {e}")
            return 0.5  # Neutral value on error

    async def update_research_dimensions_display(self):
        """Ensure research dimensions are properly updated for display"""
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")

        if research_dimensions:
            # Don't make a separate copy - just point to the actual dimension coverage
            coverage = research_dimensions.get("coverage", [])
            if coverage:
                self.update_state("latest_dimension_coverage", coverage)
                logger.info(
                    f"Updated latest dimension coverage with {len(coverage)} values"
                )
            else:
                logger.warning("Research dimensions exist but coverage is empty")
        else:
            logger.warning("No research dimensions available for display")

    async def initialize_research_dimensions(
        self, outline_items: List[str], user_query: str
    ):
        """Initialize the semantic dimensions for tracking research progress"""
        try:
            logger.debug(
                f"Starting research dimensions initialization with {len(outline_items)} items"
            )
            logger.debug(f"Outline items: {outline_items}")
            await self.emit_status("info", "Initializing research dimensions...", False)
            # Get embeddings for each outline item sequentially
            item_embeddings = []
            for i, item in enumerate(outline_items):
                logger.debug(
                    f"Getting embedding for item {i+1}/{len(outline_items)}: {item[:100]}..."
                )
                embedding = await self.get_embedding(item[:2000])
                if embedding:
                    item_embeddings.append(embedding)
                    logger.debug(f"Successfully got embedding for item {i+1}")
                else:
                    logger.warning(
                        f"Failed to get embedding for item {i+1}: {item[:100]}..."
                    )

            # Ensure we have enough embeddings for PCA
            if len(item_embeddings) < 3:
                logger.warning(
                    f"Not enough valid embeddings for research dimensions: {len(item_embeddings)}/3 required"
                )
                await self.emit_status(
                    "warning",
                    "Research will continue without dimensional analysis. This won't affect the quality of results.",
                    False,
                )
                self.update_state("research_dimensions", None)
                return

            # Apply PCA to reduce to key dimensions
            try:
                pca = PCA(n_components=min(10, len(item_embeddings)))
                embedding_array = np.array(item_embeddings)
                pca.fit(embedding_array)

                # Store the PCA model and progress trackers
                research_dimensions = {
                    "eigenvectors": pca.components_.tolist(),
                    "eigenvalues": pca.explained_variance_.tolist(),
                    "explained_variance": pca.explained_variance_ratio_.tolist(),
                    "total_variance": pca.explained_variance_ratio_.sum(),
                    "dimensions": pca.n_components_,
                    "coverage": np.zeros(
                        pca.n_components_
                    ).tolist(),  # Initialize empty coverage
                }

                self.update_state("research_dimensions", research_dimensions)
                # Immediately store a copy of coverage for display
                self.update_state(
                    "latest_dimension_coverage", research_dimensions["coverage"]
                )
                await self.emit_status(
                    "info", "Research dimensions initialized successfully", False
                )
                logger.debug(
                    "Research dimensions initialization completed successfully"
                )
            except Exception as e:
                logger.error(f"Error in dimensional analysis: {e}")
                await self.emit_status(
                    "warning",
                    "Continuing without dimensional analysis due to computation error.",
                    False,
                )
                self.update_state("research_dimensions", None)
        except Exception as e:
            logger.error(f"Failed to initialize research dimensions: {e}")
            await self.emit_status(
                "warning",
                "Proceeding without dimensional analysis due to initialization error.",
                False,
            )
            self.update_state("research_dimensions", None)

        logger.debug("Exiting initialize_research_dimensions method")

    async def update_dimension_coverage(
        self, content: str, quality_factor: float = 1.0
    ):
        """Update the coverage of research dimensions based on new content"""
        # Get current state
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if not research_dimensions:
            return

        try:
            # Get embedding for the content
            content_embedding = await self.get_embedding(content[:2000])
            if not content_embedding:
                return

            # Get current coverage
            current_coverage = research_dimensions.get("coverage", [])
            eigenvectors = research_dimensions.get("eigenvectors", [])

            if not current_coverage or not eigenvectors:
                return

            # Convert to numpy for calculations
            coverage_array = np.array(current_coverage)
            eigenvectors_array = np.array(eigenvectors)

            # Calculate projection and contribution
            projection = np.dot(np.array(content_embedding), eigenvectors_array.T)
            contribution = np.abs(projection) * quality_factor

            # Update coverage directly
            for i in range(min(len(contribution), len(coverage_array))):
                current_value = coverage_array[i]
                new_contribution = contribution[i] * (1 - current_value / 2)
                coverage_array[i] += new_contribution

            # Update the coverage in research_dimensions
            research_dimensions["coverage"] = coverage_array.tolist()

            # Update both state keys
            self.update_state("research_dimensions", research_dimensions)
            self.update_state("latest_dimension_coverage", coverage_array.tolist())

            logger.debug(
                f"Updated dimension coverage: {[round(c * 100) for c in coverage_array.tolist()]}%"
            )

        except Exception as e:
            logger.error(f"Error updating dimension coverage: {e}")

    async def identify_research_gaps(self) -> List[str]:
        """Identify semantic dimensions that need more research"""
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if not research_dimensions:
            return []

        try:
            # Find dimensions with low coverage
            coverage = np.array(research_dimensions["coverage"])

            # Sort dimensions by coverage (ascending)
            sorted_dims = np.argsort(coverage)

            # Return indices of the least covered dimensions (lowest 3 that are below 50% coverage)
            gaps = [i for i in sorted_dims[:3] if coverage[i] < 0.5]

            return gaps
        except Exception as e:
            logger.error(f"Error identifying research gaps: {e}")
            return []

    async def extract_text_from_html(self, html_content: str) -> str:
        """Extract meaningful text content from HTML with proper character handling"""
        try:
            # Try BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup
                import html
                import re  # Explicitly import re here for the closure

                # Create a task for BS4 extraction
                def extract_with_bs4():
                    # First unescape HTML entities properly
                    unescaped_content = html.unescape(html_content)

                    soup = BeautifulSoup(unescaped_content, "html.parser")

                    # Remove common navigation elements by tag
                    for element in soup(
                        [
                            "script",
                            "style",
                            "head",
                            "iframe",
                            "noscript",
                            "nav",
                            "header",
                            "footer",
                            "aside",
                            "form",
                        ]
                    ):
                        element.decompose()

                    # Remove common menu and navigation classes - expanded list
                    nav_patterns = [
                        "menu",
                        "nav",
                        "header",
                        "footer",
                        "sidebar",
                        "dropdown",
                        "ibar",
                        "navigation",
                        "navbar",
                        "topbar",
                        "tab",
                        "toolbar",
                        "section",
                        "submenu",
                        "subnav",
                        "panel",
                        "drawer",
                        "accordion",
                        "toc",
                        "login",
                        "signin",
                        "auth",
                        "user-login",
                        "authType",
                    ]

                    # Case-insensitive class matching with partial matches
                    for element in soup.find_all(
                        class_=lambda c: c
                        and any(x.lower() in c.lower() for x in nav_patterns)
                    ):
                        element.decompose()

                    # Remove all unordered lists that contain mostly links (likely menus)
                    for ul in soup.find_all("ul"):
                        links = ul.find_all("a")
                        list_items = ul.find_all("li")

                        # If it contains links and either:
                        # 1. Most children are links, or
                        # 2. There are many list items (10+)
                        # Then it's likely a navigation menu
                        if links and (
                            (list_items and len(links) / len(list_items) > 0.7)
                            or len(links) >= 10
                            or len(list_items) >= 10
                        ):
                            ul.decompose()

                    # Extract text with proper whitespace handling
                    text = soup.get_text(" ", strip=True)

                    # Normalize whitespace while preserving intended breaks
                    # Replace multiple spaces with a single space
                    text = re.sub(r" {2,}", " ", text)

                    # Fix common issues with periods and spaces
                    text = re.sub(
                        r"\.([A-Z])", ". \\1", text
                    )  # Fix "years.Today's" -> "years. Today's"

                    # Process text line by line to better handle paragraph breaks
                    lines = text.split("\n")
                    processed_lines = []

                    for line in lines:
                        # Remove excess whitespace within each line
                        line = re.sub(r"\s+", " ", line).strip()
                        if line:
                            processed_lines.append(line)

                    # Join with proper paragraph breaks
                    return "\n\n".join(processed_lines)

                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                bs4_extraction_task = loop.run_in_executor(None, extract_with_bs4)
                bs4_result = await asyncio.wait_for(bs4_extraction_task, timeout=5.0)

                # If BS4 extraction gave substantial content, use it
                if bs4_result and len(bs4_result) > len(html_content) * 0.1:
                    return bs4_result

                # Otherwise fall back to the regex version
                # Quick regex extraction first
                import re
                import html

                # First unescape HTML entities properly
                unescaped_content = html.unescape(html_content)

                # Remove script and style tags
                content = re.sub(
                    r"<script[^>]*>.*?</script>",
                    " ",
                    unescaped_content,
                    flags=re.DOTALL,
                )
                content = re.sub(
                    r"<style[^>]*>.*?</style>", " ", content, flags=re.DOTALL
                )
                content = re.sub(
                    r"<head[^>]*>.*?</head>", " ", content, flags=re.DOTALL
                )
                content = re.sub(r"<nav[^>]*>.*?</nav>", " ", content, flags=re.DOTALL)
                content = re.sub(
                    r"<header[^>]*>.*?</header>", " ", content, flags=re.DOTALL
                )
                content = re.sub(
                    r"<footer[^>]*>.*?</footer>", " ", content, flags=re.DOTALL
                )

                # Remove HTML tags
                content = re.sub(r"<[^>]*>", " ", content)

                # Fix common issues with periods and spaces
                content = re.sub(
                    r"\.([A-Z])", ". \\1", content
                )  # Fix "years.Today's" -> "years. Today's"

                # Cleanup whitespace
                content = re.sub(r"\s+", " ", content).strip()

                return content

            except (ImportError, asyncio.TimeoutError, Exception) as e:
                logger.warning(
                    f"BeautifulSoup extraction failed: {e}, using regex fallback"
                )
                # Use regex version if BS4 fails
                import re
                import html

                # First unescape HTML entities properly
                unescaped_content = (
                    html.unescape(html_content)
                    if isinstance(html_content, str)
                    else html_content
                )

                # Remove script and style tags
                content = re.sub(
                    r"<script[^>]*>.*?</script>",
                    " ",
                    unescaped_content,
                    flags=re.DOTALL,
                )
                content = re.sub(
                    r"<style[^>]*>.*?</style>", " ", content, flags=re.DOTALL
                )
                content = re.sub(
                    r"<head[^>]*>.*?</head>", " ", content, flags=re.DOTALL
                )
                content = re.sub(r"<nav[^>]*>.*?</nav>", " ", content, flags=re.DOTALL)
                content = re.sub(
                    r"<header[^>]*>.*?</header>", " ", content, flags=re.DOTALL
                )
                content = re.sub(
                    r"<footer[^>]*>.*?</footer>", " ", content, flags=re.DOTALL
                )

                # Remove HTML tags
                content = re.sub(r"<[^>]*>", " ", content)

                # Fix common issues with periods and spaces
                content = re.sub(
                    r"\.([A-Z])", ". \\1", content
                )  # Fix "years.Today's" -> "years. Today's"

                # Cleanup whitespace
                content = re.sub(r"\s+", " ", content).strip()

                return content

        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            # Simple fallback - remove all HTML tags and unescape HTML entities
            try:
                import re
                import html

                # Unescape HTML entities
                if isinstance(html_content, str):
                    unescaped = html.unescape(html_content)
                else:
                    unescaped = html_content

                # Remove HTML tags
                text = re.sub(r"<[^>]*>", " ", unescaped)

                # Normalize whitespace
                text = re.sub(r"\s+", " ", text).strip()

                return text
            except:
                return html_content

    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL with anti-blocking measures and domain-specific rate limiting"""
        try:
            state = self.get_state()
            url_considered_count = state.get("url_considered_count", {})
            url_results_cache = state.get("url_results_cache", {})
            master_source_table = state.get("master_source_table", {})
            domain_session_map = state.get("domain_session_map", {})

            # Add to considered URLs counter
            url_considered_count[url] = url_considered_count.get(url, 0) + 1
            self.update_state("url_considered_count", url_considered_count)

            # Check if URL is in cache and use that if available
            if url in url_results_cache:
                logger.info(f"Using cached content for URL: {url}")
                return url_results_cache[url]

            logger.debug(f"Using direct fetch for URL: {url}")

            # Extract domain for session management and tracking
            from urllib.parse import urlparse

            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Domain-specific rate limiting
            # Check if we've recently accessed this domain
            if domain in domain_session_map:
                domain_info = domain_session_map[domain]
                last_access_time = domain_info.get("last_visit", 0)
                current_time = time.time()
                time_since_last_access = current_time - last_access_time

                # If we accessed this domain recently, delay to avoid rate limiting
                # Only delay if less than 2-3 seconds have passed since last access
                if time_since_last_access < 3.0:
                    # Add randomness to the delay (between 2-3 seconds total between requests)
                    base_delay = 2.0
                    jitter = random.uniform(0.1, 1.0)
                    delay_time = max(0, base_delay - time_since_last_access + jitter)

                    if delay_time > 0.1:  # Only log/delay if significant
                        logger.info(
                            f"Rate limiting for domain {domain}: Delaying for {delay_time:.2f} seconds"
                        )
                        await asyncio.sleep(delay_time)

            # Import fake-useragent for better user agent rotation
            try:
                from fake_useragent import UserAgent

                ua = UserAgent()
                random_user_agent = ua.random
            except ImportError:
                # Fallback if fake-useragent is not installed
                user_agents = [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/123.0.0.0 Safari/537.36",
                ]
                random_user_agent = random.choice(user_agents)

            # Create comprehensive browser fingerprint headers
            headers = {
                "User-Agent": random_user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "sec-ch-ua": '"Chromium";v="116", "Google Chrome";v="116", "Not=A?Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            }

            # Add EZproxy-like headers
            university_ips = {
                "Harvard": "128.103.192." + str(random.randint(1, 254)),
                "Princeton": "128.112.203." + str(random.randint(1, 254)),
                "MIT": "18.7."
                + str(random.randint(1, 254))
                + "."
                + str(random.randint(1, 254)),
                "Stanford": "171.64."
                + str(random.randint(1, 254))
                + "."
                + str(random.randint(1, 254)),
            }

            chosen_university = random.choice(list(university_ips.keys()))
            headers["X-Forwarded-For"] = university_ips[chosen_university]
            headers["X-Requested-With"] = "XMLHttpRequest"

            # Add institutional cookies
            if domain not in domain_session_map:
                domain_session_map[domain] = {
                    "cookies": {},
                    "last_visit": 0,
                    "visit_count": 0,
                }

            domain_session_map[domain]["cookies"] = {
                "ezproxy_authenticated": "true",
                "institution": chosen_university,
                "access_token": "academic_access_" + str(int(time.time())),
            }

            # Use a mix of academic and standard referrers
            referrers = [
                f"https://library.{chosen_university.lower()}.edu/find/",
                "https://scholar.google.com/scholar?q=",
                "https://www.google.com/search?q=",
                "https://www.bing.com/search?q=",
                "https://search.yahoo.com/search?p=",
                "https://www.scopus.com/record/display.uri",
                "https://www.webofscience.com/wos/woscc/full-record/",
                "https://www.sciencedirect.com/search?",
                "https://www.base-search.net/Search/Results?",
            ]

            # Create rich search terms
            search_terms = [
                parsed_url.path.split("/")[-1].replace(".pdf", "").replace("-", " "),
                (
                    "doi " + parsed_url.path.split("/")[-1]
                    if "/" in parsed_url.path
                    else domain
                ),
                domain + " research",
                domain + " academic",
                domain + " publication",
            ]

            # Filter out empty or very short ones
            search_terms = [term for term in search_terms if len(term.strip()) > 3]

            # Choose a referrer and term - use hash of domain for consistency while still appearing varied
            domain_hash = hash(domain)
            chosen_referrer = referrers[domain_hash % len(referrers)]
            search_term = search_terms[0] if search_terms else domain
            if len(search_terms) > 1:
                search_term = search_terms[domain_hash % len(search_terms)]

            # Apply the search term
            search_term = search_term.replace(" ", "+")
            headers["Referer"] = chosen_referrer + search_term

            # Update domain tracking info
            if domain not in domain_session_map:
                domain_session_map[domain] = {
                    "cookies": {},
                    "last_visit": 0,
                    "visit_count": 0,
                }

            domain_session = domain_session_map[domain]
            domain_session["visit_count"] += 1

            domain_session["last_visit"] = time.time()
            self.update_state("domain_session_map", domain_session_map)

            # Create connector with SSL verification disabled and keep session open
            connector = aiohttp.TCPConnector(verify_ssl=False, force_close=True)

            # Check if URL appears to be a PDF
            is_pdf = url.lower().endswith(".pdf")

            # Get existing cookies for this domain if available
            cookie_dict = {}
            if domain in domain_session_map:
                # Convert stored cookies to dictionary format for ClientSession
                stored_cookies = domain_session_map[domain].get("cookies", {})

                # Handle both dictionary and CookieJar formats
                if isinstance(stored_cookies, dict):
                    cookie_dict = stored_cookies
                else:
                    # Try to extract cookies from CookieJar
                    try:
                        for cookie_name, cookie in stored_cookies.items():
                            cookie_dict[cookie_name] = cookie.value
                    except AttributeError:
                        # If that fails, use an empty dict
                        cookie_dict = {}

            async with aiohttp.ClientSession(
                connector=connector, cookies=cookie_dict
            ) as session:
                if is_pdf:
                    # Use binary mode for PDFs
                    async with session.get(
                        url, headers=headers, timeout=20.0
                    ) as response:
                        # Store cookies for future requests
                        if domain in domain_session_map:
                            domain_session_map[domain]["cookies"] = (
                                session.cookie_jar.filter_cookies(url)
                            )
                            self.update_state("domain_session_map", domain_session_map)

                        if response.status == 200:
                            # Get PDF content as bytes
                            pdf_content = await response.read()
                            self.is_pdf_content = True  # Set the PDF flag
                            extracted_content = await self.extract_text_from_pdf(
                                pdf_content
                            )

                            # Limit cached content to 3x MAX_RESULT_TOKENS
                            if extracted_content:
                                tokens = await self.count_tokens(extracted_content)
                                token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                if tokens > token_limit:
                                    char_limit = int(
                                        len(extracted_content) * (token_limit / tokens)
                                    )
                                    extracted_content_to_cache = extracted_content[
                                        :char_limit
                                    ]
                                    logger.info(
                                        f"Limiting cached PDF content for URL {url} from {tokens} to {token_limit} tokens"
                                    )
                                else:
                                    extracted_content_to_cache = extracted_content

                                url_results_cache[url] = extracted_content_to_cache
                            else:
                                url_results_cache[url] = extracted_content

                            self.update_state("url_results_cache", url_results_cache)

                            # Add to master source table
                            if url not in master_source_table:
                                title = (
                                    url.split("/")[-1]
                                    .replace(".pdf", "")
                                    .replace("-", " ")
                                    .replace("_", " ")
                                )
                                source_id = f"S{len(master_source_table) + 1}"
                                master_source_table[url] = {
                                    "id": source_id,
                                    "title": title,
                                    "content_preview": extracted_content[:500],
                                    "source_type": "pdf",
                                    "accessed_date": self.research_date,
                                    "cited_in_sections": set(),
                                }
                                self.update_state(
                                    "master_source_table", master_source_table
                                )

                            return extracted_content
                        elif response.status == 403 or response.status == 271:
                            # Try archive.org for 403 errors
                            logger.info(
                                f"Received 403 for PDF {url}, trying archive.org"
                            )
                            archive_content = await self.fetch_from_archive(
                                url, session
                            )
                            if archive_content:
                                return archive_content

                            # If archive fallback fails, return original error
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status} (archive fallback failed)"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )
                        else:
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status}"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )
                else:
                    # Normal text/HTML mode
                    async with session.get(
                        url, headers=headers, timeout=20.0
                    ) as response:
                        # Store cookies for future requests
                        if domain in domain_session_map:
                            domain_session_map[domain]["cookies"] = (
                                session.cookie_jar.filter_cookies(url)
                            )
                            self.update_state("domain_session_map", domain_session_map)

                        if response.status == 200:
                            # Check content type in response headers
                            content_type = response.headers.get(
                                "Content-Type", ""
                            ).lower()

                            if "application/pdf" in content_type:
                                # This is a PDF even though the URL didn't end with .pdf
                                pdf_content = await response.read()
                                self.is_pdf_content = True  # Set the PDF flag
                                extracted_content = await self.extract_text_from_pdf(
                                    pdf_content
                                )

                                # Limit cached content to 3x MAX_RESULT_TOKENS
                                if extracted_content:
                                    tokens = await self.count_tokens(extracted_content)
                                    token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                    if tokens > token_limit:
                                        char_limit = int(
                                            len(extracted_content)
                                            * (token_limit / tokens)
                                        )
                                        extracted_content_to_cache = extracted_content[
                                            :char_limit
                                        ]
                                        logger.info(
                                            f"Limiting cached PDF content for URL {url} from {tokens} to {token_limit} tokens"
                                        )
                                    else:
                                        extracted_content_to_cache = extracted_content

                                    url_results_cache[url] = extracted_content_to_cache
                                else:
                                    url_results_cache[url] = extracted_content

                                self.update_state(
                                    "url_results_cache", url_results_cache
                                )

                                # Add to master source table
                                if url not in master_source_table:
                                    title = url.split("/")[-1]
                                    if not title or title == "/":
                                        parsed_url = urlparse(url)
                                        title = f"PDF from {parsed_url.netloc}"

                                    source_id = f"S{len(master_source_table) + 1}"
                                    master_source_table[url] = {
                                        "id": source_id,
                                        "title": title,
                                        "content_preview": extracted_content[:500],
                                        "source_type": "pdf",
                                        "accessed_date": self.research_date,
                                        "cited_in_sections": set(),
                                    }
                                    self.update_state(
                                        "master_source_table", master_source_table
                                    )

                                return extracted_content

                            # Handle as normal HTML/text
                            content = await response.text()
                            self.is_pdf_content = False  # Clear the PDF flag
                            if (
                                self.valves.EXTRACT_CONTENT_ONLY
                                and content.strip().startswith("<")
                            ):
                                extracted = await self.extract_text_from_html(content)

                                # Limit cached content to 3x MAX_RESULT_TOKENS
                                if extracted:
                                    tokens = await self.count_tokens(extracted)
                                    token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                    if tokens > token_limit:
                                        char_limit = int(
                                            len(extracted) * (token_limit / tokens)
                                        )
                                        extracted_to_cache = extracted[:char_limit]
                                        logger.info(
                                            f"Limiting cached HTML content for URL {url} from {tokens} to {token_limit} tokens"
                                        )
                                    else:
                                        extracted_to_cache = extracted

                                    url_results_cache[url] = extracted_to_cache
                                else:
                                    url_results_cache[url] = extracted

                                self.update_state(
                                    "url_results_cache", url_results_cache
                                )

                                # Add to master source table
                                if url not in master_source_table:
                                    # Try to extract title
                                    title = url
                                    title_match = re.search(
                                        r"<title>(.*?)</title>",
                                        content,
                                        re.IGNORECASE | re.DOTALL,
                                    )
                                    if title_match:
                                        title = title_match.group(1).strip()
                                    else:
                                        # Use domain as title
                                        parsed_url = urlparse(url)
                                        title = parsed_url.netloc

                                    source_id = f"S{len(master_source_table) + 1}"
                                    master_source_table[url] = {
                                        "id": source_id,
                                        "title": title,
                                        "content_preview": extracted[:500],
                                        "source_type": "web",
                                        "accessed_date": self.research_date,
                                        "cited_in_sections": set(),
                                    }
                                    self.update_state(
                                        "master_source_table", master_source_table
                                    )

                                return extracted

                            # Limit cached content to 3x MAX_RESULT_TOKENS
                            if isinstance(content, str):
                                tokens = await self.count_tokens(content)
                                token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                if tokens > token_limit:
                                    char_limit = int(
                                        len(content) * (token_limit / tokens)
                                    )
                                    content_to_cache = content[:char_limit]
                                    logger.info(
                                        f"Limiting cached content for URL {url} from {tokens} to {token_limit} tokens"
                                    )
                                else:
                                    content_to_cache = content

                                url_results_cache[url] = content_to_cache
                            else:
                                url_results_cache[url] = content

                            self.update_state("url_results_cache", url_results_cache)

                            # Add to master source table
                            if url not in master_source_table:
                                # Try to extract title
                                title = url
                                title_match = re.search(
                                    r"<title>(.*?)</title>",
                                    content,
                                    re.IGNORECASE | re.DOTALL,
                                )
                                if title_match:
                                    title = title_match.group(1).strip()
                                else:
                                    # Use domain as title
                                    parsed_url = urlparse(url)
                                    title = parsed_url.netloc

                                source_id = f"S{len(master_source_table) + 1}"
                                master_source_table[url] = {
                                    "id": source_id,
                                    "title": title,
                                    "content_preview": content[:500],
                                    "source_type": "web",
                                    "accessed_date": self.research_date,
                                    "cited_in_sections": set(),
                                }
                                self.update_state(
                                    "master_source_table", master_source_table
                                )

                            return content
                        elif response.status == 403 or response.status == 271:
                            # Try archive.org for 403 errors
                            logger.info(
                                f"Received 403 for URL {url}, trying archive.org"
                            )
                            archive_content = await self.fetch_from_archive(
                                url, session
                            )
                            if archive_content:
                                return archive_content

                            # If archive fallback fails, return original error
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status} (archive fallback failed)"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )
                        else:
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status}"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching content from {url}")
            return f"Timeout while fetching content from {url}"
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error for {url}: {e}")
            return f"Connection error: {str(e)}"
        except aiohttp.ClientOSError as e:
            logger.error(f"OS error for {url}: {e}")
            return f"Connection error: {str(e)}"
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return f"Error fetching content: {str(e)}"

    async def fetch_from_archive(self, url: str, session=None) -> str:
        """Fetch content from the Internet Archive (archive.org)"""
        try:
            # Construct Wayback Machine URL
            wayback_api_url = f"https://archive.org/wayback/available?url={url}"

            # Create a new session if not provided
            close_session = False
            if session is None:
                close_session = True
                connector = aiohttp.TCPConnector(verify_ssl=False, force_close=True)
                session = aiohttp.ClientSession(connector=connector)

            try:
                # First check if the URL is archived
                async with session.get(wayback_api_url, timeout=15.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if there are archived snapshots
                        snapshots = data.get("archived_snapshots", {})
                        closest = snapshots.get("closest", {})
                        archived_url = closest.get("url")

                        if archived_url:
                            logger.info(f"Found archive for {url}: {archived_url}")
                            # Fetch the content from the archived URL
                            async with session.get(
                                archived_url, timeout=20.0
                            ) as archive_response:
                                if archive_response.status == 200:
                                    content_type = archive_response.headers.get(
                                        "Content-Type", ""
                                    ).lower()

                                    if "application/pdf" in content_type:
                                        # Handle PDF from archive
                                        pdf_content = await archive_response.read()
                                        self.is_pdf_content = True
                                        extracted_content = (
                                            await self.extract_text_from_pdf(
                                                pdf_content
                                            )
                                        )

                                        # Cache the archived content
                                        state = self.get_state()
                                        url_results_cache = state.get(
                                            "url_results_cache", {}
                                        )
                                        url_results_cache[url] = extracted_content
                                        self.update_state(
                                            "url_results_cache", url_results_cache
                                        )

                                        # Update master source table
                                        master_source_table = state.get(
                                            "master_source_table", {}
                                        )
                                        if url not in master_source_table:
                                            title = f"Archived PDF: {url.split('/')[-1].replace('.pdf','').replace('-',' ').replace('_',' ')}"
                                            source_id = (
                                                f"S{len(master_source_table) + 1}"
                                            )
                                            master_source_table[url] = {
                                                "id": source_id,
                                                "title": title,
                                                "content_preview": extracted_content[
                                                    :500
                                                ],
                                                "source_type": "pdf",
                                                "accessed_date": self.research_date,
                                                "cited_in_sections": set(),
                                                "archived": True,
                                            }
                                            self.update_state(
                                                "master_source_table",
                                                master_source_table,
                                            )

                                        return extracted_content
                                    else:
                                        # Handle HTML/text from archive
                                        content = await archive_response.text()
                                        self.is_pdf_content = False

                                        # Extract and clean text if needed
                                        if (
                                            self.valves.EXTRACT_CONTENT_ONLY
                                            and content.strip().startswith("<")
                                        ):
                                            extracted = (
                                                await self.extract_text_from_html(
                                                    content
                                                )
                                            )

                                            # Cache the extracted content
                                            state = self.get_state()
                                            url_results_cache = state.get(
                                                "url_results_cache", {}
                                            )
                                            url_results_cache[url] = extracted
                                            self.update_state(
                                                "url_results_cache", url_results_cache
                                            )

                                            # Update master source table
                                            master_source_table = state.get(
                                                "master_source_table", {}
                                            )
                                            if url not in master_source_table:
                                                title = f"Archived: {url}"
                                                title_match = re.search(
                                                    r"<title>(.*?)</title>",
                                                    content,
                                                    re.IGNORECASE | re.DOTALL,
                                                )
                                                if title_match:
                                                    title = f"Archived: {title_match.group(1).strip()}"

                                                source_id = (
                                                    f"S{len(master_source_table) + 1}"
                                                )
                                                master_source_table[url] = {
                                                    "id": source_id,
                                                    "title": title,
                                                    "content_preview": extracted[:500],
                                                    "source_type": "web",
                                                    "accessed_date": self.research_date,
                                                    "cited_in_sections": set(),
                                                    "archived": True,
                                                }
                                                self.update_state(
                                                    "master_source_table",
                                                    master_source_table,
                                                )

                                            return extracted
                                        else:
                                            # Cache the raw content
                                            state = self.get_state()
                                            url_results_cache = state.get(
                                                "url_results_cache", {}
                                            )
                                            url_results_cache[url] = content
                                            self.update_state(
                                                "url_results_cache", url_results_cache
                                            )
                                            return content
                        else:
                            logger.warning(f"No archived version found for {url}")
                            return ""
                    else:
                        logger.warning(
                            f"Error accessing archive.org API: {response.status}"
                        )
                        return ""
            finally:
                # Close the session if we created it
                if close_session and session:
                    await session.close()

        except Exception as e:
            logger.error(f"Error fetching from archive.org: {e}")
            return ""

    async def extract_text_from_pdf(self, pdf_content) -> str:
        """Extract text from PDF content using PyPDF2 or pdfplumber"""
        if not self.valves.HANDLE_PDFS:
            return "PDF processing is disabled in settings."

        # Ensure we have bytes for the PDF content
        if isinstance(pdf_content, str):
            if pdf_content.startswith("%PDF"):
                pdf_content = pdf_content.encode("utf-8", errors="ignore")
            else:
                return "Error: Invalid PDF content format"

        # Limit extraction to configured max pages to avoid too much processing
        max_pages = self.valves.PDF_MAX_PAGES

        try:
            # Try PyPDF2 first
            try:
                import io
                from PyPDF2 import PdfReader

                # Use ThreadPoolExecutor for CPU-intensive PDF processing
                def extract_with_pypdf():
                    try:
                        # Create a reader object
                        pdf_file = io.BytesIO(pdf_content)
                        pdf_reader = PdfReader(pdf_file)

                        # Get the total number of pages
                        num_pages = len(pdf_reader.pages)
                        logger.info(
                            f"PDF has {num_pages} pages, extracting up to {max_pages}"
                        )

                        # Extract text from each page up to the limit
                        text = []
                        for page_num in range(min(num_pages, max_pages)):
                            try:
                                page = pdf_reader.pages[page_num]
                                page_text = page.extract_text() or ""
                                if page_text.strip():
                                    text.append(f"Page {page_num + 1}:\n{page_text}")
                            except Exception as e:
                                logger.warning(f"Error extracting page {page_num}: {e}")

                        # Join all pages with spacing
                        full_text = "\n\n".join(text)

                        # Add a note if we limited the page count
                        if num_pages > max_pages:
                            full_text += f"\n\n[Note: This PDF has {num_pages} pages, but only the first {max_pages} were processed.]"

                        return full_text if full_text.strip() else None
                    except Exception as e:
                        logger.error(f"Error in PDF extraction with PyPDF2: {e}")
                        return None

                # Execute in thread pool
                loop = asyncio.get_event_loop()
                pdf_extract_task = loop.run_in_executor(
                    self.executor, extract_with_pypdf
                )
                full_text = await pdf_extract_task

                if full_text and full_text.strip():
                    logger.info(
                        f"Successfully extracted text from PDF using PyPDF2: {len(full_text)} chars"
                    )
                    return full_text
                else:
                    logger.warning(
                        "PyPDF2 extraction returned empty text, trying pdfplumber..."
                    )
            except (ImportError, Exception) as e:
                logger.warning(f"PyPDF2 extraction failed: {e}, trying pdfplumber...")

            # Try pdfplumber as a fallback
            try:
                import io
                import pdfplumber

                # Use ThreadPoolExecutor for CPU-intensive PDF processing
                def extract_with_pdfplumber():
                    try:
                        pdf_file = io.BytesIO(pdf_content)
                        with pdfplumber.open(pdf_file) as pdf:
                            # Get total pages
                            num_pages = len(pdf.pages)

                            text = []
                            for i, page in enumerate(pdf.pages[:max_pages]):
                                try:
                                    page_text = page.extract_text() or ""
                                    if page_text.strip():
                                        text.append(f"Page {i + 1}:\n{page_text}")
                                except Exception as page_error:
                                    logger.warning(
                                        f"Error extracting page {i} with pdfplumber: {page_error}"
                                    )

                            full_text = "\n\n".join(text)

                            # Add a note if we limited the page count
                            if num_pages > max_pages:
                                full_text += f"\n\n[Note: This PDF has {num_pages} pages, but only the first {max_pages} were processed.]"

                            return full_text
                    except Exception as e:
                        logger.error(f"Error in PDF extraction with pdfplumber: {e}")
                        return None

                # Execute in thread pool
                loop = asyncio.get_event_loop()
                pdf_extract_task = loop.run_in_executor(
                    self.executor, extract_with_pdfplumber
                )
                full_text = await pdf_extract_task

                if full_text and full_text.strip():
                    logger.info(
                        f"Successfully extracted text from PDF using pdfplumber: {len(full_text)} chars"
                    )
                    return full_text
                else:
                    logger.warning("pdfplumber extraction returned empty text")
            except (ImportError, Exception) as e:
                logger.warning(f"pdfplumber extraction failed: {e}")

            # If both methods failed but we can tell it's a PDF, provide a more useful message
            if pdf_content.startswith(b"%PDF"):
                logger.warning(
                    "PDF detected but text extraction failed. May be scanned or encrypted."
                )
                return "This appears to be a PDF document, but text extraction failed. The PDF may contain scanned images rather than text, or it may be encrypted/protected."

            return "Could not extract text from PDF. The file may not be a valid PDF or may contain security restrictions."

        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return f"Error extracting text from PDF: {str(e)}"

    async def sanitize_query(self, query: str) -> str:
        """Sanitize search query by removing quotes and handling special characters"""
        # Remove quotes that might cause problems with search engines
        sanitized = query.replace('"', " ").replace('"', " ").replace('"', " ")

        # Replace multiple spaces with a single space
        sanitized = " ".join(sanitized.split())

        # Ensure the query isn't too long
        if len(sanitized) > 250:
            sanitized = sanitized[:250]

        logger.info(f"Sanitized query: '{query}' -> '{sanitized}'")
        return sanitized

    async def identify_and_correlate_citations(
        self, section_title, content, master_source_table
    ):
        """Identify and correlate non-numeric URL citations in a section"""
        # Create a prompt for identifying and correlating URL citations
        citation_prompt = {
            "role": "system",
            "content": """You are a master librarian identifying non-exclusively-numeric citations in research content.
            
            Focus ONLY on identifying non-numeric citations that appear inside brackets, such as [https://example.com] or [Reference 1].
            IGNORE all numerical citations like [1], [2], etc. as those have already been identified and correlated.
            
            For each non-numerical citation you identify, extract:
            1. The exact content inside the brackets
            2. The citation text exactly as it appears in the original text, including brackets
            3. The surrounding sentence to which the citation pertains
            4. A representative title for the source (10 words or less)
            
            Your response must only contain the identified citations as requested. Format your response as a valid JSON object with this structure:
            {
              "citations": [
                {
                  "marker": "Source Name",
                  "raw_text": "[Source Name]",
                  "text": "surrounding sentence containing the citation",
                  "url": "https://example.com",
                  "suggested_title": "Descriptive Title for Source"
                },
                ...
              ]
            }""",
        }

        # Build context with full section content and source list
        citation_context = f"## Section: {section_title}\n\n"
        citation_context += content + "\n\n"

        citation_context += "## Available Sources for Citation:\n"
        for url, source_data in master_source_table.items():
            citation_context += f"{source_data['title']} ({url})\n"

        citation_context += "\nIdentify non-numeric citations, ignore numeric citations, and extract the requested structured information."

        # Generate identification and correlation
        try:
            # Use research model for citation identification with appropriate temperature
            citation_response = await self.generate_completion(
                self.get_research_model(),
                [citation_prompt, {"role": "user", "content": citation_context}],
                temperature=self.valves.TEMPERATURE
                * 0.3,  # Lower temperature for precision
            )

            citation_content = citation_response["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                json_str = citation_content[
                    citation_content.find("{") : citation_content.rfind("}") + 1
                ]
                citation_data = json.loads(json_str)

                section_citations = []
                for citation in citation_data.get("citations", []):
                    marker_text = citation.get("marker", "").strip()
                    raw_text = citation.get("raw_text", "").strip()
                    context = citation.get("text", "")
                    matched_url = citation.get("url", "")
                    suggested_title = citation.get("suggested_title", "")

                    # Only add valid citations with URLs (not numerical)
                    if marker_text and matched_url and not marker_text.isdigit():
                        section_citations.append(
                            {
                                "marker": marker_text,
                                "raw_text": raw_text,
                                "text": context,
                                "url": matched_url,
                                "section": section_title,
                                "suggested_title": suggested_title,
                            }
                        )

                return section_citations

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    f"Error parsing citation identification JSON for section {section_title}: {e}"
                )
                return []

        except Exception as e:
            logger.error(f"Error identifying citations in section {section_title}: {e}")
            return []

    async def process_search_result(
        self,
        result: Dict,
        query: str,
        query_embedding: List[float],
        outline_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
    ) -> Dict:
        """Process a search result to extract and compress content with token limiting"""
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("snippet", "")

        # Require a URL for all results
        if not url:
            return {
                "title": title or f"Result for '{query}'",
                "url": "",
                "content": "This result has no associated URL and cannot be processed.",
                "query": query,
                "valid": False,
            }

        await self.emit_status("info", f"Processing result: {title[:50]}...", False)

        try:
            # Get state
            state = self.get_state()
            url_selected_count = state.get("url_selected_count", {})
            url_token_counts = state.get("url_token_counts", {})
            master_source_table = state.get("master_source_table", {})

            # Check if this is a repeated URL
            repeat_count = 0
            repeat_count = url_selected_count.get(url, 0)

            # If the snippet is empty or short but we have a URL, try to fetch content
            if (not snippet or len(snippet) < 200) and url:
                await self.emit_status(
                    "info", f"Fetching content from URL: {url}...", False
                )
                content = await self.fetch_content(url)

                if content and len(content) > 200:
                    snippet = content
                    logger.debug(
                        f"Successfully fetched content from URL: {url} ({len(content)} chars)"
                    )
                else:
                    logger.warning(f"Failed to fetch useful content from URL: {url}")

            # If we still don't have useful content, mark as invalid
            if not snippet or len(snippet) < 200:
                return {
                    "title": title or f"Result for '{query}'",
                    "url": url,
                    "content": snippet
                    or f"No substantial content available for this result.",
                    "query": query,
                    "valid": False,
                }

            # For repeated URLs, apply special sliding window treatment
            if repeat_count > 0:
                snippet = await self.handle_repeated_content(
                    snippet, url, query_embedding, repeat_count
                )

            # Calculate tokens in the content
            content_tokens = await self.count_tokens(snippet)

            # Get user preferences for PDV
            state = self.get_state()
            user_preferences = state.get("user_preferences", {})
            pdv = user_preferences.get("pdv")

            # Apply token limit if needed with adaptive scaling based on relevance
            max_tokens = await self.scale_token_limit_by_relevance(
                result, query_embedding, pdv
            )

            limited_content = snippet
            tokens = content_tokens

            if content_tokens > max_tokens:
                await self.emit_status(
                    "info", "Truncating content to token limit...", False
                )
                char_ratio = max_tokens / content_tokens
                char_limit = int(len(snippet) * char_ratio)
                limited_content = snippet[:char_limit]
                tokens = await self.count_tokens(limited_content)
            # --- FIX ENDS HERE ---

            url_selected_count[url] = url_selected_count.get(url, 0) + 1
            self.update_state("url_selected_count", url_selected_count)

            if url not in url_token_counts:
                url_token_counts[url] = content_tokens
                self.update_state("url_token_counts", url_token_counts)

            if url not in master_source_table:
                source_type = "web"
                if url.endswith(".pdf") or self.is_pdf_content:
                    source_type = "pdf"

                final_title = title
                if not final_title or final_title == f"Result for '{query}'":
                    from urllib.parse import urlparse

                    parsed_url = urlparse(url)
                    if source_type == "pdf":
                        file_name = parsed_url.path.split("/")[-1]
                        final_title = (
                            file_name.replace(".pdf", "")
                            .replace("-", " ")
                            .replace("_", " ")
                        )
                    else:
                        final_title = parsed_url.netloc

                source_id = f"S{len(master_source_table) + 1}"
                master_source_table[url] = {
                    "id": source_id,
                    "title": final_title,
                    "content_preview": limited_content[:500],
                    "source_type": source_type,
                    "accessed_date": self.research_date,
                    "cited_in_sections": set(),
                }
                self.update_state("master_source_table", master_source_table)

            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return {
                "title": title,
                "url": url,
                "content": limited_content,
                "query": query,
                "repeat_count": repeat_count,
                "tokens": tokens,
                "valid": True,
            }

        except Exception as e:
            logger.error(
                f"Unhandled error in process_search_result for {url}: {e}",
                exc_info=True,
            )
            error_msg = f"Error processing search result: {str(e)}\n\nOriginal snippet: {snippet[:1000] if snippet else 'No content available'}"
            tokens = await self.count_tokens(error_msg)

            return {
                "title": title or f"Error processing result for '{query}'",
                "url": url,
                "content": error_msg,
                "query": query,
                "repeat_count": repeat_count if "repeat_count" in locals() else 0,
                "tokens": tokens,
                "valid": False,
            }

    async def _execute_search(self, query: str) -> List[Dict]:
        """
        Executes a web search by making a direct HTTP request to the configured search API.
        This method supports both JSON and HTML responses from the search engine.
        """
        try:
            # URL encode the query for safer search
            from urllib.parse import quote

            encoded_query = quote(query)
            # Construct the search URL from the valve. Ensures the ?q= is correctly handled.
            base_search_url = self.valves.SEARCH_URL
            # 修复：只允许一个q参数
            if base_search_url.endswith("?q="):
                search_url = f"{base_search_url}{encoded_query}"
            elif base_search_url.endswith("?"):
                search_url = f"{base_search_url}q={encoded_query}"
            elif "?" not in base_search_url:
                search_url = f"{base_search_url}?q={encoded_query}"
            else:
                # If there's already a query parameter, append with '&'
                search_url = f"{base_search_url}&q={encoded_query}"

            logger.debug(f"Executing direct search with URL: {search_url}")

            # Determine the number of results to fetch based on valve settings
            total_results_to_fetch = (
                self.valves.SEARCH_RESULTS_PER_QUERY
                + self.valves.EXTRA_RESULTS_PER_QUERY
            )

            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(search_url, timeout=15.0) as response:
                    if response.status != 200:
                        logger.error(
                            f"Search engine returned status code {response.status}"
                        )
                        return []

                    # First, attempt to parse the response as JSON
                    try:
                        search_json = await response.json()
                        results = []

                        # Handle different potential JSON structures
                        items = (
                            search_json
                            if isinstance(search_json, list)
                            else search_json.get("results", [])
                        )

                        for item in items[:total_results_to_fetch]:
                            results.append(
                                {
                                    "title": item.get("title", f"Result for '{query}'"),
                                    "url": item.get("url", ""),
                                    "snippet": item.get("snippet", ""),
                                }
                            )
                        return results
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        # If JSON parsing fails, it might be an HTML response
                        logger.info(
                            "JSON parsing failed, attempting to parse HTML response."
                        )
                        try:
                            from bs4 import BeautifulSoup

                            html_content = await response.text()
                            soup = BeautifulSoup(html_content, "html.parser")
                            results = []

                            # This selector is specific to SearXNG's default theme.
                            # You may need to adjust it if your search instance uses a different theme.
                            result_elements = soup.select("article.result")

                            for element in result_elements[:total_results_to_fetch]:
                                title_element = element.select_one("h3 a")
                                url_element = element.select_one("h3 a")
                                snippet_element = element.select_one("p.content")

                                results.append(
                                    {
                                        "title": (
                                            title_element.get_text(strip=True)
                                            if title_element
                                            else f"Result for '{query}'"
                                        ),
                                        "url": (
                                            url_element.get("href")
                                            if url_element
                                            else ""
                                        ),
                                        "snippet": (
                                            snippet_element.get_text(strip=True)
                                            if snippet_element
                                            else ""
                                        ),
                                    }
                                )

                            if results:
                                return results
                            else:
                                logger.warning(
                                    "HTML parsing successful, but no results found in the page structure."
                                )
                                return []

                        except ImportError:
                            logger.error(
                                "BeautifulSoup is not installed. Cannot parse HTML search results."
                            )
                            return []
                        except Exception as e:
                            logger.error(f"An error occurred during HTML parsing: {e}")
                            return []

        except asyncio.TimeoutError:
            logger.error(f"Direct search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"An error occurred in _execute_search: {e}")
            return []

    async def _fallback_search(self, query: str) -> List[Dict]:
        """Fallback search method using direct HTTP request to search API with HTML parsing support"""
        try:
            # URL encode the query for safer search
            from urllib.parse import quote

            encoded_query = quote(query)
            search_url = f"{self.valves.SEARCH_URL}{encoded_query}"

            logger.debug(f"Using fallback search with URL: {search_url}")

            # Get state for URL tracking
            state = self.get_state()
            url_selected_count = state.get("url_selected_count", {})

            # Calculate additional results to fetch based on repeat counts
            repeat_count = 0
            for url, count in url_selected_count.items():
                if count >= self.valves.REPEATS_BEFORE_EXPANSION:
                    repeat_count += 1

            # Calculate total results to fetch
            base_results = self.valves.SEARCH_RESULTS_PER_QUERY
            additional_results = min(repeat_count, self.valves.EXTRA_RESULTS_PER_QUERY)
            total_results = (
                base_results + self.valves.EXTRA_RESULTS_PER_QUERY + additional_results
            )

            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                # Set a timeout for this request
                async with session.get(search_url, timeout=15.0) as response:
                    if response.status == 200:
                        # First try to parse as JSON
                        try:
                            search_json = await response.json()
                            results = []

                            if isinstance(search_json, list):
                                for i, item in enumerate(search_json[:total_results]):
                                    results.append(
                                        {
                                            "title": item.get("title", f"Result {i+1}"),
                                            "url": item.get("url", ""),
                                            "snippet": item.get("snippet", ""),
                                        }
                                    )
                                return results
                            elif (
                                isinstance(search_json, dict)
                                and "results" in search_json
                            ):
                                for i, item in enumerate(
                                    search_json["results"][:total_results]
                                ):
                                    results.append(
                                        {
                                            "title": item.get("title", f"Result {i+1}"),
                                            "url": item.get("url", ""),
                                            "snippet": item.get("snippet", ""),
                                        }
                                    )
                                return results
                        except (json.JSONDecodeError, aiohttp.ContentTypeError):
                            # If JSON parsing fails, try HTML parsing with BeautifulSoup
                            logger.info(
                                "JSON parsing failed, trying HTML parsing for search results"
                            )
                            try:
                                from bs4 import BeautifulSoup

                                html_content = await response.text()
                                soup = BeautifulSoup(html_content, "html.parser")

                                results = []
                                # Parse SearXNG result elements
                                result_elements = soup.select("article.result")

                                for i, element in enumerate(
                                    result_elements[:total_results]
                                ):
                                    try:
                                        title_element = element.select_one("h3 a")
                                        url_element = element.select_one("h3 a")
                                        snippet_element = element.select_one(
                                            "p.content"
                                        )

                                        title = (
                                            title_element.get_text()
                                            if title_element
                                            else f"Result {i+1}"
                                        )
                                        url = (
                                            url_element.get("href")
                                            if url_element
                                            else ""
                                        )
                                        snippet = (
                                            snippet_element.get_text()
                                            if snippet_element
                                            else ""
                                        )

                                        results.append(
                                            {
                                                "title": title,
                                                "url": url,
                                                "snippet": snippet,
                                            }
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Error parsing search result {i}: {e}"
                                        )

                                if results:
                                    return results
                                else:
                                    logger.warning("No results found in HTML parsing")
                            except ImportError:
                                logger.warning(
                                    "BeautifulSoup not available for HTML parsing"
                                )
                            except Exception as e:
                                logger.error(f"Error in HTML parsing: {e}")

                    # If we got this far, the response couldn't be parsed
                    logger.error(
                        f"Fallback search returned status code {response.status} but couldn't parse content"
                    )
                    return []
        except asyncio.TimeoutError:
            logger.error(f"Fallback search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []

    async def search_web(self, query: str) -> List[Dict]:
        """
        Performs a web search using a direct, reliable method.
        """
        logger.debug(f"Starting web search for query: {query}")

        # Directly call the reliable search method
        results = await self._execute_search(query)

        if results:
            logger.debug(
                f"Search successful, found {len(results)} results for: {query}"
            )
            return results
        else:
            # If no results are found after a successful search, create a placeholder
            logger.warning(f"No search results found for query: {query}")
            return [
                {
                    "title": f"No results for '{query}'",
                    "url": "",
                    "snippet": f"No search results were found for the query: {query}",
                }
            ]

    async def select_most_relevant_results(
        self,
        results: List[Dict],
        query: str,
        query_embedding: List[float],
        outline_embedding: List[float],
        summary_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Select the most relevant results from extra results pool using semantic transformations with similarity caching"""
        if not results:
            return results

        # If we only have the base needed amount or fewer, return them all
        base_results_per_query = self.valves.SEARCH_RESULTS_PER_QUERY
        if len(results) <= base_results_per_query:
            return results

        # Get state for URL tracking
        state = self.get_state()
        url_selected_count = state.get("url_selected_count", {})

        # Count URLs that have been repeated at REPEATS_BEFORE_EXPANSION times or more
        repeat_count = 0
        for url, count in url_selected_count.items():
            if count >= self.valves.REPEATS_BEFORE_EXPANSION:
                repeat_count += 1

        # Calculate additional results to fetch based on repeat count
        additional_results = min(repeat_count, self.valves.EXTRA_RESULTS_PER_QUERY)
        results_to_select = base_results_per_query + additional_results

        # Calculate relevance scores for each result
        relevance_scores = []

        # Get transformation if available
        state = self.get_state()
        transformation = state.get("semantic_transformations")

        # Get similarity cache
        similarity_cache = state.get("similarity_cache", {})

        # Process domain priority valve value (if provided)
        priority_domains = []
        if hasattr(self.valves, "DOMAIN_PRIORITY") and self.valves.DOMAIN_PRIORITY:
            # Split by commas and/or spaces
            domain_input = self.valves.DOMAIN_PRIORITY
            # Replace commas with spaces, then split by spaces
            domain_items = domain_input.replace(",", " ").split()
            # Remove empty items and add to priority domains
            priority_domains = [
                item.strip().lower() for item in domain_items if item.strip()
            ]
            if priority_domains:
                logger.info(f"Using priority domains: {priority_domains}")

        # Process content priority valve value (if provided)
        priority_keywords = []
        if hasattr(self.valves, "CONTENT_PRIORITY") and self.valves.CONTENT_PRIORITY:
            # Split by commas and/or spaces, handling quoted phrases
            content_input = self.valves.CONTENT_PRIORITY

            # Function to parse keywords, respecting quotes
            def parse_keywords(text):
                keywords = []
                # Pattern for quoted phrases or words
                pattern = r"\'([^\']+)\'|\"([^\"]+)\"|(\S+)"

                matches = re.findall(pattern, text)
                for match in matches:
                    # Each match is a tuple with three groups (one will contain the text)
                    keyword = match[0] or match[1] or match[2]
                    if keyword:
                        keywords.append(keyword.lower())
                return keywords

            priority_keywords = parse_keywords(content_input)
            if priority_keywords:
                logger.info(f"Using priority keywords: {priority_keywords}")

        # Get multiplier values from valves or use defaults
        domain_multiplier = getattr(self.valves, "DOMAIN_MULTIPLIER", 1.5)
        keyword_multiplier_per_match = getattr(
            self.valves, "KEYWORD_MULTIPLIER_PER_MATCH", 1.1
        )
        max_keyword_multiplier = getattr(self.valves, "MAX_KEYWORD_MULTIPLIER", 2.0)

        for i, result in enumerate(results):
            try:
                # Get a snippet for evaluation
                snippet = result.get("snippet", "")
                url = result.get("url", "")

                # If snippet is too short and URL is available, fetch a bit of content
                if len(snippet) < self.valves.RELEVANCY_SNIPPET_LENGTH and url:
                    try:
                        await self.emit_status(
                            "info",
                            f"Fetching snippet for relevance check: {url[:50]}...",
                            False,
                        )
                        # Only fetch the first part of the content for evaluation
                        content_preview = await self.fetch_content(url)
                        if content_preview:
                            snippet = content_preview[
                                : self.valves.RELEVANCY_SNIPPET_LENGTH
                            ]
                    except Exception as e:
                        logger.error(f"Error fetching content for relevance check: {e}")

                # Calculate relevance if we have enough content
                if snippet and len(snippet) > 100:
                    # FIRST, CHECK FOR VOCABULARY LIST
                    words = re.findall(r"\b\w+\b", snippet[:2000].lower())
                    if len(words) > 150:  # Only check if enough words
                        unique_words = set(words)
                        unique_ratio = len(unique_words) / len(words)
                        if (
                            unique_ratio > 0.98
                        ):  # Extremely high uniqueness = vocabulary list
                            logger.warning(
                                f"Skipping likely vocabulary list: {unique_ratio:.3f} uniqueness ratio"
                            )
                            # Assign a very low similarity score
                            similarity = 0.01
                            relevance_scores.append((i, similarity))
                            result["similarity"] = similarity
                            continue  # Skip the expensive embedding calculation

                    # Get embedding for the snippet
                    snippet_embedding = await self.get_embedding(snippet)

                    if snippet_embedding:
                        # Apply transformation to query only (Alternative A)
                        if transformation:
                            # Transform the query, not the content
                            transformed_query = (
                                await self.apply_semantic_transformation(
                                    query_embedding, transformation
                                )
                            )

                            # Calculate similarity between untransformed content and transformed query
                            similarity = cosine_similarity(
                                [snippet_embedding], [transformed_query]
                            )[0][0]
                        else:
                            # Calculate basic similarity if no transformation
                            similarity = cosine_similarity(
                                [snippet_embedding], [query_embedding]
                            )[0][0]

                        # Track original similarity for logging
                        original_similarity = similarity

                        # Apply domain multiplier if priority domains are set
                        if priority_domains and url:
                            url_lower = url.lower()
                            if any(domain in url_lower for domain in priority_domains):
                                similarity *= domain_multiplier
                                logger.debug(
                                    f"Applied domain multiplier {domain_multiplier}x to URL: {url}"
                                )

                        # Apply keyword multiplier if priority keywords are set
                        if priority_keywords and snippet:
                            snippet_lower = snippet.lower()
                            # Count matching keywords
                            keyword_matches = [
                                keyword
                                for keyword in priority_keywords
                                if keyword in snippet_lower
                            ]
                            keyword_count = len(keyword_matches)

                            if keyword_count > 0:
                                # Calculate cumulative multiplier (multiply by keyword_multiplier_per_match for each match)
                                # But cap at max_keyword_multiplier
                                cumulative_multiplier = min(
                                    max_keyword_multiplier,
                                    keyword_multiplier_per_match**keyword_count,
                                )
                                similarity *= cumulative_multiplier
                                logger.debug(
                                    f"Applied keyword multiplier {cumulative_multiplier:.2f}x "
                                    f"({keyword_count} keywords matched: {', '.join(keyword_matches[:3])}) to result {i}"
                                )

                        # Cap at 0.99 to avoid perfect scores
                        similarity = min(0.99, similarity)

                        # Log the full transformation if multipliers were applied
                        if similarity != original_similarity:
                            logger.info(
                                f"Result {i} multiplied: {original_similarity:.3f} → {similarity:.3f}"
                            )

                        # Store similarity in the result object for later use in topic dampening
                        result["similarity"] = similarity

                        # Apply penalty for repeated URLs
                        repeat_penalty = 1.0
                        url_repeats = url_selected_count.get(url, 0)
                        if url_repeats > 0:
                            # Apply a progressive penalty based on number of repeats
                            # More repeats = lower score (0.9, 0.8, 0.7, etc.)
                            repeat_penalty = max(0.5, 1.0 - (0.1 * url_repeats))
                            logger.debug(
                                f"Applied repeat penalty of {repeat_penalty} to URL: {url}"
                            )

                        # Apply penalty to similarity score
                        similarity *= repeat_penalty

                        # Store score for sorting
                        relevance_scores.append((i, similarity))

                        # Also store in the result for future use
                        result["similarity"] = similarity
                    else:
                        # No embedding, assign low score
                        relevance_scores.append((i, 0.1))
                        result["similarity"] = 0.1
                else:
                    # Insufficient content, assign low score
                    relevance_scores.append((i, 0.0))
                    result["similarity"] = 0.0

            except Exception as e:
                logger.error(f"Error calculating relevance for result {i}: {e}")
                relevance_scores.append((i, 0.0))
                result["similarity"] = 0.0

        # Update similarity cache
        self.update_state("similarity_cache", similarity_cache)

        # Sort by relevance score (highest first)
        relevance_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top results based on the dynamic count
        selected_indices = [x[0] for x in relevance_scores[:results_to_select]]
        selected_results = [results[i] for i in selected_indices]

        # Log selection information
        logger.info(
            f"Selected {len(selected_results)} most relevant results from {len(results)} total (added {additional_results} due to repeats)"
        )
        # Collect all content and quality factors first
        all_content = []
        for result in selected_results:
            content = result.get("content", "")[:2000]
            if content:
                # Use similarity as quality factor, normalize between 0.5-1.0
                quality = 0.5
                if "similarity" in result:
                    quality = 0.5 + (result["similarity"] * 0.5)
                all_content.append((content, quality))

        # Update ALL coverage in a single call
        if all_content:
            # Just grab dimensions once
            state = self.get_state()
            dims = state.get("research_dimensions")
            if dims and "coverage" in dims:
                coverage = np.array(dims["coverage"])

                # Process each content item sequentially
                for content, quality in all_content:
                    embed = await self.get_embedding(content[:2000])
                    if not embed:
                        continue
                    projection = np.dot(embed, np.array(dims["eigenvectors"]).T)
                    contribution = np.abs(projection) * quality

                    # Update coverage directly
                    for i in range(min(len(contribution), len(coverage))):
                        coverage[i] += contribution[i] * (1 - coverage[i] / 2)

                # Normalize once at the end
                coverage = np.minimum(coverage, 3.0) / 3.0

                # Save back
                dims["coverage"] = coverage.tolist()
                self.update_state("research_dimensions", dims)
                self.update_state("latest_dimension_coverage", coverage.tolist())

                # Log dimension updates for debugging
                state = self.get_state()
                research_dimensions = state.get("research_dimensions")
                if research_dimensions:
                    coverage = research_dimensions.get("coverage", [])
                    logger.debug(
                        f"Dimension coverage after result: {[round(c * 100) for c in coverage[:3]]}%..."
                    )

        return selected_results

    async def check_result_relevance(
        self,
        result: Dict,
        query: str,
        outline_items: Optional[List[str]] = None,
    ) -> bool:
        """Check if a search result is relevant to the query and research outline using a lightweight model"""
        if not self.valves.QUALITY_FILTER_ENABLED:
            return True  # Skip filtering if disabled

        # Get similarity score from result - access it correctly
        similarity = result.get("similarity", 0.0)

        # Skip filtering for very high similarity scores
        if similarity >= self.valves.QUALITY_SIMILARITY_THRESHOLD:
            logger.info(
                f"Result passed quality filter automatically with similarity {similarity:.3f}"
            )
            return True

        # Get content from the result
        content = result.get("content", "")
        title = result.get("title", "")
        url = result.get("url", "")

        if not content or len(content) < 200:
            logger.warning(
                f"Content too short for quality filtering, accepting by default"
            )
            return True

        # Create prompt for relevance checking
        relevance_prompt = {
            "role": "system",
            "content": """You are evaluating the relevance of a search result to a research query. 
Your task is to determine if the content is actually relevant to what the user is researching.

Answer with ONLY "Yes" if the content is relevant to the research query or "No" if it is:
- Not related to the core topic
- An advertisement disguised as content
- About a different product/concept with similar keywords
- So general or vague that it provides no substantive information
- Littered with HTML or CSS to the point of being unreadable

Reply with JUST "Yes" or "No" - no explanation or other text.""",
        }

        # Create context with query, outline, and full content
        context = f"Research Query: {query}\n\n"

        if outline_items and len(outline_items) > 0:
            context += "Research Outline Topics:\n"
            for item in outline_items[:5]:  # Limit to first 5 items
                context += f"- {item}\n"
            context += "\n"

        context += f"Result Title: {title}\n"
        context += f"Result URL: {url}\n\n"
        context += f"Content:\n{content}\n\n"
        context += f"""Is the above content relevant to this query: "{query}"? Answer with ONLY 'Yes' or 'No'."""

        try:
            # Use quality filter model
            quality_model = self.valves.QUALITY_FILTER_MODEL

            response = await self.generate_completion(
                quality_model,
                [relevance_prompt, {"role": "user", "content": context}],
                temperature=self.valves.TEMPERATURE
                * 0.2,  # Use your valve system with adjustment
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                answer = response["choices"][0]["message"]["content"].strip().lower()

                # Parse the response to get yes/no
                is_relevant = "yes" in answer.lower() and "no" not in answer.lower()

                logger.info(
                    f"Quality check for result: {'RELEVANT' if is_relevant else 'NOT RELEVANT'} (sim={similarity:.3f})"
                )

                return is_relevant
            else:
                logger.warning(
                    "Failed to get response from quality model, accepting by default"
                )
                return True

        except Exception as e:
            logger.error(f"Error in quality filtering: {e}")
            return True  # Accept by default on error

    async def process_query(
        self,
        query: str,
        query_embedding: List[float],
        outline_embedding: List[float],
        cycle_feedback: Optional[Dict] = None,
        summary_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Process a single search query and get results with quality filtering"""
        await self.emit_status("info", f"Searching for: {query}", False)

        # Sanitize the query to make it safer for search engines
        sanitized_query = await self.sanitize_query(query)

        # Get search results for the query
        search_results = await self.search_web(sanitized_query)
        if not search_results:
            await self.emit_message(f"*No results found for query: {query}*\n\n")
            return []

        # Always select the most relevant results - this adds similarity scores
        search_results = await self.select_most_relevant_results(
            search_results,
            query,
            query_embedding,
            outline_embedding,
            summary_embedding,
        )

        # Process each search result until we have enough successful results
        successful_results = []
        failed_count = 0

        # Get state for access to research outline
        state = self.get_state()
        all_topics = state.get("all_topics", [])

        # Track rejected results for logging
        rejected_results = []

        for result in search_results:
            # Stop if we've reached our target of successful results
            if len(successful_results) >= self.valves.SUCCESSFUL_RESULTS_PER_QUERY:
                break

            # Stop if we've had too many consecutive failures
            if failed_count >= self.valves.MAX_FAILED_RESULTS:
                await self.emit_message(
                    f"*Skipping remaining results for query: {query} after {failed_count} failures*\n\n"
                )
                break

            try:
                # Process the result
                processed_result = await self.process_search_result(
                    result,
                    query,
                    query_embedding,
                    outline_embedding,
                    summary_embedding,
                )

                # Make sure similarity is preserved from original result
                if "similarity" in result and "similarity" not in processed_result:
                    processed_result["similarity"] = result["similarity"]

                # Check if processing was successful (has substantial content and valid URL)
                if (
                    processed_result
                    and processed_result.get("content")
                    and len(processed_result.get("content", "")) > 200
                    and processed_result.get("valid", False)
                    and processed_result.get("url", "")
                ):
                    # Add token count if not already present
                    if "tokens" not in processed_result:
                        processed_result["tokens"] = await self.count_tokens(
                            processed_result["content"]
                        )

                    # Skip results with less than 200 tokens
                    if processed_result["tokens"] < 200:
                        logger.info(
                            f"Skipping result with only {processed_result['tokens']} tokens (less than minimum 200)"
                        )
                        continue

                    # Only apply quality filter for results with low similarity
                    if (
                        self.valves.QUALITY_FILTER_ENABLED
                        and "similarity" in processed_result
                        and processed_result["similarity"]
                        < self.valves.QUALITY_SIMILARITY_THRESHOLD
                    ):
                        # Check if result is relevant using quality filter
                        is_relevant = await self.check_result_relevance(
                            processed_result,
                            query,
                            all_topics,
                        )

                        if not is_relevant:
                            # Track rejected result
                            rejected_results.append(
                                {
                                    "url": processed_result.get("url", ""),
                                    "title": processed_result.get("title", ""),
                                    "similarity": processed_result.get("similarity", 0),
                                    "processed_result": processed_result,
                                }
                            )
                            logger.warning(
                                f"Rejected irrelevant result: {processed_result.get('url', '')}"
                            )
                            continue
                    else:
                        # Skip filter for high similarity or when filtering is disabled
                        logger.info(
                            f"Skipping quality filter for result: {processed_result.get('similarity', 0):.3f}"
                        )

                    # Add to successful results
                    successful_results.append(processed_result)

                    # Get the document title for display
                    document_title = processed_result["title"]
                    if document_title == f"'{query}'" and processed_result["url"]:
                        # Try to get a better title from the URL
                        from urllib.parse import urlparse

                        parsed_url = urlparse(processed_result["url"])
                        path_parts = parsed_url.path.split("/")
                        if path_parts[-1]:
                            file_name = path_parts[-1]
                            # Clean up filename to use as title
                            if file_name.endswith(".pdf"):
                                document_title = (
                                    file_name[:-4].replace("-", " ").replace("_", " ")
                                )
                            elif "." in file_name:
                                document_title = (
                                    file_name.split(".")[0]
                                    .replace("-", " ")
                                    .replace("_", " ")
                                )
                            else:
                                document_title = file_name.replace("-", " ").replace(
                                    "_", " "
                                )
                        else:
                            # Use domain as title if no useful path
                            document_title = parsed_url.netloc

                    # Get token count for displaying
                    token_count = processed_result.get("tokens", 0)
                    if token_count == 0:
                        token_count = await self.count_tokens(
                            processed_result["content"]
                        )

                    # Display the result to the user with improved formatting
                    if processed_result["url"]:
                        # Show full URL in the result header
                        url = processed_result["url"]

                        # Check if this is a PDF (either by extension or by content type detection)
                        if (
                            url.endswith(".pdf")
                            or "application/pdf" in url
                            or self.is_pdf_content
                        ):
                            prefix = "PDF: "
                        else:
                            prefix = "Site: "

                        result_text = (
                            f"#### {prefix}{url}\n**Tokens:** {token_count}\n\n"
                        )
                    else:
                        result_text = (
                            f"#### {document_title} [{token_count} tokens]\n\n"
                        )

                    result_text += f"*Search query: {query}*\n\n"

                    # Format content with short line merging
                    content_to_display = processed_result["content"][
                        : self.valves.MAX_RESULT_TOKENS
                    ]
                    formatted_content = await self.clean_text_formatting(
                        content_to_display
                    )
                    result_text += f"{formatted_content}...\n\n"

                    # Add repeat indicator if this is a repeated URL
                    repeat_count = processed_result.get("repeat_count", 0)
                    if repeat_count > 1:
                        result_text += f"*Note: This URL has been processed {repeat_count} times*\n\n"

                    await self.emit_message(result_text)

                    # Reset failed count on success
                    failed_count = 0
                else:
                    # Count as a failure
                    failed_count += 1
                    logger.warning(
                        f"Failed to get substantial content from result {len(successful_results) + failed_count} for query: {query}"
                    )

            except Exception as e:
                # Count as a failure
                failed_count += 1
                logger.error(f"Error processing result for query '{query}': {e}")
                await self.emit_message(
                    f"*Error processing a result for query: {query}*\n\n"
                )

        # If we didn't get any successful results but had rejected ones, use the top rejected result
        if not successful_results and rejected_results:
            # Sort rejected results by similarity (descending)
            sorted_rejected = sorted(
                rejected_results, key=lambda x: x.get("similarity", 0), reverse=True
            )
            top_rejected = sorted_rejected[0]

            logger.info(
                f"Using top rejected result as fallback: {top_rejected.get('url', '')}"
            )

            # Get the processed result directly from the rejection record
            if "processed_result" in top_rejected:
                processed_result = top_rejected["processed_result"]
                successful_results.append(processed_result)

                # Display the result with a note that it might not be fully relevant
                document_title = processed_result.get("title", f"Result for '{query}'")
                token_count = processed_result.get(
                    "tokens", 0
                ) or await self.count_tokens(processed_result["content"])
                url = processed_result.get("url", "")

                result_text = f"#### {document_title} [{token_count} tokens]\n\n"
                if url:
                    result_text = f"#### {'PDF: ' if url.endswith('.pdf') else 'Site: '}{url}\n**Tokens:** {token_count}\n\n"

                result_text += f"*Search query: {query}*\n\n"
                result_text += f"*Note: This result was initially filtered but is used as a fallback.*\n\n"

                # Format content
                content_to_display = processed_result["content"][
                    : self.valves.MAX_RESULT_TOKENS
                ]
                formatted_content = await self.clean_text_formatting(content_to_display)
                result_text += f"{formatted_content}...\n\n"

                await self.emit_message(result_text)

        # If we still didn't get any successful results, log this
        if not successful_results:
            logger.warning(f"No valid results obtained for query: {query}")
            await self.emit_message(f"*No valid results found for query: {query}*\n\n")

        # Update token counts with new results
        await self.update_token_counts(successful_results)

        return successful_results

    def get_research_model(self):
        """Get the appropriate model for research/mechanical tasks"""
        # Always use the main research model
        return self.valves.RESEARCH_MODEL

    def get_synthesis_model(self):
        """Get the appropriate model for synthesis tasks"""
        if (
            self.valves.SYNTHESIS_MODEL
            and self.valves.SYNTHESIS_MODEL != self.valves.RESEARCH_MODEL
        ):
            return self.valves.SYNTHESIS_MODEL
        return self.valves.RESEARCH_MODEL

    async def generate_completion(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        json_mode: bool = False,  # Add a flag for expecting JSON
    ):
        """Generate a completion from the specified model"""
        try:
            if temperature is None:
                temperature = self.valves.TEMPERATURE

            form_data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "keep_alive": "10m",
            }

            # --- FIX STARTS HERE ---
            # Add format: 'json' if json_mode is requested
            if json_mode:
                form_data["format"] = "json"
            # --- FIX ENDS HERE ---

            response = await generate_chat_completions(
                self.__request__,
                form_data,
                user=self.__user__,
            )

            return response
        except Exception as e:
            logger.error(f"Error generating completion with model {model}: {e}")
            return {"choices": [{"message": {"content": f"Error: {str(e)}"}}]}

    async def emit_message(self, message: str):
        """Emit a message to the client"""
        logger.debug(
            f"★★★ ENTERING emit_message: message length={len(message)}, first 100 chars: {message[:100]}... ★★★"
        )
        try:
            # Add a small delay to ensure message is processed by the frontend
            # This helps prevent race conditions with the "completed" signal
            logger.debug("About to emit message to client")
            await self.__current_event_emitter__(
                {"type": "message", "data": {"content": message}}
            )
            logger.debug("Message emitted successfully")

            # Add a small delay after emitting the message
            # This gives the frontend time to process and render the message
            # before potentially receiving a "completed" signal
            await asyncio.sleep(0.5)
            logger.debug("Completed delay after emitting message")
        except Exception as e:
            logger.error(f"Error emitting message: {e}")
            # Can't do much if this fails, but we don't want to crash

    async def emit_status(self, level: str, message: str, done: bool = False):
        """Emit a status message to the client"""
        logger.debug(
            f"★★★ ENTERING emit_status: level={level}, message={message[:100]}..., done={done} ★★★"
        )
        try:
            # Check if research is completed
            state = self.get_state()
            logger.debug("Got state in emit_status")
            research_completed = state.get("research_completed", False)
            logger.debug(f"Research completed: {research_completed}")

            if research_completed and not done:
                status = "complete"
            else:
                status = "complete" if done else "in_progress"

            await self.__current_event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            logger.debug(f"★★★ EXITING emit_status SUCCESSFULLY: {message[:50]}... ★★★")

        except Exception as e:
            logger.error(f"★★★ ERROR IN emit_status: {e} ★★★", exc_info=True)
            logger.debug(f"★★★ EXITING emit_status WITH ERROR: {message[:50]}... ★★★")
            # Can't do much if this fails, but we don't want to crash

    async def emit_synthesis_status(self, message, is_done=False):
        """Emit both a status update and a chat message for synthesis progress"""
        await self.emit_status("info", message, is_done)
        await self.emit_message(f"*{message}*\n")

    async def rank_topics_by_research_priority(
        self,
        active_topics: List[str],
        gap_vector: Optional[List[float]] = None,
        completed_topics: Optional[Set[str]] = None,
        research_results: Optional[List[Dict]] = None,
    ) -> List[str]:
        """Rank research topics by priority using semantic dimensions and gap analysis with dampening for frequently used topics"""
        if not active_topics:
            return []

        # If we only have a few topics, keep the original order
        if len(active_topics) <= 3:
            return active_topics

        # Get cache of topic alignments
        state = self.get_state()
        topic_alignment_cache = state.get("topic_alignment_cache", {})

        # Get topic usage counts for dampening
        topic_usage_counts = state.get("topic_usage_counts", {})
        dampening_factor = 0.9  # Each use reduces priority by 10%

        # Initialize scores for each topic
        topic_scores = {}

        # Get embeddings for all topics
        logger.info(f"Getting embeddings for {len(active_topics)} topics")
        topic_embeddings = {}

        # Get embeddings for each topic
        for topic in active_topics:
            embedding = await self.get_embedding(topic)
            if embedding:
                topic_embeddings[topic] = embedding

        # Get research trajectory for alignment calculation
        research_trajectory = state.get("research_trajectory")

        # Get user preferences
        user_preferences = state.get("user_preferences", {})
        pdv = user_preferences.get("pdv")
        pdv_impact = user_preferences.get("impact", 0.0)

        # Get current cycle for adaptive weights
        current_cycle = len(state.get("cycle_summaries", [])) + 1
        max_cycles = self.valves.MAX_CYCLES

        # Calculate weights for different factors based on research progress
        trajectory_weight = self.valves.TRAJECTORY_MOMENTUM

        # PDV weight calculation
        pdv_weight = 0.0
        if pdv is not None and pdv_impact > 0.1:
            pdv_alignment_history = state.get("pdv_alignment_history", [])
            if pdv_alignment_history:
                recent_alignment = sum(pdv_alignment_history[-3:]) / max(
                    1, len(pdv_alignment_history[-3:])
                )
                alignment_factor = min(1.0, recent_alignment * 2)
                pdv_weight = pdv_impact * alignment_factor

                # Apply adaptive fade-out
                fade_start_cycle = min(5, int(0.33 * max_cycles))
                if current_cycle > fade_start_cycle:
                    remaining_cycles = max_cycles - current_cycle
                    total_fade_cycles = max_cycles - fade_start_cycle
                    if total_fade_cycles > 0:
                        fade_ratio = remaining_cycles / total_fade_cycles
                        pdv_weight *= max(0.0, fade_ratio)
                    else:
                        pdv_weight = 0.0
            else:
                pdv_weight = pdv_impact

        # Gap weight calculation
        gap_weight = 0.0
        if gap_vector is not None:
            fade_start_cycle = min(5, int(0.5 * max_cycles))
            if current_cycle <= fade_start_cycle:
                gap_weight = self.valves.GAP_EXPLORATION_WEIGHT
            else:
                remaining_cycles = max_cycles - current_cycle
                total_fade_cycles = max_cycles - fade_start_cycle
                if total_fade_cycles > 0:
                    fade_ratio = remaining_cycles / total_fade_cycles
                    gap_weight = self.valves.GAP_EXPLORATION_WEIGHT * max(
                        0.0, fade_ratio
                    )

        # Content relevance weight increases over time
        relevance_weight = 0.2 + (0.3 * min(1.0, current_cycle / (max_cycles * 0.7)))

        # Normalize weights to sum to 1.0
        total_weight = trajectory_weight + pdv_weight + gap_weight + relevance_weight
        if total_weight > 0:
            trajectory_weight /= total_weight
            pdv_weight /= total_weight
            gap_weight /= total_weight
            relevance_weight /= total_weight

        logger.info(
            f"Priority weights: trajectory={trajectory_weight:.2f}, pdv={pdv_weight:.2f}, gap={gap_weight:.2f}, relevance={relevance_weight:.2f}"
        )

        # Prepare completed topics embeddings for relevance scoring
        completed_embeddings = {}
        if completed_topics and len(completed_topics) > 0 and relevance_weight > 0.0:
            # Limit number of completed topics to consider for efficiency
            completed_sample_size = min(10, len(completed_topics))
            completed_topics_list = list(completed_topics)[:completed_sample_size]

            # Get all completed topics embeddings sequentially
            completed_embed_results = []
            for topic in completed_topics_list:
                embedding = await self.get_embedding(topic)
                if embedding:
                    completed_embed_results.append(embedding)

            # Store valid embeddings with topic keys
            for i, embedding in enumerate(completed_embed_results):
                if embedding and i < len(completed_topics_list):
                    completed_embeddings[completed_topics_list[i]] = embedding

        # Prepare recent result embeddings for relevance scoring
        result_embeddings = {}
        if research_results and len(research_results) > 0 and relevance_weight > 0.0:
            # Get limited recent results (last 8 for efficiency)
            recent_results = research_results[-8:]

            # Prepare content for embedding
            result_contents = []
            for result in recent_results:
                content = result.get("content", "")[:2000]
                result_contents.append(content)

            # Get embeddings sequentially
            result_embed_results = []
            for content in result_contents:
                embedding = await self.get_embedding(content)
                if embedding:
                    result_embed_results.append(embedding)

            # Store valid embeddings with result index as key
            for i, embedding in enumerate(result_embed_results):
                if embedding and i < len(recent_results):
                    result_id = recent_results[i].get("url", "") or f"result_{i}"
                    result_embeddings[result_id] = embedding

        # Calculate scores for each topic
        for topic, topic_embedding in topic_embeddings.items():
            # Start with a base score
            score = 0.5
            component_scores = {}

            # Factor 1: Alignment with trajectory (research direction)
            if research_trajectory is not None and trajectory_weight > 0.0:
                # Check cache first
                cache_key = f"traj_{topic}"
                if cache_key in topic_alignment_cache:
                    traj_alignment = topic_alignment_cache[cache_key]
                else:
                    traj_alignment = np.dot(topic_embedding, research_trajectory)
                    # Normalize to 0-1 range
                    traj_alignment = (traj_alignment + 1) / 2
                    # Cache the result
                    topic_alignment_cache[cache_key] = traj_alignment

                component_scores["trajectory"] = traj_alignment * trajectory_weight

            # Factor 2: Alignment with user preference direction vector
            if pdv is not None and pdv_weight > 0.0:
                # Check cache first
                cache_key = f"pdv_{topic}"
                if cache_key in topic_alignment_cache:
                    pdv_alignment = topic_alignment_cache[cache_key]
                else:
                    pdv_alignment = np.dot(topic_embedding, pdv)
                    # Normalize to 0-1 range
                    pdv_alignment = (pdv_alignment + 1) / 2
                    # Cache the result
                    topic_alignment_cache[cache_key] = pdv_alignment

                component_scores["pdv"] = pdv_alignment * pdv_weight

            # Factor 3: Alignment with gap vector (unexplored areas)
            if gap_vector is not None and gap_weight > 0.0:
                # Check cache first
                cache_key = f"gap_{topic}"
                if cache_key in topic_alignment_cache:
                    gap_alignment = topic_alignment_cache[cache_key]
                else:
                    gap_alignment = np.dot(topic_embedding, gap_vector)
                    # Normalize to 0-1 range
                    gap_alignment = (gap_alignment + 1) / 2
                    # Cache the result
                    topic_alignment_cache[cache_key] = gap_alignment

                component_scores["gap"] = gap_alignment * gap_weight

            # Factor 4: Topic novelty compared to completed research
            if completed_embeddings and relevance_weight > 0.0:
                # Calculate average similarity to completed topics
                similarity_sum = 0
                count = 0

                for (
                    completed_topic,
                    completed_embedding,
                ) in completed_embeddings.items():
                    # Check cache first
                    cache_key = f"comp_{topic}_{completed_topic}"
                    if cache_key in topic_alignment_cache:
                        sim = topic_alignment_cache[cache_key]
                    else:
                        sim = cosine_similarity(
                            [topic_embedding], [completed_embedding]
                        )[0][0]
                        # Cache the result
                        topic_alignment_cache[cache_key] = sim

                    similarity_sum += sim
                    count += 1

                if count > 0:
                    avg_similarity = similarity_sum / count
                    # Invert - lower similarity means higher novelty
                    novelty = 1.0 - avg_similarity
                    component_scores["novelty"] = novelty * (relevance_weight * 0.5)

            # Factor 5: Information need based on search results
            if result_embeddings and relevance_weight > 0.0:
                # Calculate average relevance to results
                relevance_sum = 0
                count = 0

                for result_id, result_embedding in result_embeddings.items():
                    # Create cache key using result ID
                    cache_key = f"res_{topic}_{hash(result_id) % 10000}"

                    if cache_key in topic_alignment_cache:
                        rel = topic_alignment_cache[cache_key]
                    else:
                        rel = cosine_similarity([topic_embedding], [result_embedding])[
                            0
                        ][0]
                        # Cache the result
                        topic_alignment_cache[cache_key] = rel

                    relevance_sum += rel
                    count += 1

                if count > 0:
                    avg_relevance = relevance_sum / count
                    # Invert - lower relevance means higher information need
                    info_need = 1.0 - avg_relevance
                    component_scores["info_need"] = info_need * (relevance_weight * 0.5)

            # Calculate final score as sum of all component scores
            final_score = sum(component_scores.values())
            if not component_scores:
                final_score = 0.5  # Default if no components were calculated

            # Apply dampening based on usage count and result quality
            usage_count = topic_usage_counts.get(topic, 0)
            if usage_count > 0:
                # Get all results related to this topic
                topic_results = []

                # Look for results where the topic appears in the query or result content
                for result in research_results or []:
                    # Check if this result is relevant to this topic
                    result_content = result.get("content", "")[
                        :500
                    ]  # Use first 500 chars for efficiency
                    if topic in result.get("query", "") or topic in result_content:
                        topic_results.append(result)

                # If we have results for this topic, calculate quality-based dampening
                if topic_results:
                    # Calculate average similarity for this topic's results
                    avg_similarity = 0.0
                    count = 0
                    for result in topic_results:
                        similarity = result.get("similarity", 0.0)
                        if similarity > 0:  # Only count results with valid similarity
                            avg_similarity += similarity
                            count += 1

                    if count > 0:
                        avg_similarity /= count

                    # Scale dampening factor based on result quality
                    # similarity > 0.8: no penalty (dampening_multiplier = 1.0)
                    # similarity < 0.3: 50% penalty (dampening_multiplier = 0.5)
                    # Linear scaling between
                    if avg_similarity >= 0.8:
                        dampening_multiplier = 1.0
                    elif avg_similarity <= 0.3:
                        dampening_multiplier = 0.5
                    else:
                        # Linear scaling between 0.5 and 1.0
                        dampening_multiplier = 0.5 + (
                            0.5 * (avg_similarity - 0.3) / 0.5
                        )

                    logger.debug(
                        f"Topic '{topic}' quality-based dampening: {dampening_multiplier:.3f} (avg similarity: {avg_similarity:.3f}, from {count} results)"
                    )
                else:
                    # If no results yet, use the default dampening
                    dampening_multiplier = dampening_factor**usage_count
                    logger.debug(
                        f"Topic '{topic}' default dampening: {dampening_multiplier:.3f} (used {usage_count} times)"
                    )

                # Apply the dampening multiplier
                final_score *= dampening_multiplier

            # Store the score
            topic_scores[topic] = final_score

        # Update alignment cache with size limiting
        if len(topic_alignment_cache) > 300:  # Limit cache size
            # Create new cache with only recent entries
            new_cache = {}
            count = 0
            for k, v in reversed(list(topic_alignment_cache.items())):
                new_cache[k] = v
                count += 1
                if count >= 200:  # Keep 200 most recent entries
                    break
            topic_alignment_cache = new_cache

        self.update_state("topic_alignment_cache", topic_alignment_cache)

        # Sort topics by score (highest first)
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_topics = [topic for topic, score in sorted_topics]

        logger.info(f"Ranked {len(ranked_topics)} topics by research priority")
        return ranked_topics

    async def process_user_outline_feedback(
        self, outline_items: List[Dict], original_query: str
    ) -> Dict:
        """Process user feedback on research outline items by asking for feedback in chat"""
        logger.debug("★★★ ENTERING process_user_outline_feedback ★★★")
        logger.debug(
            f"outline_items type: {type(outline_items)}, length: {len(outline_items)}"
        )

        # Number each outline item (maintain hierarchy but flatten for numbering)
        numbered_outline = []
        flat_items = []

        # Use the existing _flatten_outline_topics method to properly extract all topic strings
        logger.debug("Using _flatten_outline_topics to safely extract all topics")
        flat_items = self._flatten_outline_topics(outline_items)
        logger.debug(f"Flattened {len(flat_items)} topics from outline")

        # Create numbered outline for display
        for i, topic in enumerate(flat_items, 1):
            numbered_outline.append(f"{i}. {topic}")
            logger.debug(f"Added numbered item {i}: {topic}")

        logger.debug(f"Created numbered outline with {len(numbered_outline)} items")
        logger.debug("★★★ OUTLINE PROCESSING COMPLETE ★★★")

        # Prepare the outline display
        outline_display = "\n".join(numbered_outline)

        # Emit a message with instructions using improved slash commands
        feedback_message = (
            "### Research Outline\n\n"
            f"{outline_display}\n\n"
            "**Please provide feedback on this research outline.**\n\n"
            "You can:\n"
            "- Use commands like `/keep 1,3,5-7` or `/remove 2,4,8-10` to select specific items by number\n"
            "- Or simply describe what topics you want to focus on or avoid in natural language\n\n"
            "Examples:\n"
            "- `/k 1,3,5-7` (keep only items 1,3,5,6,7)\n"
            "- `/r 2,4,8-10` (remove items 2,4,8,9,10)\n"
            '- "Focus on historical aspects and avoid technical details"\n'
            '- "I\'m more interested in practical applications than theoretical concepts"\n\n'
            "If you want to continue with all items, just reply 'continue' or leave your message empty.\n\n"
            "**I'll pause here to await your response before continuing the research.**"
        )

        await self.emit_message(feedback_message)

        # Set flag to indicate we're waiting for feedback
        self.update_state("waiting_for_outline_feedback", True)
        self.update_state(
            "outline_feedback_data",
            {
                "outline_items": outline_items,
                "flat_items": flat_items,
                "numbered_outline": numbered_outline,
                "original_query": original_query,
            },
        )

        # Return a default response (this will be overridden in the next call)
        return {
            "kept_items": flat_items,
            "removed_items": [],
            "kept_indices": list(range(len(flat_items))),
            "removed_indices": [],
            "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
        }

    async def process_natural_language_feedback(
        self, user_message: str, flat_items: List[str]
    ) -> Dict:
        """Process natural language feedback to determine which topics to keep/remove"""

        # Create a prompt for the model to interpret user feedback
        interpret_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant analyzing user feedback on a research outline.
    Based on the user's natural language input, determine which research topics should be kept or removed.
    
    The user's message expresses preferences about the research direction. Analyze this to identify:
    1. Which specific topics from the outline align with their interests
    2. Which specific topics should be removed based on their preferences
    
    Your task is to categorize each topic as EITHER "keep" OR "remove", NEVER both, based on the user's natural language feedback.
    Don't allow your own biases or preferences to have any affect on your answer - please remain purely objective and user research-oriented.
    Provide your response as a JSON object with two lists: "keep" for indices to keep, and "remove" for indices to remove.
    Indices should be 0-based (first item is index 0).""",
        }

        # Prepare context with list of topics and user message
        topics_list = "\n".join([f"{i}. {topic}" for i, topic in enumerate(flat_items)])

        context = f"""Research outline topics:
    {topics_list}
    
    User feedback:
    "{user_message}"
    
    Based on this feedback, categorize each topic (by index) as either "keep" or "remove".
    If the user clearly expresses a preference to focus on certain topics or avoid others, use that to guide your decisions.
    If the user's feedback is ambiguous about some topics, categorize them based on their similarity to clearly mentioned preferences.
    """

        # Generate interpretation of user feedback
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [interpret_prompt, {"role": "user", "content": context}],
                temperature=self.valves.TEMPERATURE
                * 0.3,  # Low temperature for consistent interpretation
            )

            result_content = response["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                json_str = result_content[
                    result_content.find("{") : result_content.rfind("}") + 1
                ]
                result_data = json.loads(json_str)

                # Get keep and remove lists
                keep_indices = result_data.get("keep", [])
                remove_indices = result_data.get("remove", [])

                # Ensure both keep_indices and remove_indices are lists
                if not isinstance(keep_indices, list):
                    keep_indices = []
                if not isinstance(remove_indices, list):
                    remove_indices = []

                # Ensure each index is in either keep or remove
                all_indices = set(range(len(flat_items)))
                missing_indices = all_indices - set(keep_indices) - set(remove_indices)

                # By default, keep missing indices
                keep_indices.extend(missing_indices)

                # Convert to kept and removed items
                kept_items = [
                    flat_items[i] for i in keep_indices if i < len(flat_items)
                ]
                removed_items = [
                    flat_items[i] for i in remove_indices if i < len(flat_items)
                ]

                logger.info(
                    f"Natural language feedback interpretation: keep {len(kept_items)}, remove {len(removed_items)}"
                )

                return {
                    "kept_items": kept_items,
                    "removed_items": removed_items,
                    "kept_indices": keep_indices,
                    "removed_indices": remove_indices,
                }

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing feedback interpretation: {e}")
                # Default to keeping all items
                return {
                    "kept_items": flat_items,
                    "removed_items": [],
                    "kept_indices": list(range(len(flat_items))),
                    "removed_indices": [],
                }

        except Exception as e:
            logger.error(f"Error interpreting natural language feedback: {e}")
            # Default to keeping all items
            return {
                "kept_items": flat_items,
                "removed_items": [],
                "kept_indices": list(range(len(flat_items))),
                "removed_indices": [],
            }

    async def process_outline_feedback_continuation(self, user_message: str):
        """Process the user feedback received in a continuation call"""
        logger.debug("★★★ ENTERING process_outline_feedback_continuation ★★★")
        logger.debug(f"User message: '{user_message}'")

        # Get the data from the previous call
        state = self.get_state()
        feedback_data = state.get("outline_feedback_data", {})
        outline_items = feedback_data.get("outline_items", [])
        flat_items = feedback_data.get("flat_items", [])
        original_query = feedback_data.get("original_query", "")

        logger.debug(
            f"Retrieved feedback_data: outline_items={len(outline_items)}, flat_items={len(flat_items)}"
        )
        logger.debug(
            f"flat_items types: {[type(item) for item in flat_items[:5]]}"
        )  # Check first 5 items

        # Validate flat_items - ensure all items are strings
        validated_flat_items = []
        for i, item in enumerate(flat_items):
            if isinstance(item, str):
                validated_flat_items.append(item)
                logger.debug(f"Item {i} is valid string: {item[:50]}...")
            elif isinstance(item, dict):
                # Extract topic from dict if possible
                topic = item.get("topic", str(item))
                validated_flat_items.append(topic)
                logger.warning(f"Item {i} was dict, extracted topic: {topic[:50]}...")
            else:
                # Convert to string as fallback
                str_item = str(item)
                validated_flat_items.append(str_item)
                logger.warning(
                    f"Item {i} was {type(item)}, converted to string: {str_item[:50]}..."
                )

        flat_items = validated_flat_items
        logger.debug(f"Validated flat_items: {len(flat_items)} items, all strings")

        # Process the user input
        user_input = user_message.strip()
        logger.debug(f"Processing user input: '{user_input}'")

        # If user just wants to continue with all items
        if user_input.lower() == "continue" or not user_input:
            logger.debug("User chose to continue with all items")
            await self.emit_message(
                "\n*Continuing with all research outline items.*\n\n"
            )
            result = {
                "kept_items": flat_items,
                "removed_items": [],
                "kept_indices": list(range(len(flat_items))),
                "removed_indices": [],
                "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
            }
            logger.debug(
                f"Returning result with {len(result['kept_items'])} kept items"
            )
            logger.debug(
                "★★★ EXITING process_outline_feedback_continuation (CONTINUE) ★★★"
            )
            return result

        # Check if it's a slash command (keep or remove)
        slash_keep_patterns = [r"^/k\s", r"^/keep\s"]
        slash_remove_patterns = [r"^/r\s", r"^/remove\s"]

        is_keep_cmd = any(
            re.match(pattern, user_input) for pattern in slash_keep_patterns
        )
        is_remove_cmd = any(
            re.match(pattern, user_input) for pattern in slash_remove_patterns
        )

        # Process slash commands
        if is_keep_cmd or is_remove_cmd:
            # Extract the item indices/ranges part
            if is_keep_cmd:
                items_part = re.sub(r"^(/k|/keep)\s+", "", user_input).replace(",", " ")
            else:
                items_part = re.sub(r"^(/r|/remove)\s+", "", user_input).replace(
                    ",", " "
                )

            # Process the indices and ranges
            selected_indices = set()
            for part in items_part.split():
                part = part.strip()
                if not part:
                    continue

                # Check if it's a range (e.g., 5-9)
                if "-" in part:
                    try:
                        start, end = map(int, part.split("-"))
                        # Validate range bounds before converting to 0-indexed
                        if (
                            start < 1
                            or start > len(flat_items)
                            or end < 1
                            or end > len(flat_items)
                        ):
                            await self.emit_message(
                                f"Invalid range '{part}': valid range is 1-{len(flat_items)}. Skipping."
                            )
                            continue

                        # Convert to 0-indexed
                        start = start - 1
                        end = end - 1
                        selected_indices.update(range(start, end + 1))
                    except ValueError:
                        await self.emit_message(
                            f"Invalid range format: '{part}'. Skipping."
                        )
                else:
                    # Single number
                    try:
                        idx = int(part)
                        # Validate index before converting to 0-indexed
                        if idx < 1 or idx > len(flat_items):
                            await self.emit_message(
                                f"Index {idx} out of range: valid range is 1-{len(flat_items)}. Skipping."
                            )
                            continue

                        # Convert to 0-indexed
                        idx = idx - 1
                        selected_indices.add(idx)
                    except ValueError:
                        await self.emit_message(f"Invalid number: '{part}'. Skipping.")

            # Convert to lists
            selected_indices = sorted(list(selected_indices))

            # Determine kept and removed indices based on mode
            if is_keep_cmd:
                # Keep mode - selected indices are kept, others removed
                kept_indices = selected_indices
                removed_indices = [
                    i for i in range(len(flat_items)) if i not in kept_indices
                ]
            else:
                # Remove mode - selected indices are removed, others kept
                removed_indices = selected_indices
                kept_indices = [
                    i for i in range(len(flat_items)) if i not in removed_indices
                ]

            # Get the actual items
            kept_items = [flat_items[i] for i in kept_indices if i < len(flat_items)]
            removed_items = [
                flat_items[i] for i in removed_indices if i < len(flat_items)
            ]
        else:
            # Process natural language feedback
            nl_feedback = await self.process_natural_language_feedback(
                user_input, flat_items
            )

            # Make sure we have a valid response, not None
            if nl_feedback is None:
                # Default to keeping all items
                nl_feedback = {
                    "kept_items": flat_items,
                    "removed_items": [],
                    "kept_indices": list(range(len(flat_items))),
                    "removed_indices": [],
                }

            kept_items = nl_feedback.get("kept_items", flat_items)
            removed_items = nl_feedback.get("removed_items", [])
            kept_indices = nl_feedback.get("kept_indices", list(range(len(flat_items))))
            removed_indices = nl_feedback.get("removed_indices", [])

        # Calculate preference direction vector based on kept and removed items
        preference_vector = await self.calculate_preference_direction_vector(
            kept_items, removed_items, flat_items
        )

        # Update user_preferences in state with the new preference vector
        self.update_state("user_preferences", preference_vector)
        logger.info(
            f"Updated user_preferences with PDV impact: {preference_vector.get('impact', 0.0):.3f}"
        )

        # Show the user what's happening
        await self.emit_message("\n### Feedback Processed\n")

        if kept_items:
            kept_list = "\n".join([f"✓ {item}" for item in kept_items])
            await self.emit_message(
                f"**Keeping {len(kept_items)} items:**\n{kept_list}\n"
            )

        if removed_items:
            removed_list = "\n".join([f"✗ {item}" for item in removed_items])
            await self.emit_message(
                f"**Removing {len(removed_items)} items:**\n{removed_list}\n"
            )

        await self.emit_message("Generating replacement items for removed topics...\n")

        return {
            "kept_items": kept_items,
            "removed_items": removed_items,
            "kept_indices": kept_indices,
            "removed_indices": removed_indices,
            "preference_vector": preference_vector,
        }

    async def group_replacement_topics(self, replacement_topics):
        """Group replacement topics semantically into groups of 2-4 topics each"""
        # Skip if too few topics
        if len(replacement_topics) <= 4:
            return [replacement_topics]  # Just one group if 4 or fewer topics

        # Get embeddings for each topic sequentially
        topic_embeddings = []
        for i, topic in enumerate(replacement_topics):
            embedding = await self.get_embedding(topic)
            if embedding:
                topic_embeddings.append((topic, embedding))

        # If we don't have enough valid embeddings for grouping, use simple groups
        if len(topic_embeddings) < 3:
            logger.warning(
                "Not enough embeddings for semantic grouping, using simple groups"
            )
            # Just divide topics into groups of 4
            groups = []
            for i in range(0, len(replacement_topics), 4):
                groups.append(replacement_topics[i : i + 4])
            return groups

        try:
            # Extract embeddings into a numpy array
            embeddings_array = np.array([emb for _, emb in topic_embeddings])

            # Determine number of clusters (groups)
            total_topics = len(topic_embeddings)
            # Aim for groups of 3-4 topics each
            n_clusters = max(1, total_topics // 3)
            # Cap at a reasonable number
            n_clusters = min(n_clusters, 5)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings_array)

            # Group topics by cluster
            grouped_topics = {}
            for i, (topic, _) in enumerate(topic_embeddings):
                cluster_id = kmeans.labels_[i]
                if cluster_id not in grouped_topics:
                    grouped_topics[cluster_id] = []
                grouped_topics[cluster_id].append(topic)

            # Get the groups as a list
            groups_list = list(grouped_topics.values())

            # Balance any groups that are too small or large
            if len(groups_list) > 1:
                # Sort groups by size
                groups_list.sort(key=len)

                # Merge any tiny groups (fewer than 2 topics)
                while len(groups_list) > 1 and len(groups_list[0]) < 2:
                    smallest = groups_list.pop(0)
                    second_smallest = groups_list[0]  # Don't remove yet, just reference

                    # Merge with second smallest
                    groups_list[0] = second_smallest + smallest

                    # Re-sort
                    groups_list.sort(key=len)

                # Split any very large groups (more than 5 topics)
                for i, group in enumerate(groups_list):
                    if len(group) > 5:
                        # Simple split at midpoint
                        midpoint = len(group) // 2
                        groups_list[i] = group[:midpoint]  # First half
                        groups_list.append(group[midpoint:])  # Second half

            return groups_list

        except Exception as e:
            logger.error(f"Error during topic grouping: {e}")
            # Fall back to simple grouping on error
            groups = []
            for i in range(0, len(replacement_topics), 4):
                groups.append(replacement_topics[i : i + 4])
            return groups

    async def generate_group_query(self, topic_group, user_message):
        """Generate a search query that covers a group of related topics"""
        if not topic_group:
            return user_message

        topics_text = ", ".join(topic_group)

        # Create a prompt for generating the query
        prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant generating an effective search query. 
    Create a search query that will find relevant information for a group of related topics aimed at addressing the original user input.
    The query should be specific enough to find targeted information while broadly representing all topics in the group.
    Make the query concise (maximum 10 words) and focused.""",
        }

        # Create the message content
        message = {
            "role": "user",
            "content": f"""Generate a search query for this group of topics:
    {topics_text}
    
    This is related to the original user query: "{user_message}"
    
    Generate a single concise search query that will find information relevant to these topics.
    Just respond with the search query text only.""",
        }

        # Generate the query
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [prompt, message],
                temperature=self.valves.TEMPERATURE * 0.7,
            )

            query = response["choices"][0]["message"]["content"].strip()

            # Clean up the query: remove quotes and ensure it's not too long
            query = query.replace('"', "").replace('"', "").replace('"', "")

            # If the query is too long, truncate it
            if len(query.split()) > 12:
                query = " ".join(query.split()[:12])

            return query

        except Exception as e:
            logger.error(f"Error generating group query: {e}")
            # Fallback: combine the first topic with the user message
            return f"{user_message} {topic_group[0]}"

    async def extract_topic_relevant_info(self, results, topics):
        """Extract information from search results specifically relevant to given topics"""
        if not results:
            return []

        # Create a prompt for extracting relevant information
        extraction_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant extracting information from search results.
    Identify and extract information that is specifically relevant to the given topics.
    Format the extracted information as concise bullet points, focusing on facts, data, and insights.
    Ignore general information not directly related to the topics.""",
        }

        # Create context with search results and topics
        topics_str = ", ".join(topics)
        extraction_context = f"Topics: {topics_str}\n\nSearch Results:\n\n"

        for i, result in enumerate(results):
            extraction_context += f"Result {i+1}:\n"
            extraction_context += f"Title: {result.get('title', 'Untitled')}\n"
            extraction_context += f"Content: {result.get('content', '')}...\n\n"

        extraction_context += "\nExtract relevant information for the listed topics from these search results."

        # Create messages for extraction
        extraction_messages = [
            extraction_prompt,
            {"role": "user", "content": extraction_context},
        ]

        # Extract relevant information
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                extraction_messages,
                temperature=self.valves.TEMPERATURE
                * 0.4,  # Lower temperature for factual extraction
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                extracted_info = response["choices"][0]["message"]["content"]
                return extracted_info
            else:
                return "No relevant information found."

        except Exception as e:
            logger.error(f"Error extracting topic-relevant information: {e}")
            return "Error extracting information from search results."

    async def refine_topics_with_research(
        self, topics, relevant_info, pdv, original_query
    ):
        """Refine topics based on both user preferences and research results"""
        # Create a prompt for refining topics
        refine_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant refining research topics.
    Based on the extracted information and user preferences, revise each topic to:
    1. Be specific and targeted based on the research findings, while maintaining alignment with user preferences and the original query
    2. Prioritize topics that seem most relevant to answering the query and that will reasonably result in worthwhile expanded research
    3. Be phrased as clear, researchable topics in the same style as those to be replaced
    
    Your refined topics should incorporate new discoveries that heighten and expand upon the intent of the original query.
    Avoid overstating the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
    You do not need to include justification along with your refined topics.""",
        }

        # Create context with topics, research info, and preference direction
        pdv_context = ""
        if pdv is not None:
            pdv_context = "\nUser preferences are directing research toward topics similar to what was kept and away from what was removed."

        refine_context = f"""Original topics: {', '.join(topics)}
    
    Original query: {original_query}
    
    Extracted research information:
    {relevant_info}
    {pdv_context}
    
    Refine these topics based on the research findings and user preferences.
    Provide a list of the same number of refined topics."""

        # Create messages for refinement
        refine_messages = [refine_prompt, {"role": "user", "content": refine_context}]

        # Generate refined topics
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                refine_messages,
                temperature=self.valves.TEMPERATURE
                * 0.7,  # Balanced temperature for creativity with focus
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                refined_content = response["choices"][0]["message"]["content"]

                # Extract topics using regex (looking for numbered or bulleted list items)
                refined_topics = re.findall(
                    r"(?:^|\n)(?:\d+\.\s*|\*\s*|-\s*)([^\n]+)", refined_content
                )

                # If we couldn't extract enough topics, use the original ones
                if len(refined_topics) < len(topics):
                    logger.warning(
                        f"Not enough refined topics extracted ({len(refined_topics)}), using originals"
                    )
                    return topics

                # Limit to the same number as original topics
                refined_topics = refined_topics[: len(topics)]
                return refined_topics
            else:
                return topics

        except Exception as e:
            logger.error(f"Error refining topics: {e}")
            return topics

    async def continue_research_after_feedback(
        self,
        feedback_result,
        user_message,
        outline_items,
        all_topics,
        outline_embedding,
    ):
        """Continue the research process after receiving user feedback on the outline"""
        logger.debug("★★★ ENTERING continue_research_after_feedback ★★★")
        logger.debug(f"feedback_result keys: {list(feedback_result.keys())}")

        kept_items = feedback_result["kept_items"]
        removed_items = feedback_result["removed_items"]
        preference_vector = feedback_result["preference_vector"]

        logger.debug(f"kept_items: {len(kept_items)} items")
        logger.debug(
            f"kept_items types: {[type(item) for item in kept_items[:5]]}"
        )  # Check first 5 items
        logger.debug(f"removed_items: {len(removed_items)} items")
        logger.debug(f"all_topics: {len(all_topics)} items")  # Check first 5 items

        # If there are no removed items, skip the replacement logic and return original outline
        if not removed_items:
            await self.emit_message(
                "\n*No changes made to research outline. Continuing with original outline.*\n\n"
            )
            self.update_state(
                "research_state",
                {
                    "research_outline": outline_items,
                    "all_topics": all_topics,
                    "outline_embedding": outline_embedding,
                    "user_message": user_message,
                },
            )

            return outline_items, all_topics, outline_embedding

        # Generate replacement topics for removed items if needed
        if removed_items:
            await self.emit_status("info", "Generating replacement topics...", False)
            replacement_topics = await self.generate_replacement_topics(
                user_message,
                kept_items,
                removed_items,
                preference_vector,
                all_topics,
            )

            if replacement_topics:
                # Group replacement topics semantically
                topic_groups = await self.group_replacement_topics(replacement_topics)

                # Get state for tracking URLs
                state = self.get_state()
                url_selected_count = state.get("url_selected_count", {})

                # Get initial results to track URLs from previous cycles
                results_history = state.get("results_history", [])

                # Create a set of already seen URLs from all previous research
                previously_seen_urls = set()
                for result in results_history:
                    url = result.get("url", "")
                    if url:
                        previously_seen_urls.add(url)

                # Also track URLs we see during this replacement cycle
                replacement_cycle_seen_urls = set()

                # For each group, generate and execute targeted queries
                group_results = []
                for group in topic_groups:
                    # Generate a query that covers this group of topics
                    group_query = await self.generate_group_query(group, user_message)

                    # Get query embedding
                    query_embedding = await self.get_embedding(group_query)

                    # Execute search for this group
                    await self.emit_message(
                        f"**Researching topics:** {', '.join(group)}\n**Query:** {group_query}\n\n"
                    )
                    results = await self.process_query(
                        group_query, query_embedding, outline_embedding
                    )

                    # Filter out URLs we've seen in previous cycles or this replacement cycle
                    filtered_results = []
                    for result in results:
                        url = result.get("url", "")

                        # Skip if we've seen this URL in previous cycles or this replacement cycle
                        if url and (
                            url in previously_seen_urls
                            or url in replacement_cycle_seen_urls
                        ):
                            continue

                        # Keep new URLs we haven't seen before
                        filtered_results.append(result)
                        if url:
                            replacement_cycle_seen_urls.add(
                                url
                            )  # Mark as seen in this cycle

                    # If we have no results after filtering but had some initially, use fallback
                    if not filtered_results and results:
                        # Use a fallback approach - find the least seen URL
                        least_seen = None
                        min_seen_count = float("inf")

                        for result in results:
                            url = result.get("url", "")
                            seen_count = url_selected_count.get(url, 0)

                            if seen_count < min_seen_count:
                                min_seen_count = seen_count
                                least_seen = result

                        if least_seen:
                            filtered_results.append(least_seen)
                            if least_seen.get("url"):
                                replacement_cycle_seen_urls.add(least_seen.get("url"))
                            logger.info(
                                f"Using least-seen URL as fallback to ensure research continues"
                            )

                    group_results.append(
                        {
                            "topics": group,
                            "query": group_query,
                            "results": filtered_results,
                        }
                    )

                # Now refine each topic based on both PDV and search results
                refined_topics = []
                for group in group_results:
                    topics = group["topics"]
                    results = group["results"]

                    # Extract key information from results relevant to these topics
                    relevant_info = await self.extract_topic_relevant_info(
                        results, topics
                    )

                    # Generate refined topics that incorporate both user preferences and new research
                    refined = await self.refine_topics_with_research(
                        topics,
                        relevant_info,
                        self.get_state().get("user_preferences", {}).get("pdv"),
                        user_message,
                    )

                    refined_topics.extend(refined)

                # Use these refined topics in place of the original replacement topics
                replacement_topics = refined_topics

                # Create new research outline structure
                new_research_outline = []
                new_all_topics = []

                # Track the original hierarchy
                original_hierarchy = {}  # Store parent-child relationships
                original_main_topics = set()  # Track which items were main topics
                original_subtopics = set()  # Track which items were subtopics

                # Extract from the original outline structure
                for topic_item in outline_items:
                    topic = topic_item["topic"]
                    original_main_topics.add(topic)
                    subtopics = topic_item.get("subtopics", [])

                    # Track the hierarchy
                    for subtopic in subtopics:
                        original_hierarchy[subtopic] = topic
                        original_subtopics.add(subtopic)

                # Process kept items to maintain hierarchy
                for topic_item in outline_items:
                    topic = topic_item["topic"]
                    subtopics = topic_item.get("subtopics", [])

                    if topic in kept_items:
                        # Keep the original topic with its kept subtopics
                        kept_subtopics = [s for s in subtopics if s in kept_items]
                        if kept_subtopics:  # Only add if there are kept subtopics
                            new_topic_item = {
                                "topic": topic,
                                "subtopics": kept_subtopics,
                            }
                            new_research_outline.append(new_topic_item)
                            new_all_topics.append(topic)
                            new_all_topics.extend(kept_subtopics)
                        else:
                            # If main topic is kept but no subtopics, still add it
                            new_topic_item = {"topic": topic, "subtopics": []}
                            new_research_outline.append(new_topic_item)
                            new_all_topics.append(topic)
                    else:
                        # For removed main topics, check if any subtopics were kept
                        kept_subtopics = [s for s in subtopics if s in kept_items]
                        if kept_subtopics:
                            # Just restore the original main topic name teehee
                            revised_topic = f"{topic}"
                            new_topic_item = {
                                "topic": revised_topic,
                                "subtopics": kept_subtopics,
                            }
                            new_research_outline.append(new_topic_item)
                            new_all_topics.append(revised_topic)
                            new_all_topics.extend(kept_subtopics)

                # Process orphaned kept items (not already added)
                orphaned_kept_items = [
                    item for item in kept_items if item not in new_all_topics
                ]

                # Get embeddings for assignment
                if orphaned_kept_items and new_research_outline:
                    try:
                        # Try to add orphaned items to existing topics based on semantic similarity
                        main_topic_embeddings = {}
                        for outline_item in new_research_outline:
                            topic = outline_item["topic"]
                            embedding = await self.get_embedding(topic)
                            if embedding:
                                main_topic_embeddings[topic] = embedding

                        for item in orphaned_kept_items:
                            item_embedding = await self.get_embedding(item)
                            if item_embedding:
                                # Find best match
                                best_match = None
                                best_score = 0.5  # Threshold

                                for (
                                    topic,
                                    topic_embedding,
                                ) in main_topic_embeddings.items():
                                    similarity = cosine_similarity(
                                        [item_embedding], [topic_embedding]
                                    )[0][0]
                                    if similarity > best_score:
                                        best_score = similarity
                                        best_match = topic

                                if best_match:
                                    # Add to existing topic
                                    for outline_item in new_research_outline:
                                        if outline_item["topic"] == best_match:
                                            outline_item["subtopics"].append(item)
                                            new_all_topics.append(item)
                                            break
                                else:
                                    # If no good match, create a new topic from the item
                                    if item in original_main_topics:
                                        # It was a main topic, keep it that way
                                        new_research_outline.append(
                                            {"topic": item, "subtopics": []}
                                        )
                                        new_all_topics.append(item)
                                    else:
                                        # It was a subtopic, but now it's orphaned, make it a main topic
                                        new_research_outline.append(
                                            {"topic": item, "subtopics": []}
                                        )
                                        new_all_topics.append(item)
                            else:
                                # No embedding, add as a main topic
                                new_research_outline.append(
                                    {"topic": item, "subtopics": []}
                                )
                                new_all_topics.append(item)
                    except Exception as e:
                        logger.error(f"Error assigning orphaned items: {e}")
                        # Add all orphaned items as main topics on error
                        for item in orphaned_kept_items:
                            new_research_outline.append(
                                {"topic": item, "subtopics": []}
                            )
                            new_all_topics.append(item)
                elif orphaned_kept_items:
                    # No existing topics to add to, make each orphaned item a main topic
                    for item in orphaned_kept_items:
                        new_research_outline.append({"topic": item, "subtopics": []})
                        new_all_topics.append(item)

                # Add replacement topics now
                # First, try to add them to semantically similar existing main topics
                if replacement_topics and new_research_outline:
                    try:
                        # Get embeddings for existing main topics
                        main_topic_embeddings = {}
                        for outline_item in new_research_outline:
                            topic = outline_item["topic"]
                            embedding = await self.get_embedding(topic)
                            if embedding:
                                main_topic_embeddings[topic] = embedding

                        # Track which replacements have been assigned
                        assigned_replacements = set()

                        # Try to assign each replacement to a semantically similar main topic
                        for replacement in replacement_topics:
                            replacement_embedding = await self.get_embedding(
                                replacement
                            )
                            if replacement_embedding:
                                # Find best match
                                best_match = None
                                best_score = 0.65  # Higher threshold for replacements

                                for (
                                    topic,
                                    topic_embedding,
                                ) in main_topic_embeddings.items():
                                    similarity = cosine_similarity(
                                        [replacement_embedding], [topic_embedding]
                                    )[0][0]
                                    if similarity > best_score:
                                        best_score = similarity
                                        best_match = topic

                                if best_match:
                                    # Add to existing topic
                                    for outline_item in new_research_outline:
                                        if outline_item["topic"] == best_match:
                                            outline_item["subtopics"].append(
                                                replacement
                                            )
                                            new_all_topics.append(replacement)
                                            assigned_replacements.add(replacement)
                                            break

                        # Create new topics for unassigned replacements
                        unassigned_replacements = [
                            r
                            for r in replacement_topics
                            if r not in assigned_replacements
                        ]

                        # Group the unassigned replacements
                        replacement_groups = await self.group_replacement_topics(
                            unassigned_replacements
                        )

                        for group in replacement_groups:
                            # Generate title for the group
                            try:
                                group_title = await self.generate_group_title(
                                    group, user_message
                                )
                            except Exception as e:
                                logger.error(f"Error generating group title: {e}")
                                group_title = f"Additional Research Area {len(new_research_outline) - len(outline_items) + 1}"

                            # Add as a new main topic
                            new_research_outline.append(
                                {"topic": group_title, "subtopics": group}
                            )
                            new_all_topics.append(group_title)
                            new_all_topics.extend(group)

                    except Exception as e:
                        logger.error(f"Error during replacement topic assignment: {e}")
                        # Fallback: add all replacements as a new group
                        group_title = "Additional Research Topics"
                        new_research_outline.append(
                            {"topic": group_title, "subtopics": replacement_topics}
                        )
                        new_all_topics.append(group_title)
                        new_all_topics.extend(replacement_topics)
                elif replacement_topics:
                    # No existing outline to add to, create groups from replacements
                    replacement_groups = await self.group_replacement_topics(
                        replacement_topics
                    )

                    for i, group in enumerate(replacement_groups):
                        try:
                            group_title = await self.generate_group_title(
                                group, user_message
                            )
                        except Exception as e:
                            logger.error(f"Error generating group title: {e}")
                            group_title = f"Research Group {i+1}"

                        new_research_outline.append(
                            {"topic": group_title, "subtopics": group}
                        )
                        new_all_topics.append(group_title)
                        new_all_topics.extend(group)

                    # Update the research outline and topic list
                    if new_research_outline:  # Only update if we have valid content
                        research_outline = new_research_outline
                        all_topics = new_all_topics

                        # Update outline embedding based on all_topics
                        outline_text = " ".join(all_topics)
                        outline_embedding = await self.get_embedding(outline_text)

                        # Re-initialize dimension tracking with new topics
                        await self.initialize_research_dimensions(
                            all_topics, user_message
                        )

                        # Make sure to store initial coverage for later display
                        research_dimensions = state.get("research_dimensions")
                        if research_dimensions:
                            # Make a copy to avoid reference issues
                            self.update_state(
                                "latest_dimension_coverage",
                                research_dimensions["coverage"].copy(),
                            )
                            logger.info(
                                f"Updated dimension coverage after feedback with {len(research_dimensions['coverage'])} values"
                            )

                            # Also update trajectory accumulator for consistency
                            self.trajectory_accumulator = (
                                None  # Reset for fresh accumulation
                            )

                        # Show the updated outline to the user
                        updated_outline = "### Updated Research Outline\n\n"
                        for topic_item in research_outline:
                            updated_outline += f"**{topic_item['topic']}**\n"
                            for subtopic in topic_item.get("subtopics", []):
                                updated_outline += f"- {subtopic}\n"
                            updated_outline += "\n"

                        await self.emit_message(updated_outline)

                        # Updated message about continuing with main research
                        await self.emit_message(
                            "\n*Updated research outline with user preferences. Continuing to main research cycles...*\n\n"
                        )

                        # Store the updated research state
                        self.update_state(
                            "research_state",
                            {
                                "research_outline": research_outline,
                                "all_topics": all_topics,
                                "outline_embedding": outline_embedding,
                                "user_message": user_message,
                            },
                        )

                        return research_outline, all_topics, outline_embedding
                    else:
                        # If we couldn't create a valid outline, continue with original
                        await self.emit_message(
                            "\n*No valid outline could be created. Continuing with original outline.*\n\n"
                        )
                        self.update_state(
                            "research_state",
                            {
                                "research_outline": outline_items,
                                "all_topics": all_topics,
                                "outline_embedding": outline_embedding,
                                "user_message": user_message,
                            },
                        )

                        return outline_items, all_topics, outline_embedding
                else:
                    # No items were removed, continue with original outline
                    await self.emit_message(
                        "\n*No changes made to research outline. Continuing with original outline.*\n\n"
                    )
                    self.update_state(
                        "research_state",
                        {
                            "research_outline": outline_items,
                            "all_topics": all_topics,
                            "outline_embedding": outline_embedding,
                            "user_message": user_message,
                        },
                    )

                    return outline_items, all_topics, outline_embedding

    async def generate_group_title(self, topics: List[str], user_message: str) -> str:
        """Generate a descriptive title for a group of related topics"""
        if not topics:
            return ""

        # For very small groups, just combine the topics
        if len(topics) <= 2:
            combined = " & ".join(topics)
            if len(combined) > 80:
                return combined[:77] + "..."
            return combined

        # Create a prompt to generate the group title
        title_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant creating a concise descriptive title for a group of related research topics.
    Create a short, clear title (4-8 words) that captures the common theme across these topics.
    The title should be specific enough to distinguish this group from others, but general enough to encompass all topics.
    DO NOT use generic phrases like "Research Group" or "Topic Group".
    Respond with ONLY the title text.""",
        }

        # Create the message content with full topics
        topic_text = "\n- " + "\n- ".join(topics)

        message = {
            "role": "user",
            "content": f"""Create a concise title for this group of related research topics:
    {topic_text}
    
    These topics are part of research about: "{user_message}"
    
    Respond with ONLY the title (4-8 words).""",
        }

        # Generate the title
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [title_prompt, message],
                temperature=0.7,
            )

            title = response["choices"][0]["message"]["content"].strip()

            # Remove quotes if present
            title = title.strip("\"'")

            # Limit length if needed
            if len(title) > 80:
                title = title[:77] + "..."

            return title
        except Exception as e:
            logger.error(f"Error generating group title: {e}")
            # Single clean fallback that uses first topic
            return f"{topics[0][:40]}... & Related Topics"

    async def is_follow_up_query(self, messages: List[Dict]) -> bool:
        """Determine if the current query is a follow-up to a previous research session"""
        # If we have a previous comprehensive summary and research has been completed,
        # treat any new query as a follow-up
        state = self.get_state()
        prev_comprehensive_summary = state.get("prev_comprehensive_summary", "")
        research_completed = state.get("research_completed", False)

        # Check if we're waiting for outline feedback - if so, don't treat as new or follow-up
        waiting_for_outline_feedback = state.get("waiting_for_outline_feedback", False)
        if waiting_for_outline_feedback:
            return False

        # Check for fresh conversation by examining message count
        # A brand new conversation will have very few messages
        is_new_conversation = (
            len(messages) <= 2
        )  # Only 1-2 messages in a new conversation

        # If this appears to be a new conversation and we're not waiting for feedback,
        # don't treat as follow-up and reset state
        if is_new_conversation and not waiting_for_outline_feedback:
            # Reset the state for this conversation to ensure clean start
            self.reset_state()
            return False

        return bool(prev_comprehensive_summary and research_completed)

    async def generate_synthesis_outline(
        self,
        original_outline: List[Dict],
        completed_topics: Set[str],
        user_query: str,
        research_results: List[Dict],
    ) -> List[Dict]:
        """Generate a refined research outline for synthesis that better integrates additional research areas"""

        state = self.get_state()

        # Get the number of elapsed cycles
        elapsed_cycles = len(state.get("cycle_summaries", []))

        # Create a prompt for generating the synthesis outline
        synthesis_outline_prompt = {
            "role": "system",
            "content": f"""You are a post-graduate academic scholar reorganizing a research outline to be used in writing a comprehensive research report.
            
    Create a refined outline that condenses key topics/subtopics and insights from the current outline, and focuses on addressing the original query in areas best supported by the research.
    Aim to have approximately {round((elapsed_cycles * 0.25) + 2)} main topics and {round((elapsed_cycles * 0.8) + 5)} subtopics in your revised outline.

    The original user query was: "{user_query}".

    Your refined outline must:
    1. Appropriately incorporate relevant new topics discovered along the way that are directly relevant to the research "core" and original user query.
    2. Tailors the outline to reflect the progress and outcome of research activities without getting distracted by irrelevant results or specific examples, brands, locations, etc.
    3. Unite how research has evolved, and the reference material obtained during research, with the initial purpose and scope, prioritizing the initial purpose and scope.
    4. Where appropriate, reign in the representation of tangential research branches to refocus on topics more directly related to the original query.

    Your refined outline must NOT:
    1. Attempt to trump up, downplay, remove, soften, qualify, or otherwise modify the representation of research topics due to your own biases, preferences, or interests.
    2. Include main topics intended to serve as an introduction or conclusion for the full report.
    3. Focus on topics explored during research that don't actually serve to address the user's query or are fully tangent to it, or overly emphasize specific cases.
    4. Include any other text - please only respond with the outline. 
    
    The goal is to create a refined outline reflecting a logical narrative and informational flow for the final comprehensive report based on the user's query and gathered research.
    
    Format your response as a valid JSON object with the following structure:
    {{"outline": [
      {{"topic": "Main topic 1", "subtopics": ["Subtopic 1.1", "Subtopic 1.2"]}},
      {{"topic": "Main topic 2", "subtopics": ["Subtopic 2.1", "Subtopic 2.2"]}}
    ]}}""",
        }

        # Calculate similarity of research results to the research outline
        result_scores = []
        outline_text = "\n".join(
            [topic_item["topic"] for topic_item in original_outline]
        )

        # Check if we have a cached outline embedding
        state = self.get_state()
        outline_embedding_key = f"outline_embedding_{hash(outline_text)}"
        outline_embedding = state.get(outline_embedding_key)

        if not outline_embedding:
            outline_embedding = await self.get_embedding(outline_text)
            if outline_embedding:
                # Cache the outline embedding
                self.update_state(outline_embedding_key, outline_embedding)

        # Initialize outline_context
        outline_context = ""
        if outline_embedding:
            for i, result in enumerate(research_results):
                content = result.get("content", "")
                if not content:
                    continue

                # Check cache first for result embedding
                result_key = f"result_embedding_{hash(result.get('url', ''))}"
                content_embedding = state.get(result_key)

                if not content_embedding:
                    content_embedding = await self.get_embedding(content[:2000])
                    if content_embedding:
                        # Cache the result embedding
                        self.update_state(result_key, content_embedding)

                if content_embedding:
                    similarity = cosine_similarity(
                        [content_embedding], [outline_embedding]
                    )[0][0]
                    result_scores.append((i, similarity))

            # Sort results by similarity to outline in reverse order (most similar last)
            result_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_results = [research_results[i] for i, _ in result_scores]

            # Add sorted results to context
            outline_context += "\n### Research Results:\n\n"
            for result in sorted_results:
                outline_context += f"Title: {result.get('title', 'Untitled')}\n"
                outline_context += f"Content: {result.get('content', '')}\n\n"

        # Build context from the original outline and research results
        outline_context = "### Original Research Outline:\n\n"

        for topic_item in original_outline:
            outline_context += f"- {topic_item['topic']}\n"
            for subtopic in topic_item.get("subtopics", []):
                outline_context += f"  - {subtopic}\n"

        # Add semantic dimensions if available
        state = self.get_state()
        research_dimensions = state.get("research_dimensions")
        if research_dimensions:
            try:
                dimension_coverage = research_dimensions.get("coverage", [])

                # Create dimension labels for better context
                dimension_labels = await self.translate_dimensions_to_words(
                    research_dimensions, dimension_coverage
                )

                if dimension_coverage:
                    outline_context += "\n### Research Dimensions Coverage:\n"
                    for dim in dimension_labels[:10]:  # Limit to top 10 dimensions
                        outline_context += f"- {dim.get('words', 'Dimension ' + str(dim.get('dimension', 0)))}:  {dim.get('coverage', 0)}% covered\n"

            except Exception as e:
                logger.error(
                    f"Error adding research dimensions to outline context: {e}"
                )

        # Create messages for the model
        messages = [
            synthesis_outline_prompt,
            {
                "role": "user",
                "content": f"{outline_context}\n\nGenerate a refined research outline following the instructions and format in the system prompt.",
            },
        ]

        # Generate the synthesis outline
        try:
            await self.emit_status(
                "info", "Generating refined outline for synthesis...", False
            )

            # Use synthesis model for this task
            synthesis_model = self.get_synthesis_model()
            response = await self.generate_completion(
                synthesis_model, messages, temperature=self.valves.SYNTHESIS_TEMPERATURE
            )
            outline_content = response["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                # First try standard JSON extraction
                json_start = outline_content.find("{")
                json_end = outline_content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    outline_json_str = outline_content[json_start:json_end]
                    try:
                        outline_data = json.loads(outline_json_str)
                        synthesis_outline = outline_data.get("outline", [])
                        if synthesis_outline:
                            return synthesis_outline
                    except (json.JSONDecodeError, ValueError):
                        # If standard approach fails, try regex approach
                        pass

                # Use regex to find any JSON structure containing "outline" array
                import re

                json_pattern = r'(\{[^{}]*"outline"\s*:\s*\[[^\[\]]*\][^{}]*\})'
                matches = re.findall(json_pattern, outline_content, re.DOTALL)

                for match in matches:
                    try:
                        outline_data = json.loads(match)
                        synthesis_outline = outline_data.get("outline", [])
                        if synthesis_outline:
                            return synthesis_outline
                    except:
                        continue

                # If no valid JSON found, try a more aggressive repair approach
                # Look for anything that resembles the outline structure
                topic_pattern = (
                    r'"topic"\s*:\s*"([^"]*)"\s*,\s*"subtopics"\s*:\s*\[(.*?)\]'
                )
                topics_matches = re.findall(topic_pattern, outline_content, re.DOTALL)

                if topics_matches:
                    synthetic_outline = []
                    for topic_match in topics_matches:
                        topic = topic_match[0]
                        subtopics_str = topic_match[1]
                        # Extract subtopics strings - look for quoted strings
                        subtopics = re.findall(r'"([^"]*)"', subtopics_str)
                        synthetic_outline.append(
                            {"topic": topic, "subtopics": subtopics}
                        )

                    if synthetic_outline:
                        return synthetic_outline

                # All extraction methods failed, return original outline
                return original_outline

            except Exception as e:
                logger.error(f"Error parsing synthesis outline JSON: {e}")
                return original_outline

        except Exception as e:
            logger.error(f"Error generating synthesis outline: {e}")
            return original_outline

    async def generate_subtopic_content_with_citations(
        self,
        section_title: str,
        subtopic: str,
        original_query: str,
        research_results: List[Dict],
        synthesis_model: str,
        is_follow_up: bool = False,
        previous_summary: str = "",
    ) -> Dict:
        """Generate content for a single subtopic with numbered citations"""
        # Only emit status if we haven't seen this section yet
        if not hasattr(self, "_seen_subtopics"):
            self._seen_subtopics = set()

        # Only emit status if we haven't seen this subtopic yet
        if subtopic not in self._seen_subtopics:
            await self.emit_status(
                "info", f"Generating content for subtopic: {subtopic}...", False
            )
            self._seen_subtopics.add(subtopic)

        # Get state
        state = self.get_state()

        # Get relevance cache or initialize it
        relevance_cache = state.get("subtopic_relevance_cache", {})

        # Create embedding cache keys for efficiency
        query_embedding_key = f"query_embedding_{hash(original_query)}"
        subtopic_embedding_key = f"subtopic_embedding_{hash(subtopic)}"
        combined_embedding_key = (
            f"combined_embedding_{hash(original_query)}_{hash(subtopic)}"
        )

        # Create a prompt specific to this subtopic
        subtopic_prompt = {
            "role": "system",
            "content": f"""You are a post-grad research assistant writing a concise subsection (1-3 paragraphs) about "{subtopic}" 
        for a comprehensive combined research report addressing this query: "{original_query}" based on internet research results.
        
        Your subsection MUST:
        1. Focus specifically on the subtopic "{subtopic}" within the broader section "{section_title}".
        2. Make FULL use of the provided research sources, and ONLY the provided sources.
        3. Include IN-TEXT CITATIONS for all information from sources, using ONLY the numerical IDs provided in the source list, e.g. [1], [4], etc.
        4. Follow a structure that best fits the subtopic subject matter. Aim for an academic report style while tolerating flexibility as appropriate.
        5. Only be written on the subtopic matter - consider how your subsection will be combined with others in a greater research report.
        6. Be written to a length, between 1 medium paragraph and 3 long paragraphs, based on the subtopic's perceived importance to the research query.
        
        Your subsection must NOT:
        1. Interpret the content in a lofty way that exaggerates its importance or profundity, or contrives a narrative with empty sophistication. 
        2. Attempt to portray the subject matter in any particular sort of light, good or bad, especially by using apologetic or dismissive language.
        3. Focus on perceived complexities or challenges related to the topic or research process, or include appeals to future research.
        4. Ever take a preachy or moralizing tone, or take a "stance" for or against/"side" with or against anything not driven by the provided data.
        5. Overstate the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
        6. Sound to the reader as though it is overtly attempting to be diplomatic, considerate, enthusiastic, or overly-generalized.
    
        You must accurately cite your sources to avoid plagiarizing. Citations MUST be numerical and correspond to the correct source ID in the provided list.
        Do not combine multiple IDs in one citation tag. Please respond with just the subsection body, no intro or title.""",
        }

        # Create a combined embedding for query + subtopic with state-level caching
        combined_embedding = state.get(combined_embedding_key)

        if combined_embedding is None:
            # Check if we already have the individual embeddings cached in state
            query_embedding = state.get(query_embedding_key)
            subtopic_embedding = state.get(subtopic_embedding_key)

            # Get query embedding if not already cached
            if query_embedding is None:
                try:
                    query_embedding = await self.get_embedding(original_query)
                    if query_embedding:
                        # Cache at state level to avoid repeated API calls
                        self.update_state(query_embedding_key, query_embedding)
                except Exception as e:
                    logger.error(f"Error getting query embedding: {e}")

            # Get subtopic embedding if not already cached
            if subtopic_embedding is None:
                try:
                    subtopic_embedding = await self.get_embedding(subtopic)
                    if subtopic_embedding:
                        # Cache at state level to avoid repeated API calls
                        self.update_state(subtopic_embedding_key, subtopic_embedding)
                except Exception as e:
                    logger.error(f"Error getting subtopic embedding: {e}")

            # Create combined embedding if both components exist
            if query_embedding and subtopic_embedding:
                try:
                    # Combine with equal weight
                    combined_array = (
                        np.array(query_embedding) * 0.5
                        + np.array(subtopic_embedding) * 0.5
                    )
                    # Normalize
                    norm = np.linalg.norm(combined_array)
                    if norm > 1e-10:
                        combined_array = combined_array / norm
                    combined_embedding = combined_array.tolist()
                    # Cache the combined embedding
                    self.update_state(combined_embedding_key, combined_embedding)
                except Exception as e:
                    logger.error(f"Error creating combined embedding: {e}")

        # If combined embedding failed or doesn't exist, fall back to subtopic embedding
        if not combined_embedding:
            # Check if we have cached subtopic embedding
            subtopic_embedding = state.get(subtopic_embedding_key)
            if not subtopic_embedding:
                # Try to get it now
                subtopic_embedding = await self.get_embedding(subtopic)
                # Cache if successful
                if subtopic_embedding:
                    self.update_state(subtopic_embedding_key, subtopic_embedding)

            combined_embedding = subtopic_embedding

        # Build context from research results that might be relevant to this subtopic
        subtopic_context = f"# Subtopic to Write: {subtopic}\n"
        subtopic_context += f"# Within Section: {section_title}\n\n"

        # Add the research outline for context
        subtopic_context += "## Research Outline Context:\n"
        state = self.get_state()
        synthesis_outline = state.get("research_state", {}).get("research_outline", [])
        if synthesis_outline:
            for topic_item in synthesis_outline:
                topic = topic_item.get("topic", "")
                if topic == section_title:
                    subtopic_context += f"**Current Section: {topic}**\n"
                else:
                    subtopic_context += f"Section: {topic}\n"

                for st in topic_item.get("subtopics", []):
                    if st == subtopic:
                        subtopic_context += f"  - **Current Subtopic: {st}**\n"
                    else:
                        subtopic_context += f"  - {st}\n"
            subtopic_context += "\n"

        # Create a unique cache key for this subtopic
        subtopic_key = f"{section_title}_{subtopic}"

        # Calculate relevance scores for each result
        subtopic_results = []
        result_scores = []

        # Check if we have cached relevance scores for this subtopic
        if subtopic_key in relevance_cache:
            logger.info(f"Using cached relevance scores for subtopic: {subtopic}")
            result_scores = relevance_cache[subtopic_key]
            # Sort by relevance score (highest first)
            result_scores.sort(key=lambda x: x[1], reverse=True)
            # Map back to research results
            subtopic_results = [
                research_results[i]
                for i, _ in result_scores
                if i < len(research_results)
            ]
        elif combined_embedding:
            # Calculate relevance scores using combined query+subtopic embedding
            for i, result in enumerate(research_results):
                content = result.get("content", "")
                if not content:
                    continue

                # Create a cache key for this result's embedding
                result_key = f"result_{hash(result.get('url', ''))}"
                content_embedding = state.get(result_key)

                if not content_embedding:
                    content_embedding = await self.get_embedding(content[:2000])
                    # Cache the content embedding if valid
                    if content_embedding:
                        self.update_state(result_key, content_embedding)

                if content_embedding:
                    similarity = cosine_similarity(
                        [content_embedding], [combined_embedding]
                    )[0][0]
                    result_scores.append((i, similarity))

            # Cache the relevance scores for this subtopic
            relevance_cache[subtopic_key] = result_scores
            self.update_state("subtopic_relevance_cache", relevance_cache)

            # Sort by relevance score (highest first)
            result_scores.sort(key=lambda x: x[1], reverse=True)
            # Map to research results
            subtopic_results = [research_results[i] for i, _ in result_scores]
        else:
            # If no embedding, just use all results
            subtopic_results = research_results

        # Calculate how many results to include based on number of cycles and vibes
        top_results_count = max(
            3, min(len(subtopic_results), math.ceil(0.5 * self.valves.MAX_CYCLES + 3))
        )
        top_results = subtopic_results[:top_results_count]

        # Create source list with assigned IDs
        sources_for_subtopic = {}
        source_id = 1

        # Extract URLs and titles from top results, sort alphabetically by title
        sorted_results = sorted(top_results, key=lambda x: x.get("title", "").lower())

        for result in sorted_results:
            url = result.get("url", "")
            title = result.get("title", "Untitled Source")

            if url and url not in sources_for_subtopic:
                sources_for_subtopic[url] = {
                    "id": source_id,
                    "title": title,
                    "url": url,
                    "subtopic": subtopic,
                    "section": section_title,
                }
                source_id += 1

        # Add source list to context (at the beginning)
        subtopic_context += (
            "## Available Source List (Use ONLY these numerical citations):\n\n"
        )
        for url, source_data in sorted(
            sources_for_subtopic.items(), key=lambda x: x[1]["title"]
        ):
            subtopic_context += (
                f"[{source_data['id']}] {source_data['title']} - {url}\n"
            )

        subtopic_context += "\n## Research Results:\n\n"

        # Reorder results to have most relevant last (most recent in context)
        top_results.reverse()

        # Add the top results to context (most relevant last)
        for result in top_results:
            url = result.get("url", "")
            title = result.get("title", "Untitled Source")
            content = result.get("content", "")

            # Skip results without content
            if not content:
                continue

            # Get the source ID for this URL
            source_id = sources_for_subtopic.get(url, {}).get("id", "?")

            subtopic_context += f"Source ID: [{source_id}] {title}\n"
            subtopic_context += f"Content: {content}\n\n"

        # Include previous summary if this is a follow-up
        if is_follow_up and previous_summary:
            subtopic_context += "## Previous Research Summary:\n\n"
            subtopic_context += f"{previous_summary}...\n\n"

        # Prepare final instruction
        subtopic_context += f"""Using the provided research sources and referencing them with numerical citations [#], write a concise subsection about "{subtopic}" per the system prompt."""
        subtopic_context += f"""Every citation MUST be numerical (e.g., [1], [2]) corresponding to the source list provided."""
        subtopic_context += f"""Please use proper Markdown and write 1-3 focused paragraphs exclusively on this specific subtopic."""

        # Create messages array for completion
        messages = [subtopic_prompt, {"role": "user", "content": subtopic_context}]

        # Generate subtopic content
        try:
            # Calculate scaled temperature from the synthesis temperature valve
            scaled_temperature = (
                self.valves.TEMPERATURE
            )  # Use research model temperature for subtopics

            # Use research model for generating subtopics
            response = await self.generate_completion(
                synthesis_model,
                messages,
                stream=False,
                temperature=scaled_temperature,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                subtopic_content = response["choices"][0]["message"]["content"]

                # Count tokens in the subtopic content
                tokens = await self.count_tokens(subtopic_content)

                # Store content for later use
                subtopic_synthesized_content = state.get(
                    "subtopic_synthesized_content", {}
                )
                subtopic_synthesized_content[subtopic] = subtopic_content
                self.update_state(
                    "subtopic_synthesized_content", subtopic_synthesized_content
                )

                # Store source mapping for this subtopic
                subtopic_sources = state.get("subtopic_sources", {})
                subtopic_sources[subtopic] = sources_for_subtopic
                self.update_state("subtopic_sources", subtopic_sources)

                # Identify citations in this subtopic content
                subtopic_citations = []
                for url, source_data in sources_for_subtopic.items():
                    local_id = source_data.get("id")
                    if local_id is not None:
                        # Find all instances of this citation in the text
                        pattern = (
                            r"([^.!?]*(?:\["
                            + str(local_id)
                            + r"\]|&#"
                            + str(local_id)
                            + r")[^.!?]*[.!?])"
                        )
                        context_matches = re.findall(pattern, subtopic_content)

                        for match in context_matches:
                            citation = {
                                "marker": str(local_id),
                                "raw_text": f"[{local_id}]",
                                "text": match,
                                "url": url,
                                "section": section_title,
                                "subtopic": subtopic,
                                "suggested_title": source_data.get("title", ""),
                            }
                            subtopic_citations.append(citation)

                # Log the sources used
                logger.info(
                    f"Subtopic '{subtopic}' uses {len(sources_for_subtopic)} sources"
                )

                return {
                    "content": subtopic_content,  # Return original content with local IDs
                    "tokens": tokens,
                    "sources": sources_for_subtopic,
                    "citations": subtopic_citations,
                    "verified_citations": [],  # Verification happens later
                    "flagged_citations": [],  # Flagging happens later
                }
            else:
                return {
                    "content": f"*Error generating content for subtopic: {subtopic}*",
                    "tokens": 0,
                    "sources": {},
                    "citations": [],
                    "verified_citations": [],
                    "flagged_citations": [],
                }

        except Exception as e:
            logger.error(f"Error generating subtopic content for '{subtopic}': {e}")
            return {
                "content": f"*Error generating content for subtopic: {subtopic}*",
                "tokens": 0,
                "sources": {},
                "citations": [],
                "verified_citations": [],
                "flagged_citations": [],
            }

    async def generate_section_content_with_citations(
        self,
        section_title: str,
        subtopics: List[str],
        original_query: str,
        research_results: List[Dict],
        synthesis_model: str,
        is_follow_up: bool = False,
        previous_summary: str = "",
    ) -> Dict:
        """Generate content for a section by combining subtopics with citations"""
        # Use a static set to track which sections we've displayed status for
        if not hasattr(self, "_seen_sections"):
            self._seen_sections = set()

        # Only emit status if we haven't seen this section yet
        if section_title not in self._seen_sections:
            await self.emit_status(
                "info", f"Generating content for section: {section_title}...", False
            )
            self._seen_sections.add(section_title)

        # Get state
        state = self.get_state()

        # Generate content for each subtopic independently
        subtopic_contents = {}
        section_sources = {}
        all_section_citations = []
        total_tokens = 0

        for subtopic in subtopics:
            subtopic_result = await self.generate_subtopic_content_with_citations(
                section_title,
                subtopic,
                original_query,
                research_results,
                synthesis_model,
                is_follow_up,
                previous_summary if is_follow_up else "",
            )

            subtopic_contents[subtopic] = subtopic_result["content"]
            total_tokens += subtopic_result.get("tokens", 0)

            # Collect all citations from this subtopic
            if "citations" in subtopic_result:
                all_section_citations.extend(subtopic_result["citations"])

            # Merge sources with section sources, maintaining unique source IDs and tracking originals
            for url, source_data in subtopic_result["sources"].items():
                if url not in section_sources:
                    section_sources[url] = (
                        source_data.copy()
                    )  # Use copy to avoid reference issues

                # Store the original local ID with subtopic context for precise replacement
                # Add this information to the source data record
                if "original_ids" not in section_sources[url]:
                    section_sources[url]["original_ids"] = {}

                # Track which local ID was used in which subtopic
                local_id = source_data.get("id")
                if local_id is not None:
                    section_sources[url]["original_ids"][subtopic] = local_id

        # Build or update the global citation map with sources from this section
        master_source_table = state.get("master_source_table", {})
        global_citation_map = state.get("global_citation_map", {})

        # Add all section sources to global map if not already present
        for url, source_data in section_sources.items():
            if url not in global_citation_map:
                global_citation_map[url] = len(global_citation_map) + 1

            # Also add to master source table if not already there
            if url not in master_source_table:
                source_id = f"S{len(master_source_table) + 1}"
                master_source_table[url] = {
                    "id": source_id,
                    "title": source_data.get("title", "Untitled Source"),
                    "content_preview": "",
                    "source_type": "web" if not url.endswith(".pdf") else "pdf",
                    "accessed_date": self.research_date,
                    "cited_in_sections": set([section_title]),
                }
            elif section_title not in master_source_table[url].get(
                "cited_in_sections", set()
            ):
                # Update sections where this source is cited
                master_source_table[url]["cited_in_sections"].add(section_title)

        # Update state
        self.update_state("global_citation_map", global_citation_map)
        self.update_state("master_source_table", master_source_table)

        # Verify citations if enabled
        verified_citations = []
        flagged_citations = []

        if self.valves.VERIFY_CITATIONS and all_section_citations:
            # Group citations by URL for efficient verification
            citations_by_url = {}
            for citation in all_section_citations:
                url = citation.get("url")
                if url:
                    if url not in citations_by_url:
                        citations_by_url[url] = []
                    citations_by_url[url].append(citation)

            # Verify each URL's citations
            for url, citations in citations_by_url.items():
                try:
                    # Get source content
                    url_results_cache = state.get("url_results_cache", {})

                    # Check cache first
                    source_content = None
                    if url in url_results_cache:
                        source_content = url_results_cache[url]

                    # If not in cache, fetch source content
                    if not source_content or len(source_content) < 200:
                        source_content = await self.fetch_content(url)

                    if source_content and len(source_content) >= 200:
                        # Add global ID to each citation for verification tracking
                        if url in global_citation_map:
                            global_id = global_citation_map[url]
                            for citation in citations:
                                citation["global_id"] = global_id

                        # Verify citations against source content
                        verification_results = await self.verify_citation_batch(
                            url, citations, source_content
                        )

                        # Sort verified and flagged citations
                        for result in verification_results:
                            if result.get("verified", False):
                                verified_citations.append(result)
                            elif result.get("flagged", False):
                                flagged_citations.append(result)
                    else:
                        # Mark as unverified but not flagged
                        for citation in citations:
                            citation["verified"] = False
                            citation["flagged"] = False
                except Exception as e:
                    logger.error(f"Error verifying citations for URL {url}: {e}")
                    # Mark as unverified but not flagged
                    for citation in citations:
                        citation["verified"] = False
                        citation["flagged"] = False

        # Now process each subtopic content to:
        # 1. Apply strikethrough to flagged citations
        # 2. Replace local citation IDs with global IDs
        processed_subtopic_contents = {}

        for subtopic, content in subtopic_contents.items():
            processed_content = content

            # Apply strikethrough to flagged citations
            flagged_sentences_for_subtopic = set()
            for citation in flagged_citations:
                if citation.get("subtopic") == subtopic and citation.get("text"):
                    flagged_sentences_for_subtopic.add(citation.get("text"))

            if flagged_sentences_for_subtopic:
                # Split content into sentences
                sentences = re.split(r"(?<=[.!?])\s+", processed_content)
                modified_sentences = []

                for sentence in sentences:
                    modified_sentence = sentence

                    # Check if this is a flagged sentence
                    for flagged_text in flagged_sentences_for_subtopic:
                        if flagged_text in sentence:
                            # Apply strikethrough
                            modified_sentence = f"~~{modified_sentence}~~"
                            break

                    modified_sentences.append(modified_sentence)

                # Join sentences back together
                processed_content = " ".join(modified_sentences)

                # Track applied strikethroughs
                citation_fixes = state.get("citation_fixes", [])
                for flagged_text in flagged_sentences_for_subtopic:
                    citation_fixes.append(
                        {
                            "section": section_title,
                            "subtopic": subtopic,
                            "reason": "Citation could not be verified",
                            "original_text": flagged_text,
                        }
                    )
                self.update_state("citation_fixes", citation_fixes)

            # Now replace all local citation IDs with global IDs using context-aware replacement
            # First handle single citations - standard pattern [n]
            for url, source_data in section_sources.items():
                # Check if this URL has a local ID for this specific subtopic
                original_ids = source_data.get("original_ids", {})
                local_id = original_ids.get(subtopic)

                if local_id is not None and url in global_citation_map:
                    global_id = global_citation_map[url]

                    # Replace local citation ID with global ID
                    pattern = r"\[" + re.escape(str(local_id)) + r"\]"
                    processed_content = re.sub(
                        pattern, f"[{global_id}]", processed_content
                    )

            # Now handle combined citations like [1, 2] or [1,2]
            # First, extract all citation groups from content
            combined_citation_pattern = r"\[(\d+(?:\s*,\s*\d+)+)\]"
            combined_matches = re.finditer(combined_citation_pattern, processed_content)

            # Process each combined citation group
            for match in combined_matches:
                original_citation = match.group(
                    0
                )  # The full citation group e.g. "[1, 2, 3]"
                citation_ids = match.group(1)  # Just the IDs part e.g. "1, 2, 3"

                # Extract individual IDs (handles both [1,2] and [1, 2] formats)
                local_ids = [
                    int(id_str.strip()) for id_str in re.split(r"\s*,\s*", citation_ids)
                ]

                # Convert each local ID to its global ID
                global_ids = []
                for local_id in local_ids:
                    # Find the URL(s) that had this local ID in this subtopic
                    for url, source_data in section_sources.items():
                        original_ids = source_data.get("original_ids", {})
                        if (
                            subtopic in original_ids
                            and original_ids[subtopic] == local_id
                        ):
                            if url in global_citation_map:
                                global_ids.append(str(global_citation_map[url]))

                # If we found global IDs, create the replacement citation
                if global_ids:
                    global_citation = f"[{', '.join(global_ids)}]"
                    # Replace just this specific citation instance
                    processed_content = processed_content.replace(
                        original_citation, global_citation, 1
                    )

            processed_subtopic_contents[subtopic] = processed_content

        # Combine subtopic contents into a section draft
        combined_content = ""
        for subtopic, content in processed_subtopic_contents.items():
            # Add subtopic heading
            combined_content += f"\n\n### {subtopic}\n\n"
            combined_content += f"{content}\n\n"

        # Only do smoothing if we have multiple subtopics
        if len(subtopics) > 1:
            # Review and smooth transitions between subtopics
            section_content = await self.smooth_section_transitions(
                section_title,
                subtopics,
                combined_content,
                original_query,
                synthesis_model,
            )
        else:
            section_content = combined_content

        # Track token counts
        memory_stats = state.get("memory_stats", {})
        section_tokens = memory_stats.get("section_tokens", {})
        section_tokens[section_title] = total_tokens
        memory_stats["section_tokens"] = section_tokens
        self.update_state("memory_stats", memory_stats)

        # Store content for later use
        section_synthesized_content = state.get("section_synthesized_content", {})
        section_synthesized_content[section_title] = section_content
        self.update_state("section_synthesized_content", section_synthesized_content)

        # Store section sources for later citation correlation
        section_sources_map = state.get("section_sources_map", {})
        section_sources_map[section_title] = section_sources
        self.update_state("section_sources_map", section_sources_map)

        # Store all citations for this section
        section_citations = state.get("section_citations", {})
        section_citations[section_title] = all_section_citations
        self.update_state("section_citations", section_citations)

        # Show section completion status
        await self.emit_status(
            "info",
            f"Section generated: {section_title}",
            False,
        )

        return {
            "content": section_content,
            "tokens": total_tokens,
            "sources": section_sources,
            "citations": all_section_citations,
            "verified_citations": verified_citations,
            "flagged_citations": flagged_citations,
        }

    async def smooth_section_transitions(
        self,
        section_title: str,
        subtopics: List[str],
        combined_content: str,
        original_query: str,
        synthesis_model: str,
    ) -> str:
        """Review and smooth transitions between subtopics in a section"""

        # Create a prompt for smoothing transitions
        smoothing_prompt = {
            "role": "system",
            "content": f"""You are a post-grad research editor editing a section that combines multiple subtopics.
            
    Review the section content and improve it by:
    1. Restructuring subtopic content and makeup to better fit the greater context of the section and full report
    2. Ensuring consistent style and tone throughout the section and ensuring consistent use of proper Markdown
    3. Maintaining the exact factual content in sentences with numerical citations [#]
    4. Removing duplicate subtopic headings
    5. Moving sentences or concepts between subsections as appropriate and revising subsection headers to fit the content
    6. Removing any meta-commentary, e.g. "Okay, here's the section" or "I wrote the section while considering..."
    7. Making the section read as though it were written by one person with a cohesive strategy for assembling the section
    
    DO NOT:
    1. Remove, change, or edit ANY in-text citations or applied strikethrough
    2. Alter, censor, re-analyze, or edit the factual content in ANY way
    3. Add new information or qualifiers not present in the original
    4. Decouple the factual content of a sentence from its specific citation
    5. Include any introduction, conclusion, main title header, or meta-commentary - please return the section as requested with no other text
    6. Combine sentences containing in-text citations and/or strikethrough

    It is vitally important that your edits preserve the direct connection between any sentence and its in-text citation and/or applied strikethrough.
    You may relocate or lightly edit sentences with in-text citations or strikethrough if appropriate, as long as they maintain these features.""",
        }

        # Create context with the combined subtopics
        smoothing_context = f"# Section to Improve: '{section_title}'\n\n"
        smoothing_context += (
            f"This section is part of a research paper on: '{original_query}'\n\n"
        )

        # Add the research outline for better context
        state = self.get_state()
        research_outline = state.get("research_state", {}).get("research_outline", [])
        if research_outline:
            smoothing_context += f"## Full Research Outline:\n"
            for topic_item in research_outline:
                topic = topic_item.get("topic", "")
                if topic == section_title:
                    smoothing_context += f"**Current Section: {topic}**\n"
                else:
                    smoothing_context += f"Section: {topic}\n"

                for st in topic_item.get("subtopics", []):
                    smoothing_context += f"  - {st}\n"
            smoothing_context += "\n"

        smoothing_context += f"## Subtopics in this section:\n"
        for subtopic in subtopics:
            smoothing_context += f"- '{subtopic}'\n"

        smoothing_context += f"\n## Combined Section Content:\n\n{combined_content}\n\n"
        smoothing_context += f"Please improve this section by ensuring smooth transitions between subtopics while preserving all factual content and numerical citations."

        # Create messages for completion
        messages = [smoothing_prompt, {"role": "user", "content": smoothing_context}]

        try:
            # Use synthesis model for smoothing
            response = await self.generate_completion(
                synthesis_model,
                messages,
                stream=False,
                temperature=self.valves.SYNTHESIS_TEMPERATURE
                * 0.7,  # Lower temperature for editing
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                improved_content = response["choices"][0]["message"]["content"]
                return improved_content
            else:
                # Return original if synthesis fails
                return combined_content

        except Exception as e:
            logger.error(
                f"Error smoothing transitions for section '{section_title}': {e}"
            )
            # Return original content on error
            return combined_content

    async def generate_bibliography(self, master_source_table, global_citation_map):
        """Generate a bibliography using sequential numbering based on actual citations in the report"""
        if not master_source_table:
            return {
                "bibliography": [],
                "title_to_global_id": {},
                "url_to_global_id": {},
            }

        # First, scan all compiled sections to find actually cited sources
        state = self.get_state()
        compiled_sections = state.get("section_synthesized_content", {})

        # Extract all citation numbers from the compiled text
        cited_numbers = set()
        for section_content in compiled_sections.values():
            # Find all citations in format [n] where n is a number
            citation_matches = re.findall(r"\[(\d+)\]", section_content)
            for num in citation_matches:
                cited_numbers.add(int(num))

        # Filter global_citation_map to only include cited sources
        cited_urls = {}
        for url, id_num in global_citation_map.items():
            if id_num in cited_numbers:
                cited_urls[url] = id_num

        # Sort URLs by their assigned citation ID
        sorted_urls = sorted(cited_urls.items(), key=lambda x: x[1])

        # Create bibliography entries based on cited sources only
        bibliography = []
        url_to_global_id = {}
        title_to_global_id = {}

        # Use the sequential numbers already assigned in global_citation_map
        for url, global_id in sorted_urls:
            # Get source data from master_source_table if available
            if url in master_source_table:
                source_data = master_source_table[url]
                title = source_data.get("title", "Untitled Source")
            else:
                logger.warning(
                    f"URL {url} in global_citation_map not found in master_source_table"
                )
                title = f"Source {global_id}"

            # Add bibliography entry using the actual correlated URL
            bibliography.append(
                {
                    "id": global_id,
                    "title": title,
                    "url": url,
                }
            )

            # Create mappings
            url_to_global_id[url] = global_id
            title_to_global_id[title] = global_id

        # Sort bibliography by citation ID
        bibliography.sort(key=lambda x: x["id"])

        logger.info(
            f"Generated bibliography with {len(bibliography)} cited entries (from {len(global_citation_map)} total sources)"
        )
        return {
            "bibliography": bibliography,
            "title_to_global_id": title_to_global_id,
            "url_to_global_id": url_to_global_id,
        }

    async def format_bibliography_list(self, bibliography):
        """Format the bibliography as a numbered list"""
        if not bibliography:
            return "No sources were referenced in this research."

        # Create numbered list format
        bib_list = "\n\n## Bibliography\n\n"

        # Add each bibliography entry
        for entry in bibliography:
            citation_id = entry["id"]
            title = entry["title"]
            url = entry["url"]

            # Format URL for markdown linking
            if url.startswith("http"):
                url_formatted = f"[{url}]({url})"
            else:
                url_formatted = url

            bib_list += f"[{citation_id}] {title}. {url_formatted}\n\n"

        return bib_list

    async def verify_citation_batch(self, url, citations, source_content):
        """Verify a batch of citations from a single source with improved sentence context isolation"""
        try:
            # Create a verification prompt
            verify_prompt = {
                "role": "system",
                "content": f"""You are a post-grad research assistant verifying the accuracy of citations and cited sentences against source material.
                
            Examine the source content and verify accuracy of each snippet. A citation is considered verified if the source includes the cited information.
            
            It is imperative you actually confirm accuracy/applicability or lack of such for each citation via direct comparison to source - never try to rely on your own knowledge.
            
            Return your results as a JSON array with this format:
            [
              {{
                "verified": true,
                "global_id": "citation_id"
              }},
              {{
                "verified": false,
                "global_id": "citation_id"
              }}
            ]""",
            }

            # Create verification context with all citations from this source
            verify_context = (
                f"Source URL: {url}\n\nSource content excerpt:\n{source_content}...\n\n"
            )
            verify_context += "Citation contexts to verify:\n"

            for i, citation in enumerate(citations):
                text = citation.get("text", "")
                global_id = citation.get("global_id", "unknown")
                if text:
                    verify_context += f'{i+1}. "{text}" [Global ID: {global_id}]\n'

            verify_context += "\nVerify each citation context against the source content. Provide a JSON array with verification results."

            # Generate verification assessment using the research model
            response = await self.generate_completion(
                self.get_research_model(),
                [verify_prompt, {"role": "user", "content": verify_context}],
                temperature=self.valves.TEMPERATURE
                * 0.2,  # 20% of normal temperature for precise verification
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                result_content = response["choices"][0]["message"]["content"]

                # Extract JSON array from the response
                try:
                    # Find array pattern [...]
                    array_match = re.search(r"\[(.*?)\]", result_content, re.DOTALL)
                    if array_match:
                        json_array = f"[{array_match.group(1)}]"
                        verification_results = json.loads(json_array)

                        # Add additional information to each result
                        final_results = []
                        for i, result in enumerate(verification_results):
                            if i < len(citations):
                                citation = citations[i]
                                final_result = {
                                    "url": url,
                                    "verified": result.get("verified", False),
                                    "flagged": not result.get("verified", False),
                                    "citation_text": citation.get("text", ""),
                                    "section": citation.get("section", ""),
                                    "global_id": citation.get("global_id"),
                                }
                                final_results.append(final_result)

                        return final_results
                    else:
                        # Try to parse as individual JSON objects
                        json_objects = re.findall(r"{.*?}", result_content, re.DOTALL)
                        if json_objects:
                            final_results = []
                            for i, json_str in enumerate(json_objects):
                                try:
                                    result = json.loads(json_str)
                                    if i < len(citations):
                                        citation = citations[i]
                                        final_result = {
                                            "url": url,
                                            "verified": result.get("verified", False),
                                            "flagged": not result.get(
                                                "verified", False
                                            ),
                                            "citation_text": citation.get("text", ""),
                                            "section": citation.get("section", ""),
                                            "global_id": citation.get("global_id"),
                                        }
                                        final_results.append(final_result)
                                except:
                                    continue
                            return final_results

                except Exception as e:
                    logger.error(f"Error parsing verification results: {e}")

            # Fallback for failures - assume all unverified
            return [
                {
                    "url": url,
                    "verified": False,
                    "flagged": False,
                    "citation_text": citation.get("text", ""),
                    "section": citation.get("section", ""),
                    "global_id": citation.get("global_id"),
                }
                for citation in citations
            ]

        except Exception as e:
            logger.error(f"Error verifying batch of citations: {e}")
            return []

    async def verify_citations(
        self, global_citation_map, citations_by_section, master_source_table
    ):
        """Verify citations in smaller batches"""
        if not self.valves.VERIFY_CITATIONS:
            await self.emit_status("info", "Citation verification is disabled", False)
            return {"verified": [], "flagged": []}

        # Count citations to verify
        total_citations = sum(
            len(section_citations)
            for section_citations in citations_by_section.values()
        )
        if total_citations == 0:
            await self.emit_status("info", "No citations to verify", False)
            return {"verified": [], "flagged": []}

        # Group citations by source URL for efficient verification
        citations_by_source = {}
        for section, section_citations in citations_by_section.items():
            for citation in section_citations:
                url = citation.get("url")
                if not url:
                    continue

                if url not in citations_by_source:
                    citations_by_source[url] = []

                citations_by_source[url].append(citation)

        # Ensure verification uses global IDs by updating each citation
        for url, citations in citations_by_source.items():
            if url in global_citation_map:
                global_id = global_citation_map[url]
                for citation in citations:
                    # Update marker to use global ID for verification tracking
                    citation["global_id"] = global_id

        # Process numeric citations directly from section content
        state = self.get_state()
        compiled_sections = state.get("section_synthesized_content", {})
        numeric_citations_by_url = {}

        # Extract all numeric citations directly from content
        for section, section_content in compiled_sections.items():
            numeric_matches = re.findall(r"\[(\d+)\]", section_content)
            for num in set(numeric_matches):
                try:
                    numeric_id = int(num)
                    # Find URL for this citation number in master_source_table
                    for url, source_data in master_source_table.items():
                        source_id = source_data.get("id", "")
                        # Check if this source ID matches the numeric citation
                        if source_id == f"S{numeric_id}" or source_id == str(
                            numeric_id
                        ):
                            # Add to global citation map if not already there
                            if url not in global_citation_map:
                                global_citation_map[url] = len(global_citation_map) + 1

                            # Create tracking for this citation
                            if url not in numeric_citations_by_url:
                                numeric_citations_by_url[url] = []

                            # Find citation context for context checking
                            pattern = (
                                r"([^.!?]*\["
                                + re.escape(str(numeric_id))
                                + r"\][^.!?]*[.!?])"
                            )
                            context_matches = re.findall(pattern, section_content)
                            for context in context_matches:
                                numeric_citations_by_url[url].append(
                                    {
                                        "marker": str(numeric_id),
                                        "raw_text": f"[{numeric_id}]",
                                        "text": context,
                                        "url": url,
                                        "section": section,
                                        "global_id": global_citation_map[url],
                                    }
                                )
                            break
                except ValueError:
                    continue

        # Merge numeric citations with regular ones
        for url, citations in numeric_citations_by_url.items():
            if url in citations_by_source:
                citations_by_source[url].extend(citations)
            else:
                citations_by_source[url] = citations

        # Check if we have any valid citations to verify
        if not citations_by_source:
            await self.emit_status("info", "No valid citations to verify", False)
            return {"verified": [], "flagged": []}

        # Log beginning of verification process
        await self.emit_status(
            "info",
            f"Starting verification of {total_citations} citations from {len(citations_by_source)} sources...",
            False,
        )

        verification_results = {"verified": [], "flagged": []}

        # Use a semaphore to limit concurrent verifications
        semaphore = asyncio.Semaphore(1)  # Process one source at a time

        async def verify_source_with_semaphore(url, citations):
            async with semaphore:
                # Skip if URL is empty
                if not url or not citations:
                    return []

                # Process citations in batches of up to 5
                all_batch_results = []
                for i in range(0, len(citations), 5):
                    batch_citations = citations[i : i + 5]

                    try:
                        # Get state for cache access
                        state = self.get_state()
                        url_results_cache = state.get("url_results_cache", {})

                        # Check cache first
                        source_content = None
                        if url in url_results_cache:
                            source_content = url_results_cache[url]
                            logger.info(f"Using cached content for verification: {url}")

                        # If not in cache, fetch source content
                        if not source_content or len(source_content) < 200:
                            logger.info(f"Fetching content for verification: {url}")
                            source_content = await self.fetch_content(url)

                        if not source_content or len(source_content) < 200:
                            # If we couldn't fetch content, mark all citations as unverified
                            return [
                                {
                                    "url": url,
                                    "verified": False,
                                    "flagged": False,
                                    "citation_text": citation.get("text", ""),
                                    "section": citation.get("section", ""),
                                    "global_id": citation.get("global_id"),
                                }
                                for citation in batch_citations
                            ]

                        # Verify this batch of citations for this source
                        batch_results = await self.verify_citation_batch(
                            url, batch_citations, source_content
                        )

                        all_batch_results.extend(batch_results)

                    except Exception as e:
                        logger.error(f"Error verifying source {url} batch: {e}")
                        # Mark the current batch as unverified but don't flag them
                        error_results = [
                            {
                                "url": url,
                                "verified": False,
                                "flagged": False,
                                "citation_text": citation.get("text", ""),
                                "section": citation.get("section", ""),
                                "global_id": citation.get("global_id"),
                            }
                            for citation in batch_citations
                        ]
                        all_batch_results.extend(error_results)

                return all_batch_results

        # Create verification tasks for each source
        verification_tasks = []
        for url, citations in citations_by_source.items():
            verification_tasks.append(verify_source_with_semaphore(url, citations))

        # Process all sources with semaphore control
        all_results = []

        # Execute verification tasks
        if verification_tasks:
            results = await asyncio.gather(*verification_tasks)

            # Flatten all results
            for batch_result in results:
                if batch_result:
                    all_results.extend(batch_result)

        # Check for citation numbers that don't match any source
        for section, section_content in compiled_sections.items():
            numeric_matches = re.findall(r"\[(\d+)\]", section_content)
            for num in set(numeric_matches):
                try:
                    numeric_id = int(num)
                    # Check if this number appears in the global citation map values
                    found_match = False
                    for url, global_id in global_citation_map.items():
                        if global_id == numeric_id:
                            found_match = True
                            break

                    # If no matching source found, flag this citation
                    if not found_match:
                        pattern = (
                            r"([^.!?]*\["
                            + re.escape(str(numeric_id))
                            + r"\][^.!?]*[.!?])"
                        )
                        context_matches = re.findall(pattern, section_content)
                        for context in context_matches:
                            verification_results["flagged"].append(
                                {
                                    "url": "",
                                    "verified": False,
                                    "flagged": True,
                                    "citation_text": context,
                                    "section": section,
                                    "global_id": numeric_id,
                                }
                            )
                except ValueError:
                    continue

        # Categorize results
        for result in all_results:
            if result.get("verified", False):
                verification_results["verified"].append(result)
            elif result.get("flagged", False):
                verification_results["flagged"].append(result)

        # Log completion of verification
        await self.emit_status(
            "info",
            f"Citation verification complete: {len(verification_results['verified'])} verified, {len(verification_results['flagged'])} flagged",
            False,
        )

        # Store verification results for later use
        self.update_state("verification_results", verification_results)

        return verification_results

    async def export_research_data(self) -> str:
        """Formats the full research data into a markdown string for direct output."""
        import json
        from datetime import datetime

        state = self.get_state()
        results_history = state.get("results_history", [])

        export_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the data structure
        export_data = {
            "export_timestamp": export_timestamp,
            "research_date": self.research_date,
            "original_query": state.get("research_state", {}).get(
                "user_message", "Unknown query"
            ),
            "results_count": len(results_history),
            "results": [],
        }

        for i, result in enumerate(results_history):
            if "timestamp" not in result:
                from datetime import timedelta

                synthetic_time = datetime.now() - timedelta(
                    minutes=(len(results_history) - i)
                )
                result_timestamp = synthetic_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                result_timestamp = result.get("timestamp")

            export_result = {
                "index": i + 1,
                "timestamp": result_timestamp,
                "query": result.get("query", "Unknown query"),
                "url": result.get("url", ""),
                "title": result.get("title", "Untitled"),
                "tokens": result.get("tokens", 0),
                "content": result.get("content", ""),
                "similarity": result.get("similarity", 0.0),
            }
            export_data["results"].append(export_result)

        # Build the markdown string with English text inside a collapsible details tag
        markdown_string = f"\n<details>\n"
        markdown_string += f"<summary>Detailed Research Data (Click to expand)</summary>\n\n"
        markdown_string += f"#### Research Export Details\n"
        markdown_string += (
            f"- **Export Timestamp**: {export_data['export_timestamp']}\n"
        )
        markdown_string += f"- **Research Date**: {self.research_date}\n"
        markdown_string += f"- **Original Query**: {export_data['original_query']}\n"
        markdown_string += f"- **Total Results**: {export_data['results_count']}\n\n"

        markdown_string += "```json\n"
        # Pretty print the JSON data inside the code block for easy reading and copying
        markdown_string += json.dumps(export_data, indent=2, ensure_ascii=False)
        markdown_string += "\n```\n\n"
        markdown_string += "</details>"

        return markdown_string

    async def add_verification_note(self, comprehensive_answer):
        """Add a note about strikethrough citations if any were flagged"""
        state = self.get_state()
        verification_results = state.get("verification_results", {})
        flagged_citations = verification_results.get("flagged", [])

        # Only add the note if we have flagged citations AND actually applied strikethrough
        citation_fixes = state.get("citation_fixes", [])
        if flagged_citations and citation_fixes:
            # Create the note
            verification_note = "\n\n## Notes on Verification\n\n"
            verification_note += "Strikethrough text indicates claims where the provided source could not be verified or was found to misrepresent the source material. The original citation number is retained for reference."
            # Check if bibliography exists in the answer
            bib_pattern = r"## Bibliography"
            bib_match = re.search(bib_pattern, comprehensive_answer)
            if bib_match:
                bib_index = bib_match.start()
                bib_content = comprehensive_answer[bib_index:]

                # Find the end of the bibliography section by looking for the next heading
                # or the research date line
                next_section_match = re.search(
                    r"\n##\s+", bib_content[bib_match.end() - bib_index :]
                )
                research_date_match = re.search(
                    r"\*Research conducted on:.*\*", bib_content
                )

                # Determine where to insert
                if next_section_match:
                    # Insert before the next section
                    insert_position = bib_index + next_section_match.start()
                    comprehensive_answer = (
                        comprehensive_answer[:insert_position]
                        + verification_note
                        + comprehensive_answer[insert_position:]
                    )
                elif research_date_match:
                    # Insert before the research date line
                    insert_position = bib_index + research_date_match.start()
                    comprehensive_answer = (
                        comprehensive_answer[:insert_position]
                        + verification_note
                        + comprehensive_answer[insert_position:]
                    )
                else:
                    # If we can't find a good position, append to the end
                    comprehensive_answer += "\n\n" + verification_note
            else:
                # If no bibliography, add at the end
                comprehensive_answer += "\n\n" + verification_note

        return comprehensive_answer

    async def review_synthesis(
        self,
        compiled_sections: Dict[str, str],
        original_query: str,
        research_outline: List[Dict],
        synthesis_model: str,
    ) -> Dict[str, List[Dict]]:
        """Review the compiled synthesis and suggest edits"""
        review_prompt = {
            "role": "system",
            "content": """You are a post-grad research editor reviewing a comprehensive research report assembled per-section in different model contexts.
    Your task is to identify any issues with this combination of multiple sections and the flow between them.
    
    Focus on:
    1. Identifying areas needing better transitions between sections
    2. Finding obvious anomalies in section generation or stylistic discrepancies large enough to be distracting
    3. Making the report read as though it were written by one author who compiled these topics together for good purpose
    
    Do NOT:
    1. Impart your own biases, interests, or preferences onto the report
    2. Re-interpret the research information or soften its conclusions
    3. Make useless or unnecessary revisions beyond the scope of ensuring flow from start to finish
    4. Remove or edit ANY in-text citations or instances of applied strikethrough. These are for specific human review and MUST NOT be changed or decoupled
    
    For each suggested edit, provide exact text to find, and exact replacement text.
    Don't include any justification or reasoning for your replacements - they will be inserted directly, so please make sure they fit in context.
    
    Format your response as a JSON object with the following structure:
    {
      "global_edits": [
        {
          "find_text": "exact text to be replaced", 
          "replace_text": "exact replacement text"
        }
      ]
    }
    
    The find_text must be the EXACT text string as it appears in the document, and the replace_text must be the EXACT text to replace it with.""",
        }

        # Create context with all sections
        review_context = f"# Complete Research Report on: {original_query}\n\n"
        review_context += "## Research Outline:\n"
        for topic in research_outline:
            review_context += f"- {topic['topic']}\n"
            for subtopic in topic.get("subtopics", []):
                review_context += f"  - {subtopic}\n"
        review_context += "\n"

        # Add the full content of each section
        review_context += "## Complete Report Content by Section:\n\n"
        state = self.get_state()
        memory_stats = state.get("memory_stats", {})
        section_tokens = memory_stats.get("section_tokens", {})

        for section_title, content in compiled_sections.items():
            # Get token count for this section
            tokens = section_tokens.get(section_title, 0)
            if tokens == 0:
                tokens = await self.count_tokens(content)
                section_tokens[section_title] = tokens
                memory_stats["section_tokens"] = section_tokens
                self.update_state("memory_stats", memory_stats)

            review_context += f"### {section_title} [{tokens} tokens]\n\n"
            review_context += f"{content}\n\n"

        review_context += "\nReview this research report and respond with necessary edits with specified JSON structure. Please don't include any other text in your response but the edits."

        # Create messages array
        messages = [review_prompt, {"role": "user", "content": review_context}]

        # Generate the review
        try:
            await self.emit_status(
                "info", "Reviewing and improving the synthesis...", False
            )

            # Scale temperature based on synthesis temperature valve
            review_temperature = (
                self.valves.SYNTHESIS_TEMPERATURE * 0.5
            )  # Lower temperature for more consistent review

            # Use synthesis model for reviewing
            response = await self.generate_completion(
                synthesis_model,
                messages,
                stream=False,
                temperature=review_temperature,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                review_content = response["choices"][0]["message"]["content"]

                # Parse the JSON review
                try:
                    review_json_str = review_content[
                        review_content.find("{") : review_content.rfind("}") + 1
                    ]
                    review_data = json.loads(review_json_str)
                    return review_data
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing review JSON: {e}")
                    # Return a minimal structure if parsing fails
                    return {"global_edits": [], "section_edits": {}}
            else:
                return {"global_edits": [], "section_edits": {}}

        except Exception as e:
            logger.error(f"Error generating synthesis review: {e}")
            return {"global_edits": [], "section_edits": {}}

    async def apply_review_edits(
        self,
        compiled_sections: Dict[str, str],
        review_data: Dict[str, Any],
        synthesis_model: str,
    ):
        """Apply the suggested edits from the review to improve the synthesis"""
        # Create deep copy of sections to modify
        edited_sections = compiled_sections.copy()

        # Track if we made any changes
        changes_made = False

        # Apply global edits
        global_edits = review_data.get("global_edits", [])
        if global_edits:
            changes_made = True
            await self.emit_status(
                "info",
                f"Applying {len(global_edits)} global edits to synthesis...",
                False,
            )

            for edit_idx, edit in enumerate(global_edits):
                find_text = edit.get("find_text", "")
                replace_text = edit.get("replace_text", "")

                if not find_text:
                    logger.warning(f"Empty find_text in edit {edit_idx+1}, skipping")
                    continue

                # Apply to each section
                for section_title, content in edited_sections.items():
                    if find_text in content:
                        edited_sections[section_title] = content.replace(
                            find_text, replace_text
                        )
                        logger.info(
                            f"Applied edit {edit_idx+1} in section '{section_title}'"
                        )

        return edited_sections, changes_made

    async def generate_replacement_topics(
        self,
        query: str,
        kept_items: List[str],
        removed_items: List[str],
        preference_vector: Dict,
        outline_items: List[str],
    ) -> List[str]:
        """Generate replacement topics using semantic transformation"""
        # If nothing was removed, return empty list
        if not removed_items:
            return []

        # If nothing was kept, use the full original outline as reference
        if not kept_items:
            kept_items = outline_items

        # Calculate 80% of removed items count, rounded up
        num_replacements = math.ceil(len(removed_items) * 0.8)

        # Ensure at least one replacement
        num_replacements = max(1, num_replacements)

        logger.info(
            f"Generating {num_replacements} replacement topics (80% of {len(removed_items)} removed)"
        )

        # Create a prompt to generate replacements
        replacement_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant generating replacement topics for a research outline.
    Based on the kept topics, original query, and user's preferences, generate new research topics to replace removed ones.
    Each new topic should:
    1. Be directly relevant to answering or addressing the original query
    2. Be conceptually aligned with the kept topics
    3. Avoid concepts related to removed topics and their associated themes
    4. Be specific and actionable for research without devolving into hyperspecificity
    
    Generate EXACTLY the requested number of replacement topics in a numbered list format.
    Each replacement should be thoughtful and unique, exploring and expanding on different aspects of the research subject.
    """,
        }

        # Extract preference information
        pdv = preference_vector.get("pdv")
        strength = preference_vector.get("strength", 0.0)
        impact = preference_vector.get("impact", 0.0)

        # Prepare the request content
        content = f"""Original query: {query}
    
    Kept topics (conceptually preferred):
    {kept_items}
    
    Removed topics (to avoid):
    {removed_items}
    
    """

        # Pre-compute embeddings
        state = self.get_state()
        if pdv is not None and impact > 0.1:
            # Get query embedding first
            query_embedding = await self.get_embedding(query)

            # Get kept item embeddings sequentially
            kept_embeddings = []
            for item in kept_items:
                embedding = await self.get_embedding(item)
                if embedding:
                    kept_embeddings.append(embedding)

            # If we have enough embeddings, create a semantic transformation
            if query_embedding and len(kept_embeddings) >= 3:
                # Create a simple eigendecomposition
                try:
                    # Filter out any non-array elements that could cause errors
                    valid_embeddings = []
                    for emb in kept_embeddings:
                        if isinstance(emb, list) or (
                            hasattr(emb, "ndim") and emb.ndim == 1
                        ):
                            valid_embeddings.append(emb)

                    # Only proceed if we have enough valid embeddings
                    if len(valid_embeddings) >= 3:
                        kept_array = np.array(valid_embeddings)
                        # Simple PCA
                        pca = PCA(n_components=min(3, len(valid_embeddings)))
                        pca.fit(kept_array)
                    else:
                        logger.warning(
                            f"Not enough valid embeddings for PCA: {len(valid_embeddings)}/3 required"
                        )
                        return []

                    eigen_data = {
                        "eigenvectors": pca.components_.tolist(),
                        "eigenvalues": pca.explained_variance_.tolist(),
                        "explained_variance": pca.explained_variance_ratio_.tolist(),
                    }

                    # Create transformation that includes PDV
                    transformation = await self.create_semantic_transformation(
                        eigen_data, pdv=pdv
                    )

                    # Store for later use
                    self.update_state("semantic_transformations", transformation)

                    logger.info(
                        f"Created semantic transformation for replacement topics generation"
                    )
                except Exception as e:
                    logger.error(f"Error creating PCA for topic replacement: {e}")

        if pdv is not None:
            # Translate the PDV into natural language concepts
            pdv_concepts = await self.translate_pdv_to_words(pdv)
            if pdv_concepts:
                content += f"User preferences: The user prefers topics related to: {pdv_concepts}\n"
                if strength > 0.9:
                    content += f"The user has expressed a strong preference for these concepts. "
                elif strength > 0.5:
                    content += f"The user has expressed a moderate preference for these concepts. "
                else:
                    content += f"The user has expressed a slight preference for these concepts. "

        content += f"""Generate EXACTLY {num_replacements} replacement research topics in a numbered list.
    These should align with the kept topics and original query, while avoiding concepts from removed topics.
    Please don't include any other text in your response but the replacement topics. You don't need to justify them either.
    """

        messages = [replacement_prompt, {"role": "user", "content": content}]

        # Generate all replacements at once
        try:
            await self.emit_status(
                "info", f"Generating {num_replacements} replacement topics...", False
            )

            # Generate replacements
            # Use research model for generating replacements
            research_model = self.get_research_model()
            response = await self.generate_completion(
                research_model,
                messages,
                temperature=self.valves.TEMPERATURE
                * 1.1,  # Slightly higher temperature for creative replacements
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]

                # Parse the generated text to extract topics (numbered list format)
                lines = generated_text.split("\n")
                replacements = []

                for line in lines:
                    # Look for numbered list items: 1. Topic description
                    match = re.search(r"^\s*\d+\.\s*(.+)$", line)
                    if match:
                        topic = match.group(1).strip()
                        if (
                            topic and len(topic) > 10
                        ):  # Minimum length to be a valid topic
                            replacements.append(topic)

                # Ensure we have exactly the right number of replacements
                if len(replacements) > num_replacements:
                    replacements = replacements[:num_replacements]
                elif len(replacements) < num_replacements:
                    # If we didn't get enough, create generic ones to fill the gap
                    while len(replacements) < num_replacements:
                        missing_count = num_replacements - len(replacements)
                        await self.emit_status(
                            "info",
                            f"Generating {missing_count} additional topics...",
                            False,
                        )
                        replacements.append(
                            f"Additional research on {query} aspect {len(replacements)+1}"
                        )

                return replacements

        except Exception as e:
            logger.error(f"Error generating replacement topics: {e}")

        # Fallback - create generic replacements
        return [
            f"Alternative research topic {i+1} for {query}"
            for i in range(num_replacements)
        ]

    async def improved_query_generation(
        self, user_message, priority_topics, search_context
    ):
        """Generate refined search queries for research topics with improved context"""
        query_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant generating effective search queries.
    Based on the user's original question, current research needs, and context provided, generate 4 precise search queries.
    Each query should be specific, use relevant keywords, and be designed to find targeted information.
    
    Your queries should:
    1. Directly address the priority research topics
    2. Avoid redundancy with previous queries
    3. Target information gaps in the current research
    4. Be concise (6-12 words) but specific 
    5. Include specialized terminology when appropriate
    
    Focus on core conceptual terms with targeted expansions and don't return heavy, clunky queries.
    Use quotes sparingly and as a last resort. Never use multiple sets of quotes in the same query.
    
    Format your response as a valid JSON object with the following structure:
    {"queries": [
      "query": "search query 1", "topic": "related research topic", 
      "query": "search query 2", "topic": "related research topic",
      "query": "search query 3", "topic": "related research topic",
      "query": "search query 4", "topic": "related research topic"
    ]}""",
        }

        message = {
            "role": "user",
            "content": f"""Original query: "{user_message}"\n\nResearch context: "{search_context}"\n\nGenerate 4 effective search queries to gather information for the priority research topics.""",
        }

        # Generate the queries first, without any embedding operations
        try:
            response = await self.generate_completion(
                self.get_research_model(),
                [query_prompt, message],
                temperature=self.valves.TEMPERATURE,
            )

            query_content = response["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                query_json_str = query_content[
                    query_content.find("{") : query_content.rfind("}") + 1
                ]
                query_data = json.loads(query_json_str)
                queries = query_data.get("queries", [])

                # Check if queries is a list of strings or a list of objects
                if queries and isinstance(queries[0], str):
                    # Convert to objects with query and topic
                    query_strings = queries
                    query_topics = (
                        priority_topics[: len(queries)]
                        if priority_topics
                        else ["Research"] * len(queries)
                    )
                    queries = [
                        {"query": q, "topic": t}
                        for q, t in zip(query_strings, query_topics)
                    ]

                return queries

            except Exception as e:
                logger.error(f"Error parsing query JSON: {e}")
                # Fallback: generate basic queries for priority topics
                queries = []
                for i, topic in enumerate(priority_topics[:3]):
                    queries.append({"query": f"{user_message} {topic}", "topic": topic})

                return queries

        except Exception as e:
            logger.error(f"Error generating improved queries: {e}")
            # Fallback: generate basic queries
            queries = []
            for i, topic in enumerate(priority_topics[:3]):
                queries.append({"query": f"{user_message} {topic}", "topic": topic})

            return queries

    async def generate_titles(self, user_message, comprehensive_answer):
        """Generate a main title and subtitle for the research report"""
        titles_prompt = {
            "role": "system",
            "content": """You are a post-grad research writer creating compelling titles for research reports.
            
    Create a main title and subtitle for a comprehensive research report. The titles should:
    1. Be relevant and accurately reflect the content and focus of the research
    2. Be engaging and professional. Intriguing, even
    3. Follow academic/research paper conventions
    4. Avoid clickbait or sensationalism unless it's really begging for it
    
    For main title:
    - 5-12 words in length
    - Clear and focused
    - Appropriately formal for academic/research context
    
    For subtitle:
    - 8-15 words in length
    - Provides additional context and specificity
    - Complements the main title without redundancy
    
    Format your response as a JSON object with the following structure:
    {
      "main_title": "Your proposed main title",
      "subtitle": "Your proposed subtitle"
    }""",
        }

        # Create a context with the research query and a summary of the comprehensive answer
        titles_context = f"""Original Research Query: {user_message}
    
    Research Report Content Summary:
    {comprehensive_answer}...
    
    Generate an appropriate main title and subtitle for this research report."""

        try:
            # Get the research model for title generation
            research_model = self.get_research_model()

            # Generate titles
            response = await self.generate_completion(
                research_model,
                [titles_prompt, {"role": "user", "content": titles_context}],
                temperature=0.7,  # Allow some creativity for titles
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                titles_content = response["choices"][0]["message"]["content"]

                # Extract JSON from response
                try:
                    json_str = titles_content[
                        titles_content.find("{") : titles_content.rfind("}") + 1
                    ]
                    titles_data = json.loads(json_str)

                    main_title = titles_data.get(
                        "main_title", f"Research Report: {user_message}"
                    )
                    subtitle = titles_data.get(
                        "subtitle", "A Comprehensive Analysis and Synthesis"
                    )

                    return {"main_title": main_title, "subtitle": subtitle}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing titles JSON: {e}")
                    # Fallback to simple titles
                    return {
                        "main_title": f"Research Report: {user_message[:50]}",
                        "subtitle": "A Comprehensive Analysis and Synthesis",
                    }
            else:
                # Fallback titles
                return {
                    "main_title": f"Research Report: {user_message[:50]}",
                    "subtitle": "A Comprehensive Analysis and Synthesis",
                }

        except Exception as e:
            logger.error(f"Error generating titles: {e}")
            # Fallback titles
            return {
                "main_title": f"Research Report: {user_message[:50]}",
                "subtitle": "A Comprehensive Analysis and Synthesis",
            }

    async def generate_abstract(self, user_message, comprehensive_answer, bibliography):
        """Generate an abstract for the research report"""
        abstract_prompt = {
            "role": "system",
            "content": f"""You are a post-grad research assistant writing an abstract for a comprehensive research report.
            
    Create a concise academic abstract (150-250 words) that summarizes the research report. The abstract should:
    1. Outline the research objective and original intent without simply restating the original query
    2. Summarize the key findings and their significance
    3. Be written in an academic yet interesting tone
    4. Be self-contained and understandable on its own
    5. Draw you in by highlighting the interesting aspects of the research without being misleading or disingenuous

    The abstract must NOT:
    1. Interpret the content in a lofty way that exaggerates its importance or profundity, or contrives a narrative with empty sophistication. 
    2. Attempt to portray the subject matter in any particular sort of light, good or bad, especially by using apologetic or dismissive language.
    3. Focus on perceived complexities or challenges related to the topic or research process, or include appeals to future research.
    4. Ever take a preachy or moralizing tone, or take a "stance" for or against/"side" with or against anything not driven by the provided data.
    5. Overstate the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
    6. Sound to the reader as though it is overtly attempting to be diplomatic, considerate, enthusiastic, or overly-generalized.

    The abstract should follow scientific paper abstract structure but be accessible to an educated general audience.""",
        }

        # Create a context with the full report and bibliography information
        abstract_context = f"""Research Query: {user_message}
    
    Research Report Full Content:
    {comprehensive_answer}...
    
    Generate a concise, substantive abstract focusing on substantive content and key insights rather than how the research was conducted. Please don't include any other text in your response but the abstract.
    """

        try:
            # Get the synthesis model for abstract generation
            synthesis_model = self.get_synthesis_model()

            # Generate abstract with 5-minute timeout
            response = await asyncio.wait_for(
                self.generate_completion(
                    synthesis_model,
                    [abstract_prompt, {"role": "user", "content": abstract_context}],
                    temperature=self.valves.SYNTHESIS_TEMPERATURE,
                ),
                timeout=300,  # 5 minute timeout
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                abstract = response["choices"][0]["message"]["content"]
                await self.emit_message(f"*Abstract generation complete.*\n")
                return abstract
            else:
                # Fallback abstract
                await self.emit_message(f"*Abstract generation fallback used.*\n")
                return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."

        except asyncio.TimeoutError:
            logger.error("Abstract generation timed out after 5 minutes")
            # Provide a fallback abstract
            await self.emit_message(
                f"*Abstract generation timed out, using fallback.*\n"
            )
            return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."
        except Exception as e:
            logger.error(f"Error generating abstract: {e}")
            # Fallback abstract
            await self.emit_message(f"*Abstract generation error, using fallback.*\n")
            return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __event_call__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__current_event_call__ = __event_call__
        self.__user__ = User(**__user__)
        self.__model__ = __model__
        self.__request__ = __request__

        # Extract conversation ID from the message history
        messages = body.get("messages", [])
        if not messages:
            return ""

        # Get the user message for debugging
        user_message = messages[-1].get("content", "").strip()
        logger.debug("★★★ PIPE FUNCTION CALLED ★★★")
        logger.debug(f"User message: '{user_message}'")
        logger.debug(f"Messages count: {len(messages)}")

        # First message ID in the conversation serves as our conversation identifier
        first_message = messages[0] if messages else {}
        conversation_id = f"{__user__['id']}_{first_message.get('id', 'default')}"
        self.conversation_id = conversation_id

        # Check if this appears to be a completely new conversation
        state = self.get_state()
        waiting_for_outline_feedback = state.get("waiting_for_outline_feedback", False)
        if (
            len(messages) <= 2 and not waiting_for_outline_feedback
        ):  # Check we're not waiting for feedback
            logger.info(f"New conversation detected with ID: {conversation_id}")
            self.reset_state()  # Reset all state for this conversation

        # Initialize master source table if not exists
        state = self.get_state()
        if "master_source_table" not in state:
            self.update_state("master_source_table", {})

        # Initialize other critical state variables if missing
        if "memory_stats" not in state:
            self.update_state(
                "memory_stats",
                {
                    "results_tokens": 0,
                    "section_tokens": {},
                    "synthesis_tokens": 0,
                    "total_tokens": 0,
                },
            )

        if "url_selected_count" not in state:
            self.update_state("url_selected_count", {})

        if "url_token_counts" not in state:
            self.update_state("url_token_counts", {})

        # If the pipe is disabled or it's not a default task, return
        if not self.valves.ENABLED or (__task__ and __task__ != TASKS.DEFAULT):
            return ""

        # Get user query from the latest message
        user_message = messages[-1].get("content", "").strip()
        if not user_message:
            return ""

        # Set research date
        from datetime import datetime

        self.research_date = datetime.now().strftime("%Y-%m-%d")

        # Preload vocabulary embeddings in background as soon as possible
        self.vocabulary_embeddings = None  # Force reload
        asyncio.create_task(self.load_prebuilt_vocabulary_embeddings())

        # Get state for this conversation
        state = self.get_state()

        # Check waiting flag directly in state
        waiting_for_feedback = state.get("waiting_for_outline_feedback", False)
        logger.debug(
            f"★★★ CHECKING WAITING_FOR_OUTLINE_FEEDBACK: {waiting_for_feedback} ★★★"
        )

        if waiting_for_feedback:
            # This is a continuation after asking for feedback.
            self.update_state("waiting_for_outline_feedback", False)
            feedback_data = state.get("outline_feedback_data", {})
            if feedback_data:
                logger.debug("Processing user's outline feedback.")
                feedback_result = await self.process_outline_feedback_continuation(
                    user_message
                )

                original_query = feedback_data.get("original_query", "")
                outline_items = feedback_data.get("outline_items", [])

                all_topics = self._flatten_outline_topics(outline_items)
                outline_text = " ".join(all_topics)
                outline_embedding = await self.get_embedding(outline_text)

                (
                    research_outline,
                    all_topics,
                    outline_embedding,
                ) = await self.continue_research_after_feedback(
                    feedback_result,
                    original_query,
                    outline_items,
                    all_topics,
                    outline_embedding,
                )

                await self.initialize_research_state(
                    original_query, research_outline, all_topics, outline_embedding
                )
                await self.update_token_counts()
                user_message = original_query  # Restore user_message for the cycles
            else:
                logger.warning(
                    "Was waiting for feedback, but no feedback_data found. Starting over."
                )
                self.reset_state()

        # After potentially processing feedback, check if we're ready for research cycles.
        state = self.get_state()  # Refresh state
        research_state = state.get("research_state")

        if not research_state:
            # This is the first run for a new query (or a reset). Do initial setup.
            logger.debug("★★★ PROCEEDING WITH INITIAL SETUP ★★★")

            # Check if this is a follow-up query
            is_follow_up = await self.is_follow_up_query(messages)
            self.update_state("follow_up_mode", is_follow_up)

            # Get summary embedding if this is a follow-up
            summary_embedding = None
            if is_follow_up:
                prev_comprehensive_summary = state.get("prev_comprehensive_summary", "")
                if prev_comprehensive_summary:
                    try:
                        await self.emit_status(
                            "info", "Processing follow-up query...", False
                        )
                        summary_embedding = await self.get_embedding(
                            prev_comprehensive_summary
                        )
                        await self.emit_message("## Deep Research Mode: Follow-up\n\n")
                        await self.emit_message(
                            "I'll continue researching based on your follow-up query while considering our previous findings.\n\n"
                        )
                    except Exception as e:
                        logger.error(f"Error getting summary embedding: {e}")
                        # Continue without the summary embedding if there's an error
                        is_follow_up = False
                        self.update_state("follow_up_mode", False)
                        await self.emit_message("## Deep Research Mode: Activated\n\n")
                        await self.emit_message(
                            "I'll search for comprehensive information about your query. This might take a moment...\n\n"
                        )
                else:
                    is_follow_up = False
                    self.update_state("follow_up_mode", False)
            else:
                await self.emit_status("info", "Starting deep research...", False)
                await self.emit_message("## Deep Research Mode: Activated\n\n")
                await self.emit_message(
                    "I'll search for comprehensive information about your query. This might take a moment...\n\n"
                )

            # For follow-up queries, generate a new research outline
            if is_follow_up:
                outline_embedding = await self.get_embedding(
                    user_message
                )  # Create initial placeholder
                # ... (rest of follow-up setup logic)
                # This part is complex but seems correct in the original file
                # The key is that it will eventually produce research_outline, all_topics, etc.
                await self.emit_status(
                    "info",
                    "Generating initial search queries for follow-up...",
                    False,
                )

                initial_query_prompt = {
                    "role": "system",
                    "content": """You are a post-grad research assistant generating effective search queries for continued research based on an existing report.
        Based on the user's follow-up question and the previous research summary, generate 6 initial search queries.
        Each query should be specific, use relevant keywords, and be designed to find new information that builds on the previous research towards the new query.
        Use quotes sparingly and as a last resort. Never use multiple sets of quotes in the same query.

        Format your response as a valid JSON object with the following structure:
        {"queries": [
          "search query 1",
          "search query 2",
          "search query 3"
        ]}""",
                }

                initial_query_messages = [
                    initial_query_prompt,
                    {
                        "role": "user",
                        "content": f"Follow-up question: {user_message}\n\nPrevious research summary:\n{state.get('prev_comprehensive_summary', '')}...\n\nGenerate initial search queries for the follow-up question that build on the previous research.",
                    },
                ]

                # Get initial search queries
                query_response = await self.generate_completion(
                    self.get_research_model(),
                    initial_query_messages,
                    temperature=self.valves.TEMPERATURE,
                )
                query_content = query_response["choices"][0]["message"]["content"]

                # Extract JSON from response
                try:
                    query_json_str = query_content[
                        query_content.find("{") : query_content.rfind("}") + 1
                    ]
                    query_data = json.loads(query_json_str)
                    initial_queries = query_data.get("queries", [])
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing query JSON: {e}")
                    # Fallback: extract queries using regex if JSON parsing fails
                    import re

                    initial_queries = re.findall(r'"([^"]+)"', query_content)[:3]
                    if not initial_queries:
                        initial_queries = ["Information about " + user_message]

                # Create a simple outline for follow-up queries
                research_outline = [
                    {
                        "topic": f"Follow-up Analysis of {user_message}",
                        "subtopics": [
                            "Updated Information",
                            "New Developments",
                            "Additional Context",
                        ],
                    },
                    {
                        "topic": "Extended Research",
                        "subtopics": [
                            "Detailed Investigation",
                            "Related Topics",
                            "Comparative Analysis",
                        ],
                    },
                ]
                all_topics = self._flatten_outline_topics(research_outline)
                outline_text = " ".join(all_topics)
                outline_embedding = await self.get_embedding(outline_text)
                await self.initialize_research_dimensions(all_topics, user_message)

                outline_text_display = "### Research Outline for Follow-up\n\n"
                for topic in research_outline:
                    outline_text_display += f"**{topic['topic']}**\n"
                    for subtopic in topic.get("subtopics", []):
                        outline_text_display += f"- {subtopic}\n"
                    outline_text_display += "\n"
                await self.emit_message(outline_text_display)
                await self.emit_message(
                    "\n*Continuing with research based on this outline and previous findings...*\n\n"
                )
                initial_results = []  # Start fresh for follow up

            else:
                # Regular new query
                await self.emit_status(
                    "info", "Generating initial search queries...", False
                )
                # ... [Initial query generation logic from original file] ...
                initial_query_prompt = {
                    "role": "system",
                    "content": f"""You are a post-grad research assistant generating effective search queries.
        The user has submitted a research query: "{user_message}".
        Based on the user's input, generate 8 initial search queries to begin research and help us delineate the research topic.
        Half of the queries should be broad, aimed at identifying and defining the main topic and returning core characteristic information about it.
        The other half should be more specific, designed to find information to help expand on known base details of the user's query.
        Use quotes sparingly and as a last resort. Never use multiple sets of quotes in the same query.

        Format your response as a valid JSON object with the following structure:
        {{"queries": [
          "search query 1",
          "search query 2",
          "search query 3..."
        ]}}""",
                }
                initial_query_messages = [
                    initial_query_prompt,
                    {
                        "role": "user",
                        "content": f"Generate initial search queries for this user query: {user_message}",
                    },
                ]
                query_response = await self.generate_completion(
                    self.get_research_model(),
                    initial_query_messages,
                    temperature=self.valves.TEMPERATURE,
                )
                query_content = query_response["choices"][0]["message"]["content"]
                try:
                    query_json_str = query_content[
                        query_content.find("{") : query_content.rfind("}") + 1
                    ]
                    query_data = json.loads(query_json_str)
                    initial_queries = query_data.get("queries", [])
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing query JSON: {e}")
                    import re

                    initial_queries = re.findall(r'"([^"]+)"', query_content)[:3]
                    if not initial_queries:
                        initial_queries = ["Information about " + user_message]

                await self.emit_message(f"### Initial Research Queries\n\n")
                for i, query in enumerate(initial_queries):
                    await self.emit_message(f"**Query {i+1}**: {query}\n\n")

                outline_embedding = await self.get_embedding(user_message)
                initial_results = []
                for query in initial_queries:
                    query_embedding = await self.get_embedding(query)
                    if not query_embedding:
                        query_embedding = [0] * 384
                    results = await self.process_query(
                        query,
                        query_embedding,
                        outline_embedding,
                        None,
                        summary_embedding,
                    )
                    initial_results.extend(results)

                useful_results = [
                    r for r in initial_results if len(r.get("content", "")) > 200
                ]
                if not useful_results:
                    await self.emit_message(
                        f"*Unable to find initial search results. Creating research outline based on the query alone.*\n\n"
                    )
                    initial_results = [
                        {
                            "title": f"Information about {user_message}",
                            "url": "",
                            "content": f"Placeholder for research about {user_message}.",
                            "query": user_message,
                        }
                    ]
                else:
                    logger.info(
                        f"Found {len(useful_results)} useful results from initial queries"
                    )

                await self.emit_status(
                    "info",
                    "Analyzing initial results and generating research outline...",
                    False,
                )
                # ... [Outline generation logic from original file, including the two-pass method] ...
                # This part generates `research_outline` and `all_topics`.
                # ... [Assume this logic is copied and works] ...
                research_outline = []
                try:
                    import time

                    start_time = time.time()
                    logger.info("Starting two-pass outline generation.")

                    # == PASS 1: Generate outline in Markdown format ==
                    outline_generation_prompt = {
                        "role": "system",
                        "content": f"""You are a post-grad academic scholar creating a structured research outline.
Based on the user's query and initial search results, create a comprehensive conceptual outline for a research report on: \"{user_message}\".

The outline must:
1. Break down the query into key concepts, figures, methods, etc.
2. Be organized in a hierarchical structure with main topics and subtopics.
3. Incorporate relevant topics discovered in the initial search results.
4. Conceptually fill the space between an introduction and a conclusion.

The outline MUST NOT:
1. Include philosophical or theoretical approaches unless explicitly solicited.
2. Include generic topics like "considering complexities".
3. Reflect your own biases or opinions.
4. Include an introduction, conclusion, or appendix topic.

Format your response as a clear, structured Markdown list. For example:
## Main Topic 1
- Subtopic 1.1
- Subtopic 1.2
## Main Topic 2
- Subtopic 2.1
- Subtopic 2.2
""",
                    }

                    outline_context = "### Initial Search Results:\n\n"
                    for i, result in enumerate(initial_results):
                        outline_context += (
                            f"Result {i+1} (Query: '{result.get('query', 'N/A')}')\n"
                        )
                        outline_context += f"Title: {result.get('title', 'N/A')}\n"
                        outline_context += (
                            f"Content: {result.get('content', '')[:1500]}...\n\n"
                        )

                    outline_generation_messages = [
                        outline_generation_prompt,
                        {
                            "role": "user",
                            "content": f"Original query: {user_message}\n\n{outline_context}\n\nGenerate a structured research outline in Markdown format.",
                        },
                    ]

                    logger.info("Calling LLM for Pass 1 (Markdown outline)...")
                    markdown_outline_response = await self.generate_completion(
                        self.get_research_model(),
                        outline_generation_messages,
                        temperature=self.valves.TEMPERATURE,
                    )
                    logger.info("LLM Pass 1 (Markdown outline) completed.")

                    markdown_outline = ""
                    if markdown_outline_response and markdown_outline_response.get(
                        "choices"
                    ):
                        markdown_outline = markdown_outline_response["choices"][0][
                            "message"
                        ]["content"]

                    if not markdown_outline.strip():
                        raise ValueError(
                            "LLM failed to generate Markdown outline in Pass 1."
                        )
                    logger.info("Pass 1 (Markdown) successful.")

                    # == PASS 2: Convert Markdown outline to JSON ==
                    json_conversion_prompt = {
                        "role": "system",
                        "content": 'You are a data formatting expert. Convert a Markdown outline into a valid JSON object. Output ONLY the raw JSON, with no other text. The JSON structure must be: {"outline": [{"topic": "Main Topic", "subtopics": ["Subtopic 1"]}]}.',
                    }
                    json_conversion_messages = [
                        json_conversion_prompt,
                        {
                            "role": "user",
                            "content": f"Convert this Markdown to JSON:\n\n{markdown_outline}",
                        },
                    ]

                    logger.info("Calling LLM for Pass 2 (JSON conversion)...")
                    json_outline_response = await self.generate_completion(
                        self.get_research_model(),
                        json_conversion_messages,
                        json_mode=True,
                        temperature=0.0,
                    )
                    logger.info("LLM Pass 2 (JSON conversion) completed.")

                    json_outline_content = ""
                    if json_outline_response and json_outline_response.get("choices"):
                        json_outline_content = json_outline_response["choices"][0][
                            "message"
                        ]["content"]

                    if not json_outline_content or not json_outline_content.strip():
                        raise ValueError("LLM returned an empty response in Pass 2.")

                    json_start = json_outline_content.find("{")
                    json_end = json_outline_content.rfind("}") + 1

                    if json_start != -1 and json_end > json_start:
                        outline_json_str = json_outline_content[json_start:json_end]
                        logger.info(
                            f"Extracted JSON string for parsing: {outline_json_str[:200]}..."
                        )
                        research_outline, all_topics = (
                            await self.parse_json_with_fallback(
                                outline_json_str, user_message
                            )
                        )
                    else:
                        raise ValueError(
                            "No valid JSON object found in the LLM response for Pass 2."
                        )

                    if not research_outline:
                        raise ValueError(
                            "LLM returned an empty or invalid outline in Pass 2."
                        )
                    logger.info(
                        "Pass 2 (JSON conversion) successful. Outline generated."
                    )

                    end_time = time.time()
                    logger.info(
                        f"Outline generation took {end_time - start_time:.2f} seconds."
                    )
                except Exception as e:
                    logger.error(
                        f"FATAL: An error occurred during outline generation: {e}",
                        exc_info=True,
                    )
                    await self.emit_status(
                        "error", f"Failed to generate outline: {e}", True
                    )
                    await self.emit_message(
                        "*Failed to generate a detailed research outline. Proceeding with a basic structure.*"
                    )
                    research_outline = [
                        {
                            "topic": f"Key Aspects of {user_message}",
                            "subtopics": [
                                "Background and History",
                                "Core Technologies",
                                "Impact and Applications",
                            ],
                        },
                        {
                            "topic": "Related Concepts",
                            "subtopics": ["Comparative Analysis", "Future Trends"],
                        },
                    ]
                all_topics = self._flatten_outline_topics(research_outline)
                outline_text = " ".join(all_topics)
                outline_embedding = await self.get_embedding(outline_text)

            # After generating the outline, either ask for feedback or initialize state
            if self.valves.INTERACTIVE_RESEARCH:
                # The logic for this part seems correct in the original file
                # It sets the waiting flag and returns to pause execution
                await self.process_user_outline_feedback(research_outline, user_message)
                return ""
            else:
                # Not interactive, so initialize state and proceed to cycles
                await self.initialize_research_state(
                    user_message,
                    research_outline,
                    all_topics,
                    outline_embedding,
                    initial_results,
                )
                await self.update_token_counts(initial_results)
                await self.emit_message("\n*Continuing with research outline...*\n\n")

        # If we reach here, we must have a research state.
        state = self.get_state()
        research_state = state.get("research_state")

        if not research_state:
            # This can happen if interactive mode paused execution.
            # The next call will handle it, so we just return.
            return ""

        # Retrieve state variables to start the research cycles
        research_outline = research_state.get("research_outline", [])
        all_topics = research_state.get("all_topics", [])
        outline_embedding = research_state.get("outline_embedding")
        user_message = research_state.get("user_message", user_message)
        is_follow_up = state.get("follow_up_mode", False)
        summary_embedding = None
        if is_follow_up:
            prev_summary = state.get("prev_comprehensive_summary", "")
            if prev_summary:
                summary_embedding = await self.get_embedding(prev_summary)

        initial_results = state.get(
            "results_history", []
        )  # Use history as initial results

        # Update status to show we've moved beyond outline generation
        await self.emit_status(
            "info", "Research outline confirmed. Beginning research cycles...", False
        )

        # Initialize research variables for continued cycles
        cycle = 1
        max_cycles = self.valves.MAX_CYCLES
        min_cycles = self.valves.MIN_CYCLES
        completed_topics = set(state.get("completed_topics", set()))
        irrelevant_topics = set(state.get("irrelevant_topics", set()))
        search_history = state.get("search_history", [])

        # ★★★ CRITICAL FIX: Ensure initial_results is always defined ★★★
        try:
            # Check if initial_results exists and has content
            if initial_results:
                logger.debug(f"★★★ INITIAL_RESULTS LENGTH: {len(initial_results)} ★★★")
            else:
                logger.debug("★★★ INITIAL_RESULTS IS EMPTY ★★★")
        except NameError:
            # initial_results is not defined, set it to empty list
            initial_results = []
            logger.debug("★★★ INITIAL_RESULTS WAS NOT DEFINED, SET TO EMPTY LIST ★★★")

        results_history = state.get("results_history", []) + (initial_results or [])
        active_outline = list(set(all_topics) - completed_topics - irrelevant_topics)
        cycle_summaries = state.get("cycle_summaries", [])

        # Ensure consistent token counts
        await self.update_token_counts()

        # Step 4: Begin research cycles
        while cycle < max_cycles and active_outline:
            cycle += 1
            await self.emit_status(
                "info",
                f"Research cycle {cycle}/{max_cycles}: Generating search queries...",
                False,
            )

            # Calculate research trajectory from previous cycles
            if cycle > 2 and results_history:
                research_trajectory = await self.calculate_research_trajectory(
                    search_history, results_history
                )

                # Update research trajectory
                self.update_state("research_trajectory", research_trajectory)

            # Calculate gap vector for directing research toward uncovered areas
            gap_vector = await self.calculate_gap_vector()

            # Rank active topics by priority using semantic analysis
            prioritized_topics = await self.rank_topics_by_research_priority(
                active_outline, gap_vector, completed_topics, results_history
            )

            # Get most important topics for this cycle (limited to 10)
            priority_topics = prioritized_topics[:10]

            # Build context for query generation with all the improved elements
            search_context = ""

            # Include original query and user feedback
            search_context += f"### Original Query:\n{user_message}\n\n"

            # If there was user feedback, include it as clarification
            user_preferences = state.get("user_preferences", {})
            if user_preferences.get("pdv") is not None:
                # Try to translate PDV to words
                pdv_words = await self.translate_pdv_to_words(
                    user_preferences.get("pdv")
                )
                if pdv_words:
                    search_context += f"### User Preferences:\nThe user is more interested in topics related to: {pdv_words}\n\n"

            # Include prioritized research topics
            search_context += "### Priority research topics for this cycle:\n"
            for topic in priority_topics:
                search_context += f"- {topic}\n"

            # Add a separate section for all remaining topics
            if len(active_outline) > len(priority_topics):
                search_context += "\n### Additional topics still needing research:\n"
                for topic in active_outline:
                    if topic not in priority_topics:
                        search_context += f"- {topic}\n"

            # Include recent search history (only last 3 cycles)
            if search_history:
                search_context += "\n### Recent search queries:\n"
                search_context += ", ".join([f"'{q}'" for q in search_history[-9:]])
                search_context += "\n\n"

            # Include previous results summary
            if results_history:
                search_context += "### Recent research results summary:\n\n"
                # Use most recent results only
                recent_results = results_history[-6:]  # Show just the latest 6 results

                for i, result in enumerate(recent_results):
                    search_context += f"Result {i+1} (Query: '{result['query']}')\n"
                    search_context += f"URL: {result.get('url', 'No URL')}\n"
                    search_context += f"Summary: {result['content'][:2000]}...\n\n"

            # Include previous cycle summaries (last 3 only)
            if cycle_summaries:
                search_context += "\n### Previous cycle summaries:\n"
                for i, summary in enumerate(cycle_summaries[-3:]):
                    search_context += f"Cycle {cycle-3+i} Summary: {summary}\n\n"

            # Include identified research gaps from dimensional analysis
            research_dimensions = state.get("research_dimensions")
            if research_dimensions:
                gaps = await self.identify_research_gaps()
                if gaps:
                    search_context += "\n### Identified research gaps:\n"
                    for gap in gaps:
                        search_context += f"- Dimension {gap+1}\n"

            # Include previous comprehensive summary if this is a follow-up
            if is_follow_up and state.get("prev_comprehensive_summary"):
                search_context += "### Previous Research Summary:\n\n"
                summary_excerpt = state.get("prev_comprehensive_summary", "")[:5000]
                search_context += f"{summary_excerpt}...\n\n"

            # Generate new queries for this cycle
            query_objects = await self.improved_query_generation(
                user_message, priority_topics, search_context
            )

            # Extract query strings and topics
            current_cycle_queries = query_objects

            # Track topics used for queries to apply dampening in future cycles
            used_topics = [
                query_obj.get("topic", "")
                for query_obj in current_cycle_queries
                if query_obj.get("topic")
            ]
            await self.update_topic_usage_counts(used_topics)

            # Display the queries to the user
            await self.emit_message(f"### Research Cycle {cycle}: Search Queries\n\n")
            for i, query_obj in enumerate(current_cycle_queries):
                query = query_obj.get("query", "")
                topic = query_obj.get("topic", "")
                await self.emit_message(
                    f"**Query {i+1}**: {query}\n**Topic**: {topic}\n\n"
                )

            # Extract query strings for search history
            query_strings = [q.get("query", "") for q in current_cycle_queries]

            # Add queries to search history
            search_history.extend(query_strings)
            self.update_state("search_history", search_history)

            # Execute searches and process results SEQUENTIALLY
            cycle_results = []
            for query_obj in current_cycle_queries:
                query = query_obj.get("query", "")
                topic = query_obj.get("topic", "")

                # Get query embedding for content comparison
                try:
                    query_embedding = await self.get_embedding(query)
                    if not query_embedding:
                        query_embedding = [0] * 384  # Default embedding size
                except Exception as e:
                    logger.error(f"Error getting embedding: {e}")
                    query_embedding = [0] * 384  # Default embedding size

                # Apply semantic transformation if available
                semantic_transformations = state.get("semantic_transformations")
                if semantic_transformations:
                    transformed_query = await self.apply_semantic_transformation(
                        query_embedding, semantic_transformations
                    )
                    # Use transformed embedding if available
                    if transformed_query:
                        query_embedding = transformed_query

                # Process the query and get results
                results = await self.process_query(
                    query,
                    query_embedding,
                    outline_embedding,
                    None,
                    summary_embedding,
                )

                # Add successful results to the cycle results and history
                cycle_results.extend(results)
                results_history.extend(results)

            # Update in state
            self.update_state("results_history", results_history)

            # Step 6: Analyze results and update research outline
            if cycle_results:
                await self.emit_status(
                    "info",
                    "Analyzing search results and updating research outline...",
                    False,
                )
                analysis_prompt = {
                    "role": "system",
                    "content": f"""You are a post-grad researcher analyzing search results and updating a research outline.
Examine the search results and the current research outline to assess the state of research.
This is cycle {cycle} out of a maximum of {max_cycles} research cycles.
Determine which topics have been adequately addressed by the search results.
Update the research outline by classifying topics into different categories.

Topics should be classified as:
- COMPLETED: Topics that have been fully or reasonably addressed with researched information.
- PARTIAL: Topics that have minimal information and need more research. Don't let topics languish here!
  If one hasn't been addressed in a while, reconsider if it actually has been, or if it's possibly irrelevant.
- IRRELEVANT: Topics that are not actually relevant to the main query, are red herrings, 
  based on misidentified subjects, or are website artifacts rather than substantive topics.
  For example, mark as irrelevant any topics about unrelated subjects that were mistakenly
  included due to ambiguous terms, incorrect definitions for acronyms with multiple meanings,
  or page elements/advertisements from websites that don't relate to the actual query.
- NEW: New topics discovered in the search results that should be added to the research.
  Topics that feel like a logical extension of the user's line of questioning and direction of research, 
  or that are clearly important to a specific subject but aren't currently included, belong here.

Remember that the research ultimately aims to address the original query: "{user_message}".

Format your response as a valid JSON object with the following structure:
{{
  "completed_topics": ["Topic 1", "Subtopic 2.1"],
  "partial_topics": ["Topic 2"],
  "irrelevant_topics": ["Topic that's a distraction", "Misidentified subject"],
  "new_topics": ["New topic discovered"],
  "analysis": "Brief analysis of what we've learned so far with a focus on results from this past cycle"
}}""",
                }

                # Create a context with the current outline and search results
                analysis_context = "### Current Research Outline Topics:\n"
                analysis_context += "\n".join(
                    [f"- {topic}" for topic in active_outline]
                )
                analysis_context += "\n\n### Latest Search Results:\n\n"

                for i, result in enumerate(cycle_results):
                    analysis_context += f"Result {i+1} (Query: '{result['query']}')\n"
                    analysis_context += f"Title: {result['title']}\n"
                    analysis_context += f"Content: {result['content'][:2000]}...\n\n"

                # Include previous cycle summaries for continuity
                if cycle_summaries:
                    analysis_context += "\n### Previous cycle summaries:\n"
                    for i, summary in enumerate(cycle_summaries):
                        analysis_context += f"Cycle {i+1} Summary: {summary}\n\n"

                # Include lists of completed and irrelevant topics
                if completed_topics:
                    analysis_context += "\n### Already completed topics:\n"
                    for topic in completed_topics:
                        analysis_context += f"- {topic}\n"

                if irrelevant_topics:
                    analysis_context += "\n### Already identified irrelevant topics:\n"
                    for topic in irrelevant_topics:
                        analysis_context += f"- {topic}\n"

                # Include user preferences if applicable
                if (
                    self.valves.USER_PREFERENCE_THROUGHOUT
                    and state.get("user_preferences", {}).get("pdv") is not None
                ):
                    analysis_context += (
                        "\n### User preferences are being applied to research\n"
                    )

                analysis_messages = [
                    analysis_prompt,
                    {
                        "role": "user",
                        "content": f"Original query: {user_message}\n\n{analysis_context}\n\nAnalyze these results and update the research outline.",
                    },
                ]

                try:
                    analysis_response = await self.generate_completion(
                        self.get_research_model(), analysis_messages
                    )
                    analysis_content = analysis_response["choices"][0]["message"][
                        "content"
                    ]

                    # Extract JSON from response
                    analysis_json_str = analysis_content[
                        analysis_content.find("{") : analysis_content.rfind("}") + 1
                    ]
                    analysis_data = json.loads(analysis_json_str)

                    # Update completed topics
                    newly_completed = set(analysis_data.get("completed_topics", []))
                    completed_topics.update(newly_completed)
                    self.update_state("completed_topics", completed_topics)

                    # Update irrelevant topics
                    newly_irrelevant = set(analysis_data.get("irrelevant_topics", []))
                    irrelevant_topics.update(newly_irrelevant)
                    self.update_state("irrelevant_topics", irrelevant_topics)

                    # Add any new topics discovered
                    new_topics = analysis_data.get("new_topics", [])
                    for topic in new_topics:
                        if (
                            topic not in all_topics
                            and topic not in completed_topics
                            and topic not in irrelevant_topics
                        ):
                            active_outline.append(topic)
                            all_topics.append(topic)

                    # Update active outline by removing completed and irrelevant topics
                    active_outline = [
                        topic
                        for topic in active_outline
                        if topic not in completed_topics
                        and topic not in irrelevant_topics
                    ]

                    # Update in state
                    self.update_state("active_outline", active_outline)
                    self.update_state("all_topics", all_topics)

                    # Save the analysis summary
                    cycle_summaries.append(
                        analysis_data.get("analysis", f"Analysis for cycle {cycle}")
                    )
                    self.update_state("cycle_summaries", cycle_summaries)

                    # Create the current checklist for display to the user
                    current_checklist = {
                        "completed": newly_completed,
                        "partial": set(analysis_data.get("partial_topics", [])),
                        "irrelevant": newly_irrelevant,
                        "new": set(new_topics),
                        "remaining": set(active_outline),
                    }

                    # Display analysis to the user
                    analysis_text = f"### Research Analysis (Cycle {cycle})\n\n"
                    analysis_text += f"{analysis_data.get('analysis', 'Analysis not available.')}\n\n"

                    if newly_completed:
                        analysis_text += "**Topics Completed:**\n"
                        for topic in newly_completed:
                            analysis_text += f"✓ {topic}\n"
                        analysis_text += "\n"

                    if analysis_data.get("partial_topics"):
                        partial_topics = analysis_data.get("partial_topics")
                        analysis_text += "**Topics Partially Addressed:**\n"
                        # Show only first 5 partial topics
                        for topic in partial_topics[:5]:
                            analysis_text += f"⚪ {topic}\n"
                        # Add count of additional topics if there are more than 5
                        if len(partial_topics) > 5:
                            analysis_text += f"...and {len(partial_topics) - 5} more\n"
                        analysis_text += "\n"

                    # Add display for irrelevant topics
                    if newly_irrelevant:
                        analysis_text += "**Irrelevant/Distraction Topics:**\n"
                        for topic in newly_irrelevant:
                            analysis_text += f"✗ {topic}\n"
                        analysis_text += "\n"

                    if new_topics:
                        analysis_text += "**New Topics Discovered:**\n"
                        for topic in new_topics:
                            analysis_text += f"+ {topic}\n"
                        analysis_text += "\n"

                    if active_outline:
                        analysis_text += "**Remaining Topics:**\n"
                        for topic in active_outline[:5]:  # Show just the first 5
                            analysis_text += f"□ {topic}\n"
                        if len(active_outline) > 5:
                            analysis_text += f"...and {len(active_outline) - 5} more\n"
                        analysis_text += "\n"

                    # Store dimension coverage in state but don't display it during cycles
                    research_dimensions = state.get("research_dimensions")
                    if research_dimensions:
                        try:
                            # Store the coverage for later display at summary
                            state["latest_dimension_coverage"] = research_dimensions[
                                "coverage"
                            ].copy()
                            self.update_state(
                                "latest_dimension_coverage",
                                research_dimensions["coverage"],
                            )
                        except Exception as e:
                            logger.error(f"Error storing dimension coverage: {e}")

                    await self.emit_message(analysis_text)

                    # Update dimension coverage for each result to improve tracking
                    for result in cycle_results:
                        content = result.get("content", "")
                        if content:
                            # Use similarity to query as quality factor (0.5-1.0 range)
                            quality = 0.5
                            if "similarity" in result:
                                quality = 0.5 + result["similarity"] * 0.5
                            await self.update_dimension_coverage(content, quality)

                except Exception as e:
                    logger.error(f"Error analyzing results: {e}")
                    await self.emit_message(
                        f"### Research Progress (Cycle {cycle})\n\nContinuing research on remaining topics...\n\n"
                    )
                    # Mark one topic as completed to ensure progress
                    if active_outline:
                        # Find the most covered topic based on similarities to gathered results
                        topic_scores = {}

                        # Only attempt similarity analysis if we have results
                        if cycle_results:
                            for topic in active_outline:
                                topic_embedding = await self.get_embedding(topic)
                                if topic_embedding:
                                    # Calculate similarity to each result
                                    topic_score = 0.0
                                    for result in cycle_results:
                                        content = result.get("content", "")[
                                            :1000
                                        ]  # Use first 1000 chars
                                        content_embedding = await self.get_embedding(
                                            content
                                        )
                                        if content_embedding:
                                            sim = cosine_similarity(
                                                [topic_embedding], [content_embedding]
                                            )[0][0]
                                            topic_score += sim

                                    # Average the score
                                    if cycle_results:
                                        topic_score /= len(cycle_results)

                                    topic_scores[topic] = topic_score

                        # If we have scores, select the highest; otherwise just take the first one
                        if topic_scores:
                            completed_topic = max(
                                topic_scores.items(), key=lambda x: x[1]
                            )[0]
                            logger.info(
                                f"Selected most covered topic: {completed_topic} (score: {topic_scores[completed_topic]:.3f})"
                            )
                        else:
                            completed_topic = active_outline[0]
                            logger.info(
                                f"No similarity data available, selected first topic: {completed_topic}"
                            )

                        completed_topics.add(completed_topic)
                        self.update_state("completed_topics", completed_topics)

                        active_outline.remove(completed_topic)
                        self.update_state("active_outline", active_outline)

                        await self.emit_message(
                            f"**Topic Addressed:** {completed_topic}\n\n"
                        )
                        # Add minimal analysis to cycle summaries
                        cycle_summaries.append(f"Completed topic: {completed_topic}")
                        self.update_state("cycle_summaries", cycle_summaries)

            # Check termination criteria
            if not active_outline or active_outline == []:
                await self.emit_status(
                    "info", "All research topics have been addressed!", False
                )
                break

            if cycle >= min_cycles and len(completed_topics) / len(all_topics) > 0.7:
                await self.emit_status(
                    "info",
                    "Most research topics have been addressed. Finalizing...",
                    False,
                )
                break

            # Continue to next cycle if we haven't hit max_cycles
            if cycle >= max_cycles:
                await self.emit_status(
                    "info",
                    f"Maximum research cycles ({max_cycles}) reached. Finalizing...",
                    False,
                )
                break

            await self.emit_status(
                "info",
                f"Research cycle {cycle} complete. Moving to next cycle...",
                False,
            )

        # Apply stepped compression to all research results if enabled
        if self.valves.STEPPED_SYNTHESIS_COMPRESSION and len(results_history) > 2:
            await self.emit_status(
                "info", "Applying stepped compression to research results...", False
            )

            # Track token counts before compression
            total_tokens_before = 0
            for result in results_history:
                tokens = await self.count_tokens(result.get("content", ""))
                total_tokens_before += tokens

            # Apply stepped compression to results
            results_history = await self.apply_stepped_compression(
                results_history,
                query_embedding if "query_embedding" in locals() else None,
                summary_embedding,
            )

            # Calculate total tokens after compression
            total_tokens_after = sum(
                result.get("tokens", 0) for result in results_history
            )

            # Log token reduction
            token_reduction = total_tokens_before - total_tokens_after
            if total_tokens_before > 0:
                percent_reduction = (token_reduction / total_tokens_before) * 100
                logger.info(
                    f"Stepped compression: {total_tokens_before} → {total_tokens_after} tokens "
                    f"(saved {token_reduction} tokens, {percent_reduction:.1f}% reduction)"
                )

                await self.emit_status(
                    "info",
                    f"Compressed research results from {total_tokens_before} to {total_tokens_after} tokens",
                    False,
                )

        # Step 7: Generate refined synthesis outline
        await self.emit_status(
            "info", "Generating refined outline for synthesis...", False
        )

        synthesis_outline = await self.generate_synthesis_outline(
            research_outline, completed_topics, user_message, results_history
        )

        # If synthesis outline generation failed, use original
        if not synthesis_outline:
            synthesis_outline = research_outline

        # Step 8: Synthesize final answer with the selected model - Section by Section with citations
        await self.emit_synthesis_status(
            "Synthesizing comprehensive answer from research results..."
        )
        await self.emit_message("\n\n---\n\n### Research Complete\n\n")

        # Make sure dimensions data is up-to-date
        await self.update_research_dimensions_display()

        # Display the final research outline first
        await self.emit_message("### Final Research Outline\n\n")
        for topic_item in synthesis_outline:
            topic = topic_item["topic"]
            subtopics = topic_item.get("subtopics", [])

            await self.emit_message(f"**{topic}**\n")
            for subtopic in subtopics:
                await self.emit_message(f"- {subtopic}\n")
            await self.emit_message("\n")

        # Display research dimensions after the outline
        await self.emit_status(
            "info", "Displaying research dimensions coverage...", False
        )
        await self.emit_message("### Research Dimensions (Ordered)\n\n")

        research_dimensions = state.get("research_dimensions")
        latest_coverage = state.get("latest_dimension_coverage")

        if latest_coverage and research_dimensions:
            try:
                # Translate dimensions to words
                dimension_labels = await self.translate_dimensions_to_words(
                    research_dimensions, latest_coverage
                )

                # Sort dimensions by coverage (highest to lowest)
                sorted_dimensions = sorted(
                    dimension_labels, key=lambda x: x.get("coverage", 0), reverse=True
                )

                # Display dimensions without coverage percentages
                for dim in sorted_dimensions[:10]:  # Limit to top 10
                    await self.emit_message(f"- {dim.get('words', 'Dimension')}\n")

                await self.emit_message("\n")
            except Exception as e:
                logger.error(f"Error displaying final dimension coverage: {e}")
                await self.emit_message("*Error displaying research dimensions*\n\n")
        else:
            logger.warning("No research dimensions data available for display")
            await self.emit_message("*No research dimension data available*\n\n")

        # Determine which model to use for synthesis
        synthesis_model = self.get_synthesis_model()
        await self.emit_synthesis_status(
            f"Using {synthesis_model} for section generation..."
        )

        # Clear section content storage
        self.update_state("section_synthesized_content", {})
        self.update_state("subtopic_synthesized_content", {})
        self.update_state("section_sources_map", {})
        self.update_state("section_citations", {})

        # Initialize global citation map if not exists
        if "global_citation_map" not in state:
            self.update_state("global_citation_map", {})

        # Process each main topic and its subtopics
        compiled_sections = {}

        # Include only main topics that are not in irrelevant_topics
        relevant_topics = [
            topic
            for topic in synthesis_outline
            if topic["topic"] not in irrelevant_topics
        ]

        # If we have no relevant topics, use a simple structure
        if not relevant_topics:
            relevant_topics = [
                {"topic": "Research Summary", "subtopics": ["General Information"]}
            ]

        # Initialize _seen_sections and _seen_subtopics attributes
        self._seen_sections = set()
        self._seen_subtopics = set()

        # Generate content for each section with proper status updates
        all_verified_citations = []
        all_flagged_citations = []

        for topic_item in relevant_topics:
            section_title = topic_item["topic"]
            subtopics = [
                st
                for st in topic_item.get("subtopics", [])
                if st not in irrelevant_topics
            ]

            # Generate content for this section with inline citations (subtopic-based)
            section_data = await self.generate_section_content_with_citations(
                section_title,
                subtopics,
                user_message,
                results_history,
                synthesis_model,
                is_follow_up,
                state.get("prev_comprehensive_summary", "") if is_follow_up else "",
            )

            # Store in compiled sections
            compiled_sections[section_title] = section_data["content"]

            # Track citations for bibliography generation
            if "verified_citations" in section_data:
                all_verified_citations.extend(
                    section_data.get("verified_citations", [])
                )
            if "flagged_citations" in section_data:
                all_flagged_citations.extend(section_data.get("flagged_citations", []))

        # Store verification results for later use
        verification_results = {
            "verified": all_verified_citations,
            "flagged": all_flagged_citations,
        }
        self.update_state("verification_results", verification_results)

        # Process any non-standard citations that might still be in the text
        await self.emit_synthesis_status("Processing additional citation formats...")
        additional_citations = []
        master_source_table = state.get("master_source_table", {})
        global_citation_map = state.get("global_citation_map", {})

        for section_title, content in compiled_sections.items():
            # Use existing method to find non-standard citations
            section_citations = await self.identify_and_correlate_citations(
                section_title, content, master_source_table
            )

            if section_citations:
                # Add these citations to our tracking
                additional_citations.extend(section_citations)

                # Add to section citations tracking
                all_section_citations = state.get("section_citations", {})
                if section_title not in all_section_citations:
                    all_section_citations[section_title] = []
                all_section_citations[section_title].extend(section_citations)
                self.update_state("section_citations", all_section_citations)

                # Add URLs to global citation map
                for citation in section_citations:
                    url = citation.get("url", "")
                    if url and url not in global_citation_map:
                        global_citation_map[url] = len(global_citation_map) + 1

        # Update global citation map with any new URLs found
        self.update_state("global_citation_map", global_citation_map)

        # Generate bibliography from citation data
        await self.emit_synthesis_status("Generating bibliography...")
        bibliography_data = await self.generate_bibliography(
            master_source_table, global_citation_map
        )

        # Final pass to handle non-standard citations and apply strikethrough
        await self.emit_synthesis_status("Finalizing citation formatting...")
        for section_title, content in list(compiled_sections.items()):
            modified_content = content

            # Handle only non-standard citations (numeric ones were already processed)
            section_citations = [
                c for c in additional_citations if c.get("section") == section_title
            ]

            for citation in section_citations:
                url = citation.get("url", "")
                raw_text = citation.get("raw_text", "")

                if url and url in global_citation_map and raw_text:
                    global_id = global_citation_map[url]
                    # Replace the original citation text with the global ID
                    modified_content = modified_content.replace(
                        raw_text, f"[{global_id}]"
                    )

            # Update the original section content
            compiled_sections[section_title] = modified_content

        # Generate titles for the report
        await self.emit_synthesis_status("Generating report titles...")
        titles = await self.generate_titles(
            user_message, "".join(compiled_sections.values())
        )

        # After all sections are generated, perform synthesis review
        await self.emit_synthesis_status("Reviewing and improving the synthesis...")

        # Get synthesis review
        review_data = await self.review_synthesis(
            compiled_sections, user_message, synthesis_outline, synthesis_model
        )

        # Apply edits from review
        await self.emit_synthesis_status("Applying improvements to synthesis...")
        edited_sections, changes_made = await self.apply_review_edits(
            compiled_sections, review_data, synthesis_model
        )

        # Format the bibliography
        bibliography_table = await self.format_bibliography_list(
            bibliography_data["bibliography"]
        )

        # Generate abstract
        await self.emit_synthesis_status("Generating abstract...")
        abstract = await self.generate_abstract(
            user_message,
            "".join(edited_sections.values()),
            bibliography_data["bibliography"],
        )

        # Build final answer string
        comprehensive_answer = ""

        # Add title and subtitle
        main_title = titles.get("main_title", f"Research Report: {user_message}")
        subtitle = titles.get("subtitle", "A Comprehensive Analysis and Synthesis")

        comprehensive_answer += f"# {main_title}\n\n## {subtitle}\n\n"

        # Add abstract
        comprehensive_answer += f"## Abstract\n\n{abstract}\n\n"

        # Add introduction with compression
        await self.emit_synthesis_status("Generating introduction...")
        intro_prompt = {
            "role": "system",
            "content": f"""You are a post-grad research assistant writing an introduction for a research report in response to this query: "{user_message}".
                Create a concise introduction (2-3 paragraphs) that summarizes the purpose of the research and sets the stage for the report content.

                The introduction should:
                1. Set the stage for the subject matter and orient the reader toward what's to come.
                2. Introduce the research objective and original intent without simply restating the original query.
                3. Describe key details or aspects of the subject matter to be explored in the report.
            
                The introduction must NOT:
                1. Interpret the content in a lofty way that exaggerates its importance or profundity, or contrives a narrative with empty sophistication. 
                2. Attempt to portray the subject matter in any particular sort of light, good or bad, especially by using apologetic or dismissive language.
                3. Focus on perceived complexities or challenges related to the topic or research process, or include appeals to future research.
                
                The introduction should establish the context of the original query, state the research question, and briefly outline the approach taken to answering it. 
                Do not add your own bias or sentiment to the introduction. Do not include general statements about the research process itself.
                Please only respond with your introduction - do not include any segue, commentary, explanation, etc.""",
        }

        intro_context = f"Research Query: {user_message}\n\nResearch Outline:"
        for section in edited_sections:
            intro_context += f"\n- {section}"

        # Add compressed section content for better context
        section_context = "\n\nSection Content Summary:\n"
        for section_title, content in edited_sections.items():
            section_context += f"\n{section_title}: {content}...\n"

        # Compress the combined context
        combined_intro_context = intro_context + section_context
        intro_embedding = await self.get_embedding(combined_intro_context)
        compressed_intro_context = await self.compress_content_with_eigendecomposition(
            combined_intro_context, intro_embedding, None, None
        )

        intro_message = {"role": "user", "content": compressed_intro_context}

        try:
            # Use synthesis model for intro
            intro_response = await self.generate_completion(
                synthesis_model,
                [intro_prompt, intro_message],
                stream=False,
                temperature=self.valves.SYNTHESIS_TEMPERATURE * 0.83,
            )

            if (
                intro_response
                and "choices" in intro_response
                and len(intro_response["choices"]) > 0
            ):
                introduction = intro_response["choices"][0]["message"]["content"]
                comprehensive_answer += f"## Introduction\n\n{introduction}\n\n"
                await self.emit_synthesis_status("Introduction generation complete")
        except Exception as e:
            logger.error(f"Error generating introduction: {e}")
            comprehensive_answer += f"## Introduction\n\nThis research report addresses the query: '{user_message}'. The following sections present findings from a comprehensive investigation of this topic.\n\n"
            await self.emit_synthesis_status(
                "Introduction generation failed, using fallback"
            )

        # Add each section with heading
        for section_title, content in edited_sections.items():
            # Get token count for the section
            memory_stats = state.get("memory_stats", {})
            section_tokens = memory_stats.get("section_tokens", {})
            section_tokens_count = section_tokens.get(section_title, 0)
            if section_tokens_count == 0:
                section_tokens_count = await self.count_tokens(content)
                section_tokens[section_title] = section_tokens_count
                memory_stats["section_tokens"] = section_tokens
                self.update_state("memory_stats", memory_stats)

            # Check for section title duplication in various formats
            if (
                content.startswith(section_title)
                or content.startswith(f"# {section_title}")
                or content.startswith(f"## {section_title}")
            ):
                # Remove first line and any following whitespace
                content = (
                    content.split("\n", 1)[1].lstrip() if "\n" in content else content
                )

            comprehensive_answer += f"## {section_title}\n\n{content}\n\n"

        # Add conclusion with compression
        await self.emit_synthesis_status("Generating conclusion...")
        concl_prompt = {
            "role": "system",
            "content": f"""You are a post-grad research assistant writing a comprehensive conclusion for a research report in response to this query: "{user_message}".
                Create a concise conclusion (2-4 paragraphs) that synthesizes the key findings and insights from the research.
                
                The conclusion should:
                1. Restate the research objective and original intent from what has become a more knowing and researched standpoint.
                2. Highlight the most important research discoveries and their significance to the original topic and user query.
                3. Focus on the big picture characterizing the research and topic as a whole, using researched factual content as support.
                4. Definitively address the subject matter, focusing on what we know about it rather than what we don't.
                5. Acknowledge significant tangents in research, but ultimately remain focused on the original topic and user query.
            
                The conclusion must NOT:
                1. Interpret the content in a lofty way that exaggerates its importance or profundity, or contrives a narrative with empty sophistication. 
                2. Attempt to portray the subject matter in any particular sort of light, good or bad, especially by using apologetic or dismissive language.
                3. Focus on perceived complexities or challenges related to the topic or research process, or include appeals to future research.
                4. Ever take a preachy or moralizing tone, or take a "stance" for or against/"side" with or against anything not driven by the provided data.
                5. Overstate the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
                6. Sound to the reader as though it is overtly attempting to be diplomatic, considerate, enthusiastic, or overly-generalized.
    
                Please only respond with your conclusion - do not include any segue, commentary, explanation, etc.""",
        }

        concl_context = (
            f"Research Query: {user_message}\n\nKey findings from each section:\n"
        )

        # Use compression for each section based on the model's context window
        full_content = ""
        for section_title, content in edited_sections.items():
            full_content += f"\n## {section_title}\n{content}\n\n"

        # Get embedding for compression context
        content_embedding = await self.get_embedding(full_content[:2000])

        # Apply intelligent compression based on your existing logic
        compressed_content = await self.compress_content_with_eigendecomposition(
            full_content,
            content_embedding,
            None,  # No summary embedding needed
            None,  # Let the compression function decide the ratio based on content
        )

        concl_context += compressed_content

        concl_message = {"role": "user", "content": concl_context}

        try:
            # Use synthesis model for conclusion
            concl_response = await self.generate_completion(
                synthesis_model,
                [concl_prompt, concl_message],
                stream=False,
                temperature=self.valves.SYNTHESIS_TEMPERATURE,
            )

            if (
                concl_response
                and "choices" in concl_response
                and len(concl_response["choices"]) > 0
            ):
                conclusion = concl_response["choices"][0]["message"]["content"]
                comprehensive_answer += f"## Conclusion\n\n{conclusion}\n\n"
                await self.emit_synthesis_status("Conclusion generation complete")
        except Exception as e:
            logger.error(f"Error generating conclusion: {e}")
            await self.emit_synthesis_status(
                "Conclusion generation failed, using fallback"
            )

        # Add verification note if any citations were flagged
        comprehensive_answer = await self.add_verification_note(comprehensive_answer)

        # Add bibliography
        comprehensive_answer += f"{bibliography_table}\n\n"

        # Add research date
        comprehensive_answer += f"*Research conducted on: {self.research_date}*\n\n"

        # Count total tokens in the comprehensive answer
        synthesis_tokens = await self.count_tokens(comprehensive_answer)
        memory_stats = state.get("memory_stats", {})
        memory_stats["synthesis_tokens"] = synthesis_tokens
        self.update_state("memory_stats", memory_stats)

        # Calculate total tokens used in the research
        results_tokens = memory_stats.get("results_tokens", 0)
        section_tokens_sum = sum(memory_stats.get("section_tokens", {}).values())
        total_tokens = results_tokens + section_tokens_sum + synthesis_tokens
        memory_stats["total_tokens"] = total_tokens
        self.update_state("memory_stats", memory_stats)

        # Mark research as completed
        self.update_state("research_completed", True)

        # Store the comprehensive answer for potential follow-up queries
        self.update_state("prev_comprehensive_summary", comprehensive_answer)

        # Add token usage statistics
        token_stats = (
            f"\n\n---\n\n**Token Usage Statistics**\n\n"
            f"- Research Results: {results_tokens} tokens\n"
            f"- Final Synthesis: {synthesis_tokens} tokens\n"
            f"- Total: {total_tokens} tokens\n"
        )
        comprehensive_answer += token_stats

        # Share embedding cache stats
        cache_stats = self.embedding_cache.stats()
        logger.info(f"Embedding cache stats: {cache_stats}")

        # Export research data if enabled
        if self.valves.EXPORT_RESEARCH_DATA:
            try:
                await self.emit_status("info", "Formatting research data...", False)
                exported_data_string = await self.export_research_data()
                # Append the formatted data string to the final answer
                comprehensive_answer += f"\n\n{exported_data_string}"

            except Exception as e:
                logger.error(f"Error exporting research data: {e}")
                comprehensive_answer += (
                    "\n\n*An error occurred while exporting research data.*\n"
                )

        # Complete the process
        await self.emit_status("success", "Deep research complete!", True)

        # Return the final comprehensive answer string to be displayed in the UI
        return comprehensive_answer


class TrajectoryAccumulator:
    """Efficiently accumulates research trajectory across cycles"""

    def __init__(self, embedding_dim=384):
        self.query_sum = np.zeros(embedding_dim)
        self.result_sum = np.zeros(embedding_dim)
        self.count = 0
        self.embedding_dim = embedding_dim

    def add_cycle_data(self, query_embeddings, result_embeddings, weight=1.0):
        """Add data from a research cycle"""
        if not query_embeddings or not result_embeddings:
            return

        # Simple averaging of embeddings
        query_centroid = np.mean(query_embeddings, axis=0)
        result_centroid = np.mean(result_embeddings, axis=0)

        # Add to accumulators with weight
        self.query_sum += query_centroid * weight
        self.result_sum += result_centroid * weight
        self.count += 1

    def get_trajectory(self):
        """Get the current trajectory vector"""
        if self.count == 0:
            return None

        query_centroid = self.query_sum / self.count
        result_centroid = self.result_sum / self.count
        trajectory = result_centroid - query_centroid

        norm = np.linalg.norm(trajectory)
        if norm > 1e-10:
            return (trajectory / norm).tolist()
        else:
            return None
