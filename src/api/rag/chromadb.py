"""
RAG (Retrieval-Augmented Generation) using ChromaDB.

Stores product information for Q&A enhancement.
Provides efficient similarity search for retrieving relevant
product context during question answering.
"""

import logging
import os
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "products")
DEFAULT_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./chroma_db")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# Global singleton instance
_rag_store_instance: Optional["RAGStore"] = None


class RAGStore:
    """
    RAG store using ChromaDB for vector storage and retrieval.

    Manages product embeddings and provides similarity search
    for retrieving relevant context during Q&A.

    Attributes:
        client: ChromaDB persistent client.
        collection: ChromaDB collection for product documents.
        embedder: SentenceTransformer model for text embeddings.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize the RAG store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_dir: Directory to persist the database.
            embedding_model: SentenceTransformer model name for embeddings.
        """
        logger.info(f"Initializing RAGStore with collection: {collection_name}")
        logger.info(f"Persist directory: {persist_dir}")

        # Ensure persist directory exists
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "E-commerce product documents for RAG"}
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        logger.info(
            f"RAGStore initialized. Collection has {self.collection.count()} documents"
        )

    def add_products(
        self,
        products: List[Dict],
        batch_size: int = 100,
    ) -> None:
        """
        Add products to the RAG store.

        Args:
            products: List of product dictionaries. Each product should have:
                - id: Unique identifier (optional, auto-generated if missing)
                - text: Product text content (required)
                - title: Product title (optional)
                - description: Product description (optional)
                - metadata: Additional metadata (optional)
            batch_size: Number of products to process in each batch.

        Raises:
            ValueError: If products list is empty or missing required fields.
        """
        if not products:
            logger.warning("No products to add")
            return

        logger.info(f"Adding {len(products)} products to RAG store")

        # Process in batches
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]

            ids = []
            documents = []
            metadatas = []

            for idx, product in enumerate(batch):
                # Generate ID if not provided
                product_id = product.get("id", f"product_{i + idx}")
                ids.append(str(product_id))

                # Build document text
                text_parts = []
                if "title" in product:
                    text_parts.append(f"Title: {product['title']}")
                if "text" in product:
                    text_parts.append(product["text"])
                elif "description" in product:
                    text_parts.append(product["description"])

                document = "\n".join(text_parts) if text_parts else ""
                documents.append(document)

                # Build metadata
                metadata = product.get("metadata", {})
                if "title" in product:
                    metadata["title"] = product["title"]
                if "category" in product:
                    metadata["category"] = product["category"]
                metadatas.append(metadata)

            # Generate embeddings
            embeddings = self.embedder.encode(documents).tolist()

            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.info(f"Added batch {i // batch_size + 1}: {len(batch)} products")

        logger.info(
            f"Successfully added products. Total count: {self.collection.count()}"
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        include_distances: bool = True,
    ) -> List[Dict]:
        """
        Search for similar products in the RAG store.

        Args:
            query: Search query text.
            n_results: Number of results to return.
            where: Optional filter conditions for metadata.
            include_distances: Whether to include similarity distances.

        Returns:
            List of dictionaries with search results, each containing:
                - id: Document ID
                - document: Document text
                - metadata: Document metadata
                - distance: Similarity distance (if include_distances=True)
        """
        if not query.strip():
            logger.warning("Empty search query")
            return []

        logger.debug(f"Searching for: {query[:50]}...")

        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()

        # Perform search
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=include,
        )

        # Format results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
                if include_distances and results.get("distances"):
                    result["distance"] = results["distances"][0][i]
                formatted_results.append(result)

        logger.debug(f"Found {len(formatted_results)} results")
        return formatted_results

    def build_context(
        self,
        query: str,
        n_results: int = 3,
        max_context_length: int = 2000,
    ) -> str:
        """
        Build context string from search results for prompt augmentation.

        Args:
            query: Search query text.
            n_results: Number of results to include in context.
            max_context_length: Maximum character length of context.

        Returns:
            Formatted context string for prompt augmentation.
        """
        results = self.search(query, n_results=n_results)

        if not results:
            return ""

        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            document = result.get("document", "")
            title = result.get("metadata", {}).get("title", f"Source {i}")

            # Format context entry
            entry = f"[{i}] {title}:\n{document}\n"

            # Check length limit
            if total_length + len(entry) > max_context_length:
                # Truncate if needed
                remaining = max_context_length - total_length
                if remaining > 100:  # Only add if meaningful space left
                    entry = entry[:remaining] + "..."
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            total_length += len(entry)

        return "\n".join(context_parts)

    def delete_products(self, ids: List[str]) -> None:
        """
        Delete products from the RAG store.

        Args:
            ids: List of product IDs to delete.
        """
        if not ids:
            return

        logger.info(f"Deleting {len(ids)} products from RAG store")
        self.collection.delete(ids=ids)
        logger.info(f"Deleted. Remaining count: {self.collection.count()}")

    def clear(self) -> None:
        """Clear all documents from the collection."""
        logger.warning("Clearing all documents from RAG store")
        # Delete and recreate collection
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "E-commerce product documents for RAG"}
        )
        logger.info("RAG store cleared")

    def get_stats(self) -> Dict:
        """
        Get statistics about the RAG store.

        Returns:
            Dictionary with collection statistics.
        """
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "embedding_model": self.embedder.get_sentence_embedding_dimension(),
        }


def get_rag_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> RAGStore:
    """
    Get or create the singleton RAGStore instance.

    Uses lazy initialization to create the RAG store on first access.
    Subsequent calls return the same instance.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory to persist the database.
        embedding_model: SentenceTransformer model name.

    Returns:
        RAGStore singleton instance.
    """
    global _rag_store_instance

    if _rag_store_instance is None:
        _rag_store_instance = RAGStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

    return _rag_store_instance


def reset_rag_store() -> None:
    """Reset the singleton RAGStore instance (useful for testing)."""
    global _rag_store_instance
    _rag_store_instance = None


def augment_prompt_with_rag(
    prompt: str,
    rag_store: RAGStore,
    n_results: int = 3,
) -> str:
    """
    Augment a prompt with relevant context from the RAG store.

    Searches the RAG store using the prompt and appends relevant
    context to enhance the model's response.

    Args:
        prompt: Original prompt text.
        rag_store: RAGStore instance to search.
        n_results: Number of context documents to retrieve.

    Returns:
        Augmented prompt with relevant context appended.
    """
    context = rag_store.build_context(prompt, n_results=n_results)

    if not context:
        return prompt

    augmented_prompt = f"""{prompt}

Relevant Context:
{context}

Based on the above context and your knowledge, please provide a helpful response."""

    return augmented_prompt
