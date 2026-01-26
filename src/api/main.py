"""
FastAPI application for e-commerce LLM inference.

Provides endpoints for:
- /classify: Hierarchical product categorization
- /extract: Attribute-value extraction as JSON
- /qa: Product question answering with optional RAG
"""

import logging
import os
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .rag.chromadb import get_rag_store, augment_prompt_with_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "60.0"))

app = FastAPI(
    title="E-commerce LLM API",
    description="Product classification, attribute extraction, and Q&A",
    version="0.1.0",
)


# ============================================================================
# Request/Response Models
# ============================================================================


class ClassifyRequest(BaseModel):
    """Request model for product classification."""
    product_title: str = Field(..., description="Product title to classify")
    product_description: Optional[str] = Field(
        None, description="Optional product description for better accuracy"
    )


class ClassifyResponse(BaseModel):
    """Response model for product classification."""
    category: str = Field(..., description="Predicted product category")
    confidence: Optional[float] = Field(
        None, description="Confidence score (0-1) if available"
    )


class ExtractRequest(BaseModel):
    """Request model for attribute extraction."""
    product_text: str = Field(
        ..., description="Product text to extract attributes from"
    )


class ExtractResponse(BaseModel):
    """Response model for attribute extraction."""
    attributes: Dict[str, str] = Field(
        ..., description="Extracted attribute-value pairs"
    )


class QARequest(BaseModel):
    """Request model for product Q&A."""
    product_text: str = Field(..., description="Product information text")
    question: str = Field(..., description="Question about the product")
    use_rag: bool = Field(
        True, description="Whether to use RAG for enhanced context"
    )


class QAResponse(BaseModel):
    """Response model for product Q&A."""
    answer: str = Field(..., description="Generated answer")
    sources: Optional[List[str]] = Field(
        None, description="Source documents used (if RAG enabled)"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    vllm_connected: bool
    rag_available: bool


# ============================================================================
# vLLM Client Integration
# ============================================================================


async def call_vllm(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
) -> str:
    """
    Call the vLLM OpenAI-compatible API.

    Args:
        prompt: The input prompt to send to the model.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 for deterministic).
        stop: Optional stop sequences.

    Returns:
        Generated text response.

    Raises:
        HTTPException: If vLLM call fails.
    """
    payload = {
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop or [],
    }

    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{VLLM_BASE_URL}/completions",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except httpx.TimeoutException:
            logger.error("vLLM request timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Model inference timed out"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM HTTP error: {e.response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Model server error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model inference failed: {str(e)}"
            )


async def check_vllm_health() -> bool:
    """Check if vLLM server is reachable."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{VLLM_BASE_URL}/models")
            return response.status_code == 200
        except Exception:
            return False


# ============================================================================
# Prompt Templates
# ============================================================================


def build_classify_prompt(title: str, description: Optional[str] = None) -> str:
    """Build prompt for product classification."""
    product_info = title
    if description:
        product_info = f"{title}\n\nDescription: {description}"

    return f"""[CLASSIFY] Classify the following product into the most specific Google Product Taxonomy category.

Product: {product_info}

Respond with only the category path, e.g., "Electronics > Computers > Laptops"

Category:"""


def build_extract_prompt(product_text: str) -> str:
    """Build prompt for attribute extraction."""
    return f"""[EXTRACT] Extract all product attributes from the following text as JSON key-value pairs.

Product: {product_text}

Respond with a valid JSON object containing attribute names as keys and their values.
Example: {{"brand": "Nike", "size": "Large", "color": "Blue"}}

Attributes:"""


def build_qa_prompt(product_text: str, question: str, context: str = "") -> str:
    """Build prompt for product Q&A."""
    rag_section = ""
    if context:
        rag_section = f"\n\nAdditional Context:\n{context}\n"

    return f"""[QA] Answer the question about the product based on the provided information.

Product Information:
{product_text}
{rag_section}
Question: {question}

Provide a concise and accurate answer based only on the available information.

Answer:"""


# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/classify", response_model=ClassifyResponse)
async def classify_product(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify a product into Google Product Taxonomy categories.

    Uses hierarchical classification to determine the most specific
    category for the given product.
    """
    logger.info(f"Classifying product: {request.product_title[:50]}...")

    prompt = build_classify_prompt(
        request.product_title,
        request.product_description
    )

    response_text = await call_vllm(prompt, max_tokens=100, temperature=0.0)

    # Clean up response - extract just the category
    category = response_text.strip().split("\n")[0].strip()

    logger.info(f"Classification result: {category}")

    return ClassifyResponse(category=category, confidence=None)


@app.post("/extract", response_model=ExtractResponse)
async def extract_attributes(request: ExtractRequest) -> ExtractResponse:
    """
    Extract attribute-value pairs from product text.

    Returns a dictionary of extracted attributes as JSON.
    """
    import json

    logger.info(f"Extracting attributes from: {request.product_text[:50]}...")

    prompt = build_extract_prompt(request.product_text)

    response_text = await call_vllm(
        prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["\n\n"]
    )

    # Parse JSON response with error handling
    try:
        # Find JSON object in response
        response_text = response_text.strip()
        if response_text.startswith("```"):
            # Handle markdown code blocks
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        # Find the JSON object boundaries
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            attributes = json.loads(json_str)
        else:
            logger.warning("No JSON object found in response")
            attributes = {}

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        attributes = {}

    logger.info(f"Extracted {len(attributes)} attributes")

    return ExtractResponse(attributes=attributes)


@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest) -> QAResponse:
    """
    Answer a question about a product.

    Optionally uses RAG to retrieve relevant context from the
    product knowledge base for enhanced answers.
    """
    logger.info(f"Answering question: {request.question[:50]}...")

    context = ""
    sources = None

    # Use RAG if enabled
    if request.use_rag:
        try:
            rag_store = get_rag_store()
            search_results = rag_store.search(
                f"{request.product_text} {request.question}",
                n_results=3
            )

            if search_results:
                context = rag_store.build_context(
                    f"{request.product_text} {request.question}",
                    n_results=3
                )
                sources = [r.get("id", "unknown") for r in search_results]
                logger.info(f"RAG retrieved {len(search_results)} documents")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            # Continue without RAG context

    prompt = build_qa_prompt(request.product_text, request.question, context)

    response_text = await call_vllm(prompt, max_tokens=256, temperature=0.0)

    # Clean up response
    answer = response_text.strip()

    logger.info(f"Generated answer: {answer[:100]}...")

    return QAResponse(answer=answer, sources=sources)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health of the API and its dependencies.

    Returns status of vLLM connection and RAG availability.
    """
    vllm_connected = await check_vllm_health()

    rag_available = False
    try:
        rag_store = get_rag_store()
        rag_available = rag_store.collection is not None
    except Exception:
        pass

    status_str = "healthy" if vllm_connected else "degraded"

    return HealthResponse(
        status=status_str,
        vllm_connected=vllm_connected,
        rag_available=rag_available
    )


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting E-commerce LLM API...")
    logger.info(f"vLLM endpoint: {VLLM_BASE_URL}")
    logger.info(f"Model: {VLLM_MODEL}")

    # Check vLLM connection
    if await check_vllm_health():
        logger.info("vLLM server connected successfully")
    else:
        logger.warning("vLLM server not available - some endpoints may fail")

    # Initialize RAG store
    try:
        rag_store = get_rag_store()
        count = rag_store.collection.count()
        logger.info(f"RAG store initialized with {count} documents")
    except Exception as e:
        logger.warning(f"RAG store initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down E-commerce LLM API...")
