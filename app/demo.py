"""
Streamlit Demo App for E-Commerce LLM

This app demonstrates the fine-tuned model on ECInstruct tasks:
- Query-Product Relevance: Is this product relevant to this search query?
- Product Similarity: Are these two products similar?
- Document Q&A: Can this document answer this question?
"""

import streamlit as st
import requests
import time
from typing import Dict, Any
import json

# Configuration
VLLM_FINETUNED_URL = "http://localhost:8000/v1"

# Model config
MODEL = {
    "name": "Mistral-7B (Fine-tuned on ECInstruct)",
    "model_id": "ciq-model",
    "api_url": VLLM_FINETUNED_URL,
    "description": "Fine-tuned on 50K e-commerce examples from ECInstruct"
}

# Example prompts matching ECInstruct format
EXAMPLES = {
    "relevance": [
        {
            "title": "Gift Search",
            "query": "fathers day gift ideas",
            "product": "RAK Magnetic Wristband - Men's Tool Bracelet with 10 Strong Magnets to Hold Screws, Nails - Gift Ideas for Dad, Husband, Handyman"
        },
        {
            "title": "Electronics",
            "query": "wireless noise canceling headphones",
            "product": "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Bluetooth Headphones"
        },
        {
            "title": "Kitchen Appliance",
            "query": "instant pot pressure cooker",
            "product": "Ninja Foodi 10-in-1 Pressure Cooker and Air Fryer with Nesting Broil Rack"
        },
    ],
    "similarity": [
        {
            "title": "Glassware",
            "product1": "Duralex Amalfi Glasses 1004AC04/4, Set of 4, 6 oz.",
            "product2": "Duralex Made In France Provence Glass Tumbler (Set of 6), 3.125 oz, Clear"
        },
        {
            "title": "Laptops",
            "product1": "Apple MacBook Pro 14-inch M3 Pro, 18GB RAM, 512GB SSD",
            "product2": "Apple MacBook Air 13-inch M2, 8GB RAM, 256GB SSD"
        },
        {
            "title": "Different Products",
            "product1": "Sony WH-1000XM5 Wireless Headphones",
            "product2": "Apple iPhone 15 Pro Max 256GB"
        },
    ],
    "document_qa": [
        {
            "title": "Product Reviews",
            "question": "Is this product good for gifts?",
            "documents": [
                "Bought this for my dad and he loved it!",
                "Perfect gift for Father's Day",
                "My husband uses it every day in his workshop",
                "Great quality, would recommend as a gift"
            ]
        },
        {
            "title": "Battery Question",
            "question": "How long does the battery last?",
            "documents": [
                "The sound quality is amazing",
                "Very comfortable to wear",
                "Battery lasts about 30 hours on a single charge",
                "Noise cancellation is top notch"
            ]
        },
    ],
}


def format_relevance_prompt(query: str, product: str) -> str:
    """Format prompt for query-product relevance task."""
    instruction = "What is the relevance between the query and the product title below? Answer from one of the options."
    input_json = json.dumps({"query": query, "product title": product})
    return f"{instruction}\n{input_json}\n\nOptions:\nA: The product is relevant to the query, and satisfies all the query specifications.\nB: The product is somewhat relevant. It fails to fulfill some aspects of the query but can be used as a functional substitute.\nC: The product does not fulfill the query, but could be purchased together with a relevant product.\nD: The product is irrelevant to the query.\n\nAnswer:"


def format_similarity_prompt(product1: str, product2: str) -> str:
    """Format prompt for product similarity task."""
    instruction = "Analyze the titles of Product 1 and Product 2 to determine if they are similar, if they will be purchased or viewed together, and choose the corresponding option."
    input_json = json.dumps({"Product 1:": product1, "Product 2:": product2})
    return f"{instruction}\n{input_json}\n\nOptions:\nA: The products are similar and serve the same purpose.\nB: The products are related but serve different purposes.\nC: The products are complementary and might be purchased together.\nD: The products are completely unrelated.\n\nAnswer:"


def format_document_qa_prompt(question: str, documents: list) -> str:
    """Format prompt for document Q&A task."""
    instruction = "Output yes if the supporting document can answer the given question. Otherwise, output no."
    input_json = json.dumps({"question": question, "document": documents})
    return f"{instruction}\n{input_json}\n\nAnswer:"


def query_model(prompt: str, max_tokens: int = 100, temperature: float = 0.1) -> Dict[str, Any]:
    """Query the vLLM server."""
    try:
        start_time = time.time()

        response = requests.post(
            f"{MODEL['api_url']}/completions",
            json={
                "model": MODEL["model_id"],
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["\n\n", "</s>"],
            },
            timeout=60,
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "text": result["choices"][0]["text"].strip(),
                "time": elapsed_time,
                "tokens": result.get("usage", {}).get("completion_tokens", 0),
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "time": elapsed_time,
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to vLLM server. Make sure it's running.",
            "time": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0,
        }


def check_server_status() -> bool:
    """Check if vLLM server is running."""
    try:
        response = requests.get(f"{VLLM_FINETUNED_URL}/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    st.set_page_config(
        page_title="E-Commerce LLM Demo",
        page_icon="🛒",
        layout="wide",
    )

    st.title("🛒 E-Commerce LLM Demo")
    st.markdown("""
    This model was fine-tuned on **ECInstruct dataset** (50K examples) using QLoRA.

    **Tasks the model learned:**
    - **Query-Product Relevance**: Determine if a product matches a search query
    - **Product Similarity**: Analyze if two products are similar or related
    - **Document Q&A**: Check if reviews/documents can answer a question
    """)

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        task_type = st.selectbox(
            "Task Type",
            options=["relevance", "similarity", "document_qa"],
            format_func=lambda x: {
                "relevance": "🔍 Query-Product Relevance",
                "similarity": "🔄 Product Similarity",
                "document_qa": "📄 Document Q&A",
            }[x],
        )

        st.divider()

        max_tokens = st.slider("Max Tokens", 10, 200, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)

        st.divider()

        st.subheader("Server Status")
        if check_server_status():
            st.success("✅ Connected to vLLM")
        else:
            st.error("❌ vLLM server not running")

    # Main content
    col_input, col_examples = st.columns([2, 1])

    with col_examples:
        st.subheader("💡 Examples")
        examples = EXAMPLES[task_type]
        for ex in examples:
            if st.button(ex["title"], key=ex["title"]):
                st.session_state["example"] = ex
                st.rerun()

    with col_input:
        st.subheader("📝 Input")

        # Load example if selected
        example = st.session_state.get("example", None)
        if example and "example" in st.session_state:
            del st.session_state["example"]

        if task_type == "relevance":
            query = st.text_input(
                "Search Query:",
                value=example.get("query", "") if example else "",
                placeholder="e.g., wireless headphones for running"
            )
            product = st.text_area(
                "Product Title:",
                value=example.get("product", "") if example else "",
                height=100,
                placeholder="e.g., Sony WH-1000XM5 Wireless Noise Canceling Headphones"
            )

            if st.button("🔍 Check Relevance", type="primary", use_container_width=True):
                if query and product:
                    prompt = format_relevance_prompt(query, product)
                    with st.expander("📋 Formatted Prompt"):
                        st.code(prompt)

                    with st.spinner("Analyzing..."):
                        result = query_model(prompt, max_tokens, temperature)

                    if result["success"]:
                        st.success(f"✅ Response ({result['time']:.2f}s)")
                        st.info(result["text"])
                    else:
                        st.error(result["error"])
                else:
                    st.warning("Please enter both query and product.")

        elif task_type == "similarity":
            product1 = st.text_area(
                "Product 1:",
                value=example.get("product1", "") if example else "",
                height=80,
                placeholder="e.g., Apple MacBook Pro 14-inch M3"
            )
            product2 = st.text_area(
                "Product 2:",
                value=example.get("product2", "") if example else "",
                height=80,
                placeholder="e.g., Apple MacBook Air 13-inch M2"
            )

            if st.button("🔄 Compare Products", type="primary", use_container_width=True):
                if product1 and product2:
                    prompt = format_similarity_prompt(product1, product2)
                    with st.expander("📋 Formatted Prompt"):
                        st.code(prompt)

                    with st.spinner("Comparing..."):
                        result = query_model(prompt, max_tokens, temperature)

                    if result["success"]:
                        st.success(f"✅ Response ({result['time']:.2f}s)")
                        st.info(result["text"])
                    else:
                        st.error(result["error"])
                else:
                    st.warning("Please enter both products.")

        elif task_type == "document_qa":
            question = st.text_input(
                "Question:",
                value=example.get("question", "") if example else "",
                placeholder="e.g., Is this product good for gifts?"
            )
            documents_text = st.text_area(
                "Documents/Reviews (one per line):",
                value="\n".join(example.get("documents", [])) if example else "",
                height=150,
                placeholder="Enter product reviews or documents, one per line..."
            )

            if st.button("📄 Check Documents", type="primary", use_container_width=True):
                if question and documents_text:
                    documents = [d.strip() for d in documents_text.split("\n") if d.strip()]
                    prompt = format_document_qa_prompt(question, documents)
                    with st.expander("📋 Formatted Prompt"):
                        st.code(prompt)

                    with st.spinner("Analyzing..."):
                        result = query_model(prompt, max_tokens, temperature)

                    if result["success"]:
                        st.success(f"✅ Response ({result['time']:.2f}s)")
                        answer = result["text"].lower()
                        if "yes" in answer:
                            st.info("✅ YES - The documents can answer the question")
                        elif "no" in answer:
                            st.info("❌ NO - The documents cannot answer the question")
                        else:
                            st.info(result["text"])
                    else:
                        st.error(result["error"])
                else:
                    st.warning("Please enter question and documents.")

    # Footer
    st.divider()
    st.caption("""
    **About:** Fine-tuned Mistral-7B on ECInstruct dataset using QLoRA.
    Tasks include query-product relevance, product similarity, and document Q&A.

    Built with Streamlit + vLLM | [GitHub](https://github.com/nikhilsanghi/ciq)
    """)


if __name__ == "__main__":
    main()
