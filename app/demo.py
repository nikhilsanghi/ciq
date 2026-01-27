"""
E-Commerce LLM Demo - Base Model Testing

Test Mistral-7B-Instruct on e-commerce tasks BEFORE fine-tuning.
This establishes a baseline to compare against after training.

Tasks:
1. Classification - Categorize products into Google Product Taxonomy
2. Extraction - Extract product attributes as JSON
3. Q&A - Answer questions about products
"""

import streamlit as st
import requests
import time
from typing import Dict, Any
import json

# Configuration
VLLM_URL = "http://localhost:8000/v1"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Base model, no fine-tuning

# Example prompts for each task
EXAMPLES = {
    "classify": [
        {
            "title": "Laptop",
            "product": "Apple MacBook Pro 14-inch M3 Pro, 18GB RAM, 512GB SSD, Space Gray"
        },
        {
            "title": "Headphones",
            "product": "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Bluetooth Headphones"
        },
        {
            "title": "Kitchen Appliance",
            "product": "Instant Pot Duo 7-in-1 Electric Pressure Cooker, 6 Quart, Stainless Steel"
        },
        {
            "title": "Shoes",
            "product": "Nike Air Max 90 Men's Running Shoes - White/Black - Size 10"
        },
    ],
    "extract": [
        {
            "title": "Smartphone",
            "product": "Apple iPhone 15 Pro Max 256GB Natural Titanium - Unlocked"
        },
        {
            "title": "Laptop",
            "product": "Dell XPS 15 9530 - 13th Gen Intel Core i7-13700H, 16GB RAM, 512GB SSD, NVIDIA RTX 4050, 15.6\" OLED Display"
        },
        {
            "title": "TV",
            "product": "Samsung 65-inch Class OLED 4K S95D Smart TV (2024) with Dolby Atmos"
        },
        {
            "title": "Jeans",
            "product": "Levi's 501 Original Fit Men's Jeans - Dark Stonewash - 32W x 30L"
        },
    ],
    "qa": [
        {
            "title": "Battery Question",
            "product": "MacBook Air M3 15-inch with 18-hour battery life, 8GB RAM, 256GB SSD",
            "question": "How long does the battery last?"
        },
        {
            "title": "Compatibility",
            "product": "Logitech MX Master 3S Wireless Mouse - Works with macOS, Windows, Linux, Chrome OS via Bluetooth or USB receiver",
            "question": "Does this work with Mac?"
        },
        {
            "title": "Size Question",
            "product": "Samsung 65-inch Class OLED 4K S95D Smart TV",
            "question": "What is the screen size?"
        },
        {
            "title": "Material Question",
            "product": "Yeti Rambler 20 oz Tumbler, Stainless Steel, Vacuum Insulated with MagSlider Lid",
            "question": "What material is this made of?"
        },
    ],
}


def format_classify_prompt(product: str) -> str:
    """Format prompt for product classification."""
    return f"""Classify the following product into the Google Product Taxonomy.
Provide the full category path (e.g., "Electronics > Computers > Laptops").

Product: {product}

Category:"""


def format_extract_prompt(product: str) -> str:
    """Format prompt for attribute extraction."""
    return f"""Extract the product attributes from the following product title and return them as JSON.
Include attributes like: brand, model, color, size, storage, material, etc.

Product: {product}

Attributes (JSON):"""


def format_qa_prompt(product: str, question: str) -> str:
    """Format prompt for product Q&A."""
    return f"""Answer the question about the following product. Be concise and accurate.

Product: {product}

Question: {question}

Answer:"""


def query_model(prompt: str, max_tokens: int = 150, temperature: float = 0.1) -> Dict[str, Any]:
    """Query the vLLM server."""
    try:
        start_time = time.time()

        # Try to get model list first to know which model to use
        try:
            models_response = requests.get(f"{VLLM_URL}/models", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                if models_data.get("data"):
                    model_id = models_data["data"][0]["id"]
                else:
                    model_id = MODEL_ID
            else:
                model_id = MODEL_ID
        except:
            model_id = MODEL_ID

        response = requests.post(
            f"{VLLM_URL}/completions",
            json={
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["\n\n", "</s>", "Product:", "Question:"],
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
                "model": model_id,
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
            "error": "Cannot connect to vLLM server. Start it with: python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3",
            "time": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0,
        }


def get_server_status() -> Dict[str, Any]:
    """Check vLLM server status and get model info."""
    try:
        response = requests.get(f"{VLLM_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"connected": True, "models": models}
    except:
        pass
    return {"connected": False, "models": []}


def main():
    st.set_page_config(
        page_title="E-Commerce LLM Demo",
        page_icon="🛒",
        layout="wide",
    )

    st.title("🛒 E-Commerce LLM Demo")
    st.markdown("""
    **Testing base Mistral-7B-Instruct** on e-commerce tasks (no fine-tuning yet).

    This establishes a baseline before training. Tasks:
    - **Classification**: Categorize products into Google Product Taxonomy
    - **Extraction**: Extract product attributes as structured JSON
    - **Q&A**: Answer questions about products
    """)

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        task_type = st.selectbox(
            "Task Type",
            options=["classify", "extract", "qa"],
            format_func=lambda x: {
                "classify": "🏷️ Classification",
                "extract": "📋 Attribute Extraction",
                "qa": "❓ Question Answering",
            }[x],
        )

        st.divider()

        max_tokens = st.slider("Max Tokens", 50, 300, 150)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)

        st.divider()

        st.subheader("Server Status")
        status = get_server_status()
        if status["connected"]:
            st.success("✅ Connected to vLLM")
            st.caption(f"Model: {status['models'][0] if status['models'] else 'Unknown'}")
        else:
            st.error("❌ vLLM not running")
            st.code("pkill -f vllm && python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3 --port 8000", language="bash")

    # Main content
    col_input, col_examples = st.columns([2, 1])

    with col_examples:
        st.subheader("💡 Examples")
        examples = EXAMPLES[task_type]
        for ex in examples:
            if st.button(ex["title"], key=ex["title"], use_container_width=True):
                st.session_state["example"] = ex
                st.rerun()

    with col_input:
        st.subheader("📝 Input")

        example = st.session_state.pop("example", None)

        if task_type == "classify":
            product = st.text_area(
                "Product to classify:",
                value=example.get("product", "") if example else "",
                height=100,
                placeholder="e.g., Apple MacBook Pro 14-inch M3 Pro..."
            )

            if st.button("🏷️ Classify Product", type="primary", use_container_width=True):
                if product:
                    prompt = format_classify_prompt(product)
                    with st.expander("📋 Prompt sent to model", expanded=False):
                        st.code(prompt)

                    with st.spinner("Classifying..."):
                        result = query_model(prompt, max_tokens, temperature)

                    if result["success"]:
                        st.success(f"✅ Response ({result['time']:.2f}s) - Model: {result.get('model', 'unknown')}")
                        st.info(result["text"])
                        st.caption(f"Tokens: {result.get('tokens', 'N/A')}")
                    else:
                        st.error(f"❌ Error: {result['error']}")
                else:
                    st.warning("Please enter a product to classify.")

        elif task_type == "extract":
            product = st.text_area(
                "Product to extract attributes from:",
                value=example.get("product", "") if example else "",
                height=100,
                placeholder="e.g., Dell XPS 15 - Intel i7, 16GB RAM..."
            )

            if st.button("📋 Extract Attributes", type="primary", use_container_width=True):
                if product:
                    prompt = format_extract_prompt(product)
                    with st.expander("📋 Prompt sent to model", expanded=False):
                        st.code(prompt)

                    with st.spinner("Extracting..."):
                        result = query_model(prompt, max_tokens, temperature)

                    if result["success"]:
                        st.success(f"✅ Response ({result['time']:.2f}s) - Model: {result.get('model', 'unknown')}")
                        # Try to parse as JSON for nice display
                        try:
                            # Find JSON in response
                            text = result["text"]
                            if "{" in text:
                                json_start = text.find("{")
                                json_end = text.rfind("}") + 1
                                json_str = text[json_start:json_end]
                                parsed = json.loads(json_str)
                                st.json(parsed)
                            else:
                                st.code(text)
                        except:
                            st.code(result["text"])
                        st.caption(f"Tokens: {result.get('tokens', 'N/A')}")
                    else:
                        st.error(f"❌ Error: {result['error']}")
                else:
                    st.warning("Please enter a product.")

        elif task_type == "qa":
            product = st.text_area(
                "Product:",
                value=example.get("product", "") if example else "",
                height=80,
                placeholder="e.g., Sony WH-1000XM5 with 30-hour battery..."
            )
            question = st.text_input(
                "Question:",
                value=example.get("question", "") if example else "",
                placeholder="e.g., How long does the battery last?"
            )

            if st.button("❓ Answer Question", type="primary", use_container_width=True):
                if product and question:
                    prompt = format_qa_prompt(product, question)
                    with st.expander("📋 Prompt sent to model", expanded=False):
                        st.code(prompt)

                    with st.spinner("Answering..."):
                        result = query_model(prompt, max_tokens, temperature)

                    if result["success"]:
                        st.success(f"✅ Response ({result['time']:.2f}s) - Model: {result.get('model', 'unknown')}")
                        st.info(result["text"])
                        st.caption(f"Tokens: {result.get('tokens', 'N/A')}")
                    else:
                        st.error(f"❌ Error: {result['error']}")
                else:
                    st.warning("Please enter both product and question.")

    # Footer
    st.divider()
    st.caption("""
    **About:** Testing base Mistral-7B-Instruct (no fine-tuning) on e-commerce tasks.
    This establishes a baseline before training on MAVE/AmazonQA datasets.

    Built with Streamlit + vLLM | [GitHub](https://github.com/nikhilsanghi/ciq)
    """)


if __name__ == "__main__":
    main()
