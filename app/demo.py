"""
Streamlit Demo App for E-Commerce LLM Comparison

This app allows users to compare:
- Base Mistral-7B model
- Fine-tuned Mistral-7B for e-commerce tasks
- (Later) Fine-tuned LLaMA-3-8B variants

Features:
- Side-by-side model comparison
- Task type selection (Classify, Extract, Q&A)
- Example prompts for quick testing
- Response time tracking
"""

import streamlit as st
import requests
import time
from typing import Optional, Dict, Any
import json

# Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"

# Available models (will be populated dynamically)
MODELS = {
    "mistral-base": {
        "name": "Mistral-7B (Base)",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Original Mistral-7B without fine-tuning"
    },
    "mistral-finetuned": {
        "name": "Mistral-7B (Fine-tuned)",
        "model_id": "ciq-model",
        "description": "Fine-tuned on 50K e-commerce examples"
    },
    # Future models
    # "llama-finetuned": {
    #     "name": "LLaMA-3-8B (Fine-tuned)",
    #     "model_id": "llama-ciq-model",
    #     "description": "Fine-tuned LLaMA-3 for e-commerce"
    # },
}

# Example prompts for each task type
EXAMPLE_PROMPTS = {
    "classify": [
        {
            "title": "Electronics - Headphones",
            "prompt": "Product: Sony WH-1000XM5 Wireless Noise Canceling Headphones with Auto Noise Canceling Optimizer",
        },
        {
            "title": "Apparel - Shoes",
            "prompt": "Product: Nike Air Max 90 Men's Running Shoes - White/Black - Size 10",
        },
        {
            "title": "Home & Kitchen",
            "prompt": "Product: Instant Pot Duo 7-in-1 Electric Pressure Cooker, 6 Quart, Stainless Steel",
        },
    ],
    "extract": [
        {
            "title": "Smartphone",
            "prompt": "Product: Apple iPhone 15 Pro Max 256GB Natural Titanium - Unlocked",
        },
        {
            "title": "Laptop",
            "prompt": "Product: Dell XPS 15 9530 Laptop - 13th Gen Intel Core i7-13700H, 16GB RAM, 512GB SSD, NVIDIA RTX 4050",
        },
        {
            "title": "Clothing",
            "prompt": "Product: Levi's 501 Original Fit Men's Jeans - Dark Stonewash - 32W x 30L",
        },
    ],
    "qa": [
        {
            "title": "Battery Life Question",
            "prompt": "Product: MacBook Air M3 15-inch with 18-hour battery life, 8GB RAM, 256GB SSD\nQuestion: How long does the battery last?",
        },
        {
            "title": "Compatibility Question",
            "prompt": "Product: Logitech MX Master 3S Wireless Mouse - Works with macOS, Windows, Linux, Chrome OS\nQuestion: Does this work with Mac?",
        },
        {
            "title": "Size Question",
            "prompt": "Product: Samsung 65-inch Class OLED 4K S95D Smart TV (2024)\nQuestion: What is the screen size?",
        },
    ],
}

# Task type configurations
TASK_CONFIGS = {
    "classify": {
        "prefix": "[CLASSIFY]",
        "instruction": "Classify the following product into the Google Product Taxonomy.",
        "suffix": "\nCategory:",
    },
    "extract": {
        "prefix": "[EXTRACT]",
        "instruction": "Extract product attributes as JSON from the following product.",
        "suffix": "\nAttributes:",
    },
    "qa": {
        "prefix": "[QA]",
        "instruction": "Answer the question about the following product.",
        "suffix": "\nAnswer:",
    },
}


def format_prompt(product_text: str, task_type: str) -> str:
    """Format the prompt based on task type."""
    config = TASK_CONFIGS[task_type]
    return f"{config['prefix']} {config['instruction']}\n\n{product_text}{config['suffix']}"


def query_model(
    model_id: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Query the vLLM server."""
    try:
        start_time = time.time()

        response = requests.post(
            f"{VLLM_BASE_URL}/completions",
            json={
                "model": model_id,
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
            "error": "Cannot connect to vLLM server. Make sure it's running on port 8000.",
            "time": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0,
        }


def get_available_models() -> list:
    """Check which models are available on the vLLM server."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except:
        pass
    return []


def main():
    # Page configuration
    st.set_page_config(
        page_title="E-Commerce LLM Demo",
        page_icon="🛒",
        layout="wide",
    )

    # Header
    st.title("🛒 E-Commerce LLM Comparison")
    st.markdown("""
    Compare the performance of base vs fine-tuned language models on e-commerce tasks.

    **Tasks supported:**
    - **Classify**: Categorize products into Google Product Taxonomy
    - **Extract**: Extract product attributes as structured JSON
    - **Q&A**: Answer questions about products
    """)

    st.divider()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Task type selection
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

        # Model selection
        st.subheader("Models to Compare")

        available_models = get_available_models()

        col1, col2 = st.columns(2)
        with col1:
            use_base = st.checkbox("Base Mistral", value=True)
        with col2:
            use_finetuned = st.checkbox("Fine-tuned", value=True)

        st.divider()

        # Generation parameters
        st.subheader("Generation Settings")
        max_tokens = st.slider("Max Tokens", 10, 200, 100)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)

        st.divider()

        # Server status
        st.subheader("Server Status")
        if available_models:
            st.success(f"✅ Connected ({len(available_models)} models)")
            with st.expander("Available Models"):
                for m in available_models:
                    st.code(m)
        else:
            st.error("❌ Cannot connect to vLLM server")
            st.caption("Start the server with: `./scripts/start_vllm.sh`")

    # Main content area
    col_input, col_examples = st.columns([2, 1])

    with col_input:
        st.subheader("📝 Input")

        # Text input
        user_input = st.text_area(
            "Enter product information:",
            height=150,
            placeholder="e.g., Product: Apple AirPods Pro 2nd Generation with USB-C",
        )

    with col_examples:
        st.subheader("💡 Examples")

        # Example buttons
        examples = EXAMPLE_PROMPTS[task_type]
        for example in examples:
            if st.button(example["title"], key=example["title"]):
                st.session_state["user_input"] = example["prompt"]
                st.rerun()

    # Check for session state input
    if "user_input" in st.session_state:
        user_input = st.session_state["user_input"]
        del st.session_state["user_input"]

    # Generate button
    st.divider()

    if st.button("🚀 Generate Responses", type="primary", use_container_width=True):
        if not user_input:
            st.warning("Please enter product information first.")
        else:
            # Format the prompt
            formatted_prompt = format_prompt(user_input, task_type)

            # Show formatted prompt
            with st.expander("📋 Formatted Prompt"):
                st.code(formatted_prompt)

            # Create columns for side-by-side comparison
            cols = []
            models_to_query = []

            if use_base:
                models_to_query.append(("mistral-base", MODELS["mistral-base"]))
            if use_finetuned:
                models_to_query.append(("mistral-finetuned", MODELS["mistral-finetuned"]))

            if not models_to_query:
                st.warning("Please select at least one model.")
            else:
                cols = st.columns(len(models_to_query))

                for i, (model_key, model_info) in enumerate(models_to_query):
                    with cols[i]:
                        st.subheader(model_info["name"])
                        st.caption(model_info["description"])

                        with st.spinner("Generating..."):
                            result = query_model(
                                model_info["model_id"],
                                formatted_prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )

                        if result["success"]:
                            st.success(f"✅ {result['time']:.2f}s")

                            # Display result based on task type
                            if task_type == "extract":
                                try:
                                    # Try to parse as JSON
                                    parsed = json.loads(result["text"])
                                    st.json(parsed)
                                except:
                                    st.code(result["text"])
                            else:
                                st.info(result["text"])

                            # Metrics
                            st.caption(f"Tokens: {result.get('tokens', 'N/A')}")
                        else:
                            st.error(f"❌ Error")
                            st.caption(result["error"])

    # Footer
    st.divider()
    st.caption("""
    **About this demo:**
    This app compares base and fine-tuned LLMs on e-commerce tasks.
    The fine-tuned model was trained on 50K examples from ECInstruct dataset using QLoRA.

    Built with Streamlit + vLLM | [GitHub](https://github.com/nikhilsanghi/ciq)
    """)


if __name__ == "__main__":
    main()
