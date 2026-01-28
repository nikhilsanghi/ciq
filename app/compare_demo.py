"""
E-Commerce LLM Comparison Demo - Base vs Fine-tuned

Side-by-side comparison of base Mistral-7B-Instruct vs fine-tuned model.
Run both vLLM servers first:
  - Port 8000: Fine-tuned model
  - Port 8001: Base model

Usage:
    streamlit run app/compare_demo.py --server.port 8501 --server.address 0.0.0.0
"""

import streamlit as st
import requests
import time
from typing import Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
FINETUNED_URL = "http://localhost:8000/v1"
BASE_URL = "http://localhost:8001/v1"

# Task prefixes - must match training format
TASK_PREFIXES = {
    "classify": "[CLASSIFY]",
    "extract": "[EXTRACT]",
    "qa": "[QA]",
}

# Example prompts
EXAMPLES = {
    "classify": [
        "Apple MacBook Pro 14-inch M3 Pro, 18GB RAM, 512GB SSD, Space Gray",
        "Sony WH-1000XM5 Wireless Noise Canceling Bluetooth Headphones",
        "Instant Pot Duo 7-in-1 Electric Pressure Cooker, 6 Quart",
        "Nike Air Max 90 Men's Running Shoes - White/Black - Size 10",
        "LEGO Star Wars Millennium Falcon Building Kit 75375",
    ],
    "extract": [
        "Apple iPhone 15 Pro Max 256GB Natural Titanium - Unlocked",
        "Dell XPS 15 9530 - 13th Gen Intel Core i7, 16GB RAM, 512GB SSD, RTX 4050",
        "Samsung 65-inch Class OLED 4K S95D Smart TV (2024) with Dolby Atmos",
        "Levi's 501 Original Fit Men's Jeans - Dark Stonewash - 32W x 30L",
        "Dyson V15 Detect Cordless Vacuum - Yellow/Nickel - 60min runtime",
    ],
    "qa": [
        {
            "product": "MacBook Air M3 15-inch with 18-hour battery life, 8GB RAM, 256GB SSD",
            "question": "How long does the battery last?",
        },
        {
            "product": "Logitech MX Master 3S - Works with macOS, Windows, Linux via Bluetooth",
            "question": "Does this work with Mac?",
        },
        {
            "product": "Samsung 65-inch Class OLED 4K S95D Smart TV",
            "question": "What is the screen size?",
        },
        {
            "product": "Yeti Rambler 20 oz Tumbler, Stainless Steel, Vacuum Insulated",
            "question": "What material is this made of?",
        },
    ],
}


def format_prompt(task: str, product: str, question: str = None) -> str:
    """Format prompt with task prefix (matches training format)."""
    prefix = TASK_PREFIXES[task]

    if task == "classify":
        return f"{prefix} Classify into Google Product Taxonomy.\n\nProduct: {product}"
    elif task == "extract":
        return f"{prefix} Extract product attributes as JSON.\n\nProduct: {product}"
    elif task == "qa":
        return f"{prefix} Answer the question about this product.\n\nProduct: {product}\nQuestion: {question}"


def query_model(url: str, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> Dict[str, Any]:
    """Query a vLLM server."""
    try:
        start_time = time.time()

        # Get model ID from server
        try:
            models_resp = requests.get(f"{url}/models", timeout=5)
            if models_resp.status_code == 200:
                model_id = models_resp.json().get("data", [{}])[0].get("id", "unknown")
            else:
                model_id = "unknown"
        except:
            model_id = "unknown"

        response = requests.post(
            f"{url}/completions",
            json={
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["\n\n", "</s>", "Product:", "Question:", "[CLASSIFY]", "[EXTRACT]", "[QA]"],
            },
            timeout=60,
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "text": result["choices"][0]["text"].strip(),
                "time": elapsed,
                "model": model_id,
                "tokens": result.get("usage", {}).get("completion_tokens", 0),
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:100]}",
                "time": elapsed,
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Server not running",
            "time": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0,
        }


def query_both_models(prompt: str, max_tokens: int, temperature: float):
    """Query both models in parallel."""
    results = {"finetuned": None, "base": None}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(query_model, FINETUNED_URL, prompt, max_tokens, temperature): "finetuned",
            executor.submit(query_model, BASE_URL, prompt, max_tokens, temperature): "base",
        }

        for future in as_completed(futures):
            model_type = futures[future]
            results[model_type] = future.result()

    return results


def get_server_status(url: str) -> Dict[str, Any]:
    """Check server status."""
    try:
        response = requests.get(f"{url}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"connected": True, "models": models}
    except:
        pass
    return {"connected": False, "models": []}


def display_result(result: Dict[str, Any], task: str):
    """Display a model result."""
    if result["success"]:
        st.success(f"{result['time']:.2f}s | {result.get('tokens', '?')} tokens")

        # Parse JSON for extraction task
        if task == "extract":
            try:
                text = result["text"]
                if "{" in text:
                    json_start = text.find("{")
                    json_end = text.rfind("}") + 1
                    parsed = json.loads(text[json_start:json_end])
                    st.json(parsed)
                else:
                    st.code(text)
            except:
                st.code(result["text"])
        else:
            st.info(result["text"])
    else:
        st.error(f"Error: {result['error']}")


def main():
    st.set_page_config(
        page_title="Model Comparison",
        page_icon="",
        layout="wide",
    )

    st.title("Base vs Fine-tuned Model Comparison")
    st.markdown("""
    Compare **base Mistral-7B-Instruct** (port 8001) vs **fine-tuned model** (port 8000) side-by-side.

    Start both servers with: `./scripts/start_multi_model.sh`
    """)

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        task = st.selectbox(
            "Task",
            options=["classify", "extract", "qa"],
            format_func=lambda x: {
                "classify": "Classification",
                "extract": "Attribute Extraction",
                "qa": "Question Answering",
            }[x],
        )

        max_tokens = st.slider("Max Tokens", 50, 200, 100)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

        st.divider()

        st.subheader("Server Status")

        col1, col2 = st.columns(2)

        finetuned_status = get_server_status(FINETUNED_URL)
        base_status = get_server_status(BASE_URL)

        with col1:
            if finetuned_status["connected"]:
                st.success("Fine-tuned")
                st.caption(finetuned_status["models"][0][:30] + "..." if finetuned_status["models"] else "")
            else:
                st.error("Fine-tuned")

        with col2:
            if base_status["connected"]:
                st.success("Base")
                st.caption(base_status["models"][0][:30] + "..." if base_status["models"] else "")
            else:
                st.error("Base")

        if not (finetuned_status["connected"] and base_status["connected"]):
            st.warning("Start both servers:")
            st.code("./scripts/start_multi_model.sh", language="bash")

    # Main content
    st.divider()

    # Examples
    st.subheader("Quick Examples")
    examples = EXAMPLES[task]
    example_cols = st.columns(min(len(examples), 4))

    selected_example = None
    for i, ex in enumerate(examples[:4]):
        with example_cols[i]:
            label = ex if isinstance(ex, str) else ex.get("product", "")[:25] + "..."
            if st.button(label[:25], key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    st.divider()

    # Input
    if task == "qa":
        col1, col2 = st.columns([2, 1])
        with col1:
            product = st.text_area(
                "Product:",
                value=selected_example.get("product", "") if isinstance(selected_example, dict) else "",
                height=80,
            )
        with col2:
            question = st.text_input(
                "Question:",
                value=selected_example.get("question", "") if isinstance(selected_example, dict) else "",
            )
    else:
        product = st.text_area(
            "Product:",
            value=selected_example if isinstance(selected_example, str) else "",
            height=80,
        )
        question = None

    # Run comparison
    if st.button("Compare Models", type="primary", use_container_width=True):
        if not product:
            st.warning("Please enter a product.")
            return

        if task == "qa" and not question:
            st.warning("Please enter a question.")
            return

        prompt = format_prompt(task, product, question)

        with st.expander("Prompt (same for both models)", expanded=False):
            st.code(prompt)

        with st.spinner("Querying both models..."):
            results = query_both_models(prompt, max_tokens, temperature)

        # Display results side by side
        col_ft, col_base = st.columns(2)

        with col_ft:
            st.subheader("Fine-tuned Model")
            if results["finetuned"]:
                display_result(results["finetuned"], task)
            else:
                st.error("No response")

        with col_base:
            st.subheader("Base Model")
            if results["base"]:
                display_result(results["base"], task)
            else:
                st.error("No response")

        # Comparison summary
        st.divider()
        st.subheader("Comparison")

        if results["finetuned"]["success"] and results["base"]["success"]:
            ft_time = results["finetuned"]["time"]
            base_time = results["base"]["time"]

            time_diff = ((base_time - ft_time) / base_time) * 100 if base_time > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Fine-tuned Time", f"{ft_time:.2f}s")
            col2.metric("Base Time", f"{base_time:.2f}s")
            col3.metric("Time Difference", f"{abs(time_diff):.1f}%", delta=f"{'faster' if time_diff > 0 else 'slower'}")

    # Footer
    st.divider()
    st.caption("""
    **Tip:** Use temperature=0 for consistent, comparable outputs.
    Fine-tuned model should produce more structured outputs for e-commerce tasks.
    """)


if __name__ == "__main__":
    main()
