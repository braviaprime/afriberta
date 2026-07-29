import streamlit as st
from huggingface_hub import InferenceClient

# Page Configuration
st.set_page_config(
    page_title="African Language LLM Benchmarking", 
    page_icon="🌍", 
    layout="wide"
)

# Header & Research Description
st.title("🌍 African Multilingual LLM Benchmarking Suite")
st.markdown("""
**Undergraduate Research Project Showcase:** Empirical evaluation of specialized African language models 
(`castorini/afriberta_large`) against generic multilingual baselines (`xlm-roberta-base`).
""")

# Initialize the official Hugging Face Inference Client (No raw URLs needed!)
client = InferenceClient()

# Categorized Research Test Suite
examples = {
    "Yorùbá (West Africa)": "Mo fẹ́ràn lati kà <mask> gbogbo ọjọ́.",
    "Hausa (West Africa)": "Yaro yana son <mask> ruwa.",
    "Igbo (West Africa)": "Obi na-asa <mask> m mma.",
    "Swahili (East Africa)": "Mtoto anapenda <mask> kitabu.",
    "Amharic (Horn of Africa)": "እባክዎን <mask> ስጠኝ ።"
}

# Sidebar Example Picker
st.sidebar.header("📋 Research Evaluation Presets")
selected_region = st.sidebar.selectbox("Select Language Family / Region:", list(examples.keys()))

# Input text box
default_sentence = examples[selected_region]
input_text = st.text_area(
    "Input Sentence (must contain `<mask>` for the missing word):", 
    value=default_sentence, 
    height=100
)

# Benchmark Button
if st.button("Compare Model Understanding 🚀", type="primary"):
    if "<mask>" not in input_text and "<mask>" not in input_text:
        st.error("Please include `<mask>` in your sentence to test predictions.")
    else:
        # Create Side-by-Side Comparison Columns
        col1, col2 = st.columns(2)
        
        # 1. AfriBERTa Predictions
        with col1:
            st.subheader("🟢 AfriBERTa (Specialized African Model)")
            with st.spinner("Analyzing with AfriBERTa..."):
                try:
                    formatted_text = input_text.replace("<mask style=''>", "<mask>")
                    # Official Hub call for Fill-Mask
                    results = client.fill_mask(formatted_text, model="castorini/afriberta_large")
                    
                    for res in results[:5]:
                        st.write(f"**Word:** `{res['token_str']}`")
                        st.progress(float(res["score"]), text=f"Confidence: {res['score']:.1%}")
                except Exception as e:
                    st.error(f"AfriBERTa Error: {e}")

        # 2. XLM-RoBERTa Predictions
        with col2:
            st.subheader("🔵 XLM-RoBERTa (Generic Multilingual Baseline)")
            with st.spinner("Analyzing with XLM-RoBERTa..."):
                try:
                    formatted_text = input_text.replace("<mask style=''>", "<mask>")
                    # Official Hub call for Fill-Mask
                    results = client.fill_mask(formatted_text, model="xlm-roberta-base")
                    
                    for res in results[:5]:
                        st.write(f"**Word:** `{res['token_str']}`")
                        st.progress(float(res["score"]), text=f"Confidence: {res['score']:.1%}")
                except Exception as e:
                    st.error(f"XLM-RoBERTa Error: {e}")
