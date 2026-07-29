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
(`castorini/afriberta_large`) against generic multilingual baselines (`google-bert/bert-base-multilingual-cased`).
""")

# Securely grab the API Token from Streamlit Secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# Initialize Hugging Face Client with authentication
client = InferenceClient(api_key=HF_TOKEN)

# Categorized Research Test Suite (Using <mask> as the visual standard)
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
    # Check if user included any common mask token
    has_mask = any(token in input_text for token in ["<mask>", "<mask>", "[MASK]"])
    
    if not has_mask:
        st.error("Please include `<mask>` in your sentence to test predictions.")
    elif not HF_TOKEN:
        st.error("API Token missing! Please add `HF_TOKEN` inside your Streamlit App Secrets.")
    else:
        col1, col2 = st.columns(2)
        
        # 1. AfriBERTa Predictions (Requires: <mask>)
        with col1:
            st.subheader("🟢 AfriBERTa (Specialized African Model)")
            with st.spinner("Analyzing with AfriBERTa..."):
                try:
                    # Force conversion to AfriBERTa's required token: <mask>
                    afri_text = input_text.replace("<mask>", "<mask>").replace("[MASK]", "<mask>").replace("<mask>", "<mask>")
                    
                    results = client.fill_mask(afri_text, model="castorini/afriberta_large")
                    
                    for res in results[:5]:
                        st.write(f"**Word:** `{res['token_str']}`")
                        st.progress(float(res["score"]), text=f"Confidence: {res['score']:.1%}")
                except Exception as e:
                    st.error(f"AfriBERTa Error: {e}")

        # 2. Generic Multilingual Baseline (Requires: [MASK])
        with col2:
            st.subheader("🔵 mBERT (Generic Multilingual Baseline)")
            with st.spinner("Analyzing with Multilingual BERT..."):
                try:
                    # Force conversion to BERT's required token: [MASK]
                    bert_text = input_text.replace("<mask>", "[MASK]").replace("<mask>", "[MASK]").replace("<mask>", "[MASK]")
                    
                    results = client.fill_mask(bert_text, model="google-bert/bert-base-multilingual-cased")
                    
                    for res in results[:5]:
                        st.write(f"**Word:** `{res['token_str']}`")
                        st.progress(float(res["score"]), text=f"Confidence: {res['score']:.1%}")
                except Exception as e:
                    st.error(f"Baseline Error: {e}")
