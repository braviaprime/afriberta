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

# Categorized Research Test Suite
examples = {
    "Yorùbá (West Africa)": ("Mo fẹ́ràn lati kà <mask > gbogbo ọjọ́.", "Yorùbá"),
    "Hausa (West Africa)": ("Yaro yana son <mask > ruwa.", "Hausa"),
    "Igbo (West Africa)": ("Obi na-asa <mask > m mma.", "Igbo"),
    "Swahili (East Africa)": ("Mtoto anapenda <mask > kitabu.", "Swahili"),
    "Amharic (Horn of Africa)": ("እባክዎን <mask > ስጠኝ ።", "Amharic")
}

# Sidebar Example Picker
st.sidebar.header("📋 Research Evaluation Presets")
selected_region = st.sidebar.selectbox("Select Language Family / Region:", list(examples.keys()))

# Extract sentence and language name from preset
default_sentence, source_lang_name = examples[selected_region]

# Input text box
input_text = st.text_area(
    "Input Sentence (must contain `<mask >` for the missing word):", 
    value=default_sentence, 
    height=100
)

# Helper function to scrub any accidental HTML styling tags
def clean_input(text):
    return (
        text.replace("<mask style=''>", "<mask >")
            .replace("<mask  style=''>", "<mask >")
            .replace("<mask >", "<mask >")
            .replace("[MASK]", "<mask >")
    )

# Helper function for reliable translation using Serverless Chat API
def translate_sentence(text, target_language):
    messages = [
        {
            "role": "system",
            "content": "You are an expert African language translator. Translate the given text accurately. Output ONLY the direct translation and nothing else. Do not add explanations or quotes."
        },
        {
            "role": "user",
            "content": f"Translate this sentence into {target_language}: {text}"
        }
    ]
    # Use a fast, reliable instruction model supported on the free serverless tier
    response = client.chat_completion(
        messages=messages,
        model="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=100,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- 4-WAY TRANSLATION HELPER ---
with st.expander("🌐 Translate Sentence Meanings (English, Yorùbá, Hausa, Igbo)"):
    st.markdown("Translate the input sentence across major Nigerian languages and English.")
    
    if st.button("Translate Sentence Across Nigerian Languages 🇳🇬"):
        if not HF_TOKEN:
            st.error("API Token missing! Please add `HF_TOKEN` inside your Streamlit App Secrets.")
        else:
            # Replace mask token with a blank line so it translates naturally
            readable_text = clean_input(input_text).replace("<mask >", "___")
            
            with st.spinner("Translating across English, Yorùbá, Hausa, and Igbo..."):
                try:
                    targets = ["English", "Yorùbá", "Hausa", "Igbo"]
                    cols = st.columns(4)
                    
                    for idx, lang_name in enumerate(targets):
                        with cols[idx]:
                            st.markdown(f"**{lang_name}**")
                            if lang_name.lower() == source_lang_name.lower():
                                st.info(readable_text)
                            else:
                                translated_text = translate_sentence(readable_text, lang_name)
                                st.success(translated_text)
                except Exception as e:
                    st.error(f"Translation Error: {e}")

st.divider()

# --- BENCHMARKING ENGINE ---
if st.button("Compare Model Understanding 🚀", type="primary"):
    standardized_text = clean_input(input_text)
    
    if "<mask >" not in standardized_text:
        st.error("Please include `<mask >` in your sentence to test predictions.")
    elif not HF_TOKEN:
        st.error("API Token missing! Please add `HF_TOKEN` inside your Streamlit App Secrets.")
    else:
        col1, col2 = st.columns(2)
        
        # 1. AfriBERTa Predictions (Requires literal: <mask >)
        with col1:
            st.subheader("🟢 AfriBERTa (Specialized African Model)")
            with st.spinner("Analyzing with AfriBERTa..."):
                try:
                    results = client.fill_mask(standardized_text, model="castorini/afriberta_large")
                    
                    for res in results[:5]:
                        st.write(f"**Word:** `{res['token_str']}`")
                        st.progress(float(res["score"]), text=f"Confidence: {res['score']:.1%}")
                except Exception as e:
                    st.error(f"AfriBERTa Error: {e}")

        # 2. Generic Multilingual Baseline (Requires literal: [MASK])
        with col2:
            st.subheader("🔵 mBERT (Generic Multilingual Baseline)")
            with st.spinner("Analyzing with Multilingual BERT..."):
                try:
                    bert_text = standardized_text.replace("<mask >", "[MASK]")
                    
                    results = client.fill_mask(
                        bert_text, 
                        model="google-bert/bert-base-multilingual-cased"
                    )
                    
                    for res in results[:5]:
                        st.write(f"**Word:** `{res['token_str']}`")
                        st.progress(float(res["score"]), text=f"Confidence: {res['score']:.1%}")
                except Exception as e:
                    st.error(f"Baseline Error: {e}")
