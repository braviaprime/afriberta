import streamlit as st
import urllib.request
import json
from huggingface_hub import InferenceClient

# Page Configuration
st.set_page_config(
    page_title="KWASU NLP Research: African Pretrained LLM", 
    page_icon="🎓", 
    layout="wide"
)

# =====================================================================
# ACADEMIC RESEARCH HEADER
# =====================================================================
st.markdown("""
<div style="background-color:#1E3A8A;padding:20px;border-radius:10px;color:white;text-align:center;margin-bottom:25px;">
    <h3 style="margin:0;color:#F3F4F6;font-weight:600;">KWARA STATE UNIVERSITY, MALETE</h3>
    <h5 style="margin:5px 0 10px 0;color:#93C5FD;font-weight:400;">Department of Computer Science</h5>
    <hr style="border-color:#3B82F6;margin:10px auto;width:60%;">
    <h2 style="margin:10px 0;color:white;font-size:24px;">
        Design and Implementation of an African Text-Based Multilingual Pretrained Large Language Model
    </h2>
    <p style="margin:0;font-size:15px;color:#E5E7EB;">
        <b>Supervised by:</b> Dr. R. M. Isiaka &nbsp;|&nbsp; <b>Research Showcase:</b> Empirical Evaluation & Benchmarking
    </p>
</div>
""", unsafe_allow_html=True)

# Securely grab the API Token from Streamlit Secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# Initialize Hugging Face Client with authentication
client = InferenceClient(api_key=HF_TOKEN)

# Categorized Research Test Suite
examples = {
    "Yorùbá (West Africa)": ("Mo fẹ́ràn lati kà <mask > gbogbo ọjọ́.", "yor_Latn"),
    "Hausa (West Africa)": ("Yaro yana son <mask > ruwa.", "hau_Latn"),
    "Igbo (West Africa)": ("Obi na-asa <mask > m mma.", "ibo_Latn"),
    "Swahili (East Africa)": ("Mtoto anapenda <mask > kitabu.", "swh_Latn"),
    "Amharic (Horn of Africa)": ("እባክዎን <mask > ስጠኝ ።", "amh_Ethi")
}

# Sidebar Example Picker & Metadata
st.sidebar.markdown("### 🏛️ Institution")
st.sidebar.info("**Kwara State University (KWASU)**\n\nDepartment of Computer Science")
st.sidebar.markdown("### 👨‍🏫 Project Supervisor")
st.sidebar.success("**Dr. R. M. Isiaka**")
st.sidebar.divider()

st.sidebar.header("📋 Evaluation Presets")
selected_region = st.sidebar.selectbox("Select Language Family / Region:", list(examples.keys()))

# Extract sentence and language code from preset
default_sentence, source_lang_code = examples[selected_region]

# Input text box
input_text = st.text_area(
    "Input Sentence (must contain `<mask >` for the missing word):", 
    value=default_sentence, 
    height=100
)

# Helper function to scrub HTML styling and fix trailing spaces inside mask tokens
def clean_input(text):
    return (
        text.replace("<mask style=''>", "<mask >")
            .replace("<mask  style=''>", "<mask >")
            .replace("<mask >", "<mask >")
            .replace("[MASK]", "<mask >")
    )

# Helper function for FREE, direct serverless HTTP translation (Bypasses provider routing errors)
def translate_sentence(text, src_code, tgt_code):
    api_url = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    payload = json.dumps({
        "inputs": text,
        "parameters": {"src_lang": src_code, "tgt_lang": tgt_code}
    }).encode("utf-8")
    
    req = urllib.request.Request(api_url, data=payload, headers=headers)
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        if isinstance(result, list) and len(result) > 0 and "translation_text" in result[0]:
            return result[0]["translation_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"Warming up ({result['error']})... Click again!"
        return str(result)

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
                    targets = {
                        "English": "eng_Latn",
                        "Yorùbá": "yor_Latn",
                        "Hausa": "hau_Latn",
                        "Igbo": "ibo_Latn"
                    }
                    cols = st.columns(4)
                    
                    for idx, (lang_name, tgt_code) in enumerate(targets.items()):
                        with cols[idx]:
                            st.markdown(f"**{lang_name}**")
                            if tgt_code == source_lang_code:
                                st.info(readable_text)
                            else:
                                translated_text = translate_sentence(readable_text, source_lang_code, tgt_code)
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

# =====================================================================
# ACADEMIC RESEARCH FOOTER
# =====================================================================
st.markdown("""
<hr style="margin-top:40px;margin-bottom:15px;">
<div style="text-align:center;color:#6B7280;font-size:13px;">
    <b>Undergraduate Research Project</b> &nbsp;|&nbsp; Kwara State University (KWASU), Malete &nbsp;|&nbsp; Department of Computer Science<br>
    <i>Project Title: Design and Implementation of an African Text-Based Multilingual Pretrained Large Language Model</i><br>
    Supervised by <b>Dr. R. M. Isiaka</b>
</div>
""", unsafe_allow_html=True)
