import streamlit as st
from transformers import pipeline

st.title("African Multilingual Pretrained LLM (AfriBERTa Demo)")
st.write("Undergraduate Project Showcase: Using `castorini/afriberta_large` pretrained on 11 African languages.")

# Cache the model so it only loads once!
@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="castorini/afriberta_large")

with st.spinner("Loading AfriBERTa model..."):
    nlp_pipeline = load_model()

text = st.text_input(
    "Input Sentence (must contain <mask>)", 
    value="Mo fẹ́ràn lati kà <mask> gbogbo ọjọ́."
)

if st.button("Predict Masked Word"):
    if "<mask>" not in text:
        st.error("Please include '<mask>' in your sentence.")
    else:
        results = nlp_pipeline(text)
        st.subheader("Top Predictions:")
        for res in results:
            st.progress(res["score"], text=f"**{res['token_str']}** (Confidence: {res['score']:.2%})")