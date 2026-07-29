import gradio as gr
import urllib.request
import json

# Endpoints for comparison
AFRIBERTA_URL = "https://api-inference.huggingface.co/models/castorini/afriberta_large"
XLMR_URL = "https://api-inference.huggingface.co/models/xlm-roberta-base"

def query_hf_api(api_url, text):
    """Helper function to send HTTP requests to Hugging Face APIs."""
    payload = json.dumps({"inputs": text}).encode("utf-8")
    req = urllib.request.Request(
        api_url, 
        data=payload, 
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def compare_models(text):
    if "<mask>" not in text and "<mask>" not in text:
        return (
            {"Error: Please include '<mask>' or '<mask>' in your sentence.": 1.0},
            {"Error: Please include '<mask>' or '<mask>' in your sentence.": 1.0}
        )
    
    # 1. Query AfriBERTa
    try:
        # Standardize mask token for AfriBERTa (<mask>)
        afri_text = text.replace("<mask>", "<mask>")
        res_afri = query_hf_api(AFRIBERTA_URL, afri_text)
        
        if isinstance(res_afri, dict) and "error" in res_afri:
            afri_out = {f"Model Loading... Retry in 10s: {res_afri['error']}": 1.0}
        else:
            afri_out = {res["token_str"]: res["score"] for res in res_afri}
    except Exception as e:
        afri_out = {f"AfriBERTa Error: {str(e)}": 1.0}

    # 2. Query XLM-RoBERTa (Baseline)
    try:
        # Standardize mask token for XLM-R (<mask>)
        xlm_text = text.replace("<mask>", "<mask>")
        res_xlm = query_hf_api(XLMR_URL, xlm_text)
        
        if isinstance(res_xlm, dict) and "error" in res_xlm:
            xlm_out = {f"Model Loading... Retry in 10s: {res_xlm['error']}": 1.0}
        else:
            xlm_out = {res["token_str"]: res["score"] for res in res_xlm}
    except Exception as e:
        xlm_out = {f"XLMR Error: {str(e)}": 1.0}

    return afri_out, xlm_out

# Categorized Research Test Suite
examples = [
    ["Yorùbá (West Africa)", "Mo fẹ́ràn lati kà <mask> gbogbo ọjọ́."],
    ["Hausa (West Africa)", "Yaro yana son <mask> ruwa."],
    ["Igbo (West Africa)", "Obi na-asa <mask style=''> m mma."],
    ["Swahili (East Africa)", "Mtoto anapenda <mask style=''> kitabu."],
    ["Amharic (Horn of Africa)", "እባክዎን <mask style=''> ስጠኝ ።"]
]

# Build Comparative UI
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🌍 African Language LLM Benchmarking Suite")
    gr.Markdown(
        "**Undergraduate Research Project:** Empirical evaluation of "
        "`castorini/afriberta_large` against generic multilingual baselines (`xlm-roberta-base`)."
    )
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Input African Language Sentence (use <mask style=''> for missing word)",
            placeholder="e.g., Mo fẹ́ràn lati kà <mask style=''> gbogbo ọjọ́.",
            lines=2
        )
    
    submit_btn = gr.Button("Compare Model Understanding", variant="primary")
    
    with gr.Row():
        output_afri = gr.Label(label="AfriBERTa (Specialized African Model)", num_top_classes=5)
        output_xlm = gr.Label(label="XLM-RoBERTa (Generic Multilingual Baseline)", num_top_classes=5)

    submit_btn.click(fn=compare_models, inputs=input_text, outputs=[output_afri, output_xlm])
    
    gr.Examples(
        examples=examples,
        inputs=[gr.Textbox(label="Language Region", visible=False), input_text],
        label="Preset Evaluation Sentences across African Language Families"
    )

if __name__ == "__main__":
    app.launch()
