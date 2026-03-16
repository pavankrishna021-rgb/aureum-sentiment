import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="AUREUM Sentiment Engine", page_icon="📊", layout="wide")

st.markdown("""<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
.stApp{background-color:#0a0a0f;}</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def analyse(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    labels = ["Positive", "Negative", "Neutral"]
    idx = torch.argmax(probs).item()
    score = probs[0].item() - probs[1].item()
    return {
        "sentiment": labels[idx],
        "confidence": probs[idx].item(),
        "positive": probs[0].item(),
        "negative": probs[1].item(),
        "neutral": probs[2].item(),
        "score": score
    }

st.markdown('<h1 style="text-align:center;color:white">AUREUM <span style="color:#d4a04a">Sentiment Engine</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:rgba(255,255,255,0.4)">FinBERT Transformer | 110M Parameters | Financial News Analysis</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Analysis", "Batch Analysis"])

with tab1:
    text = st.text_input("Enter a financial headline:", "Federal Reserve signals interest rate cuts amid cooling inflation")
    if text:
        r = analyse(text)
        c1, c2, c3 = st.columns(3)
        emoji = "🟢" if r["sentiment"] == "Positive" else "🔴" if r["sentiment"] == "Negative" else "🟡"
        c1.metric("Sentiment", emoji + " " + r["sentiment"])
        c2.metric("Confidence", str(round(r["confidence"] * 100, 1)) + "%")
        c3.metric("Score", str(round(r["score"], 3)))
        st.progress(r["positive"], text="Positive: " + str(round(r["positive"], 2)))
        st.progress(r["negative"], text="Negative: " + str(round(r["negative"], 2)))
        st.progress(r["neutral"], text="Neutral: " + str(round(r["neutral"], 2)))

with tab2:
    headlines = st.text_area("Enter headlines (one per line):",
        "NVIDIA beats earnings expectations with record AI chip revenue\nUK inflation falls to 2.6 percent\nTesla recalls 200K vehicles over safety concerns\nGoldman Sachs upgrades semiconductor sector\nUS-China trade tensions escalate with new tariff threats")
    if headlines:
        lines = [l.strip() for l in headlines.split("\n") if l.strip()]
        results = [analyse(l) for l in lines]

        pos = sum(1 for r in results if r["sentiment"] == "Positive")
        neg = sum(1 for r in results if r["sentiment"] == "Negative")
        neu = sum(1 for r in results if r["sentiment"] == "Neutral")
        avg = np.mean([r["score"] for r in results])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall", "🟢 BULLISH" if avg > 0 else "🔴 BEARISH")
        c2.metric("Positive", pos)
        c3.metric("Negative", neg)
        c4.metric("Neutral", neu)

        for line, r in zip(lines, results):
            emoji = "🟢" if r["sentiment"] == "Positive" else "🔴" if r["sentiment"] == "Negative" else "🟡"
            conf = str(round(r["confidence"] * 100)) + "%"
            sc = str(round(r["score"], 3))
            st.markdown(emoji + " *" + line + "* — " + r["sentiment"] + " (" + conf + ") Score: " + sc)

st.markdown("---")
st.markdown('<p style="text-align:center;color:rgba(255,255,255,0.15);font-size:11px">AUREUM Sentiment Engine v1.0 | FinBERT Transformer | Built by Pavan Krishna</p>', unsafe_allow_html=True)
