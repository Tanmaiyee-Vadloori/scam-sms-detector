import streamlit as st
from datasets import load_dataset
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

st.set_page_config(page_title="Scam SMS Detector", page_icon="🛡️")
st.title("🛡️ Scam SMS Detector")
st.caption("A TF-IDF + logistic regression model, with an LLM that explains each verdict.")

@st.cache_resource
def train_model():
    df = load_dataset("ucirvine/sms_spam", split="train").to_pandas()
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(df["sms"], df["label"])
    return model

model = train_model()

try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    token = None

msg = st.text_area("Paste a message to check:")
if st.button("Check") and msg.strip():
    prob = model.predict_proba([msg])[0][1]
    if prob >= 0.5:
        st.error(f"⚠️ Likely spam ({prob:.0%})")
    else:
        st.success(f"✅ Looks safe ({1 - prob:.0%} confident)")

        if token:
        try:
            client = OpenAI(base_url="https://models.github.ai/inference", api_key=token)
            r = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a fraud analyst. In 2 sentences, explain whether this message looks like a scam and why."},
                    {"role": "user", "content": msg},
                ],
            )
            if isinstance(r, str):
                st.warning("The LLM service returned an unexpected response:")
                st.code(r[:300])
            else:
                st.info(r.choices[0].message.content)
        except Exception as e:
            st.warning(f"LLM explanation unavailable: {e}")