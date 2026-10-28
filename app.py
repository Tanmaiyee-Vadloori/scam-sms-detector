import streamlit as st
import pandas as pd
from datasets import load_dataset
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from openai import OpenAI

st.set_page_config(page_title="Scam SMS Detector", page_icon="🛡️")
st.title("🛡️ Scam SMS Detector")
st.caption("TF-IDF + logistic regression, compared before and after adding Indian scam data, with an LLM that explains each verdict.")


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


@st.cache_resource
def train_models():
    df = load_dataset("ucirvine/sms_spam", split="train").to_pandas()
    extra = pd.read_csv("indian_messages.csv")

    before = build_model()
    before.fit(df["sms"], df["label"])

    after = build_model()
    after.fit(
        pd.concat([df["sms"], extra["text"]]),
        pd.concat([df["label"], extra["label"]]),
    )
    return before, after


before, after = train_models()

try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    token = None

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Check a message ----------
msg = st.text_area("Paste a message to check:")
if st.button("Check") and msg.strip():
    p_before = before.predict_proba([msg])[0][1]
    p_after = after.predict_proba([msg])[0][1]

    if p_after >= 0.5:
        st.error(f"⚠️ Likely spam ({p_after:.0%})")
    else:
        st.success(f"✅ Looks safe ({1 - p_after:.0%} confident)")

    col1, col2 = st.columns(2)
    col1.metric("Before (original data)", f"{p_before:.0%} spam")
    col2.metric(
        "After (+ Indian data)",
        f"{p_after:.0%} spam",
        delta=f"{(p_after - p_before) * 100:+.0f} pts",
        delta_color="off",
    )

    st.session_state.history.append({
        "Message": msg[:50] + ("..." if len(msg) > 50 else ""),
        "Before %": round(p_before * 100),
        "After %": round(p_after * 100),
    })

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

# ---------- Session tracker ----------
if st.session_state.history:
    st.subheader("📈 Your checks: before vs after")
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist, hide_index=True, use_container_width=True)
    st.bar_chart(hist.set_index("Message")[["Before %", "After %"]], stack=False)
    if st.button("Clear tracker"):
        st.session_state.history = []
        st.rerun()

# ---------- Held-out evaluation ----------
with st.expander("📊 Before vs after on the held-out Indian test set"):
    test = pd.read_csv("indian_test.csv")
    pred_b = before.predict(test["text"])
    pred_a = after.predict(test["text"])

    rec_b, rec_a = recall_score(test["label"], pred_b), recall_score(test["label"], pred_a)
    prec_b = precision_score(test["label"], pred_b, zero_division=0)
    prec_a = precision_score(test["label"], pred_a, zero_division=0)

    c1, c2 = st.columns(2)
    c1.metric("Recall (scams caught)", f"{rec_a:.0%}", delta=f"{(rec_a - rec_b) * 100:+.0f} pts vs before")
    c2.metric("Precision (alerts that were real scams)", f"{prec_a:.0%}", delta=f"{(prec_a - prec_b) * 100:+.0f} pts vs before")
    st.caption("These 12 messages were never used for training, so this measures real improvement, not memorization.")

    counts = {0: 0, 1: 0}
    names = []
    for lbl in test["label"]:
        counts[lbl] += 1
        names.append(f"{'Scam' if lbl == 1 else 'Safe'} {counts[lbl]}")
    test["Name"] = names
    test["Before %"] = (before.predict_proba(test["text"])[:, 1] * 100).round()
    test["After %"] = (after.predict_proba(test["text"])[:, 1] * 100).round()

    st.bar_chart(test.set_index("Name")[["Before %", "After %"]], stack=False)
    st.dataframe(test[["Name", "text", "Before %", "After %"]], hide_index=True, use_container_width=True)