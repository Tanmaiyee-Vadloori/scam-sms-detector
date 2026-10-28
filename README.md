# 🛡️ Scam SMS Detector

A machine learning app that detects scam SMS messages, built during the AI/ML workshop at **Quark, BITS Pilani Goa**.

It uses a classic ML model (TF-IDF + logistic regression) for fast, private predictions, and an LLM to explain each verdict in plain English. It also shows what happens when a model meets data it wasn't trained on, and how to fix it.

**🔗 Live demo:** <your-app-url>.streamlit.app

---

## What you'll build

- A spam classifier trained on 5,574 real SMS messages
- An evaluation that shows why **accuracy alone is misleading**
- A test on **Indian scam messages** (KYC, UPI, fake jobs) that the original model has never seen
- An improved model trained with Indian examples, compared **before vs after** on a held-out test set
- A web app deployed for free, with a public link you can share with recruiters

## Tech stack

| Part | Tool |
|---|---|
| Language | Python |
| ML model | scikit-learn (TF-IDF + Logistic Regression) |
| Dataset | [SMS Spam Collection](https://huggingface.co/datasets/ucirvine/sms_spam) via Hugging Face `datasets` |
| LLM explanations | GitHub Models (`openai/gpt-4o-mini`) |
| Web app | Streamlit |
| Hosting | Streamlit Community Cloud (free) |
| Development | Google Colab + GitHub Codespaces |

## Project structure

```
scam-sms-detector/
├── app.py                 # The Streamlit web app
├── make_csv.py            # Generates the Indian training and test data
├── indian_messages.csv    # 24 Indian-style messages used for TRAINING
├── indian_test.csv        # 12 held-out messages used ONLY for TESTING
├── requirements.txt       # Python packages the app needs
├── .gitignore             # Keeps secrets out of GitHub
└── README.md              # This file
```

---

## Step 0: Create your accounts

You need these free accounts before starting:

1. **GitHub**: [github.com](https://github.com)
2. **Google account**: for Google Colab
3. **Streamlit Community Cloud**: [share.streamlit.io](https://share.streamlit.io), sign in with your GitHub account

---

## Step 1: Explore and train the model in Colab

Open [Google Colab](https://colab.research.google.com), create a new notebook, and run these cells one by one.

**Load the data**
```python
!pip install -q datasets
from datasets import load_dataset

df = load_dataset("ucirvine/sms_spam", split="train").to_pandas()
df = df.rename(columns={"sms": "text"})   # label: 0 = safe, 1 = spam
print(df.shape)
print(df["label"].value_counts(normalize=True))
```

**Split into training and test data**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
```

**The accuracy trap: a "model" that always says safe**
```python
import numpy as np
from sklearn.metrics import accuracy_score

print("Always-say-safe accuracy:", accuracy_score(y_test, np.zeros(len(y_test))))
```
> 💡 This scores about 87% while catching **zero** scams. When one class is rare, accuracy is misleading, so we use **precision** and **recall** instead.

**Train the real model**
```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
model.fit(X_train, y_train)
```

**Evaluate it**
```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=["safe", "spam"]))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["safe", "spam"]);
```

**See what the model learned**
```python
words = model.named_steps["tfidfvectorizer"].get_feature_names_out()
coefs = model.named_steps["logisticregression"].coef_[0]
print([words[i] for i in coefs.argsort()[-15:][::-1]])
```

**Reality check: Indian scams**
```python
tests = [
    "Dear customer, your bank KYC has expired. Update now at <link> or your account will be blocked today.",
    "Your parcel is held at customs. Pay Rs 49 via UPI to release it: <link>",
    "Part-time job: earn 5000/day by liking YouTube videos. WhatsApp HR now.",
    "Hey, are we still meeting at the library at 5?",
]
for msg in tests:
    print(f"{model.predict_proba([msg])[0][1]:5.0%} spam | {msg[:60]}")
```
> 💡 Some Indian scams slip through. The dataset is UK SMS from around 2012, with no UPI, KYC, or WhatsApp-job scams. This is called **data drift**.

---

## Step 2: Get the code into your own GitHub repo

**Option A (fastest):** click **Fork** at the top of this repository to get your own copy.

**Option B (build it yourself):** create a new public repository called `scam-sms-detector` and add each file from this repo using **Add file → Create new file**.

---

## Step 3: Create a GitHub token for the LLM

The app uses GitHub Models for free LLM explanations. It needs a token with minimal permissions.

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**
2. Click **Generate new token**
3. Under **Account permissions**, set **Models** to **Read-only**. Leave everything else as **No access**
4. Generate it and copy the token. You won't see it again

> 🔒 **Never paste this token into your code or commit it to GitHub.** We store it in secrets instead. Because the token can only read Models, even if it leaked it couldn't touch your repositories. This is the **principle of least privilege**.

---

## Step 4: Run it in GitHub Codespaces (recommended before deploying)

Testing before you deploy catches errors early.

1. In your repo, click **Code → Codespaces → Create codespace on main**
2. In the terminal, install the packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the data files:
   ```bash
   python make_csv.py
   ```
4. Add your token for local testing:
   ```bash
   mkdir -p .streamlit
   echo 'GITHUB_TOKEN = "paste_your_token_here"' > .streamlit/secrets.toml
   ```
5. Make sure the secrets file is never pushed. `.gitignore` should contain:
   ```
   .streamlit/secrets.toml
   ```
6. Run the app:
   ```bash
   streamlit run app.py
   ```
   Codespaces will offer to open it in a browser tab.

---

## Step 5: Deploy to Streamlit Community Cloud

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Scam SMS detector"
   git push
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**
3. Choose your repository, branch `main`, and main file `app.py`
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GITHUB_TOKEN = "paste_your_token_here"
   ```
5. Click **Deploy**. In a few minutes you'll have a live link

Every time you push to GitHub, the app redeploys automatically.

---

## Step 6: Test your app

Use these messages. None of them were used for training, so it's a fair test.

| Message | What it tests | Expected |
|---|---|---|
| `WINNER!! You have been selected to receive a £900 prize reward! To claim, call now. Claim code KL341. Valid 12 hours only.` | Original-style spam | Spam |
| `URGENT: Your bank account has been temporarily blocked. Complete KYC verification now to restore access: <link>` | Indian scam | Higher % after the fix |
| `Your traffic challan of Rs 2,000 is unpaid. Pay now to avoid court action: <link>` | Indian scam | Higher % after the fix |
| `Your son has been detained by police. Send Rs 50,000 immediately for bail. Do not tell anyone.` | Fear-based scam, no link | Hard for the ML model |
| `654321 is your OTP for payment of Rs 499. Do not share this OTP with anyone.` | Normal, but looks scary | Safe |
| `Rs 1,200 credited to A/c XX5678 by UPI. Available balance updated.` | Normal bank alert | Safe |
| `FREE pizza at the coding club meetup today, 6 PM, CC lab. Bring your laptop!` | Spam-like words, but genuine | Safe (watch for false alarms) |
| `Aapka bank account band ho jayega. Turant KYC update karein: <link>` | Hinglish | Likely confuses the model |

Then open **📊 Before vs after on the held-out Indian test set** in the app to see the overall improvement.

---

## How the app works

1. **Two models are trained when the app starts.** "Before" uses only the original dataset. "After" also includes 24 Indian-style messages. Training takes only a few seconds, so we train on startup instead of saving model files, which avoids version-mismatch errors between environments.
2. **The verdict comes from the "After" model.** The "Before" score is shown for comparison.
3. **The LLM explains the verdict** in two sentences. If the LLM service fails, the app shows a warning instead of crashing.
4. **The held-out test set (`indian_test.csv`) is never used for training.** It only measures whether the fix genuinely works.

---

## Stretch tasks

- [x] **Fix the drift**: add Indian examples, retrain, and compare before vs after on a held-out test set
- [ ] **Tune the threshold**: change the 0.5 cutoff and plot precision vs recall. Which matters more for scam detection?
- [ ] **Compare models**: try Naive Bayes and a linear SVM, and add a results table to this README
- [ ] **Add a feedback button**: let users mark a verdict as wrong, the first step toward collecting new training data
- [ ] **Add your own data**: write messages based on real scams you or your family received (remove personal details), add them to `make_csv.py`, and re-run it

---

## My results

*Fill this in with your own numbers.*

**Original test set**

| Metric | Score |
|---|---|
| Precision | |
| Recall | |

**Held-out Indian test set**

| Metric | Before | After |
|---|---|---|
| Recall (scams caught) | | |
| Precision | | |

**Screenshot:** *add a screenshot of the before vs after chart here*

**What I observed:** *one paragraph on data drift, which scams the model still misses, and why*

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `IndentationError: expected an indented block` | Lines inside `if`/`try` aren't indented | Indent with 4 spaces per level. Check with `python -m py_compile app.py` |
| `NameError: name 'pd' is not defined` | Missing import | Add `import pandas as pd` at the top of `app.py` |
| `FileNotFoundError: indian_messages.csv` | The CSV wasn't generated or pushed | Run `python make_csv.py`, then commit and push the CSV files |
| `AttributeError` on `r.choices` | The LLM service returned something unexpected | Check the base URL is exactly `https://models.github.ai/inference` and check your token |
| "LLM explanation unavailable" warning | Token missing, expired, or rate-limited | Check the secret name is `GITHUB_TOKEN` and the token has **Models: Read-only** |
| `Your branch and 'origin/main' have diverged` | You edited on both GitHub and Codespaces | Run `git pull --no-rebase --no-edit`, check `app.py`, then `git push` |

> 💡 `python -m py_compile app.py` catches syntax errors, but not missing imports. Always test with `streamlit run app.py` before pushing.

---

## Interview questions you should be able to answer

1. **Why not report accuracy?** Only about 13% of messages are spam. A model that always says "safe" gets around 87% accuracy and catches nothing.
2. **Why logistic regression instead of deep learning?** The dataset is small, it trains in seconds, it's interpretable, and a strong simple baseline should come first.
3. **Precision or recall, which matters more here?** Missing a scam costs more than a false alarm, so recall matters more, but too many false alarms and users stop trusting the app.
4. **Why did the model fail on Indian scams?** It was trained on old UK SMS. Indian scam patterns are different. That's data drift.
5. **How do you know your fix actually worked?** I evaluated on a held-out test set the model never trained on, and checked the original test set for regression.
6. **Why use both ML and an LLM?** The local model is fast, free, and private. The LLM is slower and costs money at scale, but can explain its reasoning.
7. **How did you handle the API key?** It's stored in secrets, never in code, and the token has minimum permissions.
8. **What would you do next?** Collect more real-world data, especially Hinglish and fear-based scams, retrain, and monitor performance over time.

---

## Credits and notes

- **Dataset:** SMS Spam Collection v.1 by Tiago A. Almeida and José María Gómez Hidalgo, via the UCI Machine Learning Repository and [Hugging Face](https://huggingface.co/datasets/ucirvine/sms_spam)
- **Indian messages** in `make_csv.py` are synthetic examples written for this project. They contain no real phone numbers, links, or personal data
- Built during the AI/ML workshop at **Quark, BITS Pilani Goa**, and extended by *<your name>*
- This is an educational project, not a production fraud-detection system. Don't rely on it for real financial decisions