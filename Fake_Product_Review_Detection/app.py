import streamlit as st
import pandas as pd
import joblib
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix

# ============================ PAGE CONFIG ============================
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)

# ============================ DARK THEME CSS ============================
st.markdown("""
<style>
    .stApp { 
        background-color:#1A1D21 !important;
        color:white !important;
        font-family:'Inter', sans-serif;
    }
    textarea, input, select, .stSelectbox div[class*="css"] {
        background-color:#2A2D31 !important;
        color:white !important;
        border:1px solid #3A3D42 !important;
        border-radius:6px !important;
    }
    .stButton>button {
        background-color:#2563EB !important;
        color:white !important;
        border-radius:6px !important;
        padding:10px 18px !important;
        border:none;
        font-weight:600;
    }
    .stButton>button:hover {
        background-color:#1E4FC9 !important;
    }
    .result-card {
        background-color:#232629 !important;
        border:1px solid #3A3D42 !important;
        padding:20px;
        border-radius:10px;
        margin-top:25px;
    }
    .stTabs [role="tab"] {
        background:#2A2D31 !important;
        color:white !important;
        border-radius:6px !important;
        padding:10px 18px !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background:#2563EB !important;
        color:white !important;
        font-weight:600;
    }
</style>
""", unsafe_allow_html=True)

# ============================ LOAD MODEL ============================
model = joblib.load("models/lr_tfidf_model.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")

# NLP resources
nltk.download("stopwords")
nltk.download("vader_lexicon")

stop = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()


# ============================ HELPERS ============================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop]
    return " ".join(tokens)


def explain_fake_reason(review: str):
    review_lower = review.lower()
    reasons = []

    exaggerated_words = [
        "amazing", "incredible", "unbelievable", "perfect",
        "best", "superb", "wonderful", "flawless",
        "must buy", "highly recommend", "life changing"
    ]
    found = [w for w in exaggerated_words if w in review_lower]
    if found:
        reasons.append("Exaggerated promotional words: " + ", ".join(found))

    if review.count("!") >= 2:
        reasons.append("Excessive exclamation marks")

    if any(w.isupper() and len(w) > 3 for w in review.split()):
        reasons.append("Contains ALL CAPS words")

    if len(review.split()) < 6:
        reasons.append("Very short / low detail")

    if len(review_lower.split()) != len(set(review_lower.split())):
        reasons.append("Repetitive wording detected")

    return reasons if reasons else ["Writing pattern suggests possible fake review"]


def get_sentiment_label(text: str) -> str:
    scores = sia.polarity_scores(text)
    comp = scores["compound"]
    if comp >= 0.05:
        return "Positive"
    elif comp <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# ============================ TITLE ============================
st.title("Fake Review Detection System")
st.write("A professional AI-powered tool for identifying fake or suspicious product reviews.")


# ============================ TABS ============================
tab_single, tab_bulk, tab_dash = st.tabs(
    ["🔍 Analyze Single Review", "📄 Analyze Multiple Reviews", "📊 Dashboard & Analytics"]
)

df = None  # will store bulk data for dashboard


# ============================ TAB 1: SINGLE REVIEW ============================
with tab_single:
    st.subheader("Product Information")

    col1, col2 = st.columns(2)
    with col1:
        product_title = st.text_input("Product Title", placeholder="e.g., Wireless Earbuds")
    with col2:
        category = st.selectbox(
            "Product Category",
            ["Electronics", "Home & Kitchen", "Beauty", "Clothing", "Books",
             "Sports", "Groceries", "Toys", "Automotive", "Other"]
        )
        if category == "Other":
            category = st.text_input("Specify Category")

    use_rating = st.checkbox("Add Rating")
    rating = st.number_input("Rating (1–5)", 1.0, 5.0, 4.0, 0.1) if use_rating else None

    review_text = st.text_area("Enter Review Text", height=160)

    colA, colB = st.columns(2)
    with colA:
        analyze = st.button("Analyze Review")
    with colB:
        if st.button("Clear Inputs"):
            st.experimental_rerun()

    if analyze:
        if not review_text.strip():
            st.warning("Enter a review.")
        else:
            cleaned = clean_text(review_text)
            X = tfidf.transform([cleaned])
            prob_fake = model.predict_proba(X)[0][1]
            pred = "Fake Review" if prob_fake >= 0.5 else "Real Review"

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader(f"{'🟥' if pred == 'Fake Review' else '🟩'} {pred}")

            st.write(f"**Product:** {product_title or 'N/A'}")
            st.write(f"**Category:** {category}")
            st.write(f"**Rating:** {rating if rating else 'Not provided'}")

            st.write(f"**Confidence (Fake):** {prob_fake*100:.2f}%")
            st.progress(float(prob_fake))

            st.write(f"**Sentiment:** {get_sentiment_label(review_text)}")

            if pred == "Fake Review":
                st.write("### Fake Review Indicators")
                for r in explain_fake_reason(review_text):
                    st.write(f"- {r}")

            st.markdown("</div>", unsafe_allow_html=True)


# ============================ TAB 2: BULK ANALYSIS ============================
with tab_bulk:
    st.subheader("Upload CSV for Bulk Analysis (Must include column `text_`)")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text_" not in df.columns:
            st.error("CSV needs a column named `text_`.")
        else:
            df["cleaned"] = df["text_"].apply(clean_text)
            X_bulk = tfidf.transform(df["cleaned"])

            df["prob_fake"] = model.predict_proba(X_bulk)[:, 1]
            df["prediction"] = df["prob_fake"].apply(lambda p: "Fake" if p >= 0.5 else "Real")
            df["confidence_%"] = (df["prob_fake"] * 100).round(2)
            df["sentiment"] = df["text_"].apply(get_sentiment_label)

            st.success("Bulk analysis complete!")
            st.dataframe(df[["text_", "prediction", "confidence_%", "sentiment"]], height=350)

            download_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", download_csv, "review_results.csv", "text/csv")


# ============================ TAB 3: DASHBOARD ============================
with tab_dash:
    st.subheader("📊 Review Analytics Dashboard")

    if df is None:
        st.info("Upload a CSV in Bulk Analysis to enable dashboard.")
    else:

        # -------- SUMMARY METRICS --------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", len(df))
        col2.metric("Fake Reviews", (df["prediction"] == "Fake").sum())
        col3.metric("Real Reviews", (df["prediction"] == "Real").sum())
        col4.metric("Negative Sentiment", (df["sentiment"] == "Negative").sum())

        st.markdown("### 📦 Compact Overview Dashboard")

        # -------- SUBPLOTS GRID --------
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=90)
        fig.patch.set_facecolor('#1A1D21')

        # --- 1: Fake vs Real ---
        sns.countplot(x="prediction", data=df, ax=axes[0, 0])
        axes[0, 0].set_title("Fake vs Real", color="white")
        axes[0, 0].tick_params(colors="white")

        # --- 2: Sentiment ---
        sns.countplot(x="sentiment", data=df, ax=axes[0, 1])
        axes[0, 1].set_title("Sentiment Distribution", color="white")
        axes[0, 1].tick_params(colors="white")

        # --- 3: Confidence % ---
        sns.histplot(df["confidence_%"], bins=15, ax=axes[0, 2], color="#4C9EE3")
        axes[0, 2].set_title("Fake Confidence %", color="white")
        axes[0, 2].tick_params(colors="white")

        # --- 4: Heatmap Rating ---
        if "rating" in df.columns:
            try:
                pivot = df.pivot_table(values="prob_fake", index="rating", columns="prediction", aggfunc="mean")
                sns.heatmap(pivot, annot=True, cmap="Blues", ax=axes[1, 0])
                axes[1, 0].set_title("Fake Probability by Rating", color="white")
            except:
                axes[1, 0].text(0.5, 0.5, "No Rating Data", ha="center", color="white")
        else:
            axes[1, 0].text(0.5, 0.5, "Rating not available", ha="center", color="white")

        # --- 5: Fake Rate by Category ---
        if "category" in df.columns:
            cat_stats = (
                df.groupby("category")["prediction"]
                .apply(lambda x: (x == "Fake").mean() * 100)
                .reset_index(name="fake_rate_%")
            )
            sns.barplot(x="fake_rate_%", y="category", data=cat_stats, ax=axes[1, 1])
            axes[1, 1].set_title("Fake Rate by Category", color="white")
            axes[1, 1].tick_params(colors="white")
        else:
            axes[1, 1].text(0.5, 0.5, "Category not available", ha="center", color="white")

        # --- 6: Confusion Matrix ---
        if "label" in df.columns:
            y_true = df["label"].map({"OR": 0, "CG": 1}) if df["label"].dtype == object else df["label"]
            y_pred = df["prediction"].map({"Real": 0, "Fake": 1})

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", ax=axes[1, 2])
            axes[1, 2].set_title("Confusion Matrix", color="white")
        else:
            axes[1, 2].text(0.5, 0.5, "No Labels", ha="center", color="white")

        # Dark theme on all subplot boxes
        for ax in axes.flat:
            ax.set_facecolor("#232629")
            for spine in ax.spines.values():
                spine.set_color("#444")

        plt.tight_layout()
        st.pyplot(fig)


# ============================ FOOTER ============================
st.markdown(
    "<p style='text-align:center; color:gray; margin-top:40px;'>© 2025 Fake Review Detection | By Mohammed Asaf CT</p>",
    unsafe_allow_html=True
)
