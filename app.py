# app.py
import streamlit as st
import pandas as pd
import re
import math
import json
from collections import Counter
from pathlib import Path
import requests
import numpy as np

# --- NLP imports ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure NLTK resources (first run may download)
nltk.download("punkt")

# ========== Config ==========
# Local path to rubric Excel (you uploaded this file)
RUBRIC_XLSX_PATH = "Casestudy.xlsx"

# Optional sample transcript path (you uploaded this)
SAMPLE_TRANSCRIPT_PATH = "Sample text for case study.txt"

# LanguageTool public endpoint (fallback). You can use LanguageToolPlus if you have a key.
LANGUAGETOOL_URL = "https://api.languagetool.org/v2/check"

# SentenceTransformer model (will download first time)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Filler word list from rubric
FILLER_WORDS = [
    "um", "uh", "like", "you know", "so", "actually", "basically",
    "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"
]

# POSITIVE WORDS (simple fallback)
POSITIVE_WORDS = ["excited", "great", "enjoy", "interesting", "happy",
                  "good", "love", "like", "confident", "grateful"]

# ========== Utilities ==========
def load_rubric(path=RUBRIC_XLSX_PATH):
    df = pd.read_excel(path, sheet_name=0)
    # Normalize column names (best-effort)
    df.columns = [c.strip() for c in df.columns]
    return df

def basic_counts(text):
    words = re.findall(r"\w+'?\w*|\w+", text)
    word_count = len(words)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    return {
        "words": words,
        "word_count": word_count,
        "sentences": sentences,
        "sentence_count": sentence_count
    }

def compute_wpm(word_count, duration_seconds):
    if not duration_seconds or duration_seconds <= 0:
        return None
    return word_count / duration_seconds * 60

def language_tool_errors(text):
    try:
        resp = requests.post(LANGUAGETOOL_URL,
                             data={"text": text, "language": "en-US"},
                             timeout=15)
        j = resp.json()
        matches = j.get("matches", [])
        return len(matches), matches
    except Exception as e:
        # fallback: no errors detected
        return 0, []

def ttr_score(words):
    if not words:
        return 0.0
    distinct = len(set([w.lower() for w in words]))
    total = len(words)
    return distinct / total

def count_fillers(text):
    lower = text.lower()
    count = 0
    for f in FILLER_WORDS:
        count += len(re.findall(r"\b"+re.escape(f)+r"\b", lower))
    return count

def sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    s = analyzer.polarity_scores(text)
    # use positive probability proxy: compound > 0.05 considered positive; but return pos score
    return s  # dict with 'neg','neu','pos','compound'

def semantic_similarity_matrix(model, texts_a, texts_b):
    # returns matrix of cosine similarities shape (len(texts_a), len(texts_b))
    a_emb = model.encode(texts_a, convert_to_numpy=True, show_progress_bar=False)
    b_emb = model.encode(texts_b, convert_to_numpy=True, show_progress_bar=False)
    sim = cosine_similarity(a_emb, b_emb)
    return sim

# Mapping functions for rubric buckets (exact from the rubric)
def salutation_score(text):
    tl = text.lower()
    if any(kw in tl for kw in ["i am excited", "feeling great", "i'm excited"]):
        return 5
    if any(kw in tl for kw in ["hello everyone", "good morning", "good afternoon", "good evening", "good day"]):
        return 4
    if any(kw in tl for kw in ["hi", "hello"]):
        return 2
    return 0

def keyword_presence_score(must_presence, good_presence):
    # must: each 4 points, cap 20; good: each 2 points, cap 10
    must_score = sum(4 for v in must_presence.values() if v)
    must_score = min(must_score, 20)
    good_score = sum(2 for v in good_presence.values() if v)
    good_score = min(good_score, 10)
    return must_score + good_score  # out of 30

def flow_score(flow_order_followed):
    return 5 if flow_order_followed else 0

def speech_rate_score(wpm):
    if wpm is None:
        return 2
    if wpm > 160:
        return 2
    elif 141 <= wpm <= 160:
        return 6
    elif 111 <= wpm <= 140:
        return 10
    elif 81 <= wpm <= 110:
        return 6
    else:
        return 2

def grammar_score_from_errors(errors_count, word_count):
    if word_count == 0:
        return 2
    errors_per_100 = (errors_count / word_count) * 100
    factor = 1 - min(errors_per_100 / 10.0, 1.0)
    # map to buckets
    if factor > 0.9:
        return 10
    elif factor >= 0.7:
        return 8
    elif factor >= 0.5:
        return 6
    elif factor >= 0.3:
        return 4
    else:
        return 2

def vocab_score_from_ttr(ttr):
    if ttr >= 0.9:
        return 10
    elif ttr >= 0.7:
        return 8
    elif ttr >= 0.5:
        return 6
    elif ttr >= 0.3:
        return 4
    else:
        return 2

def filler_score_from_percent(filler_rate_percent):
    if filler_rate_percent <= 3:
        return 15
    elif filler_rate_percent <= 6:
        return 12
    elif filler_rate_percent <= 9:
        return 9
    elif filler_rate_percent <= 12:
        return 6
    else:
        return 3

def sentiment_score_from_prob(pos_prob):
    if pos_prob >= 0.9:
        return 15
    elif pos_prob >= 0.7:
        return 12
    elif pos_prob >= 0.5:
        return 9
    elif pos_prob >= 0.3:
        return 6
    else:
        return 3

# ========== Main scoring pipeline ==========
@st.cache_resource(show_spinner=False)
def load_models():
    model = SentenceTransformer(EMBED_MODEL)
    return {"embed": model}

def analyze_transcript(transcript_text, duration_seconds=52, rubric_df=None):
    # Basic counts
    counts = basic_counts(transcript_text)
    word_count = counts["word_count"]
    sentence_count = counts["sentence_count"]
    wpm = compute_wpm(word_count, duration_seconds)

    # Salutation
    sal_score = salutation_score(transcript_text)

    # Keyword presence checks (based on the rubric columns if available)
    # We'll infer must-have keywords and good-to-have from rubric if present, else use defaults
    must_keywords = {
        "name": ["my name is", "myself", "this is"],
        "age": [r"\b\d{1,2}\s*(years?|yrs?)\s*old\b", r"\bi am \d+\b", r"\b\d{1,2}\b year old\b"],
        "school/class": ["school", "class", "section"],
        "family": ["family", "mother", "father", "parents"],
        "hobbies": ["cricket", "play", "playing", "hobbies"],
        "goals": ["ambition", "dream", "goal", "want to", "wish to"],
        "unique": ["fun fact", "funfact", "unique", "interesting", "one thing people"]
    }
    good_keywords = {
        "about_family": ["family is", "we live", "we are from", "origin"],
        "ambition_goal": ["ambition", "dream", "goal", "want to become"],
        "fun_fact": ["fun fact", "once", "surprising", "interesting"],
        "strengths": ["strength", "achievement", "won", "award", "prize"]
    }

    text_lower = transcript_text.lower()
    must_presence = {}
    for k, pats in must_keywords.items():
        found = False
        for p in pats:
            try:
                if re.search(p, text_lower):
                    found = True
                    break
            except re.error:
                if p in text_lower:
                    found = True
                    break
        must_presence[k] = found

    good_presence = {}
    for k, pats in good_keywords.items():
        found = False
        for p in pats:
            if p in text_lower:
                found = True
                break
        good_presence[k] = found

    keyword_score = keyword_presence_score(must_presence, good_presence)

    # Flow/order detection
    salutation_idx = text_lower.find("hello everyone")
    basic_idx_candidates = [text_lower.find(k) for k in ["i am", "my name", "myself", "i live"] if text_lower.find(k) != -1]
    basic_idx = min(basic_idx_candidates) if basic_idx_candidates else -1
    additional_idx_candidates = [text_lower.find(k) for k in ["family", "hobbies", "fun fact", "favorite subject", "science"] if text_lower.find(k) != -1]
    additional_idx = min(additional_idx_candidates) if additional_idx_candidates else -1
    closing_idx = text_lower.rfind("thank you")
    flow_followed = (salutation_idx != -1 and basic_idx != -1 and additional_idx != -1 and closing_idx != -1 and salutation_idx < basic_idx < additional_idx < closing_idx)
    flow_sc = flow_score(flow_followed)

    # Grammar via LanguageTool
    errors_count, matches = language_tool_errors(transcript_text)
    grammar_sc = grammar_score_from_errors(errors_count, word_count)

    # Vocab TTR
    ttr = ttr_score(counts["words"])
    vocab_sc = vocab_score_from_ttr(ttr)

    # Fillers
    filler_count = count_fillers(transcript_text)
    filler_rate_percent = (filler_count / word_count) * 100 if word_count > 0 else 0.0
    filler_sc = filler_score_from_percent(filler_rate_percent)

    # Sentiment via VADER
    vader = sentiment_vader(transcript_text)
    pos_prob = vader.get("pos", 0.0)
    sentiment_sc = sentiment_score_from_prob(pos_prob)

    # Semantic similarity: use rubric descriptions if given, else only compute a general similarity to "criterion phrases"
    models = load_models()
    embed_model = models["embed"]

    # Prepare rubric text pieces for semantic similarity if rubric_df provided
    rubric_texts = []
    rubric_names = []
    if rubric_df is not None:
        # Try to use columns 'Creteria' or 'Criteria' and 'Metric' or 'Score Attributed' or 'Key Words'
        for _, row in rubric_df.iterrows():
            # Build a short description per row
            parts = []
            for c in ["Creteria", "Criterion", "Criteria", "Metric", "Scoring creteria", "Key Words", "Key Words "]:
                if c in row.index and pd.notna(row[c]):
                    parts.append(str(row[c]))
            txt = " | ".join(parts) if parts else None
            if txt:
                rubric_texts.append(txt)
                rubric_names.append(str(row.iloc[0]))
    # fallback rubric_texts (use main rubric criteria)
    if not rubric_texts:
        rubric_texts = [
            "Salutation: greeting, excited tone",
            "Content: name, age, school, family, hobbies, goals, unique point",
            "Flow: salutation -> basic -> additional -> closing",
            "Speech rate: words per minute ideal 111-140",
            "Grammar and language correctness",
            "Vocabulary richness (TTR)",
            "Clarity: filler words rate",
            "Engagement: positivity"
        ]
        rubric_names = ["Salutation","Content","Flow","SpeechRate","Grammar","Vocab","Clarity","Engagement"]

    # Compute similarity between transcript and each rubric text
    sims = semantic_similarity_matrix(embed_model, [transcript_text], rubric_texts)[0]
    # normalize similarity [0,1] — cosine already roughly between [-1,1], but should be 0..1 for SBERT
    sim_probs = [(float(max(0.0, s))) for s in sims]  # keep non-negative part
    # Map sims to a 0..1 by dividing by 1.0 (SBERT returns 0..1 for similar sentences typically)
    # create a per-criterion semantic score (scale 0..max)
    sem_scores = {}
    for name, s in zip(rubric_names, sim_probs):
        # scale to 0..1 then to max per criterion later
        sem_scores[name] = s

    # Now combine signals for final per-criterion scores as per rubric weights
    # We'll use the rubric weights from the loaded rubric_df if available, else map manually
    # Default weight mapping (from the PDF you provided)
    default_weights = {
        "Content & Structure": 40,
        "Speech Rate": 10,
        "Language & Grammar": 20,
        "Clarity": 15,
        "Engagement": 15
    }

    # If rubric_df provides a 'Weightage' column, try to use it. Else use defaults.
    weights = {}
    if rubric_df is not None and ("Creteria" in rubric_df.columns or "Criteria" in rubric_df.columns or "Creteria " in rubric_df.columns):
        # try to parse a simple mapping by matching first column text presence in default keys
        for _, row in rubric_df.iterrows():
            key = str(row.iloc[0])
            # try to match known keys crudely
            if "Content" in key:
                weights["Content & Structure"] = 40
            if "Speech" in key:
                weights["Speech Rate"] = 10
            if "Grammar" in key or "Language" in key:
                weights["Language & Grammar"] = 20
            if "Clarity" in key:
                weights["Clarity"] = 15
            if "Engagement" in key or "Sentiment" in key:
                weights["Engagement"] = 15
    # fill missing with defaults
    for k, v in default_weights.items():
        if k not in weights:
            weights[k] = v

    # Compute each main criterion's raw score by combining rule-based and semantic where appropriate
    # Content & Structure: combine salutation(5), keywords(30), flow(5) -> total 40
    content_raw = sal_score + keyword_score + flow_sc  # out of 40
    # Speech Rate: use speech rate score (out of 10)
    speech_raw = speech_rate_score(wpm)
    # Language & Grammar: grammar_sc (10) + vocab_sc (10) -> out of 20
    language_raw = grammar_sc + vocab_sc
    # Clarity: filler_sc (15)
    clarity_raw = filler_sc
    # Engagement: sentiment_sc (15)
    engagement_raw = sentiment_sc

    # Compose per-criterion objects
    per_criterion = [
        {
            "criterion": "Content & Structure",
            "score": content_raw,
            "max": 40,
            "breakdown": {
                "salutation": sal_score,
                "keyword_presence": keyword_score,
                "flow": flow_sc,
                "found_must_have": must_presence,
                "found_good_to_have": good_presence
            },
            "semantic_similarities": {n: float(s) for n, s in zip(rubric_names, sim_probs)}
        },
        {
            "criterion": "Speech Rate",
            "score": speech_raw,
            "max": 10,
            "details": {"wpm": round(wpm,1) if wpm else None}
        },
        {
            "criterion": "Language & Grammar",
            "score": language_raw,
            "max": 20,
            "details": {
                "grammar_errors": errors_count,
                "grammar_matches_sample": matches[:3] if matches else [],
                "ttr": round(ttr,3)
            }
        },
        {
            "criterion": "Clarity",
            "score": clarity_raw,
            "max": 15,
            "details": {"filler_count": filler_count, "filler_rate_percent": round(filler_rate_percent,2)}
        },
        {
            "criterion": "Engagement",
            "score": engagement_raw,
            "max": 15,
            "details": {"vader": vader}
        }
    ]

    overall = sum(item["score"] for item in per_criterion)
    overall = round(float(overall), 2)

    # output meta
    meta = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "duration_seconds": duration_seconds,
        "wpm": round(wpm,1) if wpm else None,
        "ttr": round(ttr,3)
    }

    return {"overall_score": overall, "per_criterion": per_criterion, "meta": meta}

# ========== Streamlit UI ==========
st.set_page_config(page_title="AI Rubric Scorer", layout="wide")
st.title("AI Rubric Scorer — Self-introduction (Streamlit)")

st.markdown(
    """
    Paste a student's self-introduction transcript below, set duration in seconds (if known),
    and click **Score**. The app combines rule-based checks and semantic similarity
    (sentence-transformers) to produce per-criterion scores and an overall score.
    """
)

col1, col2 = st.columns([2,1])

with col1:
    # prefill with uploaded sample if available
    sample_transcript = ""
    try:
        p = Path(SAMPLE_TRANSCRIPT_PATH)
        if p.exists():
            sample_transcript = p.read_text(encoding="utf-8")
    except Exception:
        pass # Silently fail if sample not found
    transcript = st.text_area("Transcript text", value=sample_transcript, height=300)
    if not transcript:
        st.warning("Please paste a transcript or ensure the sample file is available.")

with col2:
    duration_seconds = st.number_input("Duration (seconds)", min_value=1, value=52)
    run_btn = st.button("Score")

if run_btn:
    with st.spinner("Scoring... (loading NLP models first time may take ~30-60s)"):
        # load rubric if available
        rubric_df = None
        try:
            rubric_df = load_rubric(RUBRIC_XLSX_PATH)
        except Exception as e:
            st.warning(f"Could not load rubric at {RUBRIC_XLSX_PATH}: {e}. Using default internal rubric.")
            rubric_df = None

        result = analyze_transcript(transcript, duration_seconds=duration_seconds, rubric_df=rubric_df)

    # Show results
    st.metric("Overall score (0-100)", result["overall_score"])
    st.subheader("Meta")
    st.json(result["meta"])

    st.subheader("Per-criterion breakdown")
    for c in result["per_criterion"]:
        st.markdown(f"### {c['criterion']} — {c['score']} / {c['max']}")
        st.json(c.get("breakdown") or c.get("details") or {})
        if "semantic_similarities" in c:
            st.write("Sample semantic similarities (transcript → rubric snippets):")
            st.json(c["semantic_similarities"])

    # Save output JSON option
    st.download_button("Download JSON result", data=json.dumps(result, indent=2), file_name="scoring_result.json")
