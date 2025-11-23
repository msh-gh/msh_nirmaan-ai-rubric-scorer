# msh_nirmaan-ai-rubric-scorer
this is task during nirmaan 1st round of hiring

AI Rubric Scorer – Self-Introduction Evaluation Tool

This project implements an AI-based scoring tool for evaluating students’ self-introduction transcripts. The goal is to analyze a spoken self-introduction (already transcribed to text) and generate a rubric-based score out of 100 along with per-criterion feedback.
The scoring logic combines rule-based checks, NLP-based semantic similarity, and rubric-driven weighting. The rubric used for evaluation is taken from the Excel file provided in the case study.

The application is built with Streamlit and runs as a real-time web app where users can paste a transcript and immediately receive a score with detailed feedback.

Project Background

This project is part of the Nirmaan Communication Program case study.
Students record short self-introductions, and the audio is transcribed into text.
The task is to build a tool that:

Accepts the transcript text as input

Scores it using the rubric provided in the Excel file

Combines rule-based evaluation and semantic similarity

Produces detailed per-criterion results

Provides a final normalized score from 0 to 100

Runs in a deployed web interface

This repository contains the complete implementation, including frontend, backend logic, and deployment configuration.

Scoring Logic Overview

The scoring system integrates three components:

1. Rule-Based Evaluation

Several direct textual features are computed:

Keyword Presence
The transcript is checked for required items such as:
name, age, class/school, family, hobbies/interests, unique point, goals.
Additional “good to have” keywords such as origins, strengths, achievements, and fun facts are also counted.

Salutation Level
Identifies the type of greeting:
no greeting → low score
simple greetings (“Hi”, “Hello”) → moderate
formal greetings (“Good morning…”, “Hello everyone”) → higher
enthusiastic introductions → highest

Flow and Structure
Checks whether the student follows the recommended order:
Salutation → Basic details → Additional details → Closing

Speech Rate (WPM)
WPM is computed using:

WPM = word_count / (duration_seconds / 60)


Scores follow the rubric buckets (ideal, fast, too fast, slow, too slow).

Grammar Errors
Uses LanguageTool to identify grammar mistakes.
Grammar score is computed using the rubric formula:

Grammar Score = 1 – min(errors_per_100_words / 10, 1)


This is later mapped to the rubric’s scoring range.

Vocabulary Richness (TTR)
TTR is computed as:

TTR = distinct_words / total_words


The value is mapped to rubric-defined score ranges.

Filler Word Rate
Counts occurrences of common filler words such as “um”, “uh”, “like”, “you know”, etc.
Filler rate is computed as:

filler_rate = filler_count / total_words


Mapped to rubric scoring ranges.

Sentiment Positivity
Uses the VADER sentiment model.
Only the positive probability is used because the rubric scores positivity.
Rubric bucket ranges are applied to convert sentiment to marks.

2. Semantic Similarity (NLP-Based)

The transcript is compared with rubric descriptions using the SentenceTransformers model
all-MiniLM-L6-v2.

Cosine similarity is calculated between transcript sentences and rubric text.
This additional semantic signal helps determine whether the student meaningfully addresses the rubric requirements, even when phrasing is different.

3. Rubric Weight Integration

The final score is produced by combining all components using the weights specified in the rubric:

Criterion	Weight
Content & Structure	40
Speech Rate	10
Language & Grammar	20
Clarity	15
Engagement	15
Total	100

Each sub-score is normalized and scaled according to these weights, and the total becomes the final score out of 100.

Project Structure
ai_rubric_app/
│
├── app.py                     # Streamlit frontend + backend scoring logic
├── requirements.txt           # Dependency list
├── casestudy.xlsx             # Rubric file used for scoring
└── README.md                  # Project documentation

Running the Application Locally

Clone the repository:

git clone https://github.com/<your-username>/nirmaan-ai-rubric-scorer.git
cd nirmaan-ai-rubric-scorer


Create a virtual environment:

python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows


Install dependencies:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run app.py


The application will open in your browser at:

http://localhost:8501


Paste a transcript, set the duration (if available), and click “Score” to generate results.

Deployment (Streamlit Cloud)

Push all project files to a public GitHub repository.

Visit https://share.streamlit.io

Log in with GitHub and create a new app.

Select your repository, branch, and app.py file.

Deploy.

Streamlit Cloud will install the dependencies, cache the NLP model, and serve the app at a public URL.

JSON Output Format

The app allows users to download the scoring result as a JSON file.
The structure is as follows:

{
  "overall_score": 73.0,
  "meta": {
    "word_count": 133,
    "sentence_count": 11,
    "duration_seconds": 52,
    "wpm": 153.5,
    "ttr": 0.45
  },
  "per_criterion": [
    {
      "criterion": "Content & Structure",
      "score": 34,
      "max": 40,
      "breakdown": {
        "salutation": 4,
        "keyword_presence": 24,
        "flow": 6
      }
    }
  ]
}

Notes

The rubric is loaded from the included Excel file.

The first run may take longer because the SentenceTransformer model is downloaded.

The tool is intended for educational evaluation and not for high-stakes assessment.
