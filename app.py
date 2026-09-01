import json
import pickle
import random
import re
from datetime import datetime
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import torch

from flask import Flask, render_template, request, jsonify
from nltk.stem.porter import PorterStemmer

from nnet import NeuralNet
from nltk_utils import bag_of_words

from symptom_questions import (
    get_follow_up_symptoms,
    build_follow_up_message
)


# ============================================================
# CONFIGURATION
# ============================================================

random.seed(datetime.now().timestamp())

device = torch.device("cpu")

NLP_MODEL_FILE = "models/data.pth"
DISEASE_MODEL_FILE = "models/fitted_model_new.pkl"
SYMPTOMS_FILE = "data/list_of_symptoms_new.pickle"
INTENTS_FILE = "intents.json"

DESCRIPTION_FILE = "data/symptom_Description.csv"
PRECAUTION_FILE = "data/symptom_precaution.csv"
SEVERITY_FILE = "data/Symptom-severity.csv"

SYMPTOM_TEXT_FILE = "static/assets/files/ds_symptoms.txt"


# ============================================================
# STEMMER
# ============================================================

stemmer = PorterStemmer()


# ============================================================
# LOAD NLP MODEL
# ============================================================

print("Loading NLP model...")

model_data = torch.load(
    NLP_MODEL_FILE,
    map_location=device
)

input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]

all_words = model_data["all_words"]
tags = model_data["tags"]
model_state = model_data["model_state"]

nlp_model = NeuralNet(
    input_size,
    hidden_size,
    output_size
).to(device)

nlp_model.load_state_dict(model_state)
nlp_model.eval()

print("NLP model loaded.")
print("NLP input size:", input_size)
print("NLP hidden size:", hidden_size)
print("NLP output size:", output_size)


# ============================================================
# LOAD DISEASE MODEL
# ============================================================

print("Loading disease prediction model...")

with open(DISEASE_MODEL_FILE, "rb") as model_file:
    prediction_model = pickle.load(model_file)

print(
    "Disease model loaded:",
    type(prediction_model).__name__
)


# ============================================================
# LOAD SYMPTOM LIST
# ============================================================

print("Loading symptom list...")

with open(SYMPTOMS_FILE, "rb") as data_file:
    symptoms_list = pickle.load(data_file)

print(
    "Number of symptoms:",
    len(symptoms_list)
)


# ============================================================
# LOAD INTENTS
# ============================================================

print("Loading intents...")

with open(
    INTENTS_FILE,
    "r",
    encoding="utf-8"
) as intent_file:
    intents_data = json.load(intent_file)

intents = intents_data.get(
    "intents",
    []
)

print(
    "Number of intents:",
    len(intents)
)


# ============================================================
# LOAD DISEASE DESCRIPTION
# ============================================================

diseases_description = pd.read_csv(
    DESCRIPTION_FILE
)

diseases_description["Disease"] = (
    diseases_description["Disease"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# LOAD PRECAUTIONS
# ============================================================

disease_precaution = pd.read_csv(
    PRECAUTION_FILE
)

disease_precaution["Disease"] = (
    disease_precaution["Disease"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# LOAD SYMPTOM SEVERITY
# ============================================================

symptom_severity = pd.read_csv(
    SEVERITY_FILE
)

symptom_severity["Symptom"] = (
    symptom_severity["Symptom"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(
        " ",
        "",
        regex=False
    )
)

symptom_severity["weight"] = pd.to_numeric(
    symptom_severity["weight"],
    errors="coerce"
)


# ============================================================
# LOAD DATASET FOR KNN NEIGHBOR ANALYSIS
# ============================================================

print("Loading unique symptom dataset...")

diagnosis_dataset = pd.read_csv(
    "data/dataset.csv"
)

diagnosis_dataset = (
    diagnosis_dataset
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    "Unique diagnosis examples:",
    len(diagnosis_dataset)
)


# ============================================================
# USER SESSION STATE
# ============================================================

user_symptoms = set()

# True when chatbot has asked follow-up questions
waiting_for_followup = False

# Number of follow-up rounds already asked
followup_round = 0

MAX_FOLLOWUP_ROUNDS = 1


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower().strip()

    text = text.replace(
        "_",
        " "
    )

    text = text.replace(
        "-",
        " "
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ORIGINAL NLP MODEL
# ============================================================

def get_symptom(sentence):

    tokens = nltk.word_tokenize(
        sentence
    )

    X = bag_of_words(
        tokens,
        all_words
    )

    X = X.reshape(
        1,
        X.shape[0]
    )

    X = torch.from_numpy(X)

    with torch.no_grad():

        output = nlp_model(X)

    _, predicted = torch.max(
        output,
        dim=1
    )

    tag = tags[
        predicted.item()
    ]

    probabilities = torch.softmax(
        output,
        dim=1
    )

    probability = probabilities[
        0
    ][
        predicted.item()
    ].item()

    return tag, probability


# ============================================================
# BUILD INTENT PATTERN MAP
# ============================================================

pattern_map = {}

for intent in intents:

    tag = intent.get("tag")

    if not tag:
        continue

    if tag not in symptoms_list:
        continue

    patterns = intent.get(
        "patterns",
        []
    )

    for pattern in patterns:

        normalized_pattern = normalize_text(
            pattern
        )

        if normalized_pattern:

            pattern_map[
                normalized_pattern
            ] = tag


print(
    "Usable intent patterns:",
    len(pattern_map)
)


# ============================================================
# WORD ALIASES
# ============================================================

word_aliases = {

    "coughing": "cough",
    "coughed": "cough",
    "coughs": "cough",

    "vomited": "vomiting",
    "vomit": "vomiting",

    "dizzy": "dizziness",

    "itchy": "itching",

    "nauseous": "nausea",

    "sweaty": "sweating",

    "shiver": "shivering",

    "breathing": "breathlessness",

}


# ============================================================
# EXTRACT SYMPTOMS
# ============================================================

def extract_symptoms(sentence):

    original_text = normalize_text(
        sentence
    )

    found_symptoms = set()


    # --------------------------------------------------------
    # 1. Direct canonical symptom matching
    # --------------------------------------------------------

    for symptom in symptoms_list:

        symptom_text = normalize_text(
            symptom
        )

        if not symptom_text:
            continue

        pattern = (
            r"\b"
            + re.escape(symptom_text)
            + r"\b"
        )

        if re.search(
            pattern,
            original_text
        ):

            found_symptoms.add(
                symptom
            )


    # --------------------------------------------------------
    # 2. Intent pattern matching
    # --------------------------------------------------------

    for pattern, tag in pattern_map.items():

        pattern_regex = (
            r"\b"
            + re.escape(pattern)
            + r"\b"
        )

        if re.search(
            pattern_regex,
            original_text
        ):

            found_symptoms.add(
                tag
            )


    # --------------------------------------------------------
    # 3. Aliases
    # --------------------------------------------------------

    words = set(
        original_text.split()
    )

    for word, symptom in word_aliases.items():

        if word in words:

            if symptom in symptoms_list:

                found_symptoms.add(
                    symptom
                )


    # --------------------------------------------------------
    # 4. Stem matching
    # --------------------------------------------------------

    try:

        user_tokens = nltk.word_tokenize(
            original_text
        )

        user_stems = {
            stemmer.stem(token)
            for token in user_tokens
            if token.isalpha()
        }

        ignored_words = {
            "i",
            "am",
            "my",
            "me",
            "have",
            "has",
            "had",
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "feel",
            "feeling",
            "got",
            "get",
            "in",
            "on",
            "of",
            "to",
            "with",
            "and",
            "also",
            "very",
            "really",
            "some",
            "this",
            "that",
        }

        for pattern, tag in pattern_map.items():

            pattern_tokens = nltk.word_tokenize(
                normalize_text(pattern)
            )

            pattern_stems = {
                stemmer.stem(token)
                for token in pattern_tokens
                if token.isalpha()
            }

            useful_pattern_stems = (
                pattern_stems
                - ignored_words
            )

            if not useful_pattern_stems:
                continue

            matched_stems = (
                useful_pattern_stems
                & user_stems
            )

            if (
                matched_stems
                == useful_pattern_stems
            ):

                found_symptoms.add(
                    tag
                )

    except Exception as error:

        print(
            "Stem matching error:",
            error
        )


    # --------------------------------------------------------
    # 5. NLP fallback
    # --------------------------------------------------------

    if not found_symptoms:

        try:

            symptom, probability = get_symptom(
                sentence
            )

            print(
                "NLP fallback:",
                symptom,
                "probability:",
                round(
                    probability,
                    4
                )
            )

            if (
                probability > 0.50
                and symptom in symptoms_list
            ):

                found_symptoms.add(
                    symptom
                )

        except Exception as error:

            print(
                "NLP fallback error:",
                error
            )


    return found_symptoms


# ============================================================
# FORMAT SYMPTOM
# ============================================================

def format_symptom(symptom):

    return symptom.replace(
        "_",
        " "
    )


# ============================================================
# FORMAT SYMPTOM LIST
# ============================================================

def format_symptom_list(symptoms):

    names = [
        format_symptom(symptom)
        for symptom in sorted(symptoms)
    ]

    if not names:
        return ""

    if len(names) == 1:

        return names[0]

    if len(names) == 2:

        return (
            names[0]
            + " and "
            + names[1]
        )

    return (
        ", ".join(names[:-1])
        + " and "
        + names[-1]
    )


# ============================================================
# GET TOP DISEASE CANDIDATES
# ============================================================

def get_top_candidates(current_symptoms):

    x_test = np.array(
        [
            [
                1 if symptom in current_symptoms
                else 0
                for symptom in symptoms_list
            ]
        ],
        dtype=np.int8
    )

    # Ask KNN for nearest examples
    try:

        distances, indices = (
            prediction_model.kneighbors(
                x_test,
                n_neighbors=min(
                    20,
                    len(diagnosis_dataset)
                )
            )
        )

    except Exception as error:

        print(
            "KNN candidate error:",
            error
        )

        return []


    diseases = []

    for index in indices[0]:

        disease = (
            diagnosis_dataset
            .iloc[index]["Disease"]
        )

        diseases.append(
            str(disease).strip()
        )


    votes = Counter(
        diseases
    )

    top_candidates = [
        disease
        for disease, count
        in votes.most_common(3)
    ]


    print()
    print(
        "Top disease candidates:"
    )

    for disease, count in votes.most_common(5):

        print(
            f"  {disease}: {count}"
        )

    return top_candidates


# ============================================================
# BUILD FEATURE VECTOR
# ============================================================

def build_feature_vector():

    x_test = [
        1 if symptom in user_symptoms
        else 0
        for symptom in symptoms_list
    ]

    return np.asarray(
        x_test,
        dtype=np.int8
    ).reshape(
        1,
        -1
    )


# ============================================================
# GET DESCRIPTION
# ============================================================

def get_description(disease):

    disease_key = (
        str(disease)
        .lower()
        .strip()
    )

    matches = diseases_description.loc[
        diseases_description["Disease"]
        == disease_key,
        "Description"
    ]

    if len(matches) == 0:

        return (
            "No description is available "
            "for this prediction."
        )

    return str(
        matches.iloc[0]
    )


# ============================================================
# GET PRECAUTIONS
# ============================================================

def get_precautions(disease):

    disease_key = (
        str(disease)
        .lower()
        .strip()
    )

    matches = disease_precaution[
        disease_precaution["Disease"]
        == disease_key
    ]

    if len(matches) == 0:

        return (
            "Precautions information "
            "is not available."
        )

    row = matches.iloc[0]

    columns = [
        "Precaution_1",
        "Precaution_2",
        "Precaution_3",
        "Precaution_4",
    ]

    precautions = []

    for column in columns:

        if column not in row:
            continue

        value = row[column]

        if pd.notna(value):

            value = str(
                value
            ).strip()

            if value:

                precautions.append(
                    value
                )

    if not precautions:

        return (
            "Precautions information "
            "is not available."
        )

    return (
        "Precautions: "
        + ", ".join(precautions)
    )


# ============================================================
# GET SEVERITY
# ============================================================

def get_severity(symptoms):

    values = []

    for symptom in symptoms:

        symptom_key = (
            symptom
            .lower()
            .strip()
            .replace(" ", "")
        )

        matches = symptom_severity.loc[
            symptom_severity["Symptom"]
            == symptom_key,
            "weight"
        ]

        if len(matches) == 0:

            print(
                "Severity not found:",
                symptom
            )

            continue

        value = matches.iloc[0]

        if pd.notna(value):

            values.append(
                float(value)
            )

    return values


# ============================================================
# FINAL DISEASE RESPONSE
# ============================================================

def generate_final_prediction():

    if not user_symptoms:

        return (
            "Please enter at least one symptom "
            "before asking for a prediction."
        )


    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    input_vector = build_feature_vector()

    print()
    print(
        "Final symptoms:",
        sorted(user_symptoms)
    )

    print(
        "Feature vector:",
        input_vector.shape
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        disease = prediction_model.predict(
            input_vector
        )[0]

        disease = str(
            disease
        ).strip()

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return (
            "Sorry, I couldn't process "
            "your symptoms right now."
        )


    print(
        "Final prediction:",
        disease
    )


    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description = get_description(
        disease
    )


    # --------------------------------------------------------
    # Precautions
    # --------------------------------------------------------

    precautions = get_precautions(
        disease
    )


    # --------------------------------------------------------
    # Professional Result Card
    # --------------------------------------------------------

    recorded_symptoms = sorted(
        symptom.replace("_", " ")
        for symptom in user_symptoms
    )

    symptoms_html = "".join(
        f'<span class="symptom-tag">{symptom}</span>'
        for symptom in recorded_symptoms
    )

    response = f"""
    <div class="diagnosis-card">

        <div class="diagnosis-header">

            <div class="diagnosis-icon">
                <i class="fas fa-stethoscope"></i>
            </div>

            <div>
                <div class="diagnosis-label">
                    MODEL PREDICTION
                </div>

                <div class="diagnosis-title">
                    {disease}
                </div>
            </div>

        </div>


        <div class="diagnosis-section">

            <div class="section-title">
                <i class="fas fa-notes-medical"></i>
                Description
            </div>

            <div class="section-content">
                {description}
            </div>

        </div>


        <div class="diagnosis-section">

            <div class="section-title">
                <i class="fas fa-shield-alt"></i>
                Precautions
            </div>

            <div class="section-content">
                {precautions}
            </div>

        </div>


        <div class="diagnosis-section">

            <div class="section-title">
                <i class="fas fa-list"></i>
                Symptoms Considered
            </div>

            <div class="symptom-tags">
                {symptoms_html}
            </div>

        </div>


        <div class="diagnosis-warning">

            <div class="warning-icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>

            <div>

                <strong>Important</strong>

                <p>
                    This is an AI/ML-based prediction and not a
                    medical diagnosis. Meddy is not a doctor.
                    Please consult a qualified healthcare
                    professional for proper diagnosis and treatment.
                </p>

            </div>

        </div>

    </div>
    """



    # --------------------------------------------------------
    # Severity warning
    # --------------------------------------------------------

    severity = get_severity(
        user_symptoms
    )

    if severity:

        mean_severity = np.mean(
            severity
        )

        max_severity = np.max(
            severity
        )

        print(
            "Mean severity:",
            mean_severity
        )

        print(
            "Max severity:",
            max_severity
        )

        if (
            mean_severity > 4
            or max_severity > 5
        ):

            response += (
                "<br><br>"
                "<b>Important:</b> "
                "Some of the symptoms you entered "
                "may be serious. This chatbot is "
                "not a doctor, so please consider "
                "seeking professional medical advice."
            )


    return response


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    global waiting_for_followup
    global followup_round

    user_symptoms.clear()

    waiting_for_followup = False
    followup_round = 0

    data = []

    try:

        with open(
            SYMPTOM_TEXT_FILE,
            "r",
            encoding="utf-8"
        ) as symptom_file:

            all_symptoms = (
                symptom_file.readlines()
            )

        for symptom in all_symptoms:

            clean_symptom = (
                symptom
                .replace("'", "")
                .replace("_", " ")
                .replace(",\n", "")
                .strip()
            )

            if clean_symptom:

                data.append(
                    clean_symptom
                )

    except FileNotFoundError:

        data = [
            format_symptom(symptom)
            for symptom in symptoms_list
        ]


    data = json.dumps(
        data
    )

    return render_template(
        "index.html",
        data=data
    )


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.route(
    "/symptom",
    methods=["GET", "POST"]
)
def predict_symptom():

    global waiting_for_followup
    global followup_round


    print()
    print("=" * 60)
    print("NEW CHAT REQUEST")
    print("=" * 60)


    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    if not request.is_json:

        return jsonify(
            "Invalid request. JSON data is required."
        ), 400


    request_data = request.get_json()

    if not request_data:

        return jsonify(
            "Invalid request data."
        ), 400


    sentence = request_data.get(
        "sentence",
        ""
    )


    if not isinstance(
        sentence,
        str
    ):

        return jsonify(
            "Invalid sentence."
        ), 400


    sentence = sentence.strip()


    print(
        "User sentence:",
        sentence
    )


    if not sentence:

        return jsonify(
            "Please describe the symptoms you are experiencing."
        )


    # ========================================================
    # CHECK DONE
    # ========================================================

    cleaned_command = (
        sentence
        .replace(".", "")
        .replace("!", "")
        .replace("?", "")
        .lower()
        .strip()
    )


    # ========================================================
    # NEGATIVE FOLLOW-UP ANSWER
    # ========================================================

    negative_answers = {
        "no",
        "nope",
        "nothing",
        "nothing else",
        "no other symptoms",
        "no more symptoms",
        "that's all",
        "thats all",
        "that is all",
        "none",
    }


    is_negative_answer = (
        normalize_text(sentence)
        in {
            normalize_text(answer)
            for answer in negative_answers
        }
    )


    # ========================================================
    # NORMAL SYMPTOM EXTRACTION
    # ========================================================

    found_symptoms = extract_symptoms(
        sentence
    )


    print(
        "Extracted symptoms:",
        sorted(found_symptoms)
    )


    # ========================================================
    # IF USER ANSWERS FOLLOW-UP
    # ========================================================

    if waiting_for_followup:

        # ----------------------------------------------------
        # User says no
        # ----------------------------------------------------

        if is_negative_answer:

            print(
                "User has no additional symptoms."
            )

            waiting_for_followup = False

            response_sentence = (
                "Okay. I'll make the prediction "
                "using the symptoms you've provided."
                "<br><br>"
                + generate_final_prediction()
            )

            user_symptoms.clear()

            followup_round = 0

            return jsonify(
                response_sentence
            )


        # ----------------------------------------------------
        # User provides additional symptoms
        # ----------------------------------------------------

        if found_symptoms:

            user_symptoms.update(
                found_symptoms
            )

            waiting_for_followup = False

            formatted = (
                format_symptom_list(
                    found_symptoms
                )
            )

            response_sentence = (
                "Got it — I've also recorded "
                "<b>"
                + formatted
                + "</b>."
                "<br><br>"
                "If you have any other symptoms, "
                "you can enter them. Otherwise, "
                "write <b>Done</b>."
            )

            return jsonify(
                response_sentence
            )


        # ----------------------------------------------------
        # Couldn't understand follow-up answer
        # ----------------------------------------------------

        return jsonify(
            "I couldn't identify a symptom in that answer. "
            "Please tell me the symptom you are experiencing, "
            "or type <b>No</b> if you don't have any of those symptoms."
        )


    # ========================================================
    # DONE COMMAND
    # ========================================================

    if cleaned_command == "done":

        if not user_symptoms:

            return jsonify(
                random.choice(
                    [
                        "Please enter at least one symptom first.",
                        "I need some symptoms before I can make a prediction.",
                        "Meddy needs at least one symptom to continue.",
                    ]
                )
            )


        # ----------------------------------------------------
        # Find top candidates
        # ----------------------------------------------------

        candidates = get_top_candidates(
            user_symptoms
        )


        # ----------------------------------------------------
        # Find useful follow-up symptoms
        # ----------------------------------------------------

        if (
            candidates
            and followup_round < MAX_FOLLOWUP_ROUNDS
        ):

            questions = get_follow_up_symptoms(
                candidates,
                user_symptoms,
                max_questions=4
            )


            if questions:

                followup_round += 1

                waiting_for_followup = True

                question = (
                    build_follow_up_message(
                        questions
                    )
                )

                response_sentence = (
                    "I've analyzed the symptoms you've provided."
                    "<br><br>"
                    + question
                    + "<br><br>"
                    "You can answer with the symptoms you have, "
                    "or type <b>No</b> if you don't have them."
                )

                print(
                    "Follow-up questions:",
                    questions
                )

                return jsonify(
                    response_sentence
                )


        # ----------------------------------------------------
        # No useful follow-up available
        # ----------------------------------------------------

        waiting_for_followup = False
        followup_round = 0

        response_sentence = (
            generate_final_prediction()
        )

        user_symptoms.clear()

        return jsonify(
            response_sentence
        )


    # ========================================================
    # NORMAL USER SYMPTOM MESSAGE
    # ========================================================

    if found_symptoms:

        user_symptoms.update(
            found_symptoms
        )

        formatted_symptoms = (
            format_symptom_list(
                found_symptoms
            )
        )

        response_sentence = (
            "Got it — I've recorded "
            "<b>"
            + formatted_symptoms
            + "</b>."
            "<br><br>"
            "Do you have any other symptoms?"
        )

        print(
            "All user symptoms:",
            sorted(user_symptoms)
        )

        return jsonify(
            response_sentence
        )


    # ========================================================
    # NOTHING UNDERSTOOD
    # ========================================================

    return jsonify(
        "I'm sorry, I couldn't identify "
        "a known symptom in that message."
        "<br><br>"
        "Please describe your symptoms more clearly "
        "or choose a symptom from the suggestions."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
