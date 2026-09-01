# ============================================================
# Medical Chatbot - Follow-up Symptom Logic
# ============================================================

DISEASE_SYMPTOMS = {
    "Pneumonia": {
        "breathlessness",
        "chills",
        "fast_heart_rate",
        "fatigue",
        "malaise",
        "phlegm",
        "rusty_sputum",
        "sweating",
    },

    "Bronchial Asthma": {
        "breathlessness",
        "family_history",
        "fatigue",
        "mucoid_sputum",
    },

    "GERD": {
        "acidity",
        "stomach_pain",
        "ulcers_on_tongue",
        "vomiting",
    },

    "Tuberculosis": {
        "blood_in_sputum",
        "breathlessness",
        "chills",
        "fatigue",
        "loss_of_appetite",
        "malaise",
        "mild_fever",
        "phlegm",
        "sweating",
        "weight_loss",
        "yellowing_of_eyes",
        "swelled_lymph_nodes",
        "vomiting",
    },
}


# Symptoms that are more useful for distinguishing
# between the candidate diseases.
PRIORITY_SYMPTOMS = {
    "Pneumonia": [
        "breathlessness",
        "phlegm",
        "rusty_sputum",
        "chills",
        "sweating",
        "fast_heart_rate",
        "fatigue",
        "malaise",
    ],

    "Bronchial Asthma": [
        "breathlessness",
        "mucoid_sputum",
        "family_history",
        "fatigue",
    ],

    "GERD": [
        "acidity",
        "stomach_pain",
        "vomiting",
        "ulcers_on_tongue",
    ],

    "Tuberculosis": [
        "weight_loss",
        "blood_in_sputum",
        "breathlessness",
        "loss_of_appetite",
        "phlegm",
        "sweating",
    ],
}


def get_follow_up_symptoms(
    candidates,
    current_symptoms,
    max_questions=4
):
    """
    Find useful symptoms that the user has not
    provided yet.
    """

    current_symptoms = set(current_symptoms)

    questions = []

    for disease in candidates:

        priority = PRIORITY_SYMPTOMS.get(
            disease,
            []
        )

        for symptom in priority:

            if symptom in current_symptoms:
                continue

            if symptom not in questions:
                questions.append(symptom)

            if len(questions) >= max_questions:
                return questions

    return questions


def format_symptom(symptom):
    """Convert symptom name into readable text."""

    return symptom.replace("_", " ")


def build_follow_up_message(symptoms):
    """
    Convert symptom list into a natural chatbot question.
    """

    if not symptoms:
        return None

    formatted = [
        format_symptom(symptom)
        for symptom in symptoms
    ]

    if len(formatted) == 1:

        return (
            "To narrow down the prediction, "
            f"do you also have {formatted[0]}?"
        )

    if len(formatted) == 2:

        return (
            "To narrow down the prediction, "
            f"do you also have {formatted[0]} "
            f"or {formatted[1]}?"
        )

    return (
        "To narrow down the prediction, "
        "do you also have "
        + ", ".join(formatted[:-1])
        + ", or "
        + formatted[-1]
        + "?"
    )
