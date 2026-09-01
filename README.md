[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://medical-chatbot-ydor.onrender.com/)


# 🩺 Meddy — AI Medical Chatbot

Meddy is an AI/ML-powered medical chatbot that analyzes symptoms provided in natural language and predicts a possible disease using machine learning.

The project combines Natural Language Processing (NLP), symptom recognition, a K-Nearest Neighbors (KNN) classifier, intelligent follow-up questions, and a Flask web interface.

> ⚠️ **Medical Disclaimer:** Meddy is an educational AI/ML project. It is not a doctor and must not be used as a replacement for professional medical diagnosis or treatment.

---
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://medical-chatbot-ydor.onrender.com/)


## ✨ Features

- 💬 Interactive medical chatbot interface
- 🧠 NLP-based symptom recognition
- 🔎 Symptom suggestions while typing
- 🩺 Machine-learning-based disease prediction
- 🤖 K-Nearest Neighbors classifier
- 📋 131 supported symptoms
- 🧪 304 unique diagnosis examples
- 🏥 41 disease classes
- ❓ Intelligent follow-up symptom questions
- 📊 131-dimensional binary symptom representation
- 📝 Disease descriptions
- 🛡️ Disease precautions
- ⚠️ Severity warnings
- 🔄 Start-over conversation functionality
- ⌨️ Enter-key message submission
- ⏳ Typing indicator
- 🚨 Error handling
- 🌐 Flask web application

---

## 🧠 How It Works

```text
User enters symptoms
        ↓
NLP processes the message
        ↓
Known symptoms are identified
        ↓
Symptoms are stored
        ↓
User enters "Done"
        ↓
Top candidate diseases are identified
        ↓
Useful follow-up symptoms are suggested
        ↓
Final symptom vector is created
        ↓
KNN model predicts a disease
        ↓
Description + precautions are displayed
````

---

## 🤖 Machine Learning

The final disease prediction model is:

```text
KNeighborsClassifier
```

Configuration:

```text
n_neighbors = 6
metric = cosine
```

The model uses **131 symptom features**.

Each symptom is represented as:

```text
1 → symptom is present
0 → symptom is absent
```

Example:

```text
chest_pain     → 1
high_fever     → 1
cough          → 1
breathlessness → 1
phlegm         → 1
```

The resulting feature vector contains 131 values.

---

## 📊 Dataset

After duplicate removal, the dataset contains:

```text
Unique diagnosis examples: 304
Unique symptoms:           131
Disease classes:           41
```

The dataset and model feature list were validated to ensure that both contain the same 131 symptoms.

---

## 🧪 Model Evaluation

Multiple machine learning models were evaluated before selecting the final KNN model.

| Model               | Training Accuracy | Test Accuracy |
| ------------------- | ----------------: | ------------: |
| Logistic Regression |            96.30% |        96.72% |
| KNN                 |           100.00% |       100.00% |
| Decision Tree       |            92.18% |        65.57% |
| SVM                 |            17.70% |        16.39% |
| Stacking Classifier |            73.66% |        72.13% |

KNN achieved the highest test accuracy on the evaluated dataset and was selected as the final model.

> These results are dataset-level evaluation results only. They do not represent clinical accuracy or real-world medical performance.

---

## 🔬 NLP Component

The project includes an NLP model for recognizing symptoms from natural-language input.

NLP model configuration:

```text
Input size:  413
Hidden size: 8
Output size: 131
```

The NLP pipeline uses:

* NLTK tokenization
* Stemming
* Bag-of-words representation
* PyTorch neural network

Example:

```text
"I have severe chest pain"

        ↓

Tokenization
        ↓
Stemming / normalization
        ↓
Symptom recognition
        ↓
chest_pain
```

---

## ❓ Intelligent Follow-Up Questions

Meddy does not always make an immediate final prediction.

When the provided symptoms could correspond to multiple diseases, the chatbot identifies useful additional symptoms and asks follow-up questions.

Example:

```text
Current symptoms:

chest pain
high fever
cough
```

Meddy may ask:

```text
Do you also have breathlessness, phlegm,
rusty sputum, or chills?
```

The user can provide additional symptoms and then enter:

```text
Done
```

The final prediction is generated using the complete symptom set.

---

## 🩺 Example Prediction

For example, a user may provide:

```text
chest pain
high fever
cough
breathlessness
phlegm
```

The system converts these symptoms into the corresponding 131-feature vector and passes it to the trained KNN model.

Example result:

```text
MODEL PREDICTION

Pneumonia
```

The chatbot then displays:

```text
Description
Pneumonia is an infection in one or both lungs...

Precautions
consult doctor, medication, rest, follow up
```

The result also includes the symptoms considered by the system and an appropriate medical disclaimer.

---

## 🖥️ Technology Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Core programming        |
| Flask        | Web backend             |
| scikit-learn | Machine learning        |
| KNN          | Disease prediction      |
| PyTorch      | NLP model               |
| NLTK         | NLP processing          |
| NumPy        | Numerical operations    |
| Pandas       | Dataset processing      |
| HTML         | Frontend structure      |
| CSS          | UI styling              |
| JavaScript   | Frontend logic          |
| jQuery       | AJAX and UI interaction |
| Bootstrap    | Responsive UI           |

---

## 📁 Project Structure

```text
Medical Chatbot [END 2 END] [NLP]
│
├── app.py
├── nltk_utils.py
├── nnet.py
├── symptom_questions.py
├── train_model.py
├── train_final_model.py
├── intents.json
├── intents_short.json
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── dataset.csv
│   ├── list_of_symptoms_new.pickle
│   ├── list_of_symptoms.pickle
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
│
├── models/
│   ├── data.pth
│   └── fitted_model_new.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── templates/
│   └── index.html
│
├── tests/
│   ├── check_model.py
│   ├── evaluate_models.py
│   ├── inspect_candidates.py
│   └── test_knn.py
│
└── backups/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Medical Chatbot [END 2 END] [NLP]"
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required NLTK data

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## ▶️ Run Locally

Start the application:

```bash
python app.py
```

If port 5000 is occupied on macOS:

```bash
PORT=5001 python app.py
```

Then open:

```text
http://127.0.0.1:5001/
```

---

## 🧪 Testing

### Validate the trained model

```bash
python tests/check_model.py
```

### Compare machine learning models

```bash
python tests/evaluate_models.py
```

### Inspect KNN candidate diseases

```bash
python tests/inspect_candidates.py
```

### Run KNN tests

```bash
python tests/test_knn.py
```

---

## 💡 Example Conversation

```text
User:
I have chest pain

Meddy:
Got it — I've recorded chest pain.

User:
I have high fever

Meddy:
Got it — I've recorded high fever.

User:
I am coughing

Meddy:
Got it — I've recorded cough.

User:
done

Meddy:
To narrow down the prediction, do you also have
breathlessness, phlegm, rusty sputum, or chills?

User:
I have breathlessness and phlegm

Meddy:
Got it — I've also recorded breathlessness and phlegm.

User:
done

Meddy:
MODEL PREDICTION

Pneumonia
```

---

## 🔄 Conversation Flow

Meddy supports an iterative symptom collection process.

```text
Start conversation
       ↓
Enter symptom
       ↓
Enter additional symptoms
       ↓
Type "Done"
       ↓
Analyze candidate diseases
       ↓
Ask follow-up questions
       ↓
Add additional symptoms
       ↓
Type "Done"
       ↓
Generate final prediction
```

The chatbot can also accept:

```text
No
```

when the user does not have the suggested follow-up symptoms.

---

## 🛡️ Safety & Medical Disclaimer

Meddy is an educational AI/ML project.

It does **not** provide:

* Professional medical diagnosis
* Medical treatment
* Emergency medical services
* Clinical decision-making
* Replacement for a qualified healthcare professional

The prediction generated by the model should only be considered an experimental AI/ML output.

If symptoms are severe, concerning, or rapidly worsening, seek appropriate professional medical care.

---

## ⚠️ Limitations

* The dataset is relatively small.
* The model only supports symptoms represented in the dataset.
* Different diseases can share similar symptoms.
* Predictions depend on the symptoms entered by the user.
* Dataset accuracy does not guarantee real-world medical accuracy.
* The chatbot does not perform clinical examination.
* The system does not use laboratory or imaging results.
* The system should not be used for emergency or clinical decision-making.
* The model is intended for educational and demonstration purposes.

---

## 🚀 Future Improvements

Possible future improvements include:

* Larger and more diverse medical datasets
* Transformer-based NLP
* Better symptom normalization
* Explainable AI
* Improved confidence estimation
* Disease probability ranking
* Multilingual support
* Voice input
* Conversation history
* User authentication
* Doctor consultation integration
* Automated testing
* CI/CD
* Cloud deployment
* Production monitoring
* Improved medical safety mechanisms

---

## 🎯 Project Goal

This project demonstrates an end-to-end AI/ML workflow:

```text
Dataset
   ↓
Data preprocessing
   ↓
NLP
   ↓
Feature engineering
   ↓
Model evaluation
   ↓
KNN model
   ↓
Flask backend
   ↓
Interactive frontend
   ↓
Deployment
```

The project was developed as an educational demonstration of integrating machine learning, NLP, and web development into a complete end-to-end application.

---

## 📌 Current Model Information

```text
Model:
KNeighborsClassifier

Neighbors:
6

Distance Metric:
Cosine

Input Features:
131

Disease Classes:
41

Unique Diagnosis Examples:
304
```

---
## 📄 Screen Shot
<img width="1119" height="783" alt="image" src="https://github.com/user-attachments/assets/573e45cd-d02a-44de-8eb4-1e28df48bfd4" />
<img width="1396" height="627" alt="Screenshot 2026-09-01 at 12 49 53 PM" src="https://github.com/user-attachments/assets/4647141e-954c-47f4-a9d1-cd723a7392a0" />
<img width="1396" height="627" alt="Screenshot 2026-09-01 at 12 50 00 PM" src="https://github.com/user-attachments/assets/848d8ea4-1356-4f09-97bd-8cd0c9045741" />
<img width="1396" height="627" alt="Screenshot 2026-09-01 at 12 50 09 PM" src="https://github.com/user-attachments/assets/93c97b6d-f609-4db7-9345-e997d530fe74" />
<img width="1396" height="627" alt="Screenshot 2026-09-01 at 12 50 17 PM" src="https://github.com/user-attachments/assets/c17774ac-047c-4a44-9b72-d41d23164161" />
<img width="1396" height="639" alt="Screenshot 2026-09-01 at 12 50 38 PM" src="https://github.com/user-attachments/assets/58b12339-f455-437c-811a-9f64a6ff0ad9" />



## 📄 License
## BUILD BY - NITESH YADAV 
@VICKYCODESWITH

