# 🩺 Skin Disease Prediction Chatbot

An AI-powered **Skin Disease Prediction Chatbot** built using **Natural Language Processing (NLP)** and **Machine Learning**.
The chatbot analyzes user-provided skin symptoms and predicts possible skin diseases using a **Naive Bayes Classification Model**.

---

## 📌 Project Overview

This project uses:

* **NLP preprocessing**
* **TF-IDF Vectorization**
* **Multinomial Naive Bayes**
* **Interactive chatbot system**

to predict skin diseases from user-entered symptoms.

The chatbot accepts symptom descriptions in natural language and provides:

* Predicted disease
* Confidence score
* Follow-up interaction support

---

## 🚀 Features

✔ Symptom-based disease prediction
✔ NLP text preprocessing
✔ Stopword removal & lemmatization
✔ TF-IDF feature extraction
✔ Naive Bayes classification
✔ Confidence score prediction
✔ Interactive chatbot conversation
✔ Follow-up symptom checking

---

# 🛠 Technologies Used

* Python
* Pandas
* Scikit-learn
* NLTK
* TF-IDF Vectorizer
* Multinomial Naive Bayes

---

# 📂 Dataset

Dataset used:
`Symptoms-Disease.csv`

Dataset contains:

* **Symptoms** → Input text
* **Disease** → Target label

Example:

| Symptoms                    | Disease  |
| --------------------------- | -------- |
| itching, skin rash, redness | Eczema   |
| pimples, oily skin          | Acne     |
| white patches on skin       | Vitiligo |

---

# ⚙️ Working Process

## 1️⃣ Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv('Symptoms-Disease.csv')
```

---

## 2️⃣ Text Preprocessing

Symptoms are cleaned using NLP techniques:

* Lowercasing
* Removing special characters
* Tokenization
* Stopword removal
* Lemmatization

Example:

Input:

```text
Red itchy skin with rashes
```

Processed:

```text
red itchy skin rash
```

---

## 3️⃣ Feature Extraction

TF-IDF converts text into numerical vectors.

```python
vectorizer = TfidfVectorizer(max_features=100)
```

---

## 4️⃣ Model Training

A Multinomial Naive Bayes model is trained.

```python
model = MultinomialNB()
model.fit(X, y)
```

---

## 5️⃣ Prediction System

The chatbot predicts diseases from user symptoms.

Example:

```text
You: I have itchy red skin and rashes

Bot: Based on your symptoms, you might have: Eczema
Confidence: 92.45%
```

---

# 💬 Chatbot Flow

```text
User Input
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Naive Bayes Prediction
     ↓
Disease + Confidence Output
```

---

# 📸 Sample Output

```text
==================================================
🩺 SKIN DISEASE PREDICTION CHATBOT
==================================================

Welcome! Tell me about your skin symptoms.

You:
I have red itchy skin and dry patches

Bot:
Based on your symptoms, you might have: Eczema
Confidence: 91.23%

⚠️ Please consult a dermatologist for proper diagnosis.
```

---

# 📊 Machine Learning Model

## Algorithm Used:

### Multinomial Naive Bayes

Why Naive Bayes?

* Works well for text classification
* Fast and efficient
* Good accuracy for symptom prediction
* Handles sparse TF-IDF vectors effectively

---

# 🧠 NLP Techniques Used

* Tokenization
* Stopword Removal
* Lemmatization
* Text Cleaning
* TF-IDF Feature Engineering

---

# 📁 Project Structure

```text
Skin-Disease-Prediction-Chatbot/
│
├── Symptoms-Disease.csv
├── chatbot.py
├── README.md
└── requirements.txt
```

---

# ▶️ How to Run the Project

## 1️⃣ Install Dependencies

```bash
pip install pandas scikit-learn nltk
```

---

## 2️⃣ Run the Python File

```bash
python chatbot.py
```

---

# 📌 Future Improvements

* Add Deep Learning models
* Use larger medical datasets
* Deploy using Streamlit/Flask
* Add multilingual support
* Add image-based skin disease detection
* Improve chatbot conversation flow

---

# ⚠️ Disclaimer

This chatbot is only for educational and research purposes.
It should not replace professional medical advice or diagnosis.

Always consult a dermatologist or healthcare professional.

---

# 👨‍💻 Author

Developed as an AI/ML project using NLP and Machine Learning techniques.

---

# ⭐ If You Like This Project

Give this repository a ⭐ on GitHub!
