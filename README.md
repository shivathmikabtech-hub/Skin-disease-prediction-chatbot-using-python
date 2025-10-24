import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

try:
    df = pd.read_csv('Symptoms-Disease.csv')
except FileNotFoundError:
    print("Error: '.csv' not found. Please upload the dataset or provide the correct path.")
    df = pd.read_csv('/content/Symptoms-Disease.csv')
    print("Using sample dataset as fallback.")

# NLP Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

if 'Symptoms' in df.columns:
    df['processed_symptoms'] = df['Symptoms'].apply(preprocess_text)
else:
    print("Error: 'symptoms' column not found in the dataset.")

if 'processed_symptoms' in df.columns:
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['processed_symptoms'])
    
    if 'Disease' in df.columns:
        y = df['Disease']
        model = MultinomialNB()
        model.fit(X, y)

        def get_prediction(user_input):
            """Predict disease from user input"""
            processed_input = preprocess_text(user_input)
            input_vector = vectorizer.transform([processed_input])
            prediction = model.predict(input_vector)[0]
            confidence = model.predict_proba(input_vector)[0].max()
            return prediction, confidence

        def chatbot():
            """Run the chatbot"""
            print("=" * 50)
            print("🩺 SKIN DISEASE PREDICTION CHATBOT")
            print("=" * 50)
            print("\nWelcome! Tell me about your skin symptoms.")
            # print("(Type 'quit' to exit)\n")

            while True:
                user_input = input("You: \n").strip()

                if user_input.lower() in ['quit', 'bye', 'exit', 'no', 'thanks','ok']:
                    print("\n✋ Thank you for using our chatbot!")
                    print("⚠️ Remember to consult a dermatologist for proper diagnosis.")
                    print("Stay healthy! 💙\n")
                    break

                if not user_input:
                    print("Bot: Please describe your symptoms.\n")
                    continue

                disease, confidence = get_prediction(user_input)

                print(f"\nBot: Based on your symptoms, you might have: **{disease}**")
                print(f"Confidence: {confidence * 100:.2f}%")
                print("⚠️ Please consult a dermatologist for proper diagnosis.\n")

                while True:
                    follow_up = input("Bot: Do you have any other symptoms or questions about this prediction? (yes/no): \n").strip().lower()
                    if follow_up in ['yes', 'yehh']:
                        more_symptoms = input("Bot: Please tell me more about your other symptoms or questions: \n").strip()
                        if more_symptoms:
                            disease, confidence = get_prediction(more_symptoms)
                            print(f"\nBot: Based on the additional information, you might have: **{disease}**")
                            print(f"Confidence: {confidence * 100:.2f}%")
                            print("⚠️ Please consult a dermatologist for proper diagnosis.\n")
                            break
                        else:
                            print("Bot: Please provide some additional information.\n")
                    elif follow_up in ['no', 'na']:
                        print("Bot: Alright. Remember to consult a dermatologist for proper diagnosis.\n")
                        break
                    else:
                        print("Bot: Please answer with 'yes' or 'no'.\n")


        if __name__ == "__main__":
            chatbot()
    else:
        print("Error: 'disease' column not found in the dataset.")
else:
    print("Error: 'processed_symptoms' column not found in the dataset after preprocessing.")
