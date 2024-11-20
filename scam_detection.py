import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Download all required NLTK data
def setup_nltk():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

# Initialize NLTK
setup_nltk()

class ScamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.tokenizer = RegexpTokenizer(r'\w+')  # Fallback tokenizer if needed

    def preprocess_text(self, message):
        """Enhanced text preprocessing with additional features"""
        if not isinstance(message, str):
            message = str(message)

        # Convert to lowercase
        message = message.lower()

        # Store some features before cleaning
        features = {
            'has_url': 1 if re.search(r"http[s]?://\S+", message) is not None else 0,
            'urgent_words': len(re.findall(r'\b(urgent|immediate|critical|important|act now|limited time)\b', message.lower())),
            'exclamation_marks': message.count('!'),
            'dollar_signs': message.count('$'),
            'message_length': len(message)
        }

        # Remove URLs but keep track of them
        message = re.sub(r"http[s]?://\S+", "URL", message)

        # Remove punctuation
        message = message.translate(str.maketrans("", "", string.punctuation))

        # Tokenize and remove stopwords
        try:
            # Try using NLTK word_tokenize
            tokens = word_tokenize(message)
        except Exception as e:
            # Fallback to simple tokenization if NLTK fails
            print(f"Falling back to simple tokenization: {e}")
            tokens = self.tokenizer.tokenize(message)

        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.lower() not in stop_words]
        except Exception as e:
            print(f"Error removing stopwords: {e}")
            # Fallback: just use the tokens as is
            pass

        return " ".join(tokens), features

    def extract_features(self, text):
        """Convert text and additional features into a format suitable for the model"""
        processed_text, additional_features = self.preprocess_text(text)
        text_features = self.vectorizer.transform([processed_text])

        # Convert sparse matrix to dense array and combine with additional features
        feature_array = np.hstack((
            text_features.toarray(),
            [[
                additional_features['has_url'],
                additional_features['urgent_words'],
                additional_features['exclamation_marks'],
                additional_features['dollar_signs'],
                additional_features['message_length']
            ]]
        ))

        return feature_array

    def train(self, messages, labels):
        """Train the model with expanded dataset"""
        # Process all texts and extract features
        processed_texts = []
        additional_features_list = []

        for message in messages:
            processed_text, features = self.preprocess_text(message)
            processed_texts.append(processed_text)
            additional_features_list.append([
                features['has_url'],
                features['urgent_words'],
                features['exclamation_marks'],
                features['dollar_signs'],
                features['message_length']
            ])

        # Create TF-IDF features
        text_features = self.vectorizer.fit_transform(processed_texts)

        # Combine with additional features
        X = np.hstack((
            text_features.toarray(),
            additional_features_list
        ))

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, message, sender_email):
        """Make a prediction with confidence score"""
        # Extract features
        features = self.extract_features(message)

        # Get prediction and probability
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        # Check sender legitimacy
        sender_legitimate = self.is_legitimate(sender_email)

        # Get confidence score
        confidence = probability[prediction]

        return {
            "message": message,
            "prediction": "Scam" if prediction == 1 else "Legitimate",
            "confidence": f"{confidence:.2%}",
            "legitimate_sender": sender_legitimate,
            "risk_level": self.calculate_risk_level(confidence, sender_legitimate)
        }

    def is_legitimate(self, sender_email):
        """Enhanced legitimate sender verification"""
        legitimate_domains = {
            "ncb.com": "NCB Bank",
            "scotiabank.com": "Scotiabank",
            "td.com": "TD Bank",
            "rbc.com": "Royal Bank",
            "cibc.com": "CIBC"
        }
        domain = sender_email.split("@")[-1].lower()
        return domain in legitimate_domains

    def calculate_risk_level(self, confidence, sender_legitimate):
        """Calculate risk level based on multiple factors"""
        if confidence >= 0.8 and not sender_legitimate:
            return "High"
        elif confidence >= 0.6 or not sender_legitimate:
            return "Medium"
        return "Low"

def main():
    print("Initializing Scam Detection System...")

    # Create expanded dataset
    data = {
        "Message": [
            "Urgent: Verify your account at http://fakeurl.com",
            "Your NCB account is secure. Visit us at ncb.com",
            "Important: Your account will be closed unless you act now!",
            "Welcome to Scotiabank. Your account setup is complete.",
            "ATTENTION: $500 has been charged to your account! Verify now!",
            "Your monthly statement is ready to view online.",
            "Limited time offer: Claim your prize now! $1000 waiting!!!",
            "Please review your recent transaction at TD Bank.",
            "Warning: Multiple login attempts detected! Verify ASAP!",
            "Your automatic payment has been processed successfully."
        ],
        "Label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Scam, 0 = Legitimate
    }
    df = pd.DataFrame(data)

    # Initialize and train the detector
    detector = ScamDetector()
    print("Training model...")
    report = detector.train(df["Message"], df["Label"])
    print("\nModel Performance:")
    print(report)

    # Test cases
    test_cases = [
        {
            "message": "Act now! Your account will be deactivated unless you verify immediately!",
            "sender": "alerts@fakebank.com"
        },
        {
            "message": "Your monthly statement is ready to view in online banking.",
            "sender": "statements@scotiabank.com"
        }
    ]

    print("\nTest Results:")
    for case in test_cases:
        result = detector.predict(case["message"], case["sender"])
        print(f"\nMessage: {case['message']}")
        print(f"Sender: {case['sender']}")
        print(f"Analysis: {result}")

if __name__ == "__main__":
    main()