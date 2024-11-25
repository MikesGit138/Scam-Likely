import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from sklearn.preprocessing import LabelEncoder
import numpy as np

def setup_nltk():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

setup_nltk()

class ScamDetector:
    def __init__(self):
        # Enhanced TF-IDF configuration
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Include phrases up to 3 words
            max_features=5000,
            min_df=2
        )
        # Increased number of trees and added class weight
        self.model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            max_depth=20
        )
        self.label_encoder = LabelEncoder()
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Enhanced scam detection patterns
        self.scam_patterns = {
            'urgency': r'\b(urgent|immediate|critical|important|act now|limited time|expires|final chance|last chance|only today)\b',
            'pressure': r'\b(verify|confirm|validate|authenticate|provide|submit|send|click|login now|activate)\b',
            'reward': r'\b(won|winner|congratulations|selected|special offer|exclusive|free|bonus|prize|gift)\b',
            'threat': r'\b(suspend|terminat|lock|restrict|cancel|delete|remov|block|compromise|unusual|suspicious)\b',
            'personal_info': r'\b(password|credential|account|ssn|social security|credit card|bank detail|identity)\b',
            'money': r'(?:^\$|(?<=\s)\$)\d+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars|usd)\b',
        }

    def extract_scam_features(self, message):
        """Extract enhanced scam-specific features"""
        message = message.lower()
        features = {
            'has_url': 1 if re.search(r"http[s]?://\S+|\[link\]|\[tracking link\]", message) is not None else 0,
            'exclamation_marks': message.count('!'),
            'dollar_signs': message.count('$'),
            'message_length': len(message),
            'capitalized_words': len(re.findall(r'\b[A-Z]{2,}\b', message)),
            'urgency_count': len(re.findall(self.scam_patterns['urgency'], message)),
            'pressure_count': len(re.findall(self.scam_patterns['pressure'], message)),
            'reward_count': len(re.findall(self.scam_patterns['reward'], message)),
            'threat_count': len(re.findall(self.scam_patterns['threat'], message)),
            'personal_info_count': len(re.findall(self.scam_patterns['personal_info'], message)),
            'money_mentions': len(re.findall(self.scam_patterns['money'], message)),
            'has_click_here': 1 if 'click here' in message.lower() else 0,
            'suspicious_puncs': len(re.findall(r'[!?]{2,}', message))
        }
        return features

    def preprocess_text(self, message):
        """Enhanced text preprocessing"""
        if not isinstance(message, str):
            message = str(message)

        # Extract features before cleaning
        features = self.extract_scam_features(message)

        # Text cleaning
        message = message.lower()
        message = re.sub(r"http[s]?://\S+|\[link\]|\[tracking link\]", "URL", message)
        message = message.translate(str.maketrans("", "", string.punctuation))

        try:
            tokens = word_tokenize(message)
        except Exception:
            tokens = self.tokenizer.tokenize(message)

        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.lower() not in stop_words]
        except Exception:
            pass

        return " ".join(tokens), features

    def extract_features(self, text):
        """Convert text and additional features into a format suitable for the model"""
        processed_text, features = self.preprocess_text(text)
        text_features = self.vectorizer.transform([processed_text])

        feature_values = [
            features['has_url'],
            features['exclamation_marks'],
            features['dollar_signs'],
            features['message_length'],
            features['capitalized_words'],
            features['urgency_count'],
            features['pressure_count'],
            features['reward_count'],
            features['threat_count'],
            features['personal_info_count'],
            features['money_mentions'],
            features['has_click_here'],
            features['suspicious_puncs']
        ]

        return np.hstack((text_features.toarray(), [feature_values]))

    def train(self, dataset):
        """Train the model with enhanced features"""
        messages = dataset['message']
        labels = dataset['class']

        processed_texts = []
        additional_features_list = []

        for message in messages:
            processed_text, features = self.preprocess_text(message)
            processed_texts.append(processed_text)
            additional_features_list.append([
                features['has_url'],
                features['exclamation_marks'],
                features['dollar_signs'],
                features['message_length'],
                features['capitalized_words'],
                features['urgency_count'],
                features['pressure_count'],
                features['reward_count'],
                features['threat_count'],
                features['personal_info_count'],
                features['money_mentions'],
                features['has_click_here'],
                features['suspicious_puncs']
            ])

        text_features = self.vectorizer.fit_transform(processed_texts)
        X = np.hstack((text_features.toarray(), additional_features_list))
        y = self.label_encoder.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, message):
        """Make a prediction with detailed analysis"""
        features = self.extract_features(message['message'])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        confidence = probability[prediction]

        # Extract feature values for explanation
        _, extracted_features = self.preprocess_text(message['message'])

        return {
            "message": message,
            "prediction": "Spam" if prediction == 1 else "Ham",
            "confidence": f"{confidence:.2%}",
            "risk_indicators": {
                k: v for k, v in extracted_features.items() if v > 0
            }
        }

def main():
    print("Initializing Enhanced Scam Detection System...")

    # Load dataset
    file_path = './scam_msg_dataset/scam.csv'
    dataset = pd.read_csv(file_path, encoding='latin-1')
    dataset = dataset[['class', 'message']]
    dataset['class'] = dataset['class'].map({'ham': 0, 'spam': 1})

    # Initialize and train detector
    detector = ScamDetector()
    print("Training model...")
    report = detector.train(dataset)
    print("\nModel Performance:")
    print(report)

    # Your test cases here...
    test_cases = [
        {"message": "You have won a $1000 gift card! Claim it now by providing your details.", "expected": "Spam"},
        {"message": "Your monthly statement is ready to view in online banking.", "expected": "Ham"}
    ]

    print("\nTest Results:")
    for case in test_cases:
        result = detector.predict(case)
        print(f"\nMessage: {case}")
        print(f"Analysis: {result}")

if __name__ == "__main__":
    main()