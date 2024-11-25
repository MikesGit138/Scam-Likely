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
            tokens = word_tokenize(message)
        except Exception as e:
            tokens = self.tokenizer.tokenize(message)

        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.lower() not in stop_words]
        except Exception as e:
            pass

        return " ".join(tokens), features

    def train(self, dataset):
        """Train the model with the given dataset"""
        messages = dataset['message']
        labels = dataset['class']

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
        X = np.hstack((text_features.toarray(), additional_features_list))

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, message):
        """Make a prediction with confidence score"""
        features = self.extract_features(message)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        confidence = probability[prediction]

        return {
            "message": message,
            "prediction": "Spam" if prediction == 1 else "Ham",
            "confidence": f"{confidence:.2%}"
        }

    def extract_features(self, text):
        """Convert text and additional features into a format suitable for the model"""
        processed_text, additional_features = self.preprocess_text(text)
        text_features = self.vectorizer.transform([processed_text])

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


def main():
    print("Initializing Scam Detection System...")

    # Load the dataset
    file_path = './scam_msg_dataset/scam.csv'
    dataset = pd.read_csv(file_path, encoding='latin-1')
    dataset = dataset[['class', 'message']]  # Select relevant columns
    dataset['class'] = dataset['class'].map({'ham': 0, 'spam': 1})  # Convert labels

    # Initialize and train the detector
    detector = ScamDetector()
    print("Training model...")
    report = detector.train(dataset)
    print("\nModel Performance:")
    print(report)

    # Test cases
    test_cases = [
        "This is not a scam! Please respond immediately. We need you to send us your bank details.",
        "Your monthly statement is ready to view in online banking."
    ]

    print("\nTest Results:")
    for case in test_cases:
        result = detector.predict(case)
        print(f"\nMessage: {case}")
        print(f"Analysis: {result}")


if __name__ == "__main__":
    main()
