import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset (specifying the first column as the index)
try:
    data = pd.read_csv('dataset.csv', index_col=0)  # Treat the first column as the index
    assert 'statement' in data.columns and 'status' in data.columns
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Drop rows with NaN values in 'statement'
data.dropna(subset=['statement'], inplace=True)

# Preprocess text data
def preprocess_text(text):
    text = text.astype(str)  # Convert to string
    text = text.str.lower()  # Lowercase
    text = text.str.replace('[^a-zA-Z\\s]', '', regex=True)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    text = text.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))  # Lemmatization
    return text

data['statement'] = preprocess_text(data['statement'])

# Vectorize the text using TF-IDF with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(data['statement'])
y = data['status']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (e.g., using Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Function to preprocess user input
def preprocess_user_input(input_text):
    input_df = pd.Series([input_text])  # Create a pandas Series from the input text
    preprocessed_input = preprocess_text(input_df)  # Preprocess the input
    return preprocessed_input

# Function to predict the category of user input
def predict_user_input(input_text):
    preprocessed_input = preprocess_user_input(input_text)  # Preprocess the user input
    input_vectorized = vectorizer.transform(preprocessed_input)  # Vectorize the preprocessed input
    prediction = model.predict(input_vectorized)  # Make prediction
    return prediction[0]  # Return the predicted category

# Test the model with user input
if __name__ == "__main__":

    while (True):
        user_input = input("Please enter your statement regarding mental health: ")
        predicted_category = predict_user_input(user_input)
        print(f"The predicted mental health category is: {predicted_category}")
