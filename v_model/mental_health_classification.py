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

# Load the dataset
try:
    data = pd.read_csv('dataset.csv', index_col=0)
    # Check the first few rows and ensure the dataset contains the necessary columns
    print(data.head())  # Check the headers and some data
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

# Function to evaluate different classifiers
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Evaluate different models
models = [
    LogisticRegression(max_iter=1000), 
    RandomForestClassifier(), 
    SVC(probability=True), 
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
]

for model in models:
    evaluate_model(model, X_train, y_train, X_test, y_test)

# Hyperparameter tuning for Random Forest
rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters for Random Forest:", grid_search.best_params_)
print("Best cross-validation score for Random Forest:", grid_search.best_score_)

# Create a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
], voting='soft')

# Train and evaluate the voting classifier
evaluate_model(voting_clf, X_train, y_train, X_test, y_test)

# Cross-validation for Logistic Regression
cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print("Cross-validation scores for Logistic Regression:", cv_scores)
print("Mean CV score for Logistic Regression:", cv_scores.mean())
