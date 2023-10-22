from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
import nltk

# Load and preprocess the data
nltk.download('movie_reviews')
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')
positive_reviews = [movie_reviews.raw(fileid) for fileid in positive_fileids]
negative_reviews = [movie_reviews.raw(fileid) for fileid in negative_fileids]
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
reviews = positive_reviews + negative_reviews

# Feature extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(reviews)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

'''# Load and preprocess the data
nltk.download('movie_reviews')
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')
positive_reviews = [movie_reviews.raw(fileid) for fileid in positive_fileids]
negative_reviews = [movie_reviews.raw(fileid) for fileid in negative_fileids]
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
reviews = positive_reviews + negative_reviews

# Feature extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(reviews)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Train Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train Support Vector Machine model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Train k-Nearest Neighbors model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Train Multi-Layer Perceptron model
mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)

# Create a voting classifier
voting_model = VotingClassifier(estimators=[
    ('lr', lr_model), 
    ('nb', nb_model), 
    ('svm', svm_model), 
    ('rf', rf_model), 
    ('knn', knn_model), 
    ('mlp', mlp_model)], 
    voting='hard')

voting_model.fit(X_train, y_train)'''


def predict_sentiment(input_text):
    input_review = [input_text]
    input_tfidf = tfidf_vectorizer.transform(input_review)
    prediction = lr_model.predict(input_tfidf)
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return sentiment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    sentiment = predict_sentiment(user_input)
    return render_template('index.html', user_input=user_input, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
