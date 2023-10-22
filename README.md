# Sentiflix

![Logo](https://pngimg.com/d/netflix_PNG22.png)

This project is aimed at sentiment analysis of movie reviews. It utilizes Natural Language Processing (NLP) techniques and Machine Learning (ML) models to classify reviews as positive or negative.

## Technologies Used
- Python
- Flask
- Scikit-learn
- NLTK
- HTML
- CSS
- Bootstrap

## Project Overview

The project consists of the following components:

1. **Data Preprocessing**: The movie reviews dataset is used to train the models. It includes positive and negative reviews.

2. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is applied to convert the text data into numerical format suitable for machine learning.

3. **Models Implemented**:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Support Vector Machine
   - Random Forest
   - k-Nearest Neighbors
   - Multi-Layer Perceptron
   - Ensemble Model (Voting Classifier)

4. **Web Application**:
   - Built using Flask, a web framework in Python.
   - Allows users to input a movie review and get sentiment prediction.

5. **Visualization**:
   - Heatmap is generated to visualize model performance.

6. **User Interface**:
   - Designed using HTML, CSS, and Bootstrap for a user-friendly experience.


## Directory Structure

- `data/`: Contains movie reviews dataset.
- `models/`: Python files for ML models.
- `static/`: Contains static files (e.g., images, CSS).
- `templates/`: HTML templates for web pages.
- `app.py`: Main Flask application file.
- `heatmap.html`: HTML template for heatmap page.
- `README.md`: Project documentation.

## Acknowledgements

- The movie reviews dataset used in this project is sourced from the NLTK corpus.

## License

This project is licensed under the [MIT License](LICENSE).

