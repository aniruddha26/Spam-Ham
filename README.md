# 📩 Spam-Ham Classifier

A simple NLP-based spam detection project built with Python. This project classifies SMS messages as either **Spam** or **Ham (Not Spam)** using traditional NLP techniques like Bag of Words (BoW), TF-IDF, and Word2Vec.

> 🚀 This is a practice project for learning and applying NLP techniques in machine learning.

## 🧠 Techniques Used

- **Text Preprocessing**: Tokenization, stopword removal, stemming
- **Feature Extraction**:
  - Bag of Words (BoW)
  - TF-IDF Vectorization
  - Word2Vec embeddings
- **Classification Models**:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Machines (SVM)

## 📁 Dataset

- Publicly available SMS Spam Collection dataset from UCI repository.
- Contains ~5,500 labeled messages.

## 🔧 Project Structure

Spam-Ham/
│
├── data/ # Contains the dataset (e.g., spam.csv)
├── notebooks/ # Jupyter notebooks for experiments
├── models/ # Trained models (optional)
├── spam_ham_classifier.py # Main script for preprocessing and classification
└── README.md

bash
Copy
Edit

## ⚙️ Installation & Setup

1. Clone the repository:
    git clone https://github.com/aniruddha26/Spam-Ham.git
    cd Spam-Ham

2. Install the dependencies:
    pip install -r requirements.txt

3. Run the main script or explore the Jupyter notebooks

## 📊 Results
Performance metrics like Accuracy, Precision, Recall, and F1 Score are used for evaluation.

Experimented with various feature extraction methods to compare performance.

## 🛠️ Future Improvements
Add deep learning models (e.g., LSTM, BERT)

## 📚 References
SMS Spam Collection Dataset - UCI ML Repo

Scikit-learn, Gensim, NLTK libraries

## 👨‍💻 Author
Aniruddha Alkari


