**Sentiment Analysis of Product Reviews using Naive Bayes**
-------------------------------------------------------------
🎯 _Overview_
This project focuses on building a text classification model to identify the sentiment of product reviews using the Naive Bayes algorithm. It leverages TF-IDF/CountVectorizer, a variety of text preprocessing steps, and clean visualizations to display evaluation metrics.
✨ Features
•	✅ Clean and preprocess text (stopword removal, stemming, punctuation stripping)
•	✅ Convert text into numeric vectors using TF-IDF or CountVectorizer
•	✅ Train a Naive Bayes classifier (MultinomialNB / BernoulliNB)
•	✅ Evaluate using accuracy, precision, recall, F1-score, and confusion matrix
•	✅ Predict sentiment for new, unseen user reviews
•	✅ Visualize class distribution, top words, and model performance
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🛠️ _Tech Stack_
Category	Libraries / Tools
Language	Python 3.x
Data Handling	Pandas, NumPy
NLP	NLTK / spaCy
ML Models	Scikit-learn
Visualization	Matplotlib, Seaborn, Plotly
Environment	Jupyter Notebook / VS Code
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🚀 _Installation_
1. Clone the repository
git clone https://github.com/your-username/sentiment-analysis-naive-bayes.git
cd sentiment-analysis-naive-bayes
2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔧 _Usage_
🧪 **Option 1**: Jupyter Notebook
jupyter notebook Sentiment_Analysis_Naive_Bayes.ipynb
⚙️**Option 2**: Python Script
python sentiment_analysis.py
**Steps Performed:**
•	✅ Load and explore the dataset
•	✅ Preprocess the text data
•	✅ Feature extraction with TF-IDF or CountVectorizer
•	✅ Split data into train/test sets
•	✅ Train a Naive Bayes model
•	✅ Evaluate and visualize results
•	✅ Predict sentiment on new input
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📂_ Dataset_
Use any labeled dataset with at least two columns: Review, Sentiment.
Example:
Review,Sentiment
"This product is fantastic!",positive
"Horrible experience, never buying again",negative
**You can use:**
•	Amazon Product Reviews - https://www.kaggle.com/snap/amazon-fine-food-reviews
•	IMDb Movie Reviews - https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧠**Model Details**
Component	Value
Algorithm	Naive Bayes (Multinomial / Bernoulli)
Vectorization	CountVectorizer / TfidfVectorizer
Evaluation	Accuracy, Precision, Recall, F1-score
Output	Binary Sentiment (Positive/Negative)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📊 _Results_
•	✅ Model Accuracy: 85% - 90% (based on dataset and preprocessing)
•	✅ Evaluation Tools: Confusion Matrix, Classification Report
•	✅ Visualizations:
•	✅   - Class distribution bar/pie charts
•	✅   - Most frequent positive/negative words
•	✅   - Model performance metrics
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔮_Future Work_
•	🚀 Add neutral sentiment support (multi-class)
•	🚀 Deploy as a Flask or Streamlit web app
•	🚀 Integrate word embeddings (Word2Vec, GloVe)
•	🚀 Try alternative models: SVM, Logistic Regression
•	🚀 Use GridSearchCV for hyperparameter tuning
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📝_ License_
This project is licensed under the MIT License. Feel free to use and modify for educational and professional purposes.
🙏 Acknowledgments
•	Scikit-learn Documentation - https://scikit-learn.org/
•	NLTK Library - https://www.nltk.org/
•	Kaggle Datasets - https://www.kaggle.com/
•	Online ML/NLP courses & tutorials
