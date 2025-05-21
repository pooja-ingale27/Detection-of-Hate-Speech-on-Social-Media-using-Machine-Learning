
📌 Overview :

With the rise of social media usage worldwide, the freedom of expression online has also led to an increase in harmful and derogatory content. This project explores the development of an efficient machine learning model to detect hate speech on social media platforms.

🎯 Aim :

The primary goal of this project is to build and evaluate machine learning models that can accurately detect hate speech in textual social media content.

✅ Objectives :

Review existing research and methodologies related to hate speech detection

Preprocess a publicly available social media dataset

Experiment with feature representation techniques (Unigram, Bigram, Trigram)

Train and evaluate machine learning models (SVM and Random Forest)

Measure performance using standard evaluation metrics

🧠 Methodology:

📂 Data Preprocessing

Dataset - https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

Cleaning and normalization

Anonymizing usernames

Censoring explicit words

Tokenization

Stopwords removal

Lemmatization


🔡 Feature Engineering:

Unigram

Bigram

Trigram representations


🤖 Models Used:

Support Vector Machine (SVM)

Random Forest (RF)


📊 Evaluation Metrics:

Accuracy

Precision

Recall

F1 Score

ROC AUC Score


🧪 Results:

Model	Representation	F1 Score (CV)	F1 Score (Unseen Data)	ROC AUC

SVM	   Unigram	      0.918	        0.895	                  0.82

RF	   Unigram	      0.916	        0.922	                  0.83

Both models showed strong performance, with Random Forest slightly outperforming SVM on unseen data.

🧾 Conclusion:

The Random Forest model was most effective in detecting hate speech, showing strong generalizability on unseen social media content. However, the study also highlights key limitations:

Class imbalance in the dataset

Difficulty in generalizing across platforms and demographics

Need to consider user intent and contextual nuances


🔭 Future Work:

Develop multilingual detection capabilities

Extend detection to multimedia content (audio, video, and images)

Explore deep learning approaches and transformer-based models
