# Sentiment Analysis of Trading 212 Reviews Using NLP

## 📌 Overview
This project applies **natural language processing (NLP)** to analyze user sentiment on the **Trading 212** platform based on **Trustpilot reviews**. Various machine learning and deep learning models were tested to classify sentiments as **positive, negative, or neutral**, providing valuable insights into customer feedback.

## 📂 Project Structure
- **`Sentiment Analysis(NLP Code).ipynb`** - Jupyter Notebook containing data preprocessing, NLP model training, and evaluation.
- **`Sentiment_Analysis.pdf`** - A detailed report on sentiment analysis techniques and model comparisons.
- **`figures/`** - Directory containing sentiment distribution plots, confusion matrices, and accuracy/loss curves.

## 🛠️ Technologies Used
- **Python (NLTK, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn)**
- **Jupyter Notebook**
- **Traditional ML Models:**
  - Naive Bayes (Gaussian, Multinomial)
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
- **Deep Learning Models:**
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)
  - BERT (Bidirectional Encoder Representations from Transformers)

## 🔹 Key Features
### **1. Data Collection & Preprocessing**
- Web-scraped **22,640 Trustpilot reviews**.
- Data cleaning, **removal of null values, stop words, punctuation, and lemmatization**.
- **Sentiment labeling** based on review ratings.

### **2. Sentiment Classification Models**
- **Traditional NLP Methods:** Applied **TF-IDF vectorization**, logistic regression, and SVM to establish baselines.
- **Deep Learning Approaches:** Implemented **LSTM, GRU, and RNN** for improved contextual understanding.
- **Transformers (BERT):** Leveraged pre-trained transformer-based models for state-of-the-art sentiment classification.

### **3. Model Evaluation & Performance Metrics**
- Accuracy and F1-scores for model comparisons.
- **Confusion matrices** to analyze misclassification.
- **ROC-AUC curves** for measuring classification performance.

## 📊 Insights & Results
- **Best-performing model:** **Support Vector Machine (SVM) with 100% accuracy** in sentiment classification.
- **Deep learning models:** **LSTM achieved 98.9% accuracy**, demonstrating strong contextual analysis.
- **Feature importance analysis:** TF-IDF-based insights on the most influential words affecting sentiment.

## 🚀 Getting Started
### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/iabidoye/Sentiment_Analysis.git
   cd Sentiment_Analysis
   
2. Install required dependencies:
   ```bash
   pip install nltk tensorflow keras scikit-learn pandas numpy matplotlib seaborn

3. Open the Jupyter Notebook and run the analysis:
   ```bash
   jupyter notebook "Sentiment Analysis(NLP Code).ipynb"

🤝 Contribution
Contributions are welcome! If you have improvements, new sentiment models, or dataset expansions, feel free to submit a pull request.

📧 Contact
For inquiries or collaborations, please reach out.
