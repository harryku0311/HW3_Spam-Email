# 📱 SMS Spam Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning application designed to classify SMS messages as either **Spam** or **Ham** (legitimate). Built with **Python** and **Streamlit**, this project demonstrates the end-to-end data science workflow: from exploratory data analysis (EDA) and text preprocessing to model training, evaluation, and live inference.

## 🌟 Key Features

The application is organized into four main sections:

### 1. 📊 Overview
- **Data Statistics**: View the total count of messages and the class distribution (Spam vs. Ham).
- **Raw Data Preview**: Inspect the first few rows of the dataset to understand the input structure.

### 2. 📈 Visualizations
- **Word Clouds**: Generate distinct word clouds for 'Spam' and 'Ham' messages to visualize frequent terms.
- **Token Frequency**: Interactive bar charts showing the top 20 most frequent words for each class.
- **Message Length Distribution**: A histogram comparing the character count distributions of spam and legitimate messages.

### 3. 🤖 Model Training & Analysis
- **Advanced Preprocessing**: Uses NLTK for stopword removal and stemming, and TF-IDF for vectorization.
- **Model Options**:
    - **Baseline SVM**: Default hyperparameters.
    - **Optimized SVM**: Linear kernel with C=1.0.
    - **Advanced SVM**: Incorporates **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance, coupled with **GridSearch** for hyperparameter tuning.
- **Performance Metrics**: Real-time display of Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Visual Evaluation**: Interactive Confusion Matrix and ROC Curves using Plotly.

### 4. 🚀 Live Inference
- **Real-time Prediction**: Type any SMS message to check if it's Spam or Ham.
- **Confidence Scores**: See the model's probability/confidence rating for the prediction.
- **Random Examples**: Quickly test the model with random samples from the dataset.

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Imbalanced-learn (SMOTE)
- **Natural Language Processing**: NLTK
- **Visualization**: Plotly, Matplotlib, WordCloud

---

## 🚀 Installation & Setup

Follow these steps to run the project locally.

### Prerequisites
- Python 3.8 or higher installed.

### 1. Clone the Repository
```bash
git clone https://github.com/harryku0311/HW3_Spam-Email.git
cd HW3_Spam-Email
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```
The app will open automatically in your web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
HW3_Spam-Email/
├── data/
│   └── sms_spam_no_header.csv  # Dataset file (Expected path)
├── .gitignore                  
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── streamlit_app.py            # Main application source code
```

## 🧠 Model Details

The core classification engine uses a **Support Vector Machine (SVM)**.

- **Text Representation**: TF-IDF (Term Frequency-Inverse Document Frequency) with a max feature limit of 3000.
- **Imbalance Handling**: The dataset is heavily skewed towards 'ham' messages. We use `imblearn`'s **SMOTE** pipeline to oversample the minority class during training, ensuring the model learns to identify spam effectively without bias.
- **Validation**: Models are evaluated on a 20% stratified test set.

## 📧 Contact

**Author**: harryku0311  
**GitHub**: [https://github.com/harryku0311](https://github.com/harryku0311)

---
*Created as part of the AIOT Coursework (HW3).*
