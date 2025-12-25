import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# --- Page Config ---
st.set_page_config(page_title="SMS Spam Classifier", layout="wide")
st.title("📱 SMS Spam Classification System")

# --- NLTK Setup ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

# --- Data Loading ---
@st.cache_data
def load_data(path):
    # Try different encodings
    try:
        df = pd.read_csv(path, names=['label', 'message'], encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, names=['label', 'message'], encoding='latin-1')
        except UnicodeDecodeError:
             st.error("Could not decode file. Please check encoding.")
             return pd.DataFrame()
    return df

DATA_PATH = "sms_spam_no_header.csv"

# --- Preprocessing Functions ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters/numbers (keep only letters)
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize (simple split or nltk)
    tokens = text.split()
    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Sidebar ---
st.sidebar.title("Navigation")
tabs = ["Overview", "Visualizations", "Model Training & Analysis", "Live Inference"]
selected_tab = st.sidebar.radio("Go to", tabs)

# Load Data
df = load_data(DATA_PATH)

if df.empty:
    st.stop()

# Basic Preprocessing for Stats
df['clean_message'] = df['message'].apply(preprocess_text)
df['length'] = df['message'].apply(len)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


# --- TAB 1: OVERVIEW ---
if selected_tab == "Overview":
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", df.shape[0])
    with col2:
        counts = df['label'].value_counts()
        st.write("Class Distribution:")
        st.dataframe(counts)
        fig = px.pie(names=counts.index, values=counts.values, title="Spam vs Ham Distribution")
        st.plotly_chart(fig)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())


# --- TAB 2: VISUALIZATIONS ---
elif selected_tab == "Visualizations":
    st.header("Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Word Clouds", "Token Frequency", "Message Length"])
    
    with tab1:
        st.subheader("Word Clouds")
        spam_text = " ".join(df[df['label'] == 'spam']['clean_message'])
        ham_text = " ".join(df[df['label'] == 'ham']['clean_message'])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spam Word Cloud**")
            wc_spam = WordCloud(width=400, height=300, background_color='black', colormap='Reds').generate(spam_text)
            fig_spam, ax_spam = plt.subplots()
            ax_spam.imshow(wc_spam, interpolation='bilinear')
            ax_spam.axis('off')
            st.pyplot(fig_spam)

        with col2:
            st.markdown("**Ham Word Cloud**")
            wc_ham = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(ham_text)
            fig_ham, ax_ham = plt.subplots()
            ax_ham.imshow(wc_ham, interpolation='bilinear')
            ax_ham.axis('off')
            st.pyplot(fig_ham)

    with tab2:
        st.subheader("Most Frequent Words")
        type_choice = st.selectbox("Select Class", ["Spam", "Ham"])
        subset = df[df['label'] == type_choice.lower()]
        all_words = " ".join(subset['clean_message']).split()
        counter = Counter(all_words)
        most_common = counter.most_common(20)
        
        freq_df = pd.DataFrame(most_common, columns=['Word', 'Count'])
        fig_bar = px.bar(freq_df, x='Word', y='Count', title=f"Top 20 Words in {type_choice} Messages")
        st.plotly_chart(fig_bar)

    with tab3:
        st.subheader("Message Length Distribution")
        fig_hist = px.histogram(df, x="length", color="label", barmode="overlay", title="Message Length: Spam vs Ham")
        st.plotly_chart(fig_hist)


# --- TAB 3: MODEL TRAINING ---
elif selected_tab == "Model Training & Analysis":
    st.header("Model Training & Analysis")

    # Train Test Split
    X = df['clean_message']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=3000)
    
    model_choice = st.selectbox("Choose Model Version", 
                                ["A. Baseline SVM (Default)", 
                                 "B. Optimized SVM (Linear Kernel, C=10)", 
                                 "C. Advanced SVM (SMOTE + GridSearch)"])
    
    train_btn = st.button("Train Model")

    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
        st.session_state.vectorizer = None
        st.session_state.test_data = None

    if train_btn:
        with st.spinner("Training model... this might take a moment"):
            # Prepare Pipeline Steps
            if "C. Advanced" in model_choice:
                # SMOTE pipeline
                # Note: SMOTE requires numerical data, so we must vectorize FIRST inside the pipeline or before.
                # Since ImbPipeline can handle it:
                
                # Define pipeline
                # 1. Vectorizer
                # 2. SMOTE
                # 3. SVM
                pipeline = ImbPipeline([
                    ('tfidf', vectorizer),
                    ('smote', SMOTE(random_state=42)),
                    ('svm', SVC(probability=True, random_state=42))
                ])
                
                # GridSearch
                param_grid = {
                    'svm__C': [0.1, 1, 10], 
                    'svm__kernel': ['linear', 'rbf']
                }
                search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                search.fit(X_train, y_train)
                final_model = search.best_estimator_
                st.success(f"Best Params: {search.best_params_}")
                
            else:
                # Standard Scikit-learn Pipeline for A and B (No SMOTE needed explicitly in pipeline if not requested, but good practice)
                # But for A and B simpler:
                svc_clf = SVC(probability=True, random_state=42) # Default
                
                if "B. Optimized" in model_choice:
                    svc_clf = SVC(probability=True, kernel='linear', C=1.0, random_state=42) # Tuning as requested
                
                from sklearn.pipeline import Pipeline
                final_model = Pipeline([
                    ('tfidf', vectorizer),
                    ('svm', svc_clf)
                ])
                final_model.fit(X_train, y_train)

            st.session_state.trained_model = final_model
            st.session_state.vectorizer = vectorizer # It's inside pipeline mainly, but keeping reference
            st.session_state.test_data = (X_test, y_test)
            st.success("Training Complete!")

    # Show Metrics if model is trained
    if st.session_state.trained_model:
        model = st.session_state.trained_model
        X_test_saved, y_test_saved = st.session_state.test_data
        
        y_pred = model.predict(X_test_saved)
        y_proba = model.predict_proba(X_test_saved)[:, 1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test_saved, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_test_saved, y_pred):.4f}")
        col3.metric("Recall", f"{recall_score(y_test_saved, y_pred):.4f}")
        col4.metric("F1 Score", f"{f1_score(y_test_saved, y_pred):.4f}")

        st.metric("ROC AUC", f"{roc_auc_score(y_test_saved, y_proba):.4f}")

        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test_saved, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Ham', 'Spam'], y=['Ham', 'Spam'], title="Confusion Matrix")
            st.plotly_chart(fig_cm)

        with col_g2:
            st.subheader("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test_saved, y_proba)
            fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc_score(y_test_saved, y_proba):.4f})",
                              labels=dict(x="False Positive Rate", y="True Positive Rate"))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc)

# --- TAB 4: LIVE INFERENCE ---
elif selected_tab == "Live Inference":
    st.header("Live Inference")
    st.write("Test the model with your own text.")

    if st.session_state.get('trained_model') is None:
        st.warning("Please train a model in the 'Model Training' tab first!")
    else:
        user_input = st.text_area("Enter SMS text here:")
        
        col1, col2 = st.columns(2)
        with col1:
            check_btn = st.button("Check Spam")
        with col2:
            rand_btn = st.button("Generate Random Example")

        if rand_btn:
            # Pick random example
            random_row = df.sample(1).iloc[0]
            user_input = random_row['message']
            st.info(f"Loaded Example (True Label: {random_row['label']}):\n{user_input}")
            # We don't auto-submit to let user click check

        if check_btn and user_input:
            model = st.session_state.trained_model
            # Pipeline handles preprocessing/vectorization if set up correctly
            # But wait, our 'clean_message' logic was manual preprocessing in df.
            # Our pipeline uses TfidfVectorizer on raw text input? 
            #   -> If we passed 'clean_message' to train, we must preprocess here too.
            #   -> In Tab 3, X = df['clean_message']. So the model expects preprocessed text?
            #   -> Yes, the pipeline is Tfidf -> Model. It doesn't include the 'preprocess_text' function.
            #   -> So we MUST preprocess user input before feeding to pipeline.
            
            processed_input = preprocess_text(user_input)
            prediction = model.predict([processed_input])[0]
            probability = model.predict_proba([processed_input])[0][1]

            result_text = "SPAM" if prediction == 1 else "HAM"
            color = "red" if prediction == 1 else "green"

            st.markdown(f"### Result: <span style='color:{color}'>{result_text}</span>", unsafe_allow_html=True)
            st.write(f"Confidence Score (Spam Probability): **{probability:.4f}**")
            
            if prediction == 1:
                st.error("This message looks like Spam!")
            else:
                st.success("This message looks safe.")

