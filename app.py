import joblib
import streamlit as st
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import numpy as np

st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app detects phishing websites using content analysis. Enter a URL to check if it\'s legitimate or phishing.')

def safe_feature_extraction(soup):
    """Extract features and fix dimension mismatch."""
    try:
        vector = fe.create_vector(soup)
        expected_features = 45
        current_features = len(vector)
        if current_features < expected_features:
            padding = [0] * (expected_features - current_features)
            return vector + padding
        elif current_features > expected_features:
            return vector[:expected_features]
        else:
            return vector
    except Exception:
        return [0] * 45

choice = st.selectbox("Select your machine learning model",
    [
        'K-Neighbours', 'Random Forest', 'Decision Tree', 
        'Gaussian Naive Bayes', 'Support Vector Machine', 
        'AdaBoost', 'Neural Network'
    ]
)

# Load the model pickles instead of importing live from machine_learning.py
if choice == 'Gaussian Naive Bayes':
    model = joblib.load("nb_model.pkl")
elif choice == 'Support Vector Machine':
    model = joblib.load("svm_model.pkl")
elif choice == 'Decision Tree':
    model = joblib.load("dt_model.pkl")
elif choice == 'Random Forest':
    model = joblib.load("rf_model.pkl")
elif choice == 'AdaBoost':
    model = joblib.load("ab_model.pkl")
elif choice == 'Neural Network':
    model = joblib.load("nn_model.pkl")
else:  # K-Neighbours
    model = joblib.load("kn_model.pkl")

st.write(f'{choice} model is selected!')

url = st.text_input('Enter the URL to check:', placeholder='https://example.com')

col1, col2 = st.columns([1, 1])
with col1:
    check_button = st.button('Check!', type='primary')
with col2:
    clear_button = st.button('Clear')

if clear_button:
    st.rerun()

if check_button and url:
    if not url.startswith(('http://', 'https://')):
        st.error("Please enter a valid URL starting with http:// or https://")
    else:
        with st.spinner('Analyzing website...'):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = re.get(url, verify=False, timeout=6, headers=headers)
                # --- If can't connect or domain doesn't resolve ---
                if response.status_code != 200:
                    st.error(f"HTTP connection was not successful for the URL: {url}")
                    st.info("The website is unreachable or offline. This does NOT mean it is phishing or legitimate. Many phishing sites go offline quickly, but some legitimate sites disappear too. Try another URL.")
                else:
                    # --- Valid, process with model ---
                    soup = BeautifulSoup(response.content, "html.parser")
                    feature_vector = safe_feature_extraction(soup)
                    vector = [feature_vector]
                    required = getattr(model, "n_features_in_", None)
                    if required:
                        actual = len(vector[0])
                        if actual < required:
                            vector[0] += [0] * (required - actual)
                        elif actual > required:
                            vector[0] = vector[0][:required]
                    try:
                        result = model.predict(vector)
                        if result[0] == 0:
                            st.success("✅ This website seems LEGITIMATE!")
                            st.balloons()
                        else:
                            st.warning("⚠️ Attention! This website is a potential PHISHING site!")
                            st.snow()
                    except Exception as pred_error:
                        st.error(f"Prediction error: {str(pred_error)}")
                        st.info("There was an issue with the model prediction. Please try a different model or site.")
            except re.exceptions.ConnectionError:
                st.error("Could not reach this website (domain unreachable, not registered, or does not exist).")
                st.info("The website appears to be expired or offline. This does NOT mean it is phishing. Many phishing domains go offline, but some legitimate sites also expire.")
            except re.exceptions.Timeout:
                st.error("Connection timed out when trying to reach the site.")
                st.info("The site may be down or very slow. No label is assigned!")
            except Exception as e:
                st.error(f"Could not analyze the URL: {str(e)}")
                st.info("Please check your internet connection or try a different URL.")

with st.expander('📝 Example URLs for Testing'):
    st.write('**✅ Safe Examples:**')
    st.code('https://www.google.com')
    st.code('https://www.github.com')
    st.code('https://www.wikipedia.org')
    st.write('**⚠️ Potential Phishing Examples:**')
    st.code('https://rtyu38.godaddysites.com/')
    st.code('http://www.aborderlaviolence.org/gtr/irz/')
    st.caption('Note: Phishing websites have short lifecycles and examples may become inactive!')
