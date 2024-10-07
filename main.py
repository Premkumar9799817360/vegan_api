import re
import pickle
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = FastAPI(title="Vegan Classification API")

# Allow CORS for specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; adjust for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic Models
class UserInput(BaseModel):
    user_input: str

class ClassificationResult(BaseModel):
    result: str
    warning: Optional[str] = None
    success: Optional[str] = None

# Load vegan and non-vegan ingredients from text files
def load_ingredients(filename: str) -> set:
    try:
        with open(filename, 'r') as file:
            ingredients = {line.strip().lower() for line in file if line.strip()}
        return ingredients
    except FileNotFoundError:
        raise RuntimeError(f"File {filename} not found.")

vegan_ingredients = load_ingredients('vegan.txt')
non_vegan_ingredients = load_ingredients('non-vegan.txt')

# Load saved models and tokenizers
def load_pickle_model(filename: str):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Model file {filename} not found.")

loaded_tokenizer = load_pickle_model('tokenizer.pkl')
loaded_vectorizer = load_pickle_model('count_vectorizer.pkl')
nb_model_loaded = load_pickle_model('nb_model.pkl')
log_reg_model_loaded = load_pickle_model('log_reg_model.pkl')
rf_model_loaded = load_pickle_model('rf_model.pkl')
xgb_model_loaded = load_pickle_model('xgb_model.pkl')

# Load TensorFlow models
def load_tf_model(filename: str):
    try:
        model = load_model(filename)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {filename}: {e}")

lstm_model_loaded = load_tf_model('lstm_model.h5')
bi_lstm_model_loaded = load_tf_model('bi_lstm_model.h5')

# Preprocess user input to handle multiple formats
def preprocess_user_input(text: str) -> List[str]:
    # Remove any numbers and extra spaces, convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s,]', '', text)  # Remove punctuation except commas
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    # Replace new lines with commas to handle multi-line text
    text = text.replace('\n', ',').replace('\r', ',')
    # Split by comma, trim each item and return as a list of ingredients
    return [item.strip() for item in text.split(',') if item.strip()]

# Function to classify ingredients based on known lists
def classify_ingredients(user_text: str) -> (List[str], List[str]):
    ingredients = preprocess_user_input(user_text)
    detected_vegan = []
    detected_non_vegan = []

    for ingredient in ingredients:
        if ingredient in vegan_ingredients:
            detected_vegan.append(ingredient)
        elif ingredient in non_vegan_ingredients:
            detected_non_vegan.append(ingredient)

    return detected_vegan, detected_non_vegan

# Model prediction functions
def predict_logistic_regression(user_text: str) -> str:
    user_text_cleaned = ' '.join(preprocess_user_input(user_text))
    user_text_vec = loaded_vectorizer.transform([user_text_cleaned])
    prediction = log_reg_model_loaded.predict(user_text_vec)
    return 'Vegan' if prediction[0] == 1 else 'Non-Vegan'

def predict_naive_bayes(user_text: str) -> str:
    user_text_cleaned = ' '.join(preprocess_user_input(user_text))
    user_text_vec = loaded_vectorizer.transform([user_text_cleaned])
    prediction = nb_model_loaded.predict(user_text_vec)
    return 'Vegan' if prediction[0] == 1 else 'Non-Vegan'

def predict_random_forest(user_text: str) -> str:
    user_text_cleaned = ' '.join(preprocess_user_input(user_text))
    user_text_vec = loaded_vectorizer.transform([user_text_cleaned])
    prediction = rf_model_loaded.predict(user_text_vec)
    return 'Vegan' if prediction[0] == 1 else 'Non-Vegan'

def predict_xgboost(user_text: str) -> str:
    user_text_cleaned = ' '.join(preprocess_user_input(user_text))
    user_text_vec = loaded_vectorizer.transform([user_text_cleaned])
    prediction = xgb_model_loaded.predict(user_text_vec)
    return 'Vegan' if prediction[0] == 1 else 'Non-Vegan'

def predict_lstm(user_text: str) -> str:
    user_text_cleaned = ' '.join(preprocess_user_input(user_text))
    user_text_seq = loaded_tokenizer.texts_to_sequences([user_text_cleaned])
    user_text_pad = pad_sequences(user_text_seq, maxlen=100)
    prediction = lstm_model_loaded.predict(user_text_pad)
    return 'Vegan' if prediction[0][0] > 0.5 else 'Non-Vegan'

def predict_bi_lstm(user_text: str) -> str:
    user_text_cleaned = ' '.join(preprocess_user_input(user_text))
    user_text_seq = loaded_tokenizer.texts_to_sequences([user_text_cleaned])
    user_text_pad = pad_sequences(user_text_seq, maxlen=100)
    prediction = bi_lstm_model_loaded.predict(user_text_pad)
    return 'Vegan' if prediction[0][0] > 0.5 else 'Non-Vegan'

@app.post("/classify", response_model=ClassificationResult)
async def classify(user_input: UserInput):
    user_text = user_input.user_input
    vegan_list, non_vegan_list = classify_ingredients(user_text)

    if non_vegan_list:
        final_result = 'Non-Vegan'
        warning_message = f"Warning: Non-Vegan ingredients found: {', '.join(non_vegan_list)}"
        return ClassificationResult(result=final_result, warning=warning_message)

    if vegan_list and not non_vegan_list:
        final_result = 'Vegan'
        success_message = "Success: Only Vegan ingredients found!"
        return ClassificationResult(result=final_result, success=success_message)

    # If no known ingredients or both categories exist, use models to predict
    predictions = [
        predict_logistic_regression(user_text),
        predict_naive_bayes(user_text),
        predict_random_forest(user_text),
        predict_xgboost(user_text),
        predict_lstm(user_text),
        predict_bi_lstm(user_text),
    ]

    # Determine the most common prediction
    final_result = max(set(predictions), key=predictions.count)
    return ClassificationResult(result=final_result)

# Optional: Root endpoint for health check or basic info
@app.get("/", summary="Root Endpoint")
async def read_root():
    return {"message": "Welcome to the Vegan Classification API. Use the /classify endpoint to classify ingredients."}
