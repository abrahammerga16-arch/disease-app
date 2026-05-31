import os
import ast
import warnings
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from googletrans import Translator

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─── Load CSVs ────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

main_df        = pd.read_csv(os.path.join(DATA_DIR, "Diseases_and_Symptoms_dataset.csv"))
description_df = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))
diets_df       = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))
medications_df = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
precautions_df = pd.read_csv(os.path.join(DATA_DIR, "precautions.csv"))
workout_df     = pd.read_csv(os.path.join(DATA_DIR, "workout.csv"))

# ─── Load Models ──────────────────────────────────────────────────────────────
loaded_svc_model = joblib.load(os.path.join(MODELS_DIR, "svc_model.pkl"))
loaded_dt_model  = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
loaded_le        = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
print("✅ Models and datasets loaded.")

# ─── Feature matrix ───────────────────────────────────────────────────────────
X            = main_df.drop(columns=["diseases"])
symptom_list = X.columns.tolist()

# ─── Disease info maps ────────────────────────────────────────────────────────
def clean_disease_name(name):
    return str(name).lower().replace("_", " ").strip()

description_map = {clean_disease_name(r["Disease"]): r["Description"]  for _, r in description_df.iterrows()}
diets_map       = {clean_disease_name(r["Disease"]): r["Diet"]          for _, r in diets_df.iterrows()}
medications_map = {clean_disease_name(r["Disease"]): r["Medication"]    for _, r in medications_df.iterrows()}
workout_col     = workout_df.columns[1]
workout_map     = {clean_disease_name(r["Disease"]): r[workout_col]     for _, r in workout_df.iterrows()}

precautions_map = {}
for _, row in precautions_df.iterrows():
    disease = clean_disease_name(row["Disease"])
    precs   = [row["Precaution_1"], row["Precaution_2"], row["Precaution_3"], row["Precaution_4"]]
    precautions_map[disease] = [p for p in precs if pd.notna(p)]

disease_names = list(description_map.keys())

disease_symptoms = {}
for disease in main_df["diseases"].unique():
    d_df = main_df[main_df["diseases"] == disease]
    cols = d_df.loc[:, d_df.columns != "diseases"]
    syms = cols.columns[(cols == 1).any()].tolist()
    disease_symptoms[disease] = sorted(syms)

# ─── Lazy Sentence Transformer ────────────────────────────────────────────────
_model = None
_symptom_emb = None
_disease_emb = None

def get_model():
    global _model, _symptom_emb, _disease_emb
    if _model is None:
        print("Loading sentence transformer...")
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _symptom_emb = _model.encode(symptom_list, show_progress_bar=False)
        _disease_emb = _model.encode(disease_names, show_progress_bar=False)
        print("✅ Embeddings ready.")
    return _model, _symptom_emb, _disease_emb

# ─── Translation ──────────────────────────────────────────────────────────────
translator = Translator()

amharic_translations = {
    "Disease": "በሽታ",
    "Description": "መግለጫ",
    "Dietary Plan": "የአመጋገብ እቅድ",
    "Medications": "መድሃኒቶች",
    "Workout/Activity": "ስፖርት/እንቅስቃሴ",
    "Precautions": "ጥንቃቄዎች",
    "Symptoms": "ምልክቶች",
    "Hello! How can I help you with health information today?": "ሰላም! ዛሬ በጤና መረጃ እንዴት ልረዳዎ እችላለሁ?",
    "Access Denied: Information is not available for users under 18.": "የመዳረሻ እገዳ: ከ18 ዓመት በታች ለሆኑ ተጠቃሚዎች መረጃ አይገኝም",
    "I couldn't find specific information for that query. Please try asking about a specific disease (e.g., 'What is Asthma?') or use the prediction system first.": "ለዚህ ጥያቄ የተለየ መረጃ ማግኘት አልቻልኩም።",
    "medical_advice_disclaimer": "ማንኛውንም መድሃኒት ከመውሰድዎ በፊት ወይም ከባድ ምልክቶች ካጋጠሙዎት ሁልጊዜ ሐኪም ያማክሩ።",
    "Please enter symptoms.": "እባክዎ ምልክቶችን ያስገቡ።",
    "Please enter a query.": "እባክዎ ጥያቄ ያስገቡ።",
}

def translate_text(text, target_language="English"):
    if pd.isna(text) or not str(text).strip():
        return text
    if target_language.lower() == "amharic":
        return amharic_translations.get(text, text)
    return text

def _safe_literal_eval(s):
    try:
        if isinstance(s, str) and s.strip().startswith("["):
            return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    return None

def translate_to_amharic(text):
    if pd.isna(text) or not str(text).strip():
        return text
    original = str(text)
    lst = _safe_literal_eval(original)
    if isinstance(lst, list):
        result = []
        for item in lst:
            try:
                result.append(translator.translate(str(item), dest="am").text)
            except Exception:
                result.append(str(item))
        return result
    try:
        return translator.translate(original, dest="am").text
    except Exception:
        return original

def translate_to_english(text):
    if pd.isna(text) or not str(text).strip():
        return text
    original = str(text)
    lst = _safe_literal_eval(original)
    if isinstance(lst, list):
        result = []
        for item in lst:
            try:
                result.append(translator.translate(str(item), dest="en").text)
            except Exception:
                result.append(str(item))
        return result
    try:
        return translator.translate(original, dest="en").text
    except Exception:
        return original

# ─── Health Recommender ───────────────────────────────────────────────────────
def health_recommender(disease_name, user_age, user_role, current_language="English"):
    if user_age < 18:
        return translate_text("Access Denied: Information is not available for users under 18.", current_language), None

    disease_key = clean_disease_name(disease_name)
    if disease_key not in description_map:
        return f"Sorry, I don't have recommendations for '{disease_name}'.", None

    description_content = description_map.get(disease_key)
    diet_content        = diets_map.get(disease_key)
    medication_content  = medications_map.get(disease_key)
    workout_content     = workout_map.get(disease_key)
    precs_list          = precautions_map.get(disease_key, ["No specific precautions recorded."])
    precautions_content = ", ".join(precs_list)

    symptoms_for_display = disease_symptoms.get(disease_key, [])
    if current_language.lower() == "amharic":
        symptoms_content = [translate_to_amharic(s.replace("_", " ").title()) for s in symptoms_for_display]
    else:
        symptoms_content = [s.replace("_", " ").title() for s in symptoms_for_display]

    if current_language.lower() == "amharic":
        description_content = translate_to_amharic(description_content)
        tl = translate_to_amharic(diet_content)
        diet_content = ", ".join(tl) if isinstance(tl, list) else tl
        tl = translate_to_amharic(medication_content)
        medication_content = ", ".join(tl) if isinstance(tl, list) else tl
        tl = translate_to_amharic(workout_content)
        workout_content = ", ".join(tl) if isinstance(tl, list) else tl
        precautions_content = translate_to_amharic(precautions_content)

    recommendations = {
        translate_text("Disease",     current_language): disease_name.title(),
        translate_text("Description", current_language): description_content,
        translate_text("Symptoms",    current_language): symptoms_content,
    }

    normal_user_advice = None
    if user_role in ["Student", "Doctor"]:
        recommendations[translate_text("Dietary Plan",     current_language)] = diet_content
        recommendations[translate_text("Medications",      current_language)] = medication_content
        recommendations[translate_text("Workout/Activity", current_language)] = workout_content
        recommendations[translate_text("Precautions",      current_language)] = precautions_content
    elif user_role == "Normal User":
        recommendations[translate_text("Workout/Activity", current_language)] = workout_content
        recommendations[translate_text("Precautions",      current_language)] = precautions_content
        normal_user_advice = translate_text("medical_advice_disclaimer", current_language)

    return recommendations, normal_user_advice

# ─── Prediction ───────────────────────────────────────────────────────────────
def integrated_prediction_system(user_input, user_age, user_role, current_language="English", similarity_threshold=0.6):
    if user_age < 18:
        return translate_text("Access Denied: Information is not available for users under 18.", current_language), None

    model, symptom_emb, disease_emb = get_model()

    processed_input = user_input
    if current_language.lower() == "amharic":
        processed_input = translate_to_english(user_input)
    processed_input = str(processed_input).lower()

    symptom_data = np.zeros((1, len(X.columns)))
    symptom_df   = pd.DataFrame(symptom_data, columns=X.columns)

    phrases = [s.strip().lower() for s in processed_input.split(",") if s.strip()]
    if not phrases:
        return "Please enter at least one symptom.", None

    found_symptoms = []
    for phrase in phrases:
        emb  = model.encode(phrase)
        sims = cosine_similarity([emb], symptom_emb)[0]
        idx  = int(np.argmax(sims))
        if sims[idx] > similarity_threshold:
            found_symptoms.append(symptom_list[idx])
    found_symptoms = list(set(found_symptoms))

    if not found_symptoms:
        return "I couldn't recognize any specific symptoms. Please try different keywords.", None

    for sym in found_symptoms:
        if sym in symptom_df.columns:
            symptom_df.iloc[0, symptom_df.columns.get_loc(sym)] = 1

    probs          = loaded_svc_model.predict_proba(symptom_df)[0]
    top_indices    = np.argsort(probs)[::-1][:4]
    top_diseases   = loaded_le.inverse_transform(top_indices)
    top_confidence = probs[top_indices]

    predicted_conditions = [
        {"disease": d.title(), "confidence": f"{c:.2%}"}
        for d, c in zip(top_diseases, top_confidence)
    ]

    recommendations_dict, advice_str = health_recommender(top_diseases[0], user_age, user_role, current_language)
    return {
        "predicted_conditions": predicted_conditions,
        "recommendations": recommendations_dict,
        "advice": advice_str,
    }, None

# ─── Chatbot ──────────────────────────────────────────────────────────────────
def chatbot_response(user_query, user_age, user_role, current_language="English", similarity_threshold=0.6):
    if user_age < 18:
        return translate_text("Access Denied: Information is not available for users under 18.", current_language)

    model, symptom_emb, disease_emb = get_model()

    processed_query = user_query
    if current_language.lower() == "amharic":
        processed_query = translate_to_english(user_query)
    processed_query = str(processed_query).lower()

    greeting_response = ""
    if any(w in processed_query for w in ["hello", "hi", "hey", "greetings"]):
        greeting_response = translate_text("Hello! How can I help you with health information today?", current_language) + " "

    found_disease = None
    query_emb = model.encode(processed_query)
    sims      = cosine_similarity([query_emb], disease_emb)[0]
    best_idx  = int(np.argmax(sims))
    if sims[best_idx] > similarity_threshold:
        found_disease = disease_names[best_idx]

    if not found_disease:
        for key in description_map:
            if key in processed_query:
                found_disease = key
                break

    response_parts = []
    if found_disease:
        if any(w in processed_query for w in ["what is", "tell me about", "description"]):
            desc = description_map.get(found_disease)
            if current_language.lower() == "amharic":
                desc = translate_to_amharic(desc)
            response_parts.append(f"**{translate_text('Description', current_language)} of {found_disease.title()}:** {desc}")

        if user_role in ["Student", "Doctor"]:
            if any(w in processed_query for w in ["diet", "eat", "food"]):
                diet = diets_map.get(found_disease)
                if current_language.lower() == "amharic":
                    tl = translate_to_amharic(diet); diet = ", ".join(tl) if isinstance(tl, list) else tl
                response_parts.append(f"**{translate_text('Dietary Plan', current_language)} for {found_disease.title()}:** {diet}")
            if any(w in processed_query for w in ["medication", "medicine", "drug", "treatment"]):
                med = medications_map.get(found_disease)
                if current_language.lower() == "amharic":
                    tl = translate_to_amharic(med); med = ", ".join(tl) if isinstance(tl, list) else tl
                response_parts.append(f"**{translate_text('Medications', current_language)} for {found_disease.title()}:** {med}")
            if any(w in processed_query for w in ["precaution", "prevent", "care"]):
                precs = ", ".join(precautions_map.get(found_disease, ["No specific precautions."]))
                if current_language.lower() == "amharic":
                    precs = translate_to_amharic(precs)
                response_parts.append(f"**{translate_text('Precautions', current_language)} for {found_disease.title()}:** {precs}")
            if any(w in processed_query for w in ["workout", "exercise", "activity"]):
                wo = workout_map.get(found_disease)
                if current_language.lower() == "amharic":
                    tl = translate_to_amharic(wo); wo = ", ".join(tl) if isinstance(tl, list) else tl
                response_parts.append(f"**{translate_text('Workout/Activity', current_language)} for {found_disease.title()}:** {wo}")
        elif user_role == "Normal User":
            if any(w in processed_query for w in ["precaution", "prevent", "care"]):
                precs = ", ".join(precautions_map.get(found_disease, ["No specific precautions."]))
                if current_language.lower() == "amharic":
                    precs = translate_to_amharic(precs)
                response_parts.append(f"**{translate_text('Precautions', current_language)} for {found_disease.title()}:** {precs}")
            if any(w in processed_query for w in ["workout", "exercise", "activity"]):
                wo = workout_map.get(found_disease)
                if current_language.lower() == "amharic":
                    tl = translate_to_amharic(wo); wo = ", ".join(tl) if isinstance(tl, list) else tl
                response_parts.append(f"**{translate_text('Workout/Activity', current_language)} for {found_disease.title()}:** {wo}")

        if response_parts:
            return greeting_response + "\n\n" + "\n\n".join(response_parts)
        return greeting_response + f"I found information for **{found_disease.title()}**. Ask about its description, precautions, diet, medications, or workout."

    if greeting_response:
        return greeting_response

    return translate_text(
        "I couldn't find specific information for that query. Please try asking about a specific disease (e.g., 'What is Asthma?') or use the prediction system first.",
        current_language,
    )

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Disease Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.json
    result, _ = integrated_prediction_system(
        data.get("symptoms"),
        int(data.get("age", 25)),
        data.get("role", "Normal User"),
        data.get("language", "English"),
        float(data.get("similarity_threshold", 0.6)),
    )
    if isinstance(result, str):
        return jsonify({"error": result}), 400
    return jsonify(result)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    result, advice = health_recommender(
        data.get("disease"),
        int(data.get("age", 25)),
        data.get("role", "Normal User"),
        data.get("language", "English"),
    )
    if isinstance(result, str):
        return jsonify({"error": result}), 400
    response = {"recommendations": result}
    if advice:
        response["advice"] = advice
    return jsonify(response)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    response_text = chatbot_response(
        data.get("query"),
        int(data.get("age", 25)),
        data.get("role", "Normal User"),
        data.get("language", "English"),
        float(data.get("similarity_threshold", 0.6)),
    )
    return jsonify({"response": response_text})

@app.route("/diseases", methods=["GET"])
def get_diseases():
    return jsonify(sorted(disease_names))

@app.route("/translations", methods=["GET"])
def get_translations():
    return jsonify(amharic_translations)

@app.route("/translated_diseases", methods=["POST"])
def get_translated_diseases():
    data            = request.json
    target_language = data.get("target_language", "English")
    result = {}
    for name in disease_names:
        result[name] = translate_to_amharic(name) if target_language.lower() == "amharic" else name
    return jsonify(result)

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
