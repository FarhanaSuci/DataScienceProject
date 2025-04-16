import streamlit as st
import numpy as np
import pickle

# ✅ Load trained model & scaler
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("xgb_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ✅ Custom CSS (Clean & Pink Theme)
st.markdown("""
<style>
body, .stApp {
    background-color: #ffe6f0;
    color: black;
}

h1, h2, h3 {
    color: #cc0066;
}

/* Input label styling */
label, .css-1y0tads, .css-1cpxqw2 {
    color: black !important;
    font-weight: bold;
}

.stButton>button {
    background-color: hsl(306, 63.10%, 79.80%);
    color: white;
    border-radius: 15px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stNumberInput>div>input {
    background-color: #fff0f5;
    color: #4b004b;
    border-radius: 10px;
}

.stSidebar {
    background-color: hsl(306, 63.10%, 79.80%);
}

/* Big Butterfly Emoji */
h1::before, h1::after {
    content: "🦋";
    font-size: 50px;
    margin: 0 10px;
    color: #cc0066;
}
</style>
""", unsafe_allow_html=True)

# ✅ Prediction Page
def predict_cancer():
    st.markdown("""
    <h1 style="text-align: center;color:#cc0066;">
        Breast Cancer InsightX:<br> 
        <span style="color:#cc0066;">AI-Powered Diagnostic Analysis</span>
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("This application predicts whether a tumor is **Benign** or **Malignant** based on diagnostic features. Please enter the values below to get a prediction.")

    feature_limits = {
        'Perimeter_mean': (0.0, 200.0),
        'Smoothness_mean': (0.0, 1.0),
        'Compactness_mean': (0.0, 1.0),
        'Texture_worst': (0.0, 50.0),
        'Perimeter_worst': (0.0, 250.0)
    }

    input_features = []

    for feature, (min_val, max_val) in feature_limits.items():
        value = st.number_input(f"{feature}:", min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
        input_features.append(value)

    submit = st.button("🦋 Predict")

    if submit:
        input_data = np.array(input_features).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)

        if prediction[0] == 1:
           st.markdown("<p style='color: black;'>🛑 The tumor is <strong>Malignant</strong>. Please consult a doctor immediately.</p>", unsafe_allow_html=True)
        else:
           st.markdown("<p style='color: black;'>✅ The tumor is <strong>Benign</strong>. No need to worry!</p>", unsafe_allow_html=True)


# ✅ Main Function
def main():
    st.sidebar.title("🦋 Select Activity")
    choice = st.sidebar.selectbox("Choose an option", ("About", "Predict Cancer"))

    if choice == "Predict Cancer":
        predict_cancer()

    elif choice == "About":
        st.markdown("""
        <h1 style="text-align: center;color:#cc0066;">
            Breast Cancer InsightX:<br> 
            <span style="color:#cc0066;">AI-Powered Diagnostic Analysis</span>
        </h1>
        """, unsafe_allow_html=True)

        

        st.markdown("""
        <div style="font-size: 20px; line-height: 1.6;">
            🌸 <strong>AI-Powered Predictions</strong> : Classifies tumors as Benign or Malignant.<br>
            🌸 <strong>Fast & Accurate</strong> : 98.25% accuracy with 5 diagnostic features.<br>
            🌸 <strong>User-Friendly</strong> : Simple form-based design.<br>
            🌸 <strong>Instant Results</strong> : Diagnosis in seconds.<br>
            🌸 <strong>Secure</strong> : No personal data is stored.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---
    <div style="background-color:hsl(306, 63.10%, 79.80%);
 padding: 25px 0; text-align: center; color: black; font-size: 20px; border-radius: 25px;">
        <p><strong>🦋 Developers:<br> Farhana Akter Suci (B190305001) & Rifah Sajida Deya (B190305004)</strong></p>
        <p><strong>CSE , JnU</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
