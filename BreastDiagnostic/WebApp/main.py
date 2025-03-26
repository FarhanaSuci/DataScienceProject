import streamlit as st
import numpy as np
import pickle

# ✅ Load trained model & scaler
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("xgb_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ✅ Function to make prediction
def predict_cancer():

    st.markdown("""
    <h1 style="text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
        📝 Breast Cancer InsightX:<br> AI-Powered Diagnostic Analysis<br>
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    This application predicts whether a tumor is **Benign** or **Malignant** based on diagnostic features.
    Please enter the values below to get a prediction.
    """)

    # Input fields for selected features
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
    
    submit = st.button("🔍 Predict")

    if submit:
        # Convert input to NumPy array and reshape
        input_data = np.array(input_features).reshape(1, -1)

        # Scale input
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display result
        if prediction[0] == 1:
            st.error(f"🛑 The tumor is **Malignant**. Please consult a doctor immediately.")
        else:
            st.success(f"✅ The tumor is **Benign**. No need to worry!")

# ✅ Main Function
def main():
    st.sidebar.title("🔍 Select Activity")
    choice = st.sidebar.selectbox("Select your choice", ("About", "Predict Cancer"))

    if choice == "Predict Cancer":
        predict_cancer()
    elif choice == "About":
        st.markdown("""
    <h1 style="text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
        🔬 Breast Cancer InsightX:<br> AI-Powered Diagnostic Analysis<br>
    </h1>
    """, unsafe_allow_html=True)
        st.markdown("""
    <ul style="font-size: 20px; line-height: 1.6;">
        <li><strong>AI-Powered Predictions</strong> : Uses machine learning to classify tumors as Benign or Malignant.</li>
        <li><strong>Fast & Accurate Diagnosis</strong> : Achieves <strong>98.25%</strong> accuracy using 5 key diagnostic features.</li>
        <li><strong>Easy-to-Use</strong> : Simple form-based input, requiring no technical expertise.</li>
        <li><strong>Instant Results</strong> : Get a diagnosis in seconds without waiting for lab results.</li>
        <li><strong>Personalized Insights</strong> : Patients can enter their own diagnostic parameters.</li>
        <li><strong>Secure & Confidential</strong> : Ensures data privacy & no storage of personal details.</li>
    </ul>
""", unsafe_allow_html=True)

        # ✅ Footer: Add Developer and Organization Information with Styling
    st.markdown("""
    ---
    <div style="background-color: ; padding: 25px 0; text-align: center; color: white; font-size: 20px; border-radius: 25px;">
        <p><br><br><strong>Developers:<br> Farhana Akter Suci (B190305001) & Rifah Sajida Deya (B190305004)</strong></p>
        <p><strong>CSE , JnU</strong></p>
        
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
