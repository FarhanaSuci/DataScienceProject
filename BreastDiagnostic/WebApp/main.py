import streamlit as st
import numpy as np
import pickle

# ✅ Load trained model & scaler
with open("BreastCancer.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ✅ Function to make prediction
def predict_cancer():
    st.sidebar.header("Breast Cancer Prediction")

    st.title("🔬 Breast Cancer Prediction App")
    st.markdown("""
    This application predicts whether a tumor is **Benign** or **Malignant** based on diagnostic features.
    Please enter the values below to get a prediction.
    """)

    name = st.text_input("Patient Name:")
    
    # Input fields for selected features
    feature_labels = [
        'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Smoothness_mean',
        'Compactness_mean', 'Concavity_mean', 'Concave_points_mean',
        'Symmetry_mean', 'Radius_se', 'Texture_se', 'Perimeter_se', 'Area_se',
        'Radius_worst', 'Texture_worst', 'Perimeter_worst', 'Smoothness_worst',
        'Compactness_worst', 'Concavity_worst', 'Concave_points_worst',
        'Symmetry_worst', 'Fractal_dimension_worst'
    ]
    
    input_features = []
    
    for feature in feature_labels:
        value = st.number_input(f"{feature}:", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
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
            st.error(f"🛑 {name}, the tumor is **Malignant**. Please consult a doctor immediately.")
        else:
            st.success(f"✅ {name}, the tumor is **Benign**. No need to worry!")

# ✅ Main Function
def main():
    st.sidebar.title("🔍 Select Activity")
    choice = st.sidebar.selectbox("MODE", ("About", "Predict Cancer"))

    if choice == "Predict Cancer":
        predict_cancer()
    elif choice == "About":
        st.title("📝 About This App")
        st.markdown("""
        - This app helps predict **breast cancer** using diagnostic features.
        - Built with **Streamlit** and **Machine Learning**.
        - Model trained on **breast cancer dataset** using **Logistic Regression**.
        - For more details, visit [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
        """)

# ✅ Run the app
if __name__ == '__main__':
    main()
