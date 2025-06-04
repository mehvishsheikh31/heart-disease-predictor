import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap

st.set_page_config(page_title="Heart Disease Prediction App")

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

data = load_data()

# Preprocessing
X = data.drop(columns=['condition'])
y = data['condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ========== App Title ==========
st.title("Heart Disease Prediction App ❤️")

# ========== Section 1: Dataset Overview ==========
st.header("1. Dataset Overview")
st.write("### Dataset Preview")
st.dataframe(data.head())
st.write(f"Dataset has {data.shape[0]} rows and {data.shape[1]} columns.")

st.write("### Target Variable Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='condition', data=data, ax=ax1)
st.pyplot(fig1)

st.write("### Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(data['age'], bins=20, kde=True, ax=ax2)
st.pyplot(fig2)

st.write("### Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

st.write("### Cholesterol Level Distribution")
fig4, ax4 = plt.subplots(figsize=(8,5))
sns.histplot(data['chol'], bins=30, kde=True, color='salmon', ax=ax4)
ax4.set_xlabel("Cholesterol (mg/dl)")
ax4.set_ylabel("Count")
st.pyplot(fig4)



# ========== Section 2: Model Accuracy ==========
st.header("2. Model Accuracy")
st.write(f"Accuracy of Logistic Regression Model: **{accuracy:.2%}**")

# ========== Section 3: Predict Heart Disease ==========
st.header("3. Predict Heart Disease")

# Collect input from user
age = st.slider("Age", 20, 100, int(data['age'].median()), help="Patient's age in years")
sex = st.selectbox("Sex (0 = female, 1 = male)", options=[0, 1], help="Sex of the patient")
cp = st.selectbox("Chest pain type (cp)", options=[0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
trestbps = st.slider("Resting BP", int(data['trestbps'].min()), int(data['trestbps'].max()), int(data['trestbps'].median()), help="Resting blood pressure (mm Hg)")
chol = st.slider("Cholesterol", int(data['chol'].min()), int(data['chol'].max()), int(data['chol'].median()), help="Serum cholesterol (mg/dl)")
fbs = st.selectbox("Fasting Blood Sugar > 120? (fbs)", options=[0, 1], help="1 = True; 0 = False")
restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], help="0: Normal, 1: ST-T abnormality, 2: Left ventricular hypertrophy")
thalach = st.slider("Max Heart Rate (thalach)", int(data['thalach'].min()), int(data['thalach'].max()), int(data['thalach'].median()), help="Maximum heart rate achieved")
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], help="1 = Yes; 0 = No")
oldpeak = st.slider("ST Depression (oldpeak)", float(data['oldpeak'].min()), float(data['oldpeak'].max()), float(data['oldpeak'].median()), help="ST depression induced by exercise")
slope = st.selectbox("Slope", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
ca = st.selectbox("Major Vessels (ca)", options=[0, 1, 2, 3, 4], help="Number of major vessels colored by fluoroscopy")
thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3], help="1: Fixed defect, 2: Normal, 3: Reversible defect")

user_input = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_df)

    # Create SHAP waterfall plot
    st.write("### Feature Impact (SHAP Explanation)")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    proba = model.predict_proba(input_df)[0][1]
    st.write(f"Prediction Probability: **{proba:.2%}**")

    if prediction == 1:
        st.error("Prediction: Patient likely HAS heart disease.")
    else:
        st.success("Prediction: Patient likely does NOT have heart disease.")

    # ========== Section 4: Health Tips ==========
    st.header("4. Health Tips & Advice")

    if prediction == 1:
        st.warning("💡 Based on your input, here are some health tips:")
        st.markdown("""
        - 🍎 Eat a balanced diet (low in saturated fats, high in fiber)
        - 🚶 Exercise regularly (at least 30 minutes most days)
        - 🚭 Avoid smoking and alcohol
        - 💊 Manage blood pressure, cholesterol, and blood sugar
        - ❤️ Regular check-ups with a cardiologist are recommended
        """)
    else:
        st.success("✅ You're likely healthy, but keep up with these habits:")
        st.markdown("""
        - 🥗 Continue eating healthy meals
        - 🧘‍♀️ Maintain a stress-free lifestyle
        - 🏃‍♂️ Stay active
        - 📅 Get regular health check-ups
        """)

    with st.expander("📘 Learn more about Heart Disease"):
        st.markdown("""
        **Heart disease** refers to various types of heart conditions. The most common type is **coronary artery disease (CAD)**, which affects blood flow to the heart. Reduced blood flow can cause a **heart attack**.

        **Symptoms may include:**
        - Chest pain
        - Shortness of breath
        - Pain in the neck, jaw, or back

        **Prevention includes:**
        - Regular exercise
        - Healthy diet
        - Avoiding tobacco
        - Managing stress
        """)
