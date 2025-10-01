import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Streamlit App
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("Breast Cancer Detection App")

st.subheader("Model Accuracy")
st.write(f"Random Forest Model Accuracy: *{accuracy_score(y_test, y_pred):.2f}*")

# User Input for Prediction
st.subheader("Enter Tumor Features for Prediction")
user_input = []
for feature in X.columns[:5]:   # Using first 5 features for simplicity
    value = st.number_input(f"{feature}", min_value=0.0, max_value=50.0, value=0.0)
    user_input.append(value)

if st.button("Predict"):
    # Pad remaining features with mean values
    remaining = X.mean()[5:].tolist()
    full_input = user_input + remaining
    prediction = model.predict([full_input])
    st.success("Prediction: *Malignant" if prediction[0]==0 else "Prediction: **Benign*")

# Data Visualization
st.subheader("Data Visualization")

# Histogram
st.write("Distribution of Mean Radius")
fig1, ax1 = plt.subplots()
sns.histplot(X['mean radius'], bins=30, ax=ax1)
st.pyplot(fig1)

# Scatterplot
st.write("Mean Radius vs Mean Texture")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=X['mean radius'], y=X['mean texture'], hue=y, palette=['red','green'], ax=ax2)
ax2.set_xlabel("Mean Radius")
ax2.set_ylabel("Mean Texture")
st.pyplot(fig2)