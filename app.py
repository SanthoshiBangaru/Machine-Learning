import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config("Multiple Linear Regression", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.markdown(
    """
    <div class="card">
        <h1>Multiple Linear Regression</h1>
        <p>Predict <b>Tip Amount</b> using multiple features from the dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

X = df.drop("tip", axis=1)
y = df["tip"]

numeric_features = ["total_bill", "size"]
categorical_features = ["sex", "smoker", "day", "time"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Actual vs Predicted Tips")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax.set_xlabel("Actual Tip")
ax.set_ylabel("Predicted Tip")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.2f}")
c4.metric("Adjusted R²", f"{adj_r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

size = st.slider(
    "Party Size",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)
sex = st.selectbox("Sex", df.sex.unique())
smoker = st.selectbox("Smoker", df.smoker.unique())
day = st.selectbox("Day", df.day.unique())
time = st.selectbox("Time", df.time.unique())

input_df = pd.DataFrame(
    {
        "total_bill": [bill],
        "size": [size],
        "sex": [sex],
        "smoker": [smoker],
        "day": [day],
        "time": [time],
    }
)

predicted_tip = model.predict(input_df)[0]

st.markdown(
    f"""
    <div class="prediction-box">
        Predicted Tip: $ {predicted_tip:.2f}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)