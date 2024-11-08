import streamlit as st
import requests
from pydantic import BaseModel
from PIL import Image

class InputData(BaseModel):
    bar: float
    baz: float
    xgt: float
    qgg: float
    lux: float
    wsg: float
    yyz: float
    drt: float
    gox: float
    foo: float
    boz: float
    fyt: float
    lgh: float
    hrt: float
    juu: float
    day: float
    month: float
    year: float
    day_of_week: float
    is_weekend: float

st.title("XGBoost Model Prediction")

# Image upload and display
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

def get_input(label, default):
    user_input = st.text_input(label, value=str(default))
    try:
        return float(user_input)
    except ValueError:
        st.warning(f"Invalid input for {label}, using default value.")
        return default

# Use a dropdown for "is_weekend" with options 0 and 1
is_weekend = st.selectbox("is_weekend", options=[0, 1], index=0)

input_data = InputData(
    bar=get_input("bar", 0.0123675769),
    baz=get_input("baz", -0.9680613912),
    xgt=get_input("xgt", -0.0112230303),
    qgg=get_input("qgg", 1.1150204493),
    lux=get_input("lux", 1.6486955982),
    wsg=get_input("wsg", -1.1855091658),
    yyz=get_input("yyz", 1.6650503797),
    drt=get_input("drt", -2.1443346141),
    gox=get_input("gox", -1.802215425),
    foo=get_input("foo", 0.80960123),
    boz=get_input("boz", -0.2854279603),
    fyt=get_input("fyt", -0.9454389185),
    lgh=get_input("lgh", 1.3640965852),
    hrt=get_input("hrt", 1.1059522563),
    juu=get_input("juu", -1.1046848669),
    day=get_input("day", 0.754945189),
    month=get_input("month", 1),
    year=get_input("year", 2016),
    day_of_week=get_input("day_of_week", 5),
    is_weekend=is_weekend
)

# Prediction button and response handling
if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=input_data.dict())
    if response.status_code == 200:
        prediction = response.json().get("prediction", "No prediction returned")
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
