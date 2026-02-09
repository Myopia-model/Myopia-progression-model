import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
from backend.backend import progression_tracker
def configure_page() -> None:
    st.set_page_config(page_title="Progression Model", layout="wide")

configure_page()

#Title and user info
st.title("Myopia Progression Model")
name = st.text_input("Enter your name")
sex = st.radio("Select sex", ["Male", "Female", "Other"])
age = st.slider("Age", 1,109)
mom_myopic = st.selectbox("Mother myopic?", ["NO", "YES"])
dad_myopic = st.selectbox("Father myopic?", ["NO", "YES"])
mommy = 1 if mom_myopic == "YES" else 0
dadmy = 1 if dad_myopic == "YES" else 0

tv_time = st.slider("TV Time (hours/day)", 1,24)
reading_time = st.slider("Reading Time (hours/day)", 1,24)
sport_time = st.slider("Sport / outdoor time (hours/day)", 0, 10)
screen_time = tv_time + 0.5 * reading_time
outdoor_time = sport_time
gender_map = {"Male": 1, "Female": 0, "Other": 0}
gender_num = gender_map[sex]

#Prescription details
odright = st.number_input("Enter your OD(Right eye) prescription", step = 0.25)
osleft = st.number_input("Enter your OS(Left eye prescription)", step = 0.25)
cylinder = st.number_input("Cylinder",min_value = -12.00, max_value = 0.00,step = 0.25 )
axis = st.number_input("Axis (0–180°)", min_value = 0, max_value=180, step=1)
sphere = st.number_input("Sphere", min_value = -30.00, max_value = 20.00, step = 0.25)
conditions = st.text_input("Enter conditions")

def compute_spheq(sphere,cylinder):
    if  cylinder!= 0:
        return sphere + (cylinder/2)
    return sphere

spheq_left = compute_spheq(osleft,cylinder)
spheq_right = compute_spheq(odright, cylinder)


if st.button("Run Prediction"):
    result = progression_tracker(
        age=age,
        gender=gender_num,
        mommy=mommy,
        dadmy=dadmy,
        screen_time=screen_time,
        outdoor_time=outdoor_time
        
    )
    left_curve = [spheq_left + d for d in result["delta"]]
    right_curve = [spheq_right + d for d in result["delta"]]


    df_plot = pd.DataFrame({
        "Age": result["ages"],
        "Left Eye (OS)": left_curve,
        "Right Eye (OD)": right_curve,
    })

    st.line_chart(df_plot.set_index("Age"))


#submit button
if st.button("Submit"):
    st.success("Inputs saved successfully.")


st.write("DEBUG FRONTEND", {
    "screen_time": screen_time,
    "outdoor_time": outdoor_time,
    "mommy": mommy,
    "dadmy": dadmy,
    "gender": gender_num
})
