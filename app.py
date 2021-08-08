import streamlit as st
import numpy as np
from PIL import Image
import requests
import base64
import cv2

max_width_str = f"max-width: 1200px;"
st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>Image Text Recognition Tool</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Hanoi, 2021 by Sun Asterisk AI Team</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
            * Click the button to upload a image file.
            * Wait for running
"""
)

url = "http://192.168.1.187:8085/predictions/ocr_model"
uploaded_file = st.file_uploader("Upload Image", type=[".png", ".jpg", ".jpeg"])
if uploaded_file is not None:
    image = np.asarray(Image.open(uploaded_file))
    img_str = cv2.imencode('.jpg', image)[1].tostring()  # Encode the picture into stream data, put it in the memory cache, and then convert it into string format
    b64_code = base64.b64encode(img_str) # Encode into base64
    response = requests.post(url, files={'body': b64_code})

    st.image(image)
    if response.status_code == 200:
        st.markdown('***Predicted result***: {}'.format(response.text))
