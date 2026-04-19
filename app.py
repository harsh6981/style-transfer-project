import streamlit as st
from PIL import Image
import io
import os

from style_transfer import (
    load_image,
    save_image,
    run_style_transfer,
    cnn,
    cnn_normalization_mean,
    cnn_normalization_std
)

st.set_page_config(page_title="Neural Style Transfer", layout="centered")

st.title("🎨 Neural Style Transfer Demo")
st.write("Upload a content image and a style image, then generate the stylized output.")

col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader("📷 Upload Content Image", type=["jpg", "jpeg", "png"])

with col2:
    style_file = st.file_uploader("🖌️ Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file is not None and style_file is not None:
    content_img_pil = Image.open(content_file).convert("RGB")
    style_img_pil = Image.open(style_file).convert("RGB")

    st.subheader("Uploaded Images")
    c1, c2 = st.columns(2)

    with c1:
        st.image(content_img_pil, caption="Content Image", use_container_width=True)

    with c2:
        st.image(style_img_pil, caption="Style Image", use_container_width=True)

    if st.button("Apply Style Transfer"):
        with st.spinner("Generating stylized image... Please wait."):
            content_temp_path = "temp_content.jpg"
            style_temp_path = "temp_style.jpg"
            output_temp_path = "temp_output.jpg"

            content_img_pil.save(content_temp_path)
            style_img_pil.save(style_temp_path)

            content_img = load_image(content_temp_path)
            style_img = load_image(style_temp_path)
            input_img = content_img.clone()

            output = run_style_transfer(
                cnn,
                cnn_normalization_mean,
                cnn_normalization_std,
                content_img,
                style_img,
                input_img,
                num_steps=150
            )

            save_image(output, output_temp_path)

            output_pil = Image.open(output_temp_path)

        st.subheader("🎯 Output Image")
        st.image(output_pil, caption="Stylized Output", use_container_width=True)

        img_buffer = io.BytesIO()
        output_pil.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        st.download_button(
            label="📥 Download Output",
            data=img_buffer,
            file_name="styled_image.png",
            mime="image/png"
        )

        if os.path.exists(content_temp_path):
            os.remove(content_temp_path)
        if os.path.exists(style_temp_path):
            os.remove(style_temp_path)
        if os.path.exists(output_temp_path):
            os.remove(output_temp_path)