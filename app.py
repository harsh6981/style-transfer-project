import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# -------------------------------
# Dummy Style Transfer Function
# Replace this with your model
# -------------------------------
def apply_style(content_img, style_img):
    # For demo: just blend images (replace with real model)
    content = content_img.resize((512, 512))
    style = style_img.resize((512, 512))
    
    output = Image.blend(content, style, alpha=0.5)
    return output

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="Neural Style Transfer 🎨", layout="centered")

st.title("🎨 Neural Style Transfer Demo")
st.write("Upload a **content image** and a **style image**, then apply style!")

# -------------------------------
# Upload Section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader("📷 Upload Content Image", type=["jpg", "png"])

with col2:
    style_file = st.file_uploader("🖌️ Upload Style Image", type=["jpg", "png"])

# -------------------------------
# Show Uploaded Images
# -------------------------------
if content_file:
    content_img = Image.open(content_file)
    st.subheader("Content Image")
    st.image(content_img, use_column_width=True)

if style_file:
    style_img = Image.open(style_file)
    st.subheader("Style Image")
    st.image(style_img, use_column_width=True)

# -------------------------------
# Apply Button
# -------------------------------
if content_file and style_file:
    if st.button("✨ Apply Style"):
        with st.spinner("Applying style..."):
            output_img = apply_style(content_img, style_img)

        st.subheader("🎯 Output Image")
        st.image(output_img, use_column_width=True)

        # Download button
        st.download_button(
            label="📥 Download Output",
            data=output_img.tobytes(),
            file_name="styled_image.png",
            mime="image/png"
        )