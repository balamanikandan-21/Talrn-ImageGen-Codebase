import streamlit as st
import os
from pathlib import Path
from PIL import Image
from image_generator import ImageGenerator

# Page Config
st.set_page_config(
    page_title="Talrn ImageGen",
    page_icon="🎨",
    layout="wide"
)

# Initialize Generator (Cached to avoid reloading model on every interaction)
@st.cache_resource
def get_generator(device_name):
    return ImageGenerator(device=device_name)

# Sidebar
st.sidebar.title("⚙️ Settings")

# Device Selection
device_selection = st.sidebar.selectbox(
    "Device",
    ["Auto", "GPU (CUDA)", "CPU"],
    index=0,
    help="Select 'CPU' if you don't have a dedicated GPU. Note: CPU generation is slower."
)

try:
    generator = get_generator(device_selection)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

style = st.sidebar.selectbox(
    "Choose Style",
    ["Photorealistic", "Artistic", "Cartoon", "Anime", "Cyberpunk"]
)

# Dynamic defaults based on device
default_steps = 20 if generator.device == "cpu" else 50
steps = st.sidebar.slider("Inference Steps", min_value=1, max_value=100, value=default_steps, step=1)
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "Talrn ImageGen uses Stable Diffusion v1.5 to generate images from text. "
    "Running on **" + generator.device.upper() + "**."
)

# Main Interface
st.title("🎨 Talrn AI Image Generator")
st.markdown("Generate stunning images from text using open-source AI.")

# Responsible Use Disclaimer
with st.expander("⚠️ Responsible Use Disclaimer"):
    st.markdown("""
    **Please use this tool responsibly.**
    - Do not generate harmful, offensive, or illegal content.
    - Respect copyright and intellectual property rights.
    - This tool includes a basic NSFW filter, but users are responsible for their inputs.
    """)

col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area("Enter your prompt:", height=100, placeholder="A futuristic city at sunset...")
    negative_prompt = st.text_input("Negative Prompt (Optional):", placeholder="blurry, low quality...")
    
    generate_btn = st.button("🚀 Generate Image", type="primary")

with col2:
    st.markdown("### Output")
    image_placeholder = st.empty()
    status_text = st.empty()

if generate_btn and prompt:
    status_text.text("⏳ Generating... Please wait.")
    progress_bar = st.progress(0)
    
    # Simulate progress (since we can't easily hook into diffusers callback in a simple way without slowing it down, 
    # we'll just show a spinner mostly, but let's do a fake progress for UX)
    for i in range(100):
        # Just a visual placeholder, real generation happens below
        progress_bar.progress(i + 1)
        if i == 10: break 

    try:
        img_path, error = generator.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            steps=steps,
            guidance_scale=guidance_scale
        )
        
        progress_bar.progress(100)
        
        if error:
            st.error(error)
            status_text.empty()
        else:
            image = Image.open(img_path)
            image_placeholder.image(image, caption=f"{style} | {steps} steps", use_column_width=True)
            status_text.success("✅ Generation Complete!")
            
            # Download Button
            with open(img_path, "rb") as file:
                btn = st.download_button(
                    label="⬇️ Download Image",
                    data=file,
                    file_name=os.path.basename(img_path),
                    mime="image/png"
                )
                
            # Show Metadata
            # Show Metadata
            meta_path = Path(img_path).with_suffix(".json")
            if meta_path.exists():
                with st.expander("View Metadata"):
                    with open(meta_path, "r") as f:
                        st.json(f.read())

    except Exception as e:
        st.error(f"An error occurred: {e}")
        status_text.empty()

elif generate_btn and not prompt:
    st.warning("Please enter a prompt first.")
