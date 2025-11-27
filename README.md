# Talrn ImageGen 🎨

**A Professional-Grade AI Image Generation System**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

**Talrn ImageGen** is a robust, open-source text-to-image generation application built to demonstrate advanced capabilities in **Generative AI** and **Machine Learning Engineering**.

Leveraging the power of **Stable Diffusion v1.5** via the Hugging Face `diffusers` library, this project provides a seamless, user-friendly interface for creating high-quality visuals from textual descriptions. It is engineered with a focus on **modularity**, **ethical AI practices**, and **cross-platform compatibility** (CPU/GPU).

### ✨ Key Features

*   **🚀 High-Performance Generation**: Utilizes state-of-the-art Latent Diffusion Models (LDM) for photorealistic output.
*   **⚙️ Adaptive Compute**: Automatically detects CUDA-enabled GPUs for acceleration, with a robust fallback to CPU for universal accessibility.
*   **🎨 Style Transfer Engine**: Integrated preset styles (Cyberpunk, Anime, Photorealistic) to instantly transform prompt aesthetics.
*   **🛡️ Ethical AI Guardrails**: Built-in NSFW content filtering and automatic watermarking to ensure responsible usage.
*   **📊 Metadata Persistence**: Every generated image is saved with a comprehensive JSON log of inference parameters (seed, steps, guidance scale) for reproducibility.
*   **🖥️ Interactive UI**: A polished, responsive web interface built with Streamlit.

---

## 🛠️ Tech Stack

*   **Core Framework**: Python, PyTorch
*   **Generative Model**: Stable Diffusion v1.5 (Hugging Face Diffusers)
*   **Frontend**: Streamlit
*   **Image Processing**: Pillow (PIL), NumPy
*   **Optimization**: `safetensors` for secure model loading, `accelerate` for hardware optimization.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.8 or higher
*   (Optional) NVIDIA GPU with CUDA drivers for optimal performance.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/talrn_image_gen.git
    cd talrn_image_gen
    ```

2.  **Set Up Environment**
    It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    > [!TIP]
    > **Troubleshooting**: If you encounter a `FileNotFoundError` for `requirements.txt`, ensure you are in the project root directory (`cd talrn_image_gen`).

### Running the Application

Launch the interface with a single command:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to start creating.

---

## 🧠 Engineering Highlights

### 1. Dynamic Device Management
The application implements intelligent resource management. It allows users to explicitly select their inference device (CPU vs. GPU) or rely on auto-detection.
*   **GPU Mode**: Optimized for speed (FP16 precision, attention slicing).
*   **CPU Mode**: Optimized for compatibility (FP32 precision, reduced default inference steps).

### 2. Prompt Engineering & Enhancement
To improve user results, the system includes a backend prompt enhancer that appends style-specific keywords (e.g., "8k resolution, dramatic lighting") to user inputs, ensuring higher fidelity outputs without complex user effort.

### 3. Production-Ready Structure
The codebase is organized for scalability:
*   `app.py`: Frontend logic and state management.
*   `image_generator.py`: Encapsulated backend logic, separating UI from ML operations.
*   `outputs/`: Structured storage with timestamped folders for asset management.
  ---

  ###GPU (Recommended)
Hardware: NVIDIA GPU with CUDA support.
VRAM:
Minimum: 4GB (The code uses float16 and enables attention slicing to support this).
Recommended: 8GB or more for faster generation and larger batch sizes.
Software: CUDA-compatible drivers installed.
###CPU (Fallback)
RAM:
Minimum: 12GB System RAM.
Recommended: 16GB+ System RAM (The model runs in float32 mode on CPU, which consumes more memory).
Performance: Generation will be significantly slower on CPU (minutes vs. seconds). A modern multi-core processor (Intel i5/i7/i9 or AMD Ryzen 5/7/9) is highly recommended.
Note: The application automatically detects if a GPU is available and defaults to it. If not, it falls back to CPU mode.

---
📘 Prompt Engineering Tips & Best Practices

Prompt engineering plays a major role in generating high-quality images with Stable Diffusion. Clear, structured prompts give the model stronger guidance and produce more reliable outputs.

🔹 1. Use a Clear Prompt Structure

A reliable prompt formula is:

[Subject] + [Details] + [Style] + [Quality Enhancements] + [Lighting] + [Camera/Art Terms]


Example:

“A futuristic cyberpunk city skyline at night, neon reflections on wet roads, ultra-realistic, 4K, cinematic lighting, sharp focus.”

🔹 2. Add Quality Boosting Keywords

These descriptors consistently improve output quality:

ultra detailed

hyper-realistic

4K / 8K resolution

dramatic lighting

sharp focus

volumetric light

masterpiece

🔹 3. Apply Style Guidance

Use artistic or technical style tags to influence the look:

photorealistic

anime style

digital art

concept art

watercolor

oil painting

3D render

🔹 4. Control Composition & Camera

Prompt terms like:

portrait / close-up / full body

wide-angle shot

macro photograph

top-down view

depth of field / bokeh

help control framing and perspective.

🔹 5. Add Context, Mood & Action

Enhances storytelling and realism:

emotions → calm, joyful, mysterious

actions → running, floating, dancing

environment → foggy forest, snowy mountain, neon-lit street

🔹 6. Use Negative Prompts

Negative prompts help avoid unwanted artifacts and distortions.

Recommended Negative Prompt:

blurry, low quality, distorted, deformed, extra limbs, bad anatomy,
text, watermark, noise, oversaturated, out of frame

🔹 7. Avoid Overloaded Prompts

Too many unrelated concepts confuse the model.
Keep prompts focused and descriptive.

🔹 8. Build Prompts Iteratively

Start simple → refine with style → add quality → adjust lighting.

Example:

“a cat wearing sunglasses”

“a fluffy orange cat wearing black sunglasses, sitting on a beach”

“digital art, soft shading”

“4K, high detail, sharp focus”

🔹 9. Reliable Prompt Formula (Highly Effective)
[Subject], [Details], [Style], ultra detailed, 4K, high resolution,
cinematic lighting, sharp focus, masterpiece

⚠️ Key Limitations of Stable Diffusion v1.5

While Stable Diffusion v1.5 is powerful and open-source, it has known performance and technical limitations. These should be considered when using or extending this project.

🔻 1. Generation Speed
Hardware	Approx. Time per Image
GPU (Recommended)	~2–10 seconds
CPU	~2–5 minutes

Reason:
Diffusion models rely on large-scale tensor operations that GPUs accelerate far more efficiently than CPUs.

🔻 2. Memory Requirements
GPU VRAM

Minimum: 4GB (with optimizations)

Recommended: 8GB+ for smooth performance

CPU System RAM

Minimum: 12GB

Recommended: 16GB+

CPU mode uses float32 precision, resulting in higher memory usage and slower computation.

🔻 3. Image Resolution Limitations

Stable Diffusion v1.5 is trained at 512 × 512 pixels.

Higher resolutions generated directly may show:

duplicated objects

warped anatomy

stretched features

inconsistent details

To upscale to larger images, a separate super-resolution or img2img pipeline is normally required (not included in this version).

🔻 4. Limited Concurrency

This application processes one generation request at a time.

Multiple users or multiple prompts may result in queued or delayed processing.

🔻 5. Model Cold Start Time

On initial launch, loading the ~4GB model into memory can take:

10–30 seconds (depending on hardware)

After loading, subsequent generations are significantly faster.
---

## ⚠️ Ethical Considerations

This project adheres to responsible AI development practices:
*   **Safety Checker**: Prevents the generation of explicit or harmful content.
*   **Transparency**: All images are watermarked "AI-generated via Talrn ImageGen" to prevent misinformation.

---

## 🔮 Future Roadmap

*   [ ] Integration of **SDXL (Stable Diffusion XL)** for higher resolution.
*   [ ] **Image-to-Image** (Img2Img) pipeline support.
*   [ ] User authentication and cloud storage integration.
*   [ ] API endpoint deployment via FastAPI.

---

*Developed for the Talrn ML Internship Task.*
