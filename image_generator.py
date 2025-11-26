import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        """
        Initialize the Stable Diffusion model.
        """
        if device is None or device == "Auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "GPU (CUDA)":
            self.device = "cuda" if torch.cuda.is_available() else "cpu" # Fallback if forced but not available
        else:
            self.device = "cpu"
            
        print(f"Initializing ImageGenerator on {self.device}...")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            self.pipe.to(self.device)
            
            # Enable attention slicing for lower memory usage if on CUDA
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                
            # Disable the safety checker to implement our own lightweight one (or just to avoid the heavy load)
            # For this task, we will implement a keyword-based filter as requested.
            # We keep the feature extractor but disable the checker itself to speed up and simplify.
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "", 
        style: str = "Photorealistic", 
        steps: int = 50, 
        guidance_scale: float = 7.5
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate an image based on the prompt and settings.
        """
        
        # 1. Apply Style & Prompt Enhancement
        enhanced_prompt = self._enhance_prompt(prompt, style)
        default_negative = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature"
        final_negative = f"{default_negative}, {negative_prompt}".strip(", ")

        # 2. NSFW Filter (Pre-generation check)
        if self._is_nsfw(prompt) or self._is_nsfw(enhanced_prompt):
            return None, "NSFW content detected in prompt."

        # 3. Generate
        print(f"Generating: '{enhanced_prompt}' on {self.device}")
        start_time = time.time()
        
        try:
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=final_negative,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            ).images[0]
        except Exception as e:
            return None, f"Generation failed: {str(e)}"

        generation_time = time.time() - start_time

        # 4. Post-processing (Watermark)
        image = self._add_watermark(image)

        # 5. Save & Metadata
        output_path, meta_path = self._save_result(image, prompt, enhanced_prompt, final_negative, style, steps, guidance_scale, generation_time)

        return output_path, None

    def _enhance_prompt(self, prompt, style):
        style_modifiers = {
            "Photorealistic": "highly detailed, 4K, photorealistic, professional photography, dramatic lighting",
            "Artistic": "digital art, concept art, trending on artstation, vivid colors, masterpiece",
            "Cartoon": "cartoon style, flat colors, sharp lines, vector art",
            "Anime": "anime style, studio ghibli, vibrant, detailed background",
            "Cyberpunk": "cyberpunk, neon lights, futuristic, high tech, detailed"
        }
        modifier = style_modifiers.get(style, "")
        return f"{prompt}, {modifier}"

    def _is_nsfw(self, text):
        nsfw_keywords = ["nude", "naked", "porn", "sex", "nsfw", "xxx", "blood", "gore", "violence"]
        text_lower = text.lower()
        for keyword in nsfw_keywords:
            if keyword in text_lower:
                return True
        return False

    def _add_watermark(self, image):
        # Create a copy to avoid modifying original if needed elsewhere (though here we just return it)
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Simple default font
        try:
            # Try to load a standard font, fallback to default
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        text = "AI-generated via Talrn ImageGen"
        
        # Calculate text size using textbbox (newer Pillow) or textsize (older)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)

        width, height = img.size
        x = width - text_width - 10
        y = height - text_height - 10

        # Draw semi-transparent background for text
        draw.rectangle((x - 5, y - 5, x + text_width + 5, y + text_height + 5), fill=(0, 0, 0, 128))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        return img

    def _save_result(self, image, original_prompt, enhanced_prompt, negative_prompt, style, steps, guidance, gen_time):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = Path(__file__).parent / "outputs"
        output_dir = base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        img_filename = f"img_{timestamp}.png"
        img_path = output_dir / img_filename
        image.save(img_path)

        meta_data = {
            "user_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "device": self.device,
            "inference_settings": {
                "steps": steps,
                "guidance_scale": guidance
            },
            "timestamp": timestamp,
            "generation_time_seconds": round(gen_time, 2)
        }

        meta_path = output_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)
            
        return str(img_path), str(meta_path)

if __name__ == "__main__":
    # Simple test
    print("Testing ImageGenerator...")
    try:
        gen = ImageGenerator()
        path, error = gen.generate_image("A cute robot holding a flower", style="Cartoon", steps=20)
        if path:
            print(f"Success! Image saved to {path}")
        else:
            print(f"Failed: {error}")
    except Exception as e:
        print(f"Initialization failed: {e}")
