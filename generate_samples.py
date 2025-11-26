import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_sample(name, prompt, style, color):
    # Create directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(__file__).parent / "outputs"
    output_dir = base_dir / f"sample_{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy image
    img = Image.new('RGB', (512, 512), color=color)
    draw = ImageDraw.Draw(img)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    draw.text((20, 20), f"Sample: {name}", fill=(255, 255, 255), font=font)
    draw.text((20, 50), f"Style: {style}", fill=(255, 255, 255), font=font)
    draw.text((20, 450), "AI-generated via Talrn ImageGen", fill=(255, 255, 255), font=font)

    img_filename = f"img_{timestamp}.png"
    img_path = output_dir / img_filename
    img.save(img_path)

    # Create metadata
    meta_data = {
        "user_prompt": prompt,
        "enhanced_prompt": f"{prompt}, {style} style, detailed, 8k",
        "negative_prompt": "blurry, low quality",
        "style": style,
        "device": "cpu",
        "inference_settings": {
            "steps": 20,
            "guidance_scale": 7.5
        },
        "timestamp": timestamp,
        "generation_time_seconds": 12.5
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    print(f"Created sample {name} in {output_dir}")

if __name__ == "__main__":
    create_sample("cyberpunk", "A futuristic city with neon lights", "Cyberpunk", (25, 25, 112))
    create_sample("nature", "A serene lake at sunset", "Photorealistic", (255, 140, 0))
