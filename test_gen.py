import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_generator import ImageGenerator

def test_generation():
    print("Initializing ImageGenerator...")
    try:
        gen = ImageGenerator()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Generating test image...")
    try:
        path, meta = gen.generate_image(
            prompt="A test image of a cat",
            steps=1, # Fast generation for testing
            style="Cartoon"
        )
        if path and Path(path).exists():
            print(f"Success! Image saved to {path}")
            print(f"Metadata saved to {meta}")
        else:
            print("Generation failed: No output path returned or file missing.")
    except Exception as e:
        print(f"Generation failed with error: {e}")

if __name__ == "__main__":
    test_generation()
