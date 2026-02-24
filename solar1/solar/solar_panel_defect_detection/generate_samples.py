import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_images():
    """Create sample solar panel images for testing."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    print(f"Data dir: {data_dir}")
    os.makedirs(os.path.join(data_dir, "defective"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "normal"), exist_ok=True)

    # Create 5 defective images
    for i in range(5):
        # Create a base image (simulate solar panel)
        img = Image.new('RGB', (224, 224), color=(100, 100, 100))  # Gray background
        draw = ImageDraw.Draw(img)

        # Add some "defects" - random dark spots
        for _ in range(np.random.randint(3, 8)):
            x = np.random.randint(0, 224)
            y = np.random.randint(0, 224)
            size = np.random.randint(5, 20)
            draw.ellipse([x, y, x+size, y+size], fill=(0, 0, 0))

        # Add text label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), f"Defective {i+1}", fill=(255, 255, 255), font=font)

        img.save(os.path.join(data_dir, "defective", f"defective_{i+1}.jpg"))

    # Create 5 normal images
    for i in range(5):
        # Create a clean solar panel image
        img = Image.new('RGB', (224, 224), color=(150, 150, 150))  # Light gray
        draw = ImageDraw.Draw(img)

        # Add some grid lines to simulate solar cells
        for x in range(0, 224, 28):
            draw.line([x, 0, x, 224], fill=(200, 200, 200), width=1)
        for y in range(0, 224, 28):
            draw.line([0, y, 224, y], fill=(200, 200, 200), width=1)

        # Add text label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), f"Normal {i+1}", fill=(255, 255, 255), font=font)

        img.save(os.path.join(data_dir, "normal", f"normal_{i+1}.jpg"))

    print("Sample images created successfully!")

if __name__ == "__main__":
    create_sample_images()
