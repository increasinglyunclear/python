# Simple image threshold filter
# pip install Pillow
# Kevin Walker 02 Jan 2025
# ported from Processing sketch: https://github.com/increasinglyunclear/processing/tree/main/bitmap

from PIL import Image

try:
    # Load the image
    img = Image.open("fossil.jpg")
    
    # Convert to grayscale first
    gray_image = img.convert('L')
    
    # Convert to 1-bit using custom threshold
    threshold = 128
    bw_image = gray_image.point(lambda x: 0 if x < threshold else 255, '1')
    
    # Save the result
    bw_image.save("output.jpg")
    print("Conversion completed successfully!")
    
except FileNotFoundError:
    print("Error: Image file not found. Make sure 'fossil.jpg' is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}") 
