# I needed some randomized circles for https://increasinglyunclear.world/planets/
# ultimately switched to js - though this does create nice circles!
# Kevin Walker 23 Dec 2024

from PIL import Image, ImageDraw
import random
import os

print("Starting circle art generation...")

# Create a new black image
width = 800
height = 800
image = Image.new('RGB', (width, height), 'black')
draw = ImageDraw.Draw(image)

# Define circle parameters
num_circles = 8
min_radius = 100
max_radius = 250

print(f"Drawing {num_circles} circles...")

# Generate random circles
for i in range(num_circles):
    # Random radius first
    radius = random.randint(min_radius, max_radius)
    
    # Random center position (adjusted to keep circles fully on canvas)
    x = random.randint(radius, width - radius)
    y = random.randint(radius, height - radius)
    
    # Calculate bounding box for the circle
    left = x - radius
    top = y - radius
    right = x + radius
    bottom = y + radius
    
    # Draw circle outline in white
    draw.ellipse([left, top, right, bottom], outline='white', width=2)
    print(f"Drew circle {i+1} of {num_circles}")

# Save the image
output_path = 'circle_art.png'
image.save(output_path)
print(f"Saved image to: {os.path.abspath(output_path)}") 
