import cv2
import numpy as np
from pathlib import Path
from scene_understanding import SceneUnderstanding
import matplotlib.pyplot as plt
import json
from datetime import datetime

def save_results(image_name, results, output_file):
    """Save analysis results to a JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        'classification': {
            'top_categories': results['classification']['top_categories'],
            'raw_output': results['classification']['raw_output'].tolist() if 'raw_output' in results['classification'] else None
        },
        'segmentation': results['segmentation']  # This will be None for now
    }
    
    with open(output_file, 'a') as f:
        result_entry = {
            'image': image_name,
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }
        f.write(json.dumps(result_entry) + '\n')

def display_results(image, results, image_path):
    """Display the image and classification results"""
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    
    # Display classification results
    plt.subplot(1, 2, 2)
    categories = results['classification']['top_categories']
    y_pos = np.arange(len(categories))
    probabilities = [cat['probability'] for cat in categories]
    names = [cat['category'] for cat in categories]
    
    plt.barh(y_pos, probabilities, align='center')
    plt.yticks(y_pos, names)
    plt.xlabel('Probability')
    plt.title('Top Scene Categories')
    
    plt.tight_layout()
    plt.savefig(f'results/{Path(image_path).stem}_analysis.png')
    plt.close()

def list_all_categories(analyzer):
    """List all available scene categories"""
    categories = analyzer.categories  # Access categories from SceneUnderstanding instance
    print("\nAvailable Scene Categories:")
    print("==========================")
    for idx, category in sorted(categories.items()):
        print(f"{idx:3d}: {category}")
    print(f"\nTotal number of categories: {len(categories)}")

def process_single_image(analyzer, image_path, output_file):
    """Process a single image and save results"""
    print(f"\nProcessing {image_path.name}...")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Analyze scene
    results = analyzer.analyze_frame(image)
    
    # Print results
    print("\nTop Scene Categories:")
    for cat in results['classification']['top_categories']:
        print(f"{cat['category']}: {cat['probability']:.4f}")
    
    # Save results
    save_results(image_path.name, results, output_file)
    
    # Generate visualization
    display_results(image, results, image_path)

def main():
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    # Initialize output file
    output_file = f'results/analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # Initialize scene understanding module
    analyzer = SceneUnderstanding()
    
    # List all available categories
    list_all_categories(analyzer)
    
    # Test images
    test_dir = Path('test_images')
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg')) + list(test_dir.glob('*.png'))
    
    if not image_files:
        print("No test images found. Please add some images to the test_images directory.")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Results will be saved to {output_file}")
    
    # Process each image
    for image_file in image_files:
        try:
            process_single_image(analyzer, image_file, output_file)
        except Exception as e:
            print(f"\nError processing {image_file.name}: {str(e)}")
            continue
    
    print("\nProcessing complete!")
    print(f"Results saved to {output_file}")
    print("Visualizations saved in the 'results' directory")

if __name__ == "__main__":
    main() 