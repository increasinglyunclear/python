import os
import json
import cv2
import logging
from datetime import datetime
from scene_understanding import SceneUnderstanding
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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

def process_image(image_path, analyzer):
    """Process a single image and return scene analysis results."""
    try:
        # Read and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Get scene analysis
        results = analyzer.analyze_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Format results
        formatted_results = {
            'image_path': image_path,
            'primary_scene': results['scene_category'],
            'confidence': results['confidence'],
            'alternative_scenes': results['alternative_scenes']
        }
        
        # Print results
        print("\nTop Scene Categories:")
        print(f"1. {results['scene_category']} ({results['confidence']:.2f})")
        for i, alt in enumerate(results['alternative_scenes'], 2):
            print(f"{i}. {alt['category']} ({alt['confidence']:.2f})")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')
        
        # Display classification results
        plt.subplot(1, 2, 2)
        categories = [results['scene_category']] + [alt['category'] for alt in results['alternative_scenes']]
        confidences = [results['confidence']] + [alt['confidence'] for alt in results['alternative_scenes']]
        y_pos = np.arange(len(categories))
        
        plt.barh(y_pos, confidences, align='center')
        plt.yticks(y_pos, [cat.replace('/', '\n') for cat in categories])
        plt.xlabel('Confidence')
        plt.title('Top Scene Categories')
        
        # Save visualization
        plt.tight_layout()
        vis_path = os.path.join('results', f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return formatted_results
        
    except Exception as e:
        print(f"\nError processing {os.path.basename(image_path)}: {str(e)}")
        return None

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize scene analyzer
    analyzer = SceneUnderstanding()
    
    # Print available categories
    print("\nAvailable Scene Categories:")
    print("==========================")
    for idx, category in analyzer.categories.items():
        print(f"{idx:3d}: {category}")
    print(f"\nTotal number of categories: {len(analyzer.categories)}\n")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Get list of images to process
    image_dir = 'test_images'
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/analysis_{timestamp}.json'
    print(f"Results will be saved to {results_file}\n")
    
    # Process each image
    all_results = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing {image_file}...")
        
        results = process_image(image_path, analyzer)
        if results:
            all_results.append(results)
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nProcessing complete!")
    print(f"Results saved to {results_file}")
    print("Visualizations saved in the 'results' directory")

if __name__ == "__main__":
    main() 