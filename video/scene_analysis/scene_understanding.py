import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import logging
import urllib.request
from torchvision import models
from torch import nn
from PIL import Image

logger = logging.getLogger(__name__)

class SceneUnderstanding:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
        self._load_categories()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _initialize_model(self):
        """Initialize the scene classification model."""
        try:
            # Initialize ResNet50 model
            self.model = models.resnet50(pretrained=False)
            
            # Modify the final layer for 365 categories
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 365)
            
            # Load Places365 weights
            model_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
            state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=self.device)
            
            # Remove 'module.' prefix from state dict keys
            new_state_dict = {}
            for k, v in state_dict['state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logging.info("Scene classification model initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing scene classification model: {str(e)}")
            raise

    def _load_categories(self):
        """Load Places365 categories."""
        try:
            # Download categories file
            categories_url = 'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt'
            categories_file = 'categories_places365.txt'
            urllib.request.urlretrieve(categories_url, categories_file)
            
            # Load categories
            self.categories = {}
            with open(categories_file, 'r') as f:
                for i, line in enumerate(f):
                    category = line.strip().split(' ')[0]
                    # Remove the /a/ prefix
                    if category.startswith('/a/'):
                        category = category[3:]
                    self.categories[i] = category
            
            logging.info(f"Loaded {len(self.categories)} scene categories")
            
        except Exception as e:
            logging.error(f"Error loading scene categories: {str(e)}")
            raise

    def preprocess_frame(self, frame):
        """Preprocess frame for scene classification."""
        try:
            # Convert frame to PIL Image
            frame_pil = Image.fromarray(frame)
            
            # Apply different preprocessing techniques for ensemble
            transforms_list = [
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            ]
            
            # Apply each transform and stack the results
            processed_frames = []
            for transform in transforms_list:
                processed_frame = transform(frame_pil)
                processed_frames.append(processed_frame)
            
            return torch.stack(processed_frames)
            
        except Exception as e:
            logging.error(f"Error preprocessing frame: {str(e)}")
            raise

    def classify_scene(self, frame):
        """Classify scene using ensemble of predictions."""
        try:
            # Preprocess frame
            processed_frames = self.preprocess_frame(frame)
            
            # Move to device
            processed_frames = processed_frames.to(self.device)
            
            # Get predictions from each transform
            with torch.no_grad():
                outputs = []
                for i in range(processed_frames.size(0)):
                    output = self.model(processed_frames[i].unsqueeze(0))
                    outputs.append(output)
                
                # Average predictions
                ensemble_output = torch.mean(torch.stack(outputs), dim=0)
                probabilities = torch.softmax(ensemble_output, dim=1)
                
                # Get top predictions
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                top5_prob = top5_prob.squeeze().cpu().numpy()
                top5_catid = top5_catid.squeeze().cpu().numpy()
                
                # Convert to category names
                predictions = []
                for i in range(5):
                    category_id = top5_catid[i]
                    category_name = self.categories[category_id]
                    confidence = float(top5_prob[i])
                    predictions.append({
                        'category': category_name,
                        'confidence': confidence
                    })
                
                return predictions
                
        except Exception as e:
            logging.error(f"Error classifying scene: {str(e)}")
            raise

    def analyze_frame(self, frame):
        """Analyze a single frame for scene understanding."""
        try:
            # Get scene classification
            scene_predictions = self.classify_scene(frame)
            
            # Get primary scene category
            primary_scene = scene_predictions[0]['category']
            confidence = scene_predictions[0]['confidence']
            
            # Format results
            results = {
                'scene_category': primary_scene,
                'confidence': confidence,
                'alternative_scenes': [
                    {
                        'category': pred['category'],
                        'confidence': pred['confidence']
                    } for pred in scene_predictions[1:]
                ]
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error analyzing frame: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the scene understanding module
    analyzer = SceneUnderstanding()
    
    # Test with a sample frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder frame
    results = analyzer.analyze_frame(test_frame)
    
    print("Scene understanding test results:")
    print(results) 