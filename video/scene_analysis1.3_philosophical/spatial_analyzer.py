import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import torch
import torchvision.transforms as T
from PIL import Image
import torchvision
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add COCO classes list at the top of the file, after imports
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class PhilosophicalAnalyzer:
    """
    Analyzer that uses fine-tuned GPT-2 model to generate philosophical insights
    """
    def __init__(self, model_path="philosophical_llm/models/gpt2_finetuned"):
        """
        Initialize the philosophical analyzer with the fine-tuned GPT-2 model
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
        
    def generate_insight(self, context):
        # Construct a more focused prompt that guides the model toward philosophical interpretation
        prompt = f"""In this architectural space, we observe the following elements:

{context}

From a philosophical perspective, this spatial arrangement suggests:"""
        
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text with improved parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=100,  # Generate up to 100 new tokens beyond the prompt
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.92,
                temperature=0.85,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean up the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part after the prompt
        insight = generated_text[len(prompt):].strip()
        
        # Ensure we have a complete sentence
        if insight and not insight.endswith(('.', '!', '?')):
            insight += '.'
            
        return insight if insight else "The spatial arrangement invites contemplation of the relationship between form and function."

class SpatialAnalyzer:
    """
    Analyzer for spatial practices and visual elements in video frames
    """
    def __init__(self, knowledge_base_path=None):
        """
        Initialize the analyzer with spatial analysis capabilities
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.analysis_categories = {
            'spatial_practice': [],
            'critical_engagement': [],
            'poetic_observations': [],
            'theoretical_frameworks': [],
            'urban_insights': []
        }
        
        # Initialize Faster R-CNN model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.CLASSES = COCO_CLASSES
        self.model.eval()
        self.transform = T.Compose([
            T.ToTensor()
        ])
        
        # Initialize philosophical analyzer
        self.philosophical_analyzer = PhilosophicalAnalyzer()
        
    def _load_knowledge_base(self, path: str) -> Dict:
        """Load the processed philosophical texts"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_frame(self, frame_data: Dict) -> Dict:
        """
        Analyze a single video frame using Rendell's critical spatial practice approach
        Args:
            frame_data: Dictionary containing frame analysis (objects, scenes, poses, etc.)
        Returns:
            Dictionary containing Rendell-inspired analysis
        """
        # Extract timestamp if available
        timestamp = frame_data.get('timestamp', None)
        if timestamp:
            try:
                timestamp = datetime.fromtimestamp(timestamp)
            except:
                timestamp = None

        # Get spatial analysis using Faster R-CNN
        frame = frame_data.get('frame')
        if frame is None:
            raise ValueError("Frame data must contain a 'frame' key with the image data.")
        # Convert frame to PIL Image if it's not already
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)
        # Transform frame for model
        img_tensor = self.transform(frame)
        # Run model inference
        with torch.no_grad():
            outputs = self.model([img_tensor])
        
        # Process model outputs to extract objects and relationships
        objects = []
        for score, label, box in zip(outputs[0]['scores'], outputs[0]['labels'], outputs[0]['boxes']):
            if score > 0.7:  # Confidence threshold
                objects.append({
                    'name': COCO_CLASSES[label.item()],
                    'confidence': score.item(),
                    'bbox': box.tolist(),
                    'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                })
        
        # Extract spatial relationships
        spatial_relationships = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    rel = self._compute_relationship(obj1, obj2)
                    if rel:
                        spatial_relationships.append({
                            'object1': obj1['name'],
                            'object2': obj2['name'],
                            'relationship': rel
                        })
        
        # Combine spatial analysis with Rendell's insights
        spatial_analysis = {
            'objects': objects,
            'spatial_relationships': spatial_relationships,
            'spatial_practices': []
        }
        
        # Generate philosophical insights using GPT-2
        philosophical_insights = self._generate_philosophical_insights(spatial_analysis)
        
        analysis = {
            'timestamp': timestamp,
            'spatial_practice': self._analyze_spatial_practice(spatial_analysis),
            'critical_engagement': self._analyze_critical_engagement(spatial_analysis),
            'poetic_observations': self._generate_poetic_observations(spatial_analysis),
            'theoretical_frameworks': self._apply_theoretical_frameworks(spatial_analysis),
            'urban_insights': self._generate_urban_insights(spatial_analysis),
            'philosophical_insights': philosophical_insights
        }
        return analysis
    
    def _compute_relationship(self, obj1, obj2):
        """Compute spatial relationship between two objects based on their bounding boxes."""
        bbox1 = obj1['bbox']
        bbox2 = obj2['bbox']
        # Check for overlapping
        if (bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0] and bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1]):
            return 'overlapping'
        # Check for left/right
        if bbox1[2] < bbox2[0]:
            return 'left'
        if bbox1[0] > bbox2[2]:
            return 'right'
        # Check for inside
        if (bbox1[0] >= bbox2[0] and bbox1[2] <= bbox2[2] and bbox1[1] >= bbox2[1] and bbox1[3] <= bbox2[3]):
            return 'inside'
        return None
    
    def _analyze_spatial_practice(self, spatial_analysis: Dict) -> List[str]:
        """
        Analyze spatial practice in the frame
        Focuses on:
        - How space is used and transformed
        - Relationships between objects and space
        - Material and immaterial aspects of space
        """
        insights = []
        
        # Add spatial practices from spatial analyzer
        insights.extend(spatial_analysis['spatial_practices'])
        
        # Add Rendell-inspired insights
        for text in self.knowledge_base.values():
            for insight in text.get('urban_insights', []):
                if any(term in insight.lower() for term in ['space', 'place', 'site', 'location']):
                    # Connect with detected objects
                    for obj in spatial_analysis['objects']:
                        insights.append(f"Spatial Practice: {obj['name']} exists in relation to {insight}")
        
        return insights
    
    def _analyze_critical_engagement(self, spatial_analysis: Dict) -> List[str]:
        """
        Analyze critical engagement with the space
        Focuses on:
        - Power relations in space
        - Social and political implications
        - Questioning dominant spatial practices
        """
        insights = []
        
        # Analyze spatial relationships for power dynamics
        for rel in spatial_analysis['spatial_relationships']:
            for text in self.knowledge_base.values():
                for insight in text.get('critical_insights', []):
                    if any(term in insight.lower() for term in ['power', 'politics', 'social', 'critical']):
                        insights.append(
                            f"Critical Engagement: {rel['object1']} {rel['relationship']} {rel['object2']} "
                            f"raises questions about {insight}"
                        )
        
        return insights
    
    def _generate_poetic_observations(self, spatial_analysis: Dict) -> List[str]:
        """
        Generate poetic observations about the space
        Focuses on:
        - Sensory experiences
        - Emotional responses
        - Metaphorical connections
        """
        observations = []
        
        # Analyze spatial patterns for poetic qualities
        for pattern in spatial_analysis['spatial_practices']:
            for text in self.knowledge_base.values():
                for insight in text.get('urban_insights', []):
                    if any(term in insight.lower() for term in ['feel', 'sense', 'experience', 'imagine']):
                        observations.append(f"Poetic Observation: {pattern} evokes {insight}")
        
        return observations
    
    def _apply_theoretical_frameworks(self, spatial_analysis: Dict) -> List[str]:
        """
        Apply theoretical frameworks to the analysis
        Focuses on:
        - Theoretical questions about the space
        - Conceptual frameworks for understanding
        - Critical theory applications
        """
        frameworks = []
        
        # Apply theoretical frameworks to spatial patterns
        for pattern in spatial_analysis['spatial_practices']:
            for text in self.knowledge_base.values():
                for framework in text.get('theoretical_frameworks', []):
                    if '?' in framework:  # Focus on theoretical questions
                        frameworks.append(f"Theoretical Framework: {pattern} raises {framework}")
        
        return frameworks
    
    def _generate_urban_insights(self, spatial_analysis: Dict) -> List[str]:
        """
        Generate urban-specific insights
        Focuses on:
        - Urban space characteristics
        - City-specific observations
        - Urban theory applications
        """
        insights = []
        
        # Analyze objects in urban context
        for obj in spatial_analysis['objects']:
            for text in self.knowledge_base.values():
                for insight in text.get('urban_insights', []):
                    insights.append(f"Urban Insight: {obj['name']} relates to {insight}")
        
        # Analyze spatial patterns in urban context
        for pattern in spatial_analysis['spatial_practices']:
            for text in self.knowledge_base.values():
                for insight in text.get('urban_insights', []):
                    insights.append(f"Urban Pattern: {pattern} reflects {insight}")
        
        return insights
    
    def _generate_philosophical_insights(self, analysis):
        """Generate philosophical insights based on spatial analysis."""
        # Create a more structured context from the analysis
        context_parts = []
        
        # Add detected objects
        if analysis['objects']:
            context_parts.append("Objects present:")
            for obj in analysis['objects']:
                context_parts.append(f"- {obj['name']} ({obj['confidence']:.2f})")
        
        # Add spatial relationships
        if analysis['spatial_relationships']:
            context_parts.append("\nSpatial relationships:")
            for rel in analysis['spatial_relationships']:
                context_parts.append(f"- {rel['object1']} is {rel['relationship']} {rel['object2']}")
        
        # Add spatial practices
        if analysis['spatial_practices']:
            context_parts.append("\nSpatial practices:")
            for practice in analysis['spatial_practices']:
                context_parts.append(f"- {practice}")
        
        # Combine all parts into a single context string
        context = "\n".join(context_parts)
        # Generate philosophical insights
        return self.philosophical_analyzer.generate_insight(context)
    
    def analyze_video(self, video_data: List[Dict]) -> Dict:
        """
        Analyze a video using Rendell's approach
        Args:
            video_data: List of frame analyses
        Returns:
            Dictionary containing comprehensive analysis
        """
        analysis = {
            'frame_analyses': [],
            'temporal_insights': [],
            'spatial_narrative': [],
            'critical_reflections': []
        }
        
        # Analyze each frame
        for frame in video_data:
            frame_analysis = self.analyze_frame(frame)
            analysis['frame_analyses'].append(frame_analysis)
        
        # Generate temporal insights
        analysis['temporal_insights'] = self._analyze_temporal_aspects(analysis['frame_analyses'])
        
        # Generate spatial narrative
        analysis['spatial_narrative'] = self._generate_spatial_narrative(analysis['frame_analyses'])
        
        # Generate critical reflections
        analysis['critical_reflections'] = self._generate_critical_reflections(analysis['frame_analyses'])
        
        return analysis
    
    def _analyze_temporal_aspects(self, frame_analyses: List[Dict]) -> List[str]:
        """Analyze how space changes over time"""
        insights = []
        
        # Track changes in objects and scenes over time
        object_history = {}
        scene_history = {}
        
        for frame in frame_analyses:
            timestamp = frame.get('timestamp')
            if not timestamp:
                continue
                
            # Track object appearances
            for insight in frame['spatial_practice']:
                if 'exists in relation to' in insight:
                    obj_name = insight.split('exists in relation to')[0].replace('Spatial Practice: ', '')
                    if obj_name not in object_history:
                        object_history[obj_name] = []
                    object_history[obj_name].append(timestamp)
            
            # Track scene changes
            for insight in frame['urban_insights']:
                if 'reflects' in insight:
                    scene_type = insight.split('reflects')[0].replace('Urban Context: ', '')
                    if scene_type not in scene_history:
                        scene_history[scene_type] = []
                    scene_history[scene_type].append(timestamp)
        
        # Generate temporal insights
        for obj, timestamps in object_history.items():
            if len(timestamps) > 1:
                insights.append(f"Temporal Change: {obj} persists through multiple moments")
        
        for scene, timestamps in scene_history.items():
            if len(timestamps) > 1:
                insights.append(f"Temporal Context: {scene} maintains presence over time")
        
        return insights
    
    def _generate_spatial_narrative(self, frame_analyses: List[Dict]) -> List[str]:
        """Generate a narrative about the space's story"""
        narrative = []
        
        # Collect all spatial practices
        spatial_practices = []
        for frame in frame_analyses:
            spatial_practices.extend(frame['spatial_practice'])
        
        # Collect all urban insights
        urban_insights = []
        for frame in frame_analyses:
            urban_insights.extend(frame['urban_insights'])
        
        # Generate narrative elements
        if spatial_practices:
            narrative.append("Spatial Narrative: The space reveals multiple layers of practice and meaning")
        
        if urban_insights:
            narrative.append("Urban Story: The environment tells a story of urban experience")
        
        return narrative
    
    def _generate_critical_reflections(self, frame_analyses: List[Dict]) -> List[str]:
        """Generate critical reflections on the video"""
        reflections = []
        
        # Collect all critical engagements
        critical_engagements = []
        for frame in frame_analyses:
            critical_engagements.extend(frame['critical_engagement'])
        
        # Collect all theoretical frameworks
        theoretical_frameworks = []
        for frame in frame_analyses:
            theoretical_frameworks.extend(frame['theoretical_frameworks'])
        
        # Generate critical reflections
        if critical_engagements:
            reflections.append("Critical Reflection: The space invites critical questioning of urban practices")
        
        if theoretical_frameworks:
            reflections.append("Theoretical Reflection: The environment raises important theoretical questions")
        
        return reflections 