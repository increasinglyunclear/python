import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from .spatial_analyzer import SpatialAnalyzer

class SpatialAnalyzer:
    """
    Analyzer for spatial practices and visual elements in video frames
    """
    def __init__(self, knowledge_base_path=None):
        """
        Initialize the analyzer with spatial analysis capabilities
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.spatial_analyzer = SpatialAnalyzer()
        self.analysis_categories = {
            'spatial_practice': [],
            'critical_engagement': [],
            'poetic_observations': [],
            'theoretical_frameworks': [],
            'urban_insights': []
        }
        
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

        # Get spatial analysis
        spatial_analysis = self.spatial_analyzer.analyze_frame(frame_data['frame'])
        
        # Combine spatial analysis with Rendell's insights
        analysis = {
            'timestamp': timestamp,
            'spatial_practice': self._analyze_spatial_practice(spatial_analysis),
            'critical_engagement': self._analyze_critical_engagement(spatial_analysis),
            'poetic_observations': self._generate_poetic_observations(spatial_analysis),
            'theoretical_frameworks': self._apply_theoretical_frameworks(spatial_analysis),
            'urban_insights': self._generate_urban_insights(spatial_analysis)
        }
        return analysis
    
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