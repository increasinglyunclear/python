"""
Spatial Philosophical Interpretation Module
==========================================

This module creates a final one-paragraph spatial-philosophical interpretation
based on the video frame analyses and philosophical results.
"""

from collections import Counter
import re

def create_interpretation(frame_analyses, philosophical_results):
    """
    Create a one-paragraph spatial-philosophical interpretation.
    
    Args:
        frame_analyses: List of frame analysis dictionaries
        philosophical_results: Dictionary of philosophical analysis results
    
    Returns:
        str: One-paragraph interpretation
    """
    # Extract key themes from frame analyses
    themes = extract_themes_from_frames(frame_analyses)
    
    # Extract key insights from philosophical results
    insights = extract_insights_from_philosophy(philosophical_results)
    
    # Get video metadata
    video_info = get_video_metadata(frame_analyses)
    
    # Create interpretation
    interpretation = generate_interpretation(video_info, themes, insights)
    
    return interpretation

def extract_themes_from_frames(frame_analyses):
    """
    Extract recurring themes from frame analyses.
    """
    themes = {
        'vulnerability': 0,
        'dog_presence': 0,
        'natural_environment': 0,
        'cloth_covering': 0,
        'face_visible': 0,
        'peaceful': 0
    }
    
    for frame in frame_analyses:
        analysis = frame['analysis'].lower()
        
        # Count theme occurrences
        if any(word in analysis for word in ['vulnerability', 'exposed', 'unconscious', 'resting']):
            themes['vulnerability'] += 1
        
        if 'dog' in analysis:
            themes['dog_presence'] += 1
        
        if any(word in analysis for word in ['natural', 'grass', 'outdoor', 'peaceful', 'serene']):
            themes['natural_environment'] += 1
        
        if any(word in analysis for word in ['cloth', 'covering', 'blanket', 'protection']):
            themes['cloth_covering'] += 1
        
        if 'face' in analysis and 'visible' in analysis:
            themes['face_visible'] += 1
        
        if any(word in analysis for word in ['peaceful', 'serene', 'calm', 'tranquil']):
            themes['peaceful'] += 1
    
    # Convert to percentages
    total_frames = len(frame_analyses)
    for theme in themes:
        themes[theme] = (themes[theme] / total_frames) * 100
    
    return themes

def extract_insights_from_philosophy(philosophical_results):
    """
    Extract key philosophical insights from the results.
    """
    insights = []
    
    # Look for common philosophical themes in the results
    if 'merged_analysis' in philosophical_results:
        merged = philosophical_results['merged_analysis'].lower()
        
        if any(word in merged for word in ['vulnerability', 'exposure']):
            insights.append('human vulnerability in natural settings')
        
        if any(word in merged for word in ['companionship', 'bond', 'dog']):
            insights.append('human-animal relationships and companionship')
        
        if any(word in merged for word in ['covering', 'cloth', 'protection']):
            insights.append('politics of visibility and material concealment')
        
        if any(word in merged for word in ['natural', 'environment', 'nature']):
            insights.append('human connection to natural environments')
    
    return insights

def get_video_metadata(frame_analyses):
    """
    Extract basic video metadata from frame analyses.
    """
    if not frame_analyses:
        return {}
    
    # Get duration from timestamps
    timestamps = [frame['timestamp'] for frame in frame_analyses]
    duration = max(timestamps) if timestamps else 0
    
    # Get basic content description from first frame
    first_frame = frame_analyses[0]['analysis'] if frame_analyses else ""
    
    # Extract basic description
    description = extract_basic_description(first_frame)
    
    return {
        'duration': duration,
        'frames_analyzed': len(frame_analyses),
        'description': description
    }

def extract_basic_description(frame_analysis):
    """
    Extract a basic description from frame analysis.
    """
    analysis = frame_analysis.lower()
    
    # Look for key elements
    elements = []
    
    if 'person' in analysis and 'lying' in analysis:
        elements.append('person lying on the ground')
    
    if 'natural' in analysis or 'grass' in analysis:
        elements.append('natural environment')
    
    if 'dog' in analysis:
        elements.append('dog present')
    
    if 'cloth' in analysis or 'covering' in analysis:
        elements.append('covered with cloth')
    
    return ', '.join(elements) if elements else 'video content'

def generate_interpretation(video_info, themes, insights):
    """
    Generate the final one-paragraph interpretation.
    """
    duration = video_info.get('duration', 0)
    description = video_info.get('description', 'video content')
    
    # Build interpretation based on strongest themes
    interpretation_parts = []
    
    # Start with basic description
    interpretation_parts.append(f"The video depicts {description} over a duration of {duration:.1f} seconds.")
    
    # Add theme analysis
    if themes['vulnerability'] > 50:
        interpretation_parts.append("The fine-tuned Phi model's analysis reveals a consistent focus on themes of vulnerability and human-animal relationships.")
    
    if themes['dog_presence'] > 50:
        interpretation_parts.append("The model repeatedly identifies the person's position as suggesting vulnerability or exposure, while emphasizing the dog's presence as indicating companionship or concern.")
    
    if themes['cloth_covering'] > 50:
        interpretation_parts.append("The covering cloth appears frequently and is interpreted as providing protection from the elements, modesty, or symbolic concealment.")
    
    if themes['natural_environment'] > 50:
        interpretation_parts.append("The natural environment is consistently described as peaceful, serene, or creating a connection to nature.")
    
    # Add philosophical insights
    if insights:
        insight_text = ', '.join(insights)
        interpretation_parts.append(f"The model's philosophical reflections center on themes of {insight_text}.")
    
    # Add conclusion
    interpretation_parts.append("This suggests the fine-tuned model has learned to emphasize immediate spatial relationships and material politics, focusing on the body's vulnerability, the protective role of materials, and the companionship of animals in natural environments.")
    
    # Join into one paragraph
    interpretation = ' '.join(interpretation_parts)
    
    return interpretation 