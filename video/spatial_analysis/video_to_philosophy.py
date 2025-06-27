#!/usr/bin/env python3
"""
Video to Philosophy Pipeline
============================

This script provides a streamlined workflow to analyze a video and generate
a spatial-philosophical interpretation using AI models.

Usage:
    python video_to_philosophy.py <video_file>
    
Example:
    python video_to_philosophy.py test04b.mov
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Import our custom modules
from video_analysis import analyze_video_frames
from philosophical_pipeline import run_spatial_philosophical_analysis
from spatial_philosophical_interpretation import create_interpretation

def main():
    """
    Main function that orchestrates the entire video-to-philosophy pipeline.
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python video_to_philosophy.py <video_file>")
        print("Example: python video_to_philosophy.py test04b.mov")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    # Validate input file
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found.")
        sys.exit(1)
    
    print("=" * 80)
    print("VIDEO TO PHILOSOPHY PIPELINE")
    print("=" * 80)
    print(f"Input video: {video_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Analyze video frames using Kimi-VL-A3B-Thinking
        print("Step 1: Analyzing video frames with vision-language model...")
        frame_analyses = analyze_video_frames(video_file)
        
        if not frame_analyses:
            print("Error: No frame analyses generated.")
            sys.exit(1)
        
        print(f"✓ Analyzed {len(frame_analyses)} frames")
        
        # Step 2: Run spatial-philosophical analysis using fine-tuned Phi model
        print("\nStep 2: Running spatial-philosophical analysis...")
        philosophical_results = run_spatial_philosophical_analysis(frame_analyses)
        
        if not philosophical_results:
            print("Error: No philosophical analysis generated.")
            sys.exit(1)
        
        print("✓ Generated philosophical interpretations")
        
        # Step 3: Create final interpretation
        print("\nStep 3: Creating final interpretation...")
        interpretation = create_interpretation(frame_analyses, philosophical_results)
        
        # Step 4: Save results
        print("\nStep 4: Saving results...")
        save_results(video_file, frame_analyses, philosophical_results, interpretation)
        
        # Step 5: Display final interpretation
        print("\n" + "=" * 80)
        print("FINAL SPATIAL-PHILOSOPHICAL INTERPRETATION")
        print("=" * 80)
        print(interpretation)
        print("=" * 80)
        
        print(f"\n✓ Pipeline complete! Results saved to 'results/' directory")
        
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        sys.exit(1)

def save_results(video_file, frame_analyses, philosophical_results, interpretation):
    """
    Save all pipeline results to organized files.
    """
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_file).stem
    
    # Save frame analyses
    frame_file = results_dir / f"{video_name}_frame_analyses_{timestamp}.json"
    with open(frame_file, 'w') as f:
        json.dump({
            "video_file": video_file,
            "timestamp": timestamp,
            "frame_analyses": frame_analyses
        }, f, indent=2)
    
    # Save philosophical results
    philosophy_file = results_dir / f"{video_name}_philosophical_{timestamp}.json"
    with open(philosophy_file, 'w') as f:
        json.dump({
            "video_file": video_file,
            "timestamp": timestamp,
            "philosophical_results": philosophical_results
        }, f, indent=2)
    
    # Save final interpretation
    interpretation_file = results_dir / f"{video_name}_interpretation_{timestamp}.txt"
    with open(interpretation_file, 'w') as f:
        f.write("SPATIAL-PHILOSOPHICAL INTERPRETATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(interpretation)
    
    print(f"  - Frame analyses: {frame_file}")
    print(f"  - Philosophical results: {philosophy_file}")
    print(f"  - Final interpretation: {interpretation_file}")

if __name__ == "__main__":
    main() 