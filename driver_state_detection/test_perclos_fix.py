#!/usr/bin/env python3
"""
Test script for PERCLOS functionality fixes
"""

import time
import sys
import os

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Attention_Scorer_Module import AttentionScorer

def test_perclos_functionality():
    """Test the PERCLOS calculation with various scenarios"""
    
    print("=" * 60)
    print("TESTING PERCLOS FUNCTIONALITY")
    print("=" * 60)
    
    # Test parameters
    fps = 30  # 30 FPS
    ear_threshold = 0.2  # EAR below this means eyes are closed
    perclos_threshold = 0.3  # 30% closure time threshold
    
    # Create AttentionScorer instance
    scorer = AttentionScorer(
        capture_fps=fps,
        ear_tresh=ear_threshold,
        gaze_tresh=0.5,  # Not used in this test
        perclos_tresh=perclos_threshold,
        verbose=True
    )
    
    print(f"\\nTest Setup:")
    print(f"- FPS: {fps}")
    print(f"- EAR Threshold: {ear_threshold}")
    print(f"- PERCLOS Threshold: {perclos_threshold}")
    print(f"- Time window: {scorer.perclos_time_period} seconds")
    
    print("\\n" + "-" * 40)
    print("Test 1: Normal eye state (EAR > threshold)")
    print("-" * 40)
    
    current_time = time.time()
    
    # Simulate normal eyes (EAR > threshold) for 10 frames
    for i in range(10):
        ear_score = 0.25  # Above threshold
        tired, perclos_score = scorer.get_PERCLOS(ear_score, current_time + i * (1.0/fps))
        if i == 9:  # Show result for last frame
            print(f"Frame {i+1}: EAR={ear_score}, TIRED={tired}, PERCLOS={perclos_score:.3f}")
    
    print("\\n" + "-" * 40)
    print("Test 2: Closed eyes (EAR < threshold)")
    print("-" * 40)
    
    # Reset scorer for new test
    scorer.eye_closure_counter = 0
    scorer.prev_time = 0
    current_time = time.time()
    
    # Simulate closed eyes for 60 frames (2 seconds at 30fps)
    for i in range(60):
        ear_score = 0.15  # Below threshold (closed eyes)
        tired, perclos_score = scorer.get_PERCLOS(ear_score, current_time + i * (1.0/fps))
        if i in [29, 59]:  # Show results at 1s and 2s
            print(f"Frame {i+1}: EAR={ear_score}, TIRED={tired}, PERCLOS={perclos_score:.3f}")
    
    print("\\n" + "-" * 40)
    print("Test 3: Mixed scenario (alternating open/closed)")
    print("-" * 40)
    
    # Reset scorer for new test
    scorer.eye_closure_counter = 0
    scorer.prev_time = 0
    current_time = time.time()
    
    # Simulate alternating pattern: 15 frames closed, 15 frames open, repeat
    for i in range(120):  # 4 seconds
        if (i // 15) % 2 == 0:
            ear_score = 0.15  # Closed eyes
        else:
            ear_score = 0.25  # Open eyes
            
        tired, perclos_score = scorer.get_PERCLOS(ear_score, current_time + i * (1.0/fps))
        
        if i in [29, 59, 89, 119]:  # Show results every second
            state = "CLOSED" if ear_score < ear_threshold else "OPEN"
            print(f"Frame {i+1}: EAR={ear_score} ({state}), TIRED={tired}, PERCLOS={perclos_score:.3f}")
    
    print("\\n" + "=" * 60)
    print("PERCLOS FUNCTIONALITY TEST COMPLETED")
    print("=" * 60)
    
    # Expected behavior explanation
    print("\\nExpected Behavior:")
    print("- Test 1: PERCLOS should stay at 0.0, TIRED should be False")
    print("- Test 2: PERCLOS should increase, TIRED should become True when PERCLOS > 0.3")
    print("- Test 3: PERCLOS should reflect 50% closure time, TIRED should be True")
    
    print("\\nKey Improvements Made:")
    print("✓ Fixed time handling - now uses consistent current_time parameter")
    print("✓ Improved PERCLOS calculation using elapsed time vs fixed period")
    print("✓ Added bounds checking (PERCLOS capped at 1.0)")
    print("✓ Better verbose output for debugging")
    print("✓ Added perclos_tresh parameter to command line arguments")

if __name__ == "__main__":
    test_perclos_functionality()
