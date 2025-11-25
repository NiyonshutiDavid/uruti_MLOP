"""
Script for video presentation - run this during your recording
"""
import time
from play_demo import PitchCoachDemo

def video_presentation():
    print("=== PITCH COACH RL VIDEO PRESENTATION ===")
    print("\n1. PROBLEM STATEMENT:")
    print("   Founders struggle with pitch delivery without objective feedback")
    time.sleep(2)
    
    print("\n2. AGENT BEHAVIOR:")
    print("   RL agent learns optimal presentation strategies through trial and error")
    print("   Actions: adjust energy, use gestures, eye contact, slide transitions")
    time.sleep(3)
    
    print("\n3. REWARD STRUCTURE:")
    print("   Based on confidence, engagement, and clarity metrics")
    print("   Positive rewards for good presentation behaviors")
    print("   Negative rewards for poor delivery and timing")
    time.sleep(3)
    
    print("\n4. AGENT OBJECTIVE:")
    print("   Maximize presentation quality and audience engagement")
    print("   Complete pitch within time constraints")
    print("   Maintain high confidence, engagement, and clarity scores")
    time.sleep(3)
    
    print("\n5. STARTING DEMONSTRATION...")
    time.sleep(2)
    
    # Run the actual demo
    # Replace with your actual best model path
    demo = PitchCoachDemo("models/dqn/best_model.zip", "dqn")
    demo.run_demo(episodes=1, save_video=True)
    
    print("\n6. PERFORMANCE ANALYSIS:")
    print("   Agent successfully learned presentation optimization")
    print("   Shows adaptive behavior based on presentation metrics")
    print("   Demonstrates effective use of presentation techniques")

if __name__ == "__main__":
    video_presentation()