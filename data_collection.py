"""
Data Collection Script for Driver Drowsiness Detection
Records labeled video sessions for model training
"""

import cv2
import time
import argparse
import os
from datetime import datetime


def record_session(label, duration=300, output_dir='data', camera_id=0):
    """
    Record a labeled video session
    
    Args:
        label: Session label ('alert', 'warning', or 'drowsy')
        duration: Recording duration in seconds
        output_dir: Base output directory
        camera_id: Camera device ID
    """
    # Validate label
    valid_labels = ['alert', 'warning', 'drowsy']
    if label not in valid_labels:
        print(f"Error: Label must be one of {valid_labels}")
        return
    
    # Create output directory
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(label_dir, f"{label}_{timestamp}.avi")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    print("\n" + "="*70)
    print("Data Collection Session")
    print("="*70)
    print(f"Label: {label.upper()}")
    print(f"Duration: {duration} seconds ({duration // 60} minutes)")
    print(f"Output: {filename}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print("="*70)
    
    # Instructions based on label
    instructions = {
        'alert': [
            "Stay ALERT and focused",
            "Look straight ahead",
            "Blink normally (10-15 times/min)",
            "Keep eyes open"
        ],
        'warning': [
            "Simulate MODERATE drowsiness",
            "Blink slowly occasionally",
            "Slight head movement",
            "Occasional quick eye closure"
        ],
        'drowsy': [
            "Simulate SEVERE drowsiness",
            "Close eyes for 1-2 seconds repeatedly",
            "Yawn frequently",
            "Let head droop forward",
            "Look away from camera occasionally"
        ]
    }
    
    print("\nInstructions:")
    for i, instruction in enumerate(instructions[label], 1):
        print(f"  {i}. {instruction}")
    
    print("\n" + "="*70)
    input("Press ENTER to start recording...")
    print("="*70)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("\n🔴 RECORDING STARTED\n")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            if remaining <= 0:
                print("\n✓ Recording completed!")
                break
            
            # Add overlay information
            overlay = frame.copy()
            
            # Status bar
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Label and timer
            label_text = f"Label: {label.upper()}"
            timer_text = f"Time: {int(elapsed)}s / {duration}s"
            remaining_text = f"Remaining: {int(remaining)}s"
            
            cv2.putText(frame, label_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, timer_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, remaining_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Progress bar
            bar_width = width - 20
            bar_height = 10
            bar_x = 10
            bar_y = height - 20
            
            progress = elapsed / duration
            filled_width = int(bar_width * progress)
            
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + filled_width, bar_y + bar_height),
                         (0, 255, 0), -1)
            
            # Write frame
            out.write(frame)
            
            # Display frame
            cv2.imshow('Data Collection', frame)
            
            frame_count += 1
            
            # Progress updates every 30 seconds
            if frame_count % (fps * 30) == 0:
                print(f"  {int(elapsed)}s elapsed, {int(remaining)}s remaining...")
            
            # Check for early termination
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRecording stopped by user")
                break
    
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Summary
        actual_duration = time.time() - start_time
        
        print("\n" + "="*70)
        print("Recording Summary")
        print("="*70)
        print(f"Label: {label}")
        print(f"Frames recorded: {frame_count}")
        print(f"Duration: {actual_duration:.1f} seconds")
        print(f"Actual FPS: {frame_count / actual_duration:.1f}")
        print(f"File saved: {filename}")
        print(f"File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Record labeled video sessions for drowsiness detection training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 5-minute alert session
  python data_collection.py --label alert --duration 300
  
  # Record 3-minute drowsy session
  python data_collection.py --label drowsy --duration 180
  
  # Record with specific camera
  python data_collection.py --label warning --duration 120 --camera 1

Recommended Recording Plan:
  1. Alert state: 20 minutes (1200s) - normal, focused driving
  2. Warning state: 10 minutes (600s) - moderate drowsiness simulation
  3. Drowsy state: 10 minutes (600s) - severe drowsiness simulation
  
Total: ~40 minutes of labeled data for training
        """
    )
    
    parser.add_argument('--label', type=str, required=True,
                       choices=['alert', 'warning', 'drowsy'],
                       help='Session label')
    parser.add_argument('--duration', type=int, default=300,
                       help='Recording duration in seconds (default: 300)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    record_session(
        label=args.label,
        duration=args.duration,
        output_dir=args.output_dir,
        camera_id=args.camera
    )


if __name__ == "__main__":
    main()
