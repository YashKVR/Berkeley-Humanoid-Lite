import cv2
import mediapipe as mp
import numpy as np
from move_actuator_util import start_continuous_motor_control, update_motor_angles, stop_continuous_motor_control

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

cap = cv2.VideoCapture(0)

# Initialize motor control
# Motor-to-bus mapping: can0 = left arm (motors 3, 7), can1 = right arm (motors 4, 8)
motor_bus_mapping = {
    3: "can0",  # Left shoulder on can0
    7: "can0",  # Left elbow on can0
    4: "can1",  # Right shoulder on can1
    8: "can1",  # Right elbow on can1
}

control_thread = None
try:
    # Start with no motors - they will be added dynamically when update_motor_angles is called
    # Pass the motor-to-bus mapping so motors are initialized on the correct CAN bus
    control_thread = start_continuous_motor_control(
        initial_motors=None, 
        update_interval_ms=50,
        motor_bus_mapping=motor_bus_mapping
    )
    print("Motor control started. Press 'q' to quit.")
    print("Motor bus mapping: can0 (motors 3, 7), can1 (motors 4, 8)")
    print("You can update any motor ID using update_motor_angles([(motor_id, angle), ...])")
except Exception as e:
    print(f"Warning: Could not start motor control: {e}")
    print("Continuing with vision only...")

## Setup mediapipe instance
try:
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
          
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for left arm
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates for right arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Get coordinates for hips (needed for shoulder angle calculation)
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # Calculate angles
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

                #convert angles to radians
                left_elbow_angle = np.deg2rad(180) - np.deg2rad(left_elbow_angle)
                right_elbow_angle = np.deg2rad(180) - np.deg2rad(right_elbow_angle)
                left_shoulder_angle = np.deg2rad(left_shoulder_angle)
                right_shoulder_angle = np.deg2rad(right_shoulder_angle)
                
                # Update motor angles - you can update ANY motor ID on the bus
                # Example: update_motor_angles([(2, right_shoulder_angle), (8, right_elbow_angle)])
                # Motors will be automatically initialized if they haven't been initialized yet
                if control_thread is not None:
                    try:
                        # Update any motor IDs you want - they will be initialized automatically
                        update_motor_angles([(3, left_shoulder_angle),(4, -right_shoulder_angle),(7, left_elbow_angle),(8, -right_elbow_angle)])
                        # You can add more motors here, e.g.:
                        # update_motor_angles([(1, left_shoulder_angle), (7, left_elbow_angle)])
                    except Exception as e:
                        print(f"Warning: Could not update motor angles: {e}")
                
                # Get image dimensions for positioning text
                h, w, _ = image.shape
                
                # Visualize angles - positioning text near each joint
                # Left elbow
                cv2.putText(image, f"L Elbow: {left_elbow_angle:.1f}", 
                           tuple(np.multiply(left_elbow, [w, h]).astype(int) + np.array([10, -10])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Right elbow
                cv2.putText(image, f"R Elbow: {right_elbow_angle:.1f}", 
                           tuple(np.multiply(right_elbow, [w, h]).astype(int) + np.array([10, -10])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Left shoulder
                cv2.putText(image, f"L Shoulder: {left_shoulder_angle:.1f}", 
                           tuple(np.multiply(left_shoulder, [w, h]).astype(int) + np.array([10, -10])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Right shoulder
                cv2.putText(image, f"R Shoulder: {right_shoulder_angle:.1f}", 
                           tuple(np.multiply(right_shoulder, [w, h]).astype(int) + np.array([10, -10])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                           
            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # Stop motor control
    if control_thread is not None:
        try:
            stop_continuous_motor_control()
            print("Motor control stopped.")
        except Exception as e:
            print(f"Warning: Error stopping motor control: {e}")