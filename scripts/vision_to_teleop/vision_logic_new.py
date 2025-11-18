import cv2
import mediapipe as mp
import numpy as np
from move_actuator_util import start_continuous_motor_control, update_motor_angles, stop_continuous_motor_control

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def calculate_forearm_rotation(hand_landmarks, wrist_pos, elbow_pos):
    """
    Calculate forearm rotation (pronation/supination) from hand orientation.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks (21 points)
        wrist_pos: Wrist position [x, y]
        elbow_pos: Elbow position [x, y]
    
    Returns:
        rotation_angle: Rotation angle in degrees (0-180, where 90 = neutral)
    """
    if hand_landmarks is None:
        return None
    
    # Get key hand landmarks for orientation
    # MediaPipe hand landmarks: https://google.github.io/mediapipe/solutions/hands.html
    # 0: Wrist, 5: Index MCP, 9: Middle MCP, 13: Ring MCP, 17: Pinky MCP
    # 4: Thumb tip, 8: Index tip
    
    try:
        # Get key hand landmarks
        # Wrist (0), Index MCP (5), Pinky MCP (17), Middle MCP (9)
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
        index_mcp = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])
        pinky_mcp = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])
        middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])
        
        # Calculate palm direction (perpendicular to the line from index to pinky MCP)
        # This represents the palm normal direction
        palm_vec = pinky_mcp - index_mcp
        palm_length = np.linalg.norm(palm_vec)
        
        if palm_length < 0.01:  # Too small, invalid
            return None
        
        # Calculate forearm direction (elbow to wrist)
        forearm_vec = np.array([wrist_pos[0] - elbow_pos[0], wrist_pos[1] - elbow_pos[1]])
        forearm_length = np.linalg.norm(forearm_vec)
        
        if forearm_length < 0.01:  # Too small, invalid
            return None
        
        # Normalize vectors
        palm_vec_norm = palm_vec / palm_length
        forearm_vec_norm = forearm_vec / forearm_length
        
        # Calculate the angle between palm direction and a perpendicular to forearm
        # Rotate palm vector 90 degrees to get palm normal
        palm_normal = np.array([-palm_vec_norm[1], palm_vec_norm[0]])  # 90 degree rotation
        
        # Calculate angle between forearm and palm normal
        # This gives us the rotation angle
        dot_product = np.clip(np.dot(forearm_vec_norm, palm_normal), -1.0, 1.0)
        angle_rad = np.arcsin(dot_product)  # Use arcsin for better range
        
        # Convert to degrees and normalize to 0-180 range
        angle_deg = np.rad2deg(angle_rad)
        
        # Map to 0-180 range where:
        # 0 = fully pronated (palm facing down/back)
        # 90 = neutral (palm facing side)
        # 180 = fully supinated (palm facing up/front)
        rotation_angle = (angle_deg + 90) % 180
        
        return rotation_angle
        
    except Exception as e:
        print(f"Error calculating forearm rotation: {e}")
        return None 

cap = cv2.VideoCapture(0)

# Initialize motor control
# You can start with any motor IDs, or start with None and add motors dynamically
# Motor IDs can be any valid motor ID on the bus (e.g., 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, etc.)
# Example: [(2, 0.0), (8, 0.0)] for right shoulder and elbow
# You can also add more motors later by calling update_motor_angles with new motor IDs
control_thread = None
try:
    # Start with no motors - they will be added dynamically when update_motor_angles is called
    # Or you can initialize specific motors: start_continuous_motor_control([(2, 0.0), (8, 0.0)], update_interval_ms=50)
    control_thread = start_continuous_motor_control(initial_motors=None, update_interval_ms=50)
    print("Motor control started. Press 'q' to quit.")
    print("You can update any motor ID using update_motor_angles([(motor_id, angle), ...])")
except Exception as e:
    print(f"Warning: Could not start motor control: {e}")
    print("Continuing with vision only...")

## Setup mediapipe instances
try:
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    with pose, hands:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
          
            # Make detections
            pose_results = pose.process(image)
            hand_results = hands.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = pose_results.pose_landmarks.landmark
                
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
                left_elbow_angle = np.deg2rad(left_elbow_angle)
                right_elbow_angle = np.deg2rad(180) - np.deg2rad(right_elbow_angle)
                left_shoulder_angle = np.deg2rad(left_shoulder_angle)
                right_shoulder_angle = np.deg2rad(right_shoulder_angle)
                
                # Calculate forearm rotation from hand tracking
                left_forearm_rotation = None
                right_forearm_rotation = None
                
                if hand_results.multi_hand_landmarks:
                    # Process each detected hand
                    for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, 
                                                          hand_results.multi_handedness):
                        hand_label = handedness.classification[0].label  # "Left" or "Right"
                        
                        # Get wrist position from hand landmarks
                        wrist_hand = [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y]
                        
                        # Calculate rotation based on which hand it is
                        if hand_label == "Right":
                            # Use right arm positions
                            rotation = calculate_forearm_rotation(hand_landmarks, wrist_hand, right_elbow)
                            if rotation is not None:
                                right_forearm_rotation = rotation
                        else:  # Left hand
                            # Use left arm positions
                            rotation = calculate_forearm_rotation(hand_landmarks, wrist_hand, left_elbow)
                            if rotation is not None:
                                left_forearm_rotation = rotation
                
                # Convert forearm rotation to radians if available
                left_forearm_rotation_rad = np.deg2rad(left_forearm_rotation) if left_forearm_rotation is not None else None
                right_forearm_rotation_rad = np.deg2rad(right_forearm_rotation) if right_forearm_rotation is not None else None
                
                # Update motor angles - you can update ANY motor ID on the bus
                # Example: update_motor_angles([(2, right_shoulder_angle), (8, right_elbow_angle)])
                # Motors will be automatically initialized if they haven't been initialized yet
                # 
                # Forearm rotation (pronation/supination) is now available:
                # - right_forearm_rotation_rad: Right forearm rotation in radians
                # - left_forearm_rotation_rad: Left forearm rotation in radians
                if control_thread is not None:
                    try:
                        # Update any motor IDs you want - they will be initialized automatically
                        # Example: Motor 4 for shoulder, Motor 8 for elbow, Motor 9 for forearm rotation
                        motor_updates = [(4, -right_shoulder_angle), (8, -right_elbow_angle)]
                        
                        # Add forearm rotation if available (example: motor 9 for right forearm rotation)
                        if right_forearm_rotation_rad is not None:
                            # Map rotation to motor range (-3.14 to 3.14)
                            # Adjust the mapping based on your needs
                            rotation_mapped = np.clip((right_forearm_rotation_rad - np.pi/2) * 2, -3.14, 3.14)
                            # motor_updates.append((9, rotation_mapped))  # Uncomment to enable
                        
                        # update_motor_angles(motor_updates)
                        pass
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
                
                # Display forearm rotation angles
                if right_forearm_rotation is not None:
                    cv2.putText(image, f"R Forearm Rot: {right_forearm_rotation:.1f}° ({right_forearm_rotation_rad:.2f} rad)", 
                               tuple(np.multiply(right_wrist, [w, h]).astype(int) + np.array([10, 20])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                
                if left_forearm_rotation is not None:
                    cv2.putText(image, f"L Forearm Rot: {left_forearm_rotation:.1f}° ({left_forearm_rotation_rad:.2f} rad)", 
                               tuple(np.multiply(left_wrist, [w, h]).astype(int) + np.array([10, 20])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                           
            except:
                pass
            
            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
            
            # Render pose detections
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
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