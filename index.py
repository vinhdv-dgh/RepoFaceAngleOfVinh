import numpy as np
import os
import cv2
import time
import mediapipe as mp

savePath = './face_recognition/FaceDetected'

def collate_fn(x):
    return x[0] 

cap = cv2.VideoCapture(0)

# Ensure video file is opened
if not cap.isOpened():
    print("Error opening video file")
    exit()



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

angleFaces = []
Forward = 'Forward'
Below = 'Below'
Above = 'Above'
Left = 'Left'
Right = 'Right'
LeftAbove = 'LeftAbove'
LeftBelow = 'LeftBelow'
RightAbove = 'RightAbove'
RightBelow = 'RightBelow'



number = 0

def saveImage(image, faceAngle, currentAngle):
    if image is not None:
        image_file = f'{str(savePath)}/{str(faceAngle)}.png'
        cv2.imwrite(image_file, image)
        print(f"Captured frame at {number}")
        angleFaces.append(currentAngle)


capture_countdown = 0
capture_duration = 1  # 3 seconds countdown
currentFaceAngle = ""
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    start = time.time()
        # To improve performance
    frame.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])   
            # Convert it to the NumPy array
            print(face_2d)
            face_2d = np.array(face_2d, dtype=np.float64)
            print(face_2d)

            # Convert it to the NumPy array
            print(face_3d)
            face_3d = np.array(face_3d, dtype=np.float64)
            print(face_3d)
            

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                text = "Right"
                if x < -5:
                    text += "Above"
                    currentAngle = RightBelow
                    if RightBelow not in angleFaces:
                        if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration
                elif x > 10:
                    text += "Bellow"
                    currentAngle = RightAbove
                    if RightAbove not in angleFaces:
                        if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration
                else:
                    currentAngle = Right
                    if Right not in angleFaces:
                        if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration

            elif y > 10:
                text = "Left"
                if x < -5:
                    text += "Above"
                    currentAngle = LeftBelow
                    if LeftBelow not in angleFaces:
                        if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration
                elif x > 10:
                    text += "Bellow"
                    currentAngle = LeftAbove
                    if LeftAbove not in angleFaces:
                        if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration
                else:
                    currentAngle = Left
                    if Left not in angleFaces:
                        if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration

            elif x < -10:
                text = "Above"
                currentAngle = Above
                if Above not in angleFaces:
                        if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration
            elif x > 20:
                text = "Bellow"
                currentAngle = Below
                if Below not in angleFaces:
                    if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration
            else:
                text = "Forward"
                currentAngle = Forward
                if Forward not in angleFaces:
                    if capture_countdown == 0:
                            capture_countdown = time.time() + capture_duration
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            if Forward not in angleFaces:
                cv2.circle(image, (img_w // 2, img_h // 2), 100, (0, 0, 255), -1)
            if Below not in angleFaces:
                cv2.circle(image, (img_w // 2, 20), 100, (0, 0, 255), -1)
            if Above not in angleFaces:
                cv2.circle(image, (img_w // 2, img_h - 20), 100, (0, 0, 255), -1)
            if Right not in angleFaces:
                cv2.circle(image, (20, img_h // 2), 100, (0, 0, 255), -1)
            if Left not in angleFaces:
                cv2.circle(image, (img_w - 20, img_h // 2), 100, (0, 0, 255), -1)
            if RightAbove not in angleFaces:
                cv2.circle(image, (20, 20), 100, (0, 0, 255), -1)
            if RightBelow not in angleFaces:
                cv2.circle(image, (20, img_h - 20), 100, (0, 0, 255), -1)
            if LeftAbove not in angleFaces:
                cv2.circle(image, (img_w - 20, 20), 100, (0, 0, 255), -1)
            if LeftBelow not in angleFaces:
                cv2.circle(image, (img_w - 20, img_h - 20), 100, (0, 0, 255), -1)
        
        # Countdown for capturing image
        if currentAngle is not currentFaceAngle:
            capture_countdown = 0
            currentFaceAngle = currentAngle

        if capture_countdown > 0:
            remaining_time = capture_countdown - time.time()
            if remaining_time > 0:
                # Draw countdown text on the image
                cv2.putText(image, f"Capturing in {int(remaining_time)} seconds", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Capture the image after countdown is finished
                number+=1
                saveImage(frame, text, currentAngle)
                capture_countdown = 0

        end = time.time()
        totalTime = end - start

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            landmark_drawing_spec=drawing_spec)
                
    # Display the frame (optional)
    cv2.imshow('Head Pose Estimation', image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if len(angleFaces) == 9:
        print(angleFaces)
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()