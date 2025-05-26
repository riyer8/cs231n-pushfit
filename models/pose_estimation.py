'''
This file contains the pose_estimation (green dots on the video) that determine where the joints of the 
person performing the exercise are.

Utilizes the movenet_lightning.tflite interpreter and stores the keypoints
'''
import cv2 # type: ignore
import numpy as np
import tensorflow as tf # type: ignore
import json

def pose_estimation(video_path, file_name, output_json):

    # using the movenet_lightning pose estimation model
    interpreter = tf.lite.Interpreter(model_path="models/movenet_lightning.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    print("File name:", file_name, "with frame dimensions", frame_width, frame_height)

    all_keypoints = []
    frame_count = 0

    def preprocess_frame(frame):
        img = cv2.resize(frame, (192, 192))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        img = img[np.newaxis, ...]
        return img

    # reading all of the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = preprocess_frame(frame)
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()

        # size (17, 3) with keypoints providing the 17 main keypoints of (y coordinate, x coordinate, confidence score)
        keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

        kp_list = []
        for kp in keypoints:
            y, x, confidence = kp
            kp_list.append({
                'x': float(x * frame_width),
                'y': float(y * frame_height),
                'confidence': float(confidence)
            })

        # adds keypoints along with frame number
        all_keypoints.append({
            'frame': frame_count,
            'keypoints': kp_list
        })

        # visualization
        '''
        for pt in kp_list:
            if pt['confidence'] > 0.3:
                cv2.circle(frame, (int(pt['x']), int(pt['y'])), 4, (0, 255, 0), -1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''

        frame_count += 1

    cap.release()
    # cv2.destroyAllWindows()

    # Save keypoints
    with open(output_json, "w") as f:
        json.dump(all_keypoints, f, indent=2)

    print(f"✅ Processed {frame_count} frames. Keypoints saved to {output_json}")
