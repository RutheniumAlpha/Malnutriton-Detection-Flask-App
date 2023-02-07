from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
app = Flask(__name__)


def dist_xy(point1, point2):
    """ Euclidean distance between two points point1, point2 """
    diff_point1 = (point1[0] - point2[0]) ** 2
    diff_point2 = (point1[1] - point2[1]) ** 2
    return (diff_point1 + diff_point2) ** 0.5


def find_mid_arm_diameter(frame, mid_point, padding):
    """ Calculate the mid-upper arm diameter. """
    mid_arm = frame[mid_point[1] - padding: mid_point[1] +
                    padding, mid_point[0] - padding: mid_point[0] + padding]
    mid_arm = cv2.Canny(mid_arm, 100, 250)
    top_part = mid_arm[0:round(mid_arm.shape[1] // 2), 0:]
    print("Top Part: ", top_part)
    bottom_part = mid_arm[round(mid_arm.shape[1] // 2):, 0:]
    print("Bottom Part: ", bottom_part)
    tpp, bpp = top_part[0:, top_part.shape[0] //
                        2], bottom_part[0:, bottom_part.shape[0] // 2]
    print("TPP: ", tpp)
    print("BPP: ", bpp)
    tpe, bpe = cv2.findNonZero(tpp), cv2.findNonZero(bpp)
    print("TPE: ", tpe)
    print("BPE: ", bpe)
    top_height = padding - tpe[0][len(tpe[0]) - 1][1]
    print("Top Height: ", top_height)
    bottom_height = padding - bpe[0][0][1]
    print("Bottom Height: ", bottom_height)
    return top_height + bottom_height


# Load YOLOV model
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights",
                      "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Initialise Mediapipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Initialise Camera
cap = cv2.VideoCapture(0)

# Setup Aruco Detector
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Constants
marker_size_mm = 149  # The size of the Aruco marker in mm

process_this_frame = True


def gen_frames():
    while True:
        success, img = cap.read()  # read the camera frame
        if not success:
            break
        else:
            newImg = img.copy()
            # A boolean value for checking whether all essential data is available
            # Use YOLOV model to detect people
            (class_ids, scores, bboxes) = model.detect(
                img, confThreshold=0.3, nmsThreshold=.4)
            markerCorners, _, _ = detector.detectMarkers(
                img)  # Aruco detection
            if markerCorners:  # Continue if any one Aruco marker is found
                int_corners = np.int0(markerCorners)
                # Draw lines to join Aruco corners
                cv2.polylines(img, int_corners, True, (0, 255, 0), 1)
                parsed_corners = int_corners[0][0]
                # Calculate the marker size in px
                marker_h_px = (parsed_corners[2][1] - parsed_corners[1][1])
                # Find the ratio of marker height in px to marker height in mm
                ratio_px_mm = marker_h_px / marker_size_mm
                # Convert frame to RGB since Mediapipe processes only RGB images
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Process the RGB frame to get the landmarks
                results = pose.process(imgRGB)
                if results.pose_landmarks:  # Continue only if result is proper
                    # Right Shoulder landmark
                    rs = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
                    # Right Elbow landmark
                    re = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW]
                    # Find the shoulder x and y coordinate
                    rsPos = (
                        round(rs.x * img.shape[1]), round(rs.y * img.shape[0]))
                    # Find the elbow x and y coordinate
                    rePos = (
                        round(re.x * img.shape[1]), round(re.y * img.shape[0]))
                    # Calculate the midpoint of line
                    mid = (round((rePos[0] + rsPos[0]) / 2),
                           round((rePos[1] + rsPos[1]) / 2))
                    # Drawing the features on the frame
                    cv2.circle(img, rsPos, 5, (0, 255, 0), -1)
                    cv2.circle(img, rePos, 5, (0, 255, 0), -1)
                    cv2.line(img, rsPos, rePos, (0, 255, 0), 1)
                    cv2.circle(img, mid,
                               5, (0, 255, 0), -1)
                    cv2.putText(img, f"{round((dist_xy(rePos, rsPos) / ratio_px_mm) / 10, 1)}cm",
                                (mid[0] + 10, mid[1] - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1)
                    
                    # ---- Find Mid-Upper Arm Diameter ----
                    # Calculate the padding to be given to the midpoint
                    x_padding = round(dist_xy(rsPos, mid) / 2)
                    # mid_upper_arm_diameter = find_mid_arm_diameter(
                    #     newImg, mid, x_padding)  # Find the mid-upper arm diameter
                    # --------

                    # Drawing the shoulder, elbow and midpoint positions with the Mid-Arm diameter and shoulder-elbow length
                    cv2.circle(newImg, rsPos, 5, (0, 255, 0), -1)
                    cv2.circle(newImg, rePos, 5, (0, 255, 0), -1)
                    cv2.line(newImg, rsPos, rePos, (0, 255, 0), 1)
                    cv2.circle(newImg, mid,
                               5, (0, 255, 0), -1)
                    cv2.putText(newImg, f"Length: {round(dist_xy(rePos, rsPos))}px",
                                (mid[0] + 10, mid[1] - 20),
                                cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1)
                    # cv2.putText(newImg, f"Mid-Arm Diameter: {mid_upper_arm_diameter}px",
                    #             (mid[0] + 10, mid[1] - 10),
                    #             cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1)

                # Check for person in the frame
                for class_id, _, bbox in zip(class_ids, scores, bboxes):
                    if class_id == 0:  # ID 0 means person
                        # Get the x, y, w, h from the bounding box
                        (x, y, w, h) = bbox
                        #  Convert the height (px) to height (cm) using the ratio
                        cv2.putText(img, "Height", (x, y - 35),
                                    cv2.FONT_HERSHEY_PLAIN, 1, 255, 2)
                        cv2.putText(img, f"{round((h / ratio_px_mm) / 10, 1)}cm", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                    255,
                                    2)
                        # Draw a rectangle around the person
                        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)
            _, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
