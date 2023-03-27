from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import math
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def crop(frame):
    if frame.shape[0] > 480 and frame.shape[1] > 640:
        return frame[0:480, 0:640]
    else:
        return frame


def dist_xy(point1, point2):
    """ Euclidean distance between two points point1, point2 """
    diff_point1 = (point1[0] - point2[0]) ** 2
    diff_point2 = (point1[1] - point2[1]) ** 2
    return (diff_point1 + diff_point2) ** 0.5


def find_mid_arm_distance(frame, mid_point, padding):
    try:
        """ Calculate the mid-upper arm diameter. """
        mid_arm = frame[mid_point[1] - padding: mid_point[1] +
                        padding, mid_point[0] - padding: mid_point[0] + padding]
        if mid_arm.size == 0:
            return 0
        mid_arm = cv2.Canny(mid_arm, 100, 250)
        top_part = mid_arm[0:round(mid_arm.shape[1] // 2), 0:]

        bottom_part = mid_arm[round(mid_arm.shape[1] // 2):, 0:]
        tpp, bpp = top_part[0:, top_part.shape[0] //
                            2], bottom_part[0:, bottom_part.shape[0] // 2]
        tpe, bpe = cv2.findNonZero(tpp), cv2.findNonZero(bpp)
        try:
            if (tpe.size == 0 or bpe.size == 0):
                return 0
        except:
            return 0
        top_height = padding - tpe[0][len(tpe[0]) - 1][1]
        bottom_height = padding - bpe[0][0][1]
        print(top_height, bottom_height)
        return top_height + bottom_height
    except:
        return 0


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


def gen_calibration_frames():
    recent = None
    while True:
        success, img = cap.read()  # read the camera frame
        img = crop(img)
        cv2.flip(img, 1)
        if not success:
            break
        else:
            # A boolean value for checking whether all essential data is available
            # Use YOLOV model to detect people
            markerCorners, ids, _ = detector.detectMarkers(
                img)  # Aruco detection
            # Continue if any one Aruco marker is found
            if ids is not None and len(ids) == 3:
                int_corners = np.intp(markerCorners)
                # Draw lines to join Aruco corners
                cv2.polylines(img, int_corners, True, (0, 255, 0), 1)
                caIndex = np.where(ids == [10])[0][0]
                taIndex = np.where(ids == [1])[0][0]
                baIndex = np.where(ids == [2])[0][0]
                # Calculate the marker size in px
                marker_h_px = (
                    int_corners[caIndex][0][2][1] - int_corners[caIndex][0][1][1])
                cv2.circle(
                    img, (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1]), 5, 255, -1)
                cv2.circle(
                    img, (int_corners[baIndex][0][2][0], int_corners[baIndex][0][2][1]), 5, 255, -1)
                cv2.line(img, (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1]), (
                    int_corners[baIndex][0][2][0], int_corners[baIndex][0][2][1]), 255, 3)
                person_h_px = dist_xy((int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1]), (
                    int_corners[baIndex][0][2][0], int_corners[baIndex][0][2][1]))

                # Find the ratio of marker height in px to marker height in mm
                ratio_px_mm = marker_h_px / marker_size_mm
                # Check for person in the frame
                # Get the x, y, w, h from the bounding box
                height_cm = ((person_h_px / ratio_px_mm) / 10) - 3
                if recent is not None:
                    if height_cm > (recent + 5) or height_cm < (recent - 5):
                        recent = height_cm
                    else:
                        height_cm = recent
                else:
                    recent = height_cm
                #  Convert the height (px) to height (cm) using the ratio
                cv2.putText(img, "Height", (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1] - 35),
                            cv2.FONT_HERSHEY_PLAIN, 1, 255, 2)
                cv2.putText(img, f"{round(height_cm, 1)}cm", (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                            255,
                            2)

            _, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


def gen_height_frames():
    while True:
        success, img = cap.read()  # read the camera frame
        img = crop(img)
        cv2.flip(img, 1)
        if not success:
            break
        else:
            # A boolean value for checking whether all essential data is available
            # Use YOLOV model to detect people
            markerCorners, ids, _ = detector.detectMarkers(
                img)  # Aruco detection
            # Continue if any one Aruco marker is found
            if ids is not None and len(ids) == 3:
                int_corners = np.intp(markerCorners)
                # Draw lines to join Aruco corners
                cv2.polylines(img, int_corners, True, (0, 255, 0), 1)
                caIndex = np.where(ids == [10])[0][0]
                taIndex = np.where(ids == [1])[0][0]
                baIndex = np.where(ids == [2])[0][0]
                # Calculate the marker size in px
                marker_h_px = (
                    int_corners[caIndex][0][2][1] - int_corners[caIndex][0][1][1])
                cv2.circle(
                    img, (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1]), 5, 255, -1)
                cv2.circle(
                    img, (int_corners[baIndex][0][2][0], int_corners[baIndex][0][2][1]), 5, 255, -1)
                cv2.line(img, (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1]), (
                    int_corners[baIndex][0][2][0], int_corners[baIndex][0][2][1]), 255, 3)
                person_h_px = dist_xy((int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1]), (
                    int_corners[baIndex][0][2][0], int_corners[baIndex][0][2][1]))

                # Find the ratio of marker height in px to marker height in mm
                ratio_px_mm = marker_h_px / marker_size_mm
                # Check for person in the frame
                # Get the x, y, w, h from the bounding box
                height_cm = ((person_h_px / ratio_px_mm) / 10) - 3
                if recent is not None:
                    if height_cm > (recent + 5) or height_cm < (recent - 5):
                        recent = height_cm
                    else:
                        height_cm = recent
                else:
                    recent = height_cm
                #  Convert the height (px) to height (cm) using the ratio
                cv2.putText(img, "Height", (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1] - 35),
                            cv2.FONT_HERSHEY_PLAIN, 1, 255, 2)
                cv2.putText(img, f"{round(height_cm, 1)}cm", (int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                            255,
                            2)

            _, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


def gen_muac_frames():
    recent = None
    while True:
        success, img = cap.read()  # read the camera frame
        img = crop(img)
        cv2.flip(img, 1)
        if not success:
            break
        else:
            newImg = img.copy()
            markerCorners, _, _ = detector.detectMarkers(
                img)  # Aruco detection
            if markerCorners:  # Continue if any one Aruco marker is found
                int_corners = np.intp(markerCorners)
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
                    rs = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
                    # Right Elbow landmark
                    re = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW]
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

                    # ---- Find Mid-Upper Arm Diameter ----
                    # Calculate the padding to be given to the midpoint
                    x_padding = round(dist_xy(rsPos, mid) / 2)
                    mid_upper_arm_distance = find_mid_arm_distance(
                        newImg, mid, x_padding)  # Find the mid-upper arm diameter
                    # --------
                    muad_cm = mid_upper_arm_distance / ratio_px_mm

                    if recent is not None:
                        if muad_cm > (recent + 5) or muad_cm < (recent - 5):
                            recent = muad_cm
                        else:
                            muad_cm = recent
                    else:
                        recent = muad_cm

                    # Drawing the shoulder, elbow and midpoint positions with the Mid-Arm diameter and shoulder-elbow length
                    cv2.circle(img, rsPos, 5, (0, 255, 0), -1)
                    cv2.circle(img, rePos, 5, (0, 255, 0), -1)
                    cv2.line(img, rsPos, rePos, (0, 255, 0), 1)
                    cv2.circle(img, mid,
                               5, (0, 255, 0), -1)
                    cv2.putText(img, f"MUAC: {round(2 * math.pi * (muad_cm / 2) / 10, 1)} cm",
                                (mid[0] + 10, mid[1] - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
            _, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


def get_data():
    height_cm = None
    muac_cm = None
    _, img = cap.read()  # read the camera frame
    img = crop(img)
    cv2.flip(img, 1)
    # A boolean value for checking whether all essential data is available
    # Use YOLOV model to detect people
    markerCorners, ids, _ = detector.detectMarkers(
        img)  # Aruco detection
    # Continue if any one Aruco marker is found
    if ids is not None and len(ids) == 3:
        int_corners = np.intp(markerCorners)
        # Draw lines to join Aruco corners
        caIndex = np.where(ids == [10])[0][0]
        taIndex = np.where(ids == [1])[0][0]
        baIndex = np.where(ids == [2])[0][0]
        # Calculate the marker size in px
        marker_h_px = (
            int_corners[caIndex][0][2][1] - int_corners[caIndex][0][1][1])
        person_h_px = dist_xy((int_corners[taIndex][0][2][0], int_corners[taIndex][0][2][1]), (
            int_corners[baIndex][0][2][0], int_corners[baIndex][0][2][1]))
        # Find the ratio of marker height in px to marker height in mm
        ratio_px_mm = marker_h_px / marker_size_mm
        # Check for person in the frame
        # Get the x, y, w, h from the bounding box
        height_cm = ((person_h_px / ratio_px_mm) / 10) - 3
        #  Convert the height (px) to height (cm) using the ratio
        # Convert frame to RGB since Mediapipe processes only RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB frame to get the landmarks
        results = pose.process(imgRGB)
        if results.pose_landmarks:  # Continue only if result is proper
            # Right Shoulder landmark
            rs = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
            # Right Elbow landmark
            re = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW]
            # Find the shoulder x and y coordinate
            rsPos = (
                round(rs.x * img.shape[1]), round(rs.y * img.shape[0]))
            # Find the elbow x and y coordinate
            rePos = (
                round(re.x * img.shape[1]), round(re.y * img.shape[0]))
            # Calculate the midpoint of line
            mid = (round((rePos[0] + rsPos[0]) / 2),
                   round((rePos[1] + rsPos[1]) / 2))
            # ---- Find Mid-Upper Arm Diameter ----
            # Calculate the padding to be given to the midpoint
            x_padding = round(dist_xy(rsPos, mid) / 2)
            mid_upper_arm_distance = find_mid_arm_distance(
                img, mid, x_padding)  # Find the mid-upper arm diameter
            muac_cm = round(
                2 * math.pi * ((mid_upper_arm_distance/ratio_px_mm) / 2) / 10, 1)
            # --------
    return {
        "height_cm": height_cm,
        "muac_cm": muac_cm
    }


@app.route('/')
def home():
    return "Cam Detection (Aruco Method)"


@app.route('/get')
def data():
    return get_data()


@app.route('/calibration')
def calibration():
    return Response(gen_calibration_frames(),  mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/height')
def height():
    return Response(gen_height_frames(),  mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/muac')
def muac():
    return Response(gen_muac_frames(),  mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=5000)
