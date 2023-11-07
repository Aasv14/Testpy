import cv2
import imutils
from imutils import face_utils
import dlib  # You need to import dlib for face detection and shape prediction
from scipy.spatial import distance

print(cv2.__version__)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

thresh = 0.25
flag = 0
(iStart, iEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detector = dlib.get_frontal_face_detector()  # Fixed function name
predictor = dlib.shape_predictor("file.dat")  # Fixed function name

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Fixed typo in cvtColor
    subjects = detector(gray, 0)
    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[iStart:iEnd]
        rightEye = shape[rStart:rEnd]
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar) / 2.0
        lefteyehull = cv2.convexHull(leftEye)
        righteyehull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [lefteyehull], -1, (0, 255, 0), 1)  # Fixed typo in drawContours
        cv2.drawContours(frame, [righteyehull], -1, (0, 255, 0), 1)  # Fixed typo in drawContours
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= 2:  # Changed frame.check to a fixed value (you can adjust it)
                cv2.putText(frame, '*********ALERT*********', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
