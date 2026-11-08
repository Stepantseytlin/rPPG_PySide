import cv2 as cv


def getHaarROI(frame, face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")):
    # print("start extracting")
    if frame is None:
        return None

    if len(frame.shape) == 2:
        gray = frame
    else:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    min_size = (max(30, frame.shape[1] // 8), max(30, frame.shape[0] // 8))
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=min_size,
    )
    if len(faces) > 0:
        return tuple(faces[0])
    return None
