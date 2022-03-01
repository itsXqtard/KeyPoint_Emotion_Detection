# Thanks to https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ for setting up dlib and providing example code
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import math
from keras.models import load_model
#emoDict = {"angry":0, "anger":0, "disgust":0,
#            "scared":1, "fear":1,
#            "happy":2, "happiness":2,
#            "sad":3, "sadness":3,
#            "surprised":4, "surprise":4,
#            "neutral":5}
emoDict = {"angry":0,"scared":1,"happy":2,"sad":3,"surprised":4,"neutral":5}
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

# Load machine learning model
modelPath = "./emotion_model_dlib.hdf5"
model = load_model(modelPath)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    image = imutils.resize(frame, width=1400)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Display the resulting frame
#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # show the face number
#        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
#        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (X, Y) in shape:
            cv2.circle(image, (X, Y), 2, (0, 0, 255), -2)
        
        normalizedCoordinates = shape
#        # Normalize keypoints
#        minX = min(shape, key=lambda pair: pair[0])[0]
#        minY = min(shape, key=lambda pair: pair[1])[1]
#        maxX = max(shape, key=lambda pair: pair[0])[0]
#        maxY = max(shape, key=lambda pair: pair[1])[1]
#        maxDim = max(max(maxX - minX, maxY - minY), 1)
##        print(minX, minY, maxX, maxY, maxDim)
#        normalizedCoordinates = []
##        print("num shape coords: ", len(shape))
#        for index, item in enumerate(shape):
#            nx = (item[0] - minX) / maxDim
#            ny = (item[1] - minY) / maxDim
#            normalizedCoordinates.append((nx, ny))
##            imageData["keypoints"] = normalizedCoordinates
##        print("num coords: ", len(normalizedCoordinates))

        distances = []
        for j in range(0, len(normalizedCoordinates)):
            for k in range(j+1, len(normalizedCoordinates)):
                p1 = normalizedCoordinates[j]
                p2 = normalizedCoordinates[k]
                dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                distances.append(dist)
#        print("num dists: ", len(distances))

        distances = np.expand_dims(distances, axis=0)
        preds = model.predict(distances)[0]
#        print(preds)
        prediction = preds.argmax()  # get the index that has max value
        cv2.putText(image, "{}".format([k for k,v in emoDict.items() if v == prediction][0]), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
#    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

## load the input image, resize it, and convert it to grayscale
#image = cv2.imread(args["image"])
#image = imutils.resize(image, width=500)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
## detect faces in the grayscale image
#rects = detector(gray, 1)
#
## loop over the face detections
#for (i, rect) in enumerate(rects):
#    # determine the facial landmarks for the face region, then
#    # convert the facial landmark (x, y)-coordinates to a NumPy
#    # array
#    shape = predictor(gray, rect)
#    shape = face_utils.shape_to_np(shape)
#
#    # convert dlib's rectangle to a OpenCV-style bounding box
#    # [i.e., (x, y, w, h)], then draw the face bounding box
#    (x, y, w, h) = face_utils.rect_to_bb(rect)
#    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#    # show the face number
#    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
#        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#    # loop over the (x, y)-coordinates for the facial landmarks
#    # and draw them on the image
#    for (x, y) in shape:
#        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
#
## show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)
#cv2.waitKey(0)
