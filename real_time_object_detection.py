#!How to run?: 
#*python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

#~import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from numpy.random.mtrand import random

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--MobileNetSSD_deploy.prototxt.txt", required=True,
#	help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--MobileNetSSD_deploy.caffemodel", required=True,
#	help="path to Caffe pre-trained model")
#ap.add_argument("-c", "--20%", type=float, default=0.2,
#	help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())

# Arguments used here:
# prototxt = MobileNetSSD_deploy.prototxt.txt (required)
# model = MobileNetSSD_deploy.caffemodel (required)
# confidence = 0.2 (default)


video= cv2.VideoCapture(1)



CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#*Assigning random colors to each of the classes
color= np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")
#!print(net)


#*initialize the video stream,
#*and initialize the FPS counter
# !print("[INFO] starting video stream...")



#! Consider the video stream as a series of frames. We capture each frame based on a certain FPS, and loop over each frame
#! loop over the frames from the video stream
while True:
	#? grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	
	ret, frame = video.read()
	frame = cv2.resize(frame, (640,480))
	
	#* grab the frame dimensions and convert it to a blob
	#& First 2 values are the h and w of the frame. Here h = 225 and w = 400

	(h, w) = frame.shape[:2]
	#! Resize each frame
	# !resized_image = cv2.resize(frame, (300, 300))
	# !Creating the blob
	# !The function:

	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),(1/127.5), (300, 300), 127.5, swapRB=True)
	net.setInput(blob) 
	
	
	#^Predictions:
	predictions = net.forward()

	#*loop over the predictions

	for i in np.arange(0, predictions.shape[2]):
		#! extract the confidence (i.e., probability) associated with the prediction
		#* predictions.shape[2] = 100 here
		confidence = predictions[0, 0, i, 2]

		#^ Filter out predictions lesser than the minimum confidence level
		#? Here, we set the default confidence as 0.2. Anything lesser than 0.2 will be filtered
		if confidence > 0.5:

			# !extract the index of the class label from the 'predictions'
			# &idx is the index of the class label
			#~ E.g. for person, idx = 15, for chair, idx = 9, etc.
			id = int(predictions[0, 0, i, 1])

			
			# &then compute the (x, y)-coordinates of the bounding box for the object
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])


			#* Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
			#! Convert them to integers: 130 76 393 224
			(startX, startY, endX, endY) = box.astype("int")

			# ?Get the label with the confidence score

			label = "{}: {:.2f}%".format(CLASSES[id], confidence * 100)
			print("Object detected: ", label)

			# ~Draw a rectangle across the boundary of the object
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				color[id], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[id], 2)

	#& show the output frame
	cv2.imshow("Frame", frame)

	
	key = cv2.waitKey(1) & 0xFF

	#! Press 'q' key to break the loop
	if key == ord("q"):
		break

#?Destroy windows and cleanup
cv2.destroyAllWindows()
#&Stop the video stream
video.stop()

#* We are DONE !
