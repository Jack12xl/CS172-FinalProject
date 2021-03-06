import numpy as np
import cv2

cap = cv2.VideoCapture(0)


# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

def nothing(x):
    pass

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)

while(cap.isOpened()):
	ret,img = cap.read()
	# load a statistical model, which is the XML file classifier for frontal
	# faces provided by OpenCV to detect the faces from the frames captured
	# from a webcam during the testing stage.
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(faces)
	for (x,y,w,h) in faces:
		cv2.circle(img,(int(x+w/2),int(y+h/2)),int(max(w/2+10,h/2+10)),(0,0,0),-1)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		# eyes = eye_cascade.detectMultiScale(roi_gray)
		# for (ex,ey,ew,eh) in eyes:
		# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	# cv2.imshow('marked',img)
	cv2.imwrite('faceSubtraction.jpg',img)
	# Blur the image
	blur = cv2.blur(img,(3,3))
	#Convert to HSV color space
	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	#Create a binary image with where white will be skin colors and rest is black
	mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
	#Kernel matrices for morphological transformation
	kernel_square = np.ones((11,11),np.uint8)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	#Perform morphological transformations to filter out the background noise
	#Dilation increase skin color area
	#Erosion increase skin color area
	dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
	erosion = cv2.erode(dilation,kernel_square,iterations = 1)
	dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
	filtered = cv2.medianBlur(dilation2,5)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
	dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
	median = cv2.medianBlur(dilation2,5)
	ret,thresh = cv2.threshold(median,127,255,0)
	#Find contours of the filtered frame
	_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#Find Max contour area (Assume that hand is in the frame)
	max_area=100
	ci=0
	for i in range(len(contours)):
		cnt=contours[i]
		area = cv2.contourArea(cnt)
		if(area>max_area):
			max_area=area
			ci=i
	#Largest area contour
	cnts = contours[ci]
	#Find convex hull
	hull = cv2.convexHull(cnts)
	#Find convex defects
	hull2 = cv2.convexHull(cnts,returnPoints = False)
	defects = cv2.convexityDefects(cnts,hull2)
	#Get defect points and draw them in the original image
	FarDefect = []
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnts[s][0])
		end = tuple(cnts[e][0])
		far = tuple(cnts[f][0])
		FarDefect.append(far)
		# cv2.line(img,start,end,[0,255,0],1)
		# cv2.circle(img,far,10,[100,255,255],3)

	#Find moments of the largest contour
	moments = cv2.moments(cnts)

	#Central mass of first order moments
	if moments['m00']!=0:
		cx = int(moments['m10']/moments['m00']) # cx = M10/M00
		cy = int(moments['m01']/moments['m00']) # cy = M01/M00
	centerMass=(cx,cy)

	#Draw center mass
	cv2.circle(img,centerMass,7,[100,0,255],2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,'Center',tuple(centerMass),font,2,(255,255,255),2)
	#Distance from each finger defect(finger webbing) to the center mass
	distanceBetweenDefectsToCenter = []
	for i in range(0,len(FarDefect)):
		x =  np.array(FarDefect[i])
		centerMass = np.array(centerMass)
		distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
		distanceBetweenDefectsToCenter.append(distance)

	#Get an average of three shortest distances from finger webbing to center mass
	sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
	AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

	#Get fingertip points from contour hull
	#If points are in proximity of 80 pixels, consider as a single point in the group
	finger = []
	for i in range(0,len(hull)-1):
		if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
			if hull[i][0][1] <500 :
				finger.append(hull[i][0])

	#The fingertip points are 5 hull points with largest y coordinates
	finger =  sorted(finger,key=lambda x: x[1])
	fingers = finger[0:5]

	#Calculate distance of each finger tip to the center mass
	fingerDistance = []
	for i in range(0,len(fingers)):
		distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
		fingerDistance.append(distance)

	#Finger is pointed/raised if the distance of between fingertip to the center mass is larger
	#than the distance of average finger webbing to center mass by 130 pixels
	result = 0
	for i in range(0,len(fingers)):
		if fingerDistance[i] > AverageDefectDistance+130:
			result = result +1

	#Print number of pointed fingers
	# cv2.putText(img,str(result),(100,100),font,2,(255,255,255),2)

	#Print bounding rectangle
	x,y,w,h = cv2.boundingRect(cnts)
	img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.drawContours(img,[hull],-1,(255,255,255),2)

	##### Show final image ########
	# cv2.imshow('Dilation',img)

	###############################

	#Print execution time
	#print time.time()-start_time

	#close the output video by pressing 'ESC'
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
	    break
	cv2.imshow('test',img)
	cv2.imwrite('HandDetection.jpg',img)



	interrupt=cv2.waitKey(10)
	if interrupt & 0xFF == ord('q'):
		break;



cap.release()
cv2.destroyAllWindows()

