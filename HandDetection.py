import numpy as np
import cv2

cap = cv2.VideoCapture(0)


# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# # Starting with 100's to prevent error while masking
# h,s,v = 100,100,100

# def nothing(x):
#     pass

# # Creating track bar
# cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
# cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
# cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)

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
	blur = cv2.blur(img,(5,5))
	#Convert to HSV color space
	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	#Create a binary image with where white will be skin colors and rest is black
	mask2 = cv2.inRange(hsv,np.array([2,30,20]),np.array([15,255,255]))
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

	mul = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mul = np.multiply(mul,thresh)

	cv2.imshow('mul',mul)

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


	# Print bounding rectangle
	x,y,w,h = cv2.boundingRect(cnts)
	img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	roi = mul[y:y+h,x:x+w]
	pic = np.zeros((120,180),np.uint8)
	# roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

	if w/180 < h/120 :
		width = int(120*w/h)
		roi = cv2.resize(roi,(width,120),interpolation=cv2.INTER_CUBIC)
		height = h
		pic[0:120,90-int(width/2):90-int(width/2)+width] = roi
	# else:
	# 	height = int(180*h/w)
	# 	roi = cv2.resize(roi,(180,height),interpolation=cv2.INTER_CUBIC)
	# 	width = w
	# 	pic[0:height,0:120] = roi


	cv2.imshow('ROI',pic)


	# cv2.drawContours(img,[hull],-1,(255,255,255),2)

	###############################

	#Print execution time
	#print time.time()-start_time

	#close the output video by pressing 'ESC'
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
	    break
	cv2.imshow('test',img)
	cv2.imwrite('HandDetection.jpg',pic)

	interrupt=cv2.waitKey(10)
	if interrupt & 0xFF == ord('q'):
		break;



cap.release()
cv2.destroyAllWindows()

