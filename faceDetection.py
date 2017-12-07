import numpy as np
import cv2

cap=cv2.VideoCapture(0)

while(cap.isOpened()):
	ret,img = cap.read()
	cv2.imshow('img',img)
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
	cv2.imshow('marked',img)
	interrupt=cv2.waitKey(10)
	if interrupt & 0xFF == ord('q'):
		break;

cap.release()
cv2.destroyAllWindows()

