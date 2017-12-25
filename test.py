#!/usr/bin/env python

import cv2
import numpy as np
refPt=[]
cropping= False

def click_and_crop(event,x,y,flags,param):
	global refPt,cropping
	if len(refPt)==2:
		return
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt =[(x,y)]
		cropping = True
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x,y))
		cropping = False
		#cv2.rectangle(img,refPt[0],refPt[1],(0,0,255),2)


# cv2.namedWindow('background')
# cv2.setMouseCallback('background',draw_circle)


def remove_bg(frame):
	print('111')
	fg_mask=bg_model.apply(frame)
	kernel = np.ones((3,3),np.uint8)
	fg_mask=cv2.erode(fg_mask,kernel,iterations = 1)
	frame=cv2.bitwise_and(frame,frame,mask=fg_mask)
	return frame


cap=cv2.VideoCapture(0)
bg_captured=0;

while(cap.isOpened()):
	ret,img=cap.read()
	cv2.rectangle(img,(0,0),(400,400),(255,0,0),3)
	cv2.imshow("input",img)
	interrupt=cv2.waitKey(10)
	#if bg_captured:
	#	cv2.putText(img,"Place hand inside boxes and press 'c' to capture hand histogram",)	
	roi=img[0:400,0:400]
	if interrupt & 0xFF == ord('b'):
		bg_model=cv2.createBackgroundSubtractorMOG2()
		remove_bg(roi)
		bg_captured=1;
	elif interrupt & 0xFF == ord('c'):
		if(bg_captured):
			im=remove_bg(roi);
			print(im);
			cv2.imshow("removed",im)
	elif interrupt & 0xFF == ord('r'):
		bg_captured=0;
		cv2.destroyWindow("removed");
		print('Reset finished.')
	elif interrupt & 0xFF == ord('q'):
		break;

# cv2.namedWindow("ROI");	

# while True:
# 	cv2.imshow('ROI',img);
# 	key=cv2.waitKey(1) & 0xFF
# 	if key==ord("c"):
# 		break



# ret1,im=cap.read();
# cv2.imshow("ROI",);
# cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
