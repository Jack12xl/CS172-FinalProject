import cv2
import os
import numpy as np

def siftExtrator(poseName,poseId):
	# os.makedirs('./test')
	descriptors=np.zeros((1,128))
	for i in range(10):
		img=cv2.imread('./hand-reader-dataset/%c/0%d00%d.jpg'%(poseName,poseId,i))
		print('./hand-reader-dataset/%c/0%d00%d.jpg'%(poseName,poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		descriptors=np.concatenate((descriptors,des))

		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseName,i),img)
	for i in range(11,50):
		img=cv2.imread('./hand-reader-dataset/%c/0%d0%d.jpg'%(poseName,poseId,i))
		print('./hand-reader-dataset/%c/0%d0%d.jpg'%(poseName,poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		descriptors=np.concatenate((descriptors,des))

		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseName,i),img)
		print(des.shape)
	return descriptors

siftExtrator('C',2)

