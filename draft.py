import cv2
import numpy as np

def siftExtrator(poseId):
	descriptors=np.zeros((1,128))
	for i in range(10):
		img=cv2.imread('./Hand-Reader-Dataset/hand-reader-dataset/%c/0000%d.jpg'%(poseId,i))
		print('./Hand-Reader-Dataset/hand-reader-dataset/%c/0000%d.jpg'%(poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		descriptors=np.concatenate((descriptors,des))

		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseId,i),img)
	for i in range(11,50):
		img=cv2.imread('./Hand-Reader-Dataset/hand-reader-dataset/%c/000%d.jpg'%(poseId,i))
		print('./Hand-Reader-Dataset/hand-reader-dataset/%c/000%d.jpg'%(poseId,i))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift=cv2.xfeatures2d.SIFT_create()
		kp,des=sift.detectAndCompute(gray,None)
		descriptors=np.concatenate((descriptors,des))

		img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('./test/sift%c%d.jpg'%(poseId,i),img)
		print(descriptors.shape)

siftExtrator('A')

