# -*- coding: utf-8 -*-
#!/usr/bin/python

import time
import cv2
import pyzbar.pyzbar as pyzbar
from time import sleep
import numpy as np
import math
import types
import ctypes
import time
#import matplotlib.image as mpimg
#from pytesseract import *
import datetime
import collections
from PIL import Image

#读取人脸识别模型库
faceCascade = cv2.CascadeClassifier('/home/pi/Desktop/shijueshibie_lyt/haarcascade_frontalface_default.xml')

#1、形状
def xingzhuang(): #形状
	xz = ''
	for i in range(0,5):
		#摄像头激活sedfwqe
		camera = cv2.VideoCapture(0)
		sleep(1)
		ret, frame = camera.read()
		h, w, ch = frame.shape
		result = np.zeros((h, w, ch), dtype=np.uint8)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
		ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	
		image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		#contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in range(len(contours)):
			epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
			approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
			corners = len(approx)
			if corners == 3:
				xz = '三角形'
			if corners == 4:
				xz = '矩形'
			if corners >= 10:
				xz = '圆形'
			if 4 < corners < 10:
				xz = '多边形'
		if(1):
			print(xz)			
			return
		
		cv2.imshow('1',frame)
		k = cv2.waitKey(2)& 0xff
		if k == 27:     # 键盘Esc键退出
		    break
		#cv2.destroyAllWindows()
		camera.release()
			
#2、人脸识别
def face():
	for i in range(0,5):
		#摄像头激活sedfwqe
		camera = cv2.VideoCapture(0)
		ret, img = camera.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,     
			scaleFactor=1.2,
			minNeighbors=5,     
			minSize=(20, 20)
		)
		if(len(faces) != 0):
			print("ok")
			return
		
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]  
		cv2.imshow('1',img)
		
		#开启弹窗
		k = cv2.waitKey(2) & 0xff
		if k == 27:
			break
		#cv2.destroyAllWindows()
		camera.release()

#3、扫码
def saoma(): #二维码或者条形码
	barcodeData = ''
	for i in range(0,5):
		#摄像头激活sedfwqe
		camera = cv2.VideoCapture(0)
		sleep(1)
		ret, frame = camera.read()
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#cv2.imshow('1',image)
		barcodes = pyzbar.decode(image)
		for barcode in barcodes:
			barcodeData = barcode.data.decode("utf-8")
			if(barcodeData == "ABC"):
				print(barcodeData)
				return
		
		cv2.imshow('1',frame)
		k = cv2.waitKey(2)& 0xff
		if k == 27:     # 键盘Esc键退出
		    break
		#cv2.destroyAllWindows()
		camera.release()
		
#4、手势识别
def shoushi():
	for i in range(0,5):
		#摄像头激活sedfwqe
		camera = cv2.VideoCapture(0)
		ret,frame = camera.read() # 读取摄像头每帧图片
		kernel = np.ones((2,2),np.uint8)
		roi = frame[0:450,0:600] # 选取图片中固定位置作为手势输入

		cv2.rectangle(frame,(0,0),(600,450),(0,0,255),0) # 用红线画出手势识别框
		# 基于hsv的肤色检测
		hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
		lower_skin = np.array([0,28,70],dtype=np.uint8)
		upper_skin = np.array([20, 255, 255],dtype=np.uint8)
		
		# 进行高斯滤波
		mask = cv2.inRange(hsv,lower_skin,upper_skin)
		mask = cv2.dilate(mask,kernel,iterations=4)
		mask = cv2.GaussianBlur(mask,(5,5),100)
		try:
			# 找出轮廓
			img,contours,h = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			cnt = max(contours,key=lambda x:cv2.contourArea(x))
			epsilon = 0.0005*cv2.arcLength(cnt,True)
			approx = cv2.approxPolyDP(cnt,epsilon,True)
			hull = cv2.convexHull(cnt)
			areahull = cv2.contourArea(hull)
			areacnt = cv2.contourArea(cnt)
			arearatio = ((areahull-areacnt)/areacnt)*100
			# 求出凹凸点
			hull = cv2.convexHull(approx,returnPoints=False)
			defects = cv2.convexityDefects(approx,hull)
			l=0 #定义凹凸点个数初始值为0

			for i in range(defects.shape[0]):
				s,e,f,d, = defects[i,0]
				start = tuple(approx[s][0])
				end = tuple(approx[e][0])
				far = tuple(approx[f][0])
				pt = (100,100)

				a = math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
				b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
				c = math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
				s = (a+b+c)/2
				ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
				# 手指间角度求取
				angle = math.acos((b**2 + c**2 -a**2)/(2*b*c))*57

				if angle<=90 and d>20:
					l+=1
					cv2.circle(roi,far,3,[255,0,0],-1)
				cv2.line(roi,start,end,[0,255,0],2) # 画出包络线
			l+=1
			font = cv2.FONT_HERSHEY_SIMPLEX
			# 下面的都是条件判断
			if l==1:
				if areacnt<2000:
						cv2.putText(frame,"put hand in the window",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
				else:
					if arearatio<12:
						cv2.putText(frame,'0',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
					elif arearatio<17.5:
						cv2.putText(frame,"1",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
					else:
						cv2.putText(frame,'1',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
			elif l==2:
				 cv2.putText(frame,'2',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
			elif l==3:
				 if arearatio<27:
					cv2.putText(frame,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
				 else:
					cv2.putText(frame,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
			elif l==4:
				 cv2.putText(frame,'4',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
			elif l==5:
				 cv2.putText(frame,'5',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
			cv2.imshow('1',frame)
			#cv2.imshow('mask', mask)
			if(1):
				print(l)
				return
		except:
				 continue
				 
		k = cv2.waitKey(2) & 0xff
		if k == 27:     # 键盘Esc键退出
			break
		
		#cv2.destroyAllWindows()
		camera.release()

while(1):
	shoushi()
	xingzhuang()
	face()
	saoma()

