# -*- coding: utf-8 -*-
import numpy as np
import cv2
from time import time
from time import sleep
import re
import os

from scipy.ndimage import zoom
from scipy.spatial import distance
from scipy import ndimage

import dlib

from tensorflow.keras.models import load_model
import tensorflow as tf
from imutils import face_utils

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		self.yamur = cv2.VideoCapture('basgul_animation.mp4')
		self.model = load_model('keras_model.h5')
		self.face_detect = dlib.get_frontal_face_detector()
		self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
		self.shape_x = 48
		self.shape_y = 48
		self.wbgr = [255, 255, 255]
		self.bbgr = [24, 24, 24]
		self.count = 0

	def __del__(self):
		self.video.release()
		self.yamur.release()

	def change_color(self, frame, white, black):
		r = frame[:,:,2]
		np.place(r, r==255, white[0])
		np.place(r, r < 50, black[0])
		g = frame[:,:,1]
		np.place(g, g==255, white[1])
		np.place(g, g < 50, black[1])
		b = frame[:,:,0]
		np.place(b, b==255, white[2])
		np.place(b, b < 50, black[2])
		return np.stack((r,g,b), 2)

	def get_frame(self):
		prediction_result = 6
		ret, frame = self.video.read()
		r, f = self.yamur.read()
		f = cv2.resize(f, (960,540))

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = self.face_detect(gray, 1)
		if ret:
			self.count += 1
		if self.count % 3 == 0:
			for (i, rect) in enumerate(rects):
				# Identify face coordinates
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				face = gray[y:y+h,x:x+w]
				
				#Zoom on extracted face
				face = zoom(face, (self.shape_x / face.shape[0], self.shape_y / face.shape[1]))
				
				#Cast type float
				face = face.astype(np.float32)
				
				#Scale
				face /= float(face.max())
				face = np.reshape(face.flatten(), (1, self.shape_x, self.shape_y, 1))
				
				#Make Prediction
				prediction = self.model.predict(face)
				prediction_result = np.argmax(prediction)
			self.count = 0
			if prediction_result == 0:
				# Angry
				self.wbgr = [64, 38, 237]
				self.bbgr = [25, 25, 148]
			elif prediction_result == 1:
				# Disgust
				self.wbgr = [83, 177, 0]
				self.bbgr = [75, 109, 41]

			elif prediction_result == 2:
				# Fear
				self.wbgr = [255, 112, 132]
				self.bbgr = [97, 57, 32]
			elif prediction_result == 3:
				# Happy
				self.wbgr = [62, 167, 255]
				self.bbgr = [0, 118, 238]
			elif prediction_result == 4:
				# Sad
				self.wbgr = [216, 119, 22]
				self.bbgr = [144, 75, 0]
			elif prediction_result == 5:
				# Surprise
				self.wbgr = [153, 246, 255]
				self.bbgr = [0, 198, 255]
			elif self.count % 6 == 0:
				# Neutral
				self.wbgr = [255, 255, 255]
				self.bgr = [28, 15, 252]

		self.bbgr = [24,24,24]
		f = self.change_color(f, self.wbgr, self.bbgr)

		ret, jpeg = cv2.imencode('.jpg', f)

		return jpeg.tobytes()
