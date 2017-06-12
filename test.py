import numpy as np
import sys
sys.path.append("game/")

import pygame
import wrapped_flappy_bird as game

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K

BETA = 0.01
const = 1e-5

def bce(y_true, y_pred):
	return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1) 
	#+ BETA * K.sum(y_pred * K.log(y_pred) + (1-y_pred) * K.log(1-y_pred))     #regularisation term

def ss(y_true, y_pred):
	return K.sum(K.square(y_pred - y_true), axis=-1)

def preprocess(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image, (84,84), mode = 'constant')
	image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
	image = image.reshape(1, image.shape[0], image.shape[1], 1)
	return image

model = load_model("model2", custom_objects={'binarycrossentropy': bce, 'sumofsquares': ss})
#You can choose which model to run among model1, model2,... and model5.
game_state = game.GameState()

a_t = [1,0]
FIRST_FRAME = True

while True:	
	x_t, r_t, terminal = game_state.frame_step(a_t)	
	x_t = preprocess(x_t)

	if FIRST_FRAME:
		s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)		
	else:
		s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

	y = model.predict(s_t)
	no = np.random.random()
	
	print(y)
	if FIRST_FRAME:
		a_t = [0,1]
		FIRST_FRAME = False
	else:
		no = np.random.rand()
		a_t = [0,1] if no < y[0] else [1,0]    #stochastic policy
		#a_t = [0,1] if 0.5 <y[0] else [1,0]   #deterministic policy
	
	if r_t == -1:
		FIRST_FRAME = True

#-------------- code for checking performance of saved models by finding average scores for 10 runs------------------

# for i in range(1,6):
# 	model = load_model("model" + str(i), custom_objects={'binarycrossentropy': bce, 'sumofsquares': ss})
# 	score = 0
# 	counter = 0
# 	while counter<10:	
# 		x_t, r_t, terminal = game_state.frame_step(a_t)

# 		score += 1
# 		if r_t == -1:
# 			counter += 1
 	
# 		x_t = preprocess(x_t)

# 		if FIRST_FRAME:
# 			s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
			
# 		else:
# 			s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

# 		y = model.predict(s_t)
# 		no = np.random.random()
		
# 		print(y)
# 		if FIRST_FRAME:
# 			a_t = [0,1]
# 			FIRST_FRAME = False
# 		else:
# 			no = np.random.rand()
# 			a_t = [0,1] if no < y[0] else [1,0]	
# 			#a_t = [0,1] if 0.5 <y[0] else [1,0]
		
# 		if r_t == -1:
# 			FIRST_FRAME = True

# 	f = open("test_rewards.txt","a")
# 	f.write(str(score/10)+ "\n")
# 	f.close()
