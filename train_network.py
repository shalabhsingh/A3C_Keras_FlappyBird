import numpy as np
import sys
sys.path.append("game/")

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Activation, Input
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K
from keras.callbacks import LearningRateScheduler, History
import tensorflow as tf

import pygame
import wrapped_flappy_bird as game

import threading
import time
import math

GAMMA = 0.99                #discount value
BETA = 0.01                 #regularisation coefficient
IMAGE_ROWS = 84
IMAGE_COLS = 84
IMAGE_CHANNELS = 4
LEARNING_RATE = 7e-4	
EPISODE = 0
THREADS = 16
t_max = 5
const = 1e-5
T = 0

episode_r = []
episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
episode_output = []
episode_critic = []

ACTIONS = 2
FIRST_FRAME = True 
a_t = np.zeros(ACTIONS)

def binarycrossentropy(y_true, y_pred):     #policy loss
	return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1) 
	# BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

def sumofsquares(y_true, y_pred):        #critic loss
	return K.sum(K.square(y_pred - y_true), axis=-1)

def buildmodel():
	print("Model buliding begins")

	model = Sequential()
	keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Input')
	h0 = Convolution2D(16, kernel_size = (8,8), strides = (4,4), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(S)
	h1 = Convolution2D(32, kernel_size = (4,4), strides = (2,2), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h0)
	h2 = Flatten()(h1)
	h3 = Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h2)
	P = Dense(1, name = 'o_P', activation = 'sigmoid', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h3)
	V = Dense(1, name = 'o_V', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h3)

	model = Model(inputs = S, outputs = [P,V])
	rms = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	model.compile(loss = {'o_P': binarycrossentropy, 'o_V': sumofsquares}, loss_weights = {'o_P': 1., 'o_V' : 0.5}, optimizer = rms)
	return model

def preprocess(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image, (IMAGE_ROWS,IMAGE_COLS), mode = 'constant')	
	image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
	image = image.reshape(1, image.shape[0], image.shape[1], 1)
	return image


model = buildmodel()
#model = load_model("saved_models/model_updates", custom_objects={'binarycrossentropy': binarycrossentropy, 'sumofsquares': sumofsquares})
model._make_predict_function()
graph = tf.get_default_graph()

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('o_P').output)


a_t[0] = 1 #index 0 = no flap, 1= flap
#output of network represents probability of flap

game_state = []
for i in range(0,THREADS):
	game_state.append(game.GameState())


def runprocess(thread_id, s_t, FIRST_FRAME, t):
	global T
	global a_t
	global model

	t_start = t
	terminal = False
	r_t = 0
	r_store = []
	state_store = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	output_store = []
	critic_store = []
	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

	if FIRST_FRAME:
		a_t = [1,0]

	while t-t_start < t_max and terminal == False:
		t += 1
		T += 1
		intermediate_output = 0
		
		if FIRST_FRAME == False:
			with graph.as_default():
				out = model.predict(s_t)[0]			
				intermediate_output = intermediate_layer_model.predict(s_t)
			no = np.random.rand()
			a_t = [0,1] if no < out else [1,0]  #stochastic action
			#a_t = [0,1] if 0.5 <y[0] else [1,0]  #deterministic action

		x_t, r_t, terminal = game_state[thread_id].frame_step(a_t)
		x_t = preprocess(x_t)

		if FIRST_FRAME:
			s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
			FIRST_FRAME = False
		else:
			s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

		y = 0 if a_t[0] == 1 else 1
		
		with graph.as_default():
			critic_reward = model.predict(s_t)[1]

		r_store = np.append(r_store, r_t)
		state_store = np.append(state_store, s_t, axis = 0)
		output_store = np.append(output_store, y)
		critic_store = np.append(critic_store, critic_reward)

		print("Frame = " + str(T) + ", Updates = " + str(EPISODE) + ", Thread = " + str(thread_id) + ", Output = "+ str(intermediate_output))
	
	if terminal == False:
		r_store[len(r_store)-1] = critic_store[len(r_store)-1]
	else:
		r_store[len(r_store)-1] = -1
		FIRST_FRAME = True
	
	for i in range(2,len(r_store)+1):
		r_store[len(r_store)-i] = r_store[len(r_store)-i] + GAMMA*r_store[len(r_store)-i + 1]

	return t, state_store, output_store, r_store, critic_store, FIRST_FRAME


def step_decay(epoch):
	decay = 3.2e-8
	lrate = LEARNING_RATE - epoch*decay
	lrate = math.fabs(lrate)
	return lrate

class actorthread(threading.Thread):
	def __init__(self,thread_id, s_t, FIRST_FRAME, t):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.s_t = s_t
		self.FIRST_FRAME = FIRST_FRAME
		self.t = t

	def run(self):
		global episode_output
		global episode_r
		global episode_critic
		global episode_state

		threadLock.acquire()
		self.t, state_store, output_store, r_store, critic_store, self.FIRST_FRAME = runprocess(self.thread_id, self.s_t, self.FIRST_FRAME, self.t)
		self.s_t = state_store[-1]
		self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])

		episode_r = np.append(episode_r, r_store)
		episode_output = np.append(episode_output, output_store)
		episode_state = np.append(episode_state, state_store, axis = 0)
		episode_critic = np.append(episode_critic, critic_store)

		threadLock.release()

states = np.zeros((16, IMAGE_ROWS, IMAGE_COLS, 4))
FF = []
tt = []

for i in range(0,THREADS):
	FF.append(True)
	tt.append(0)

while True:	
	threadLock = threading.Lock()
	threads = []
	for i in range(0,THREADS):
		threads.append(actorthread(i,states[i], FF[i], tt[i]))

	states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))

	for i in range(0,THREADS):
		threads[i].start()

	for i in range(0,THREADS):
		threads[i].join()

	for i in range(0,THREADS):
		states = np.append(states, threads[i].s_t, axis = 0)
		FF[i] = threads[i].FIRST_FRAME
		tt[i] = threads[i].t

	e_mean = np.mean(episode_r)
	advantage = episode_r - episode_critic
	print("backpropagating")

	lrate = LearningRateScheduler(step_decay)
	callbacks_list = [lrate]

	#loss = model.train_on_batch(episode_state, [episode_output, episode_critic], sample_weight = advantage)
	weights = {'o_P':advantage, 'o_V':np.ones(len(advantage))}
	
	history = model.fit(episode_state, [episode_output, episode_r], epochs = EPISODE + 1, batch_size = len(episode_output), callbacks = callbacks_list, sample_weight = weights, initial_epoch = EPISODE)

	episode_r = []
	episode_output = []
	episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	episode_critic = []

	f = open("rewards.txt","a")
	f.write("Update: " + str(EPISODE) + ", Reward_mean: " + str(e_mean) + ", Loss: " + str(history.history['loss']) + "\n")
	f.close()

	if EPISODE % 50 == 0:
		model.save("saved_models/model_updates" + str(EPISODE))
	EPISODE += 1
