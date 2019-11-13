# -*- coding: utf-8 -*-

from itertools import cycle
from numpy.random import randint,choice

import torch 
import torch.nn as nn
import torch.nn.functional as F

import sys

import numpy as np

from collections import deque

import random


#argument parsing for bash script
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lr", help="learning rate used for training agent", type = float)
args = parser.parse_args()


import pygame
from pygame.locals import *

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 160 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
SCORE = 0

BACKGROUND = pygame.image.load('background.png')


class Bird(pygame.sprite.Sprite):

	def __init__(self,displayScreen):

		pygame.sprite.Sprite.__init__(self)

		self.image = pygame.image.load('redbird.png')

		self.x = int(SCREENWIDTH * 0.2)
		self.y = SCREENHEIGHT*0.5
		
		self.rect = self.image.get_rect()
		self.height = self.rect.height
		self.screen = displayScreen
		
		self.playerVelY = -9
		self.playerMaxVelY = 10
		self.playerMinVelY = -8
		self.playerAccY = 1
		self.playerFlapAcc = -9
		self.playerFlapped = False

		self.display(self.x, self.y)

	def display(self,x,y):

		self.screen.blit(self.image, (x,y))
		self.rect.x, self.rect.y = x,y


	def move(self,input):

		if input != None:
			self.playerVelY = self.playerFlapAcc
			self.playerFlapped = True

		if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
			self.playerVelY += self.playerAccY
		if self.playerFlapped:
			self.playerFlapped = False

		self.y += min(self.playerVelY, SCREENHEIGHT - self.y - self.height)
		self.y = max(self.y,0)
		self.display(self.x,self.y)


class PipeBlock(pygame.sprite.Sprite):

	def __init__(self,image,upper):

		pygame.sprite.Sprite.__init__(self)

		if upper == False:
			self.image = pygame.image.load(image)
		else:
			self.image = pygame.transform.rotate(pygame.image.load(image),180)

		self.rect = self.image.get_rect()



class Pipe(pygame.sprite.Sprite):
	
	
	def __init__(self,screen,x):

		pygame.sprite.Sprite.__init__(self)

		self.screen = screen
		self.lowerBlock = PipeBlock('pipe-red.png',False)
		self.upperBlock = PipeBlock('pipe-red.png',True)
		

		self.pipeWidth = self.upperBlock.rect.width
		self.x = x
		

		heights = self.getHeight()
		self.upperY, self.lowerY = heights[0], heights[1]

		self.behindBird = 0
		self.display()


	def getHeight(self):

		# randVal = randint(1,10)
		randVal = choice([1,2,3,4,5,6,7,8,9], p =[0.04,0.04*2,0.04*3,0.04*4,0.04*5,0.04*4,0.04*3,0.04*2,0.04] )
		#randVal = 4
		midYPos = 106 + 30*randVal

		upperPos = midYPos - (PIPEGAPSIZE/2)
		lowerPos = midYPos + (PIPEGAPSIZE/2)

		# print(upperPos)
		# print(lowerPos)
		# print('-------')
		return([upperPos,lowerPos])

	def display(self):

		self.screen.blit(self.lowerBlock.image, (self.x, self.lowerY))
		self.screen.blit(self.upperBlock.image, (self.x, self.upperY - self.upperBlock.rect.height))
		self.upperBlock.rect.x, self.upperBlock.rect.y = self.x, (self.upperY - self.upperBlock.rect.height)
		self.lowerBlock.rect.x, self.lowerBlock.rect.y = self.x, self.lowerY

	def move(self):

		self.x -= 3

		if self.x <= 0:
			self.x = SCREENWIDTH
			heights = self.getHeight()
			self.upperY, self.lowerY = heights[0], heights[1]
			self.behindBird = 0

		self.display()
		return([self.x+(self.pipeWidth/2), self.upperY, self.lowerY])

def observe(Mem):

	n_obs = 3500 #how many obervations are done before starting training 
	#n_obs = 100

	while len(Mem) < n_obs:

		#print("memory size is ", len(Mem))

		#print("observe")

		pygame.init()

		FPSCLOCK = pygame.time.Clock()
		DISPLAY  = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
		pygame.display.set_caption('Flappy Bird')

		global SCORE

		bird = Bird(DISPLAY)
		pipe1 = Pipe(DISPLAY, SCREENWIDTH+100)
		pipe2 = Pipe(DISPLAY, SCREENWIDTH+100+(SCREENWIDTH/2))

		pipeGroup = pygame.sprite.Group()
		pipeGroup.add(pipe1.upperBlock)
		pipeGroup.add(pipe2.upperBlock)
		pipeGroup.add(pipe1.lowerBlock)
		pipeGroup.add(pipe2.lowerBlock)

		# birdGroup = pygame.sprite.Group()
		# birdGroup.add(bird1)
		
		pause =0
		action = [False, True]
		reward = 0
		cnt = 0
		run = True

		BATCH_SIZE = 32

		gamma = 0.99 

		while run == True:

			b_flap = False #flap boolean value
			moved = False

			cnt +=1 

			DISPLAY.blit(BACKGROUND,(0,0))

			t = pygame.sprite.spritecollideany(bird,pipeGroup)

			if t!=None or (bird.y== 512 - bird.height) or (bird.y == 0):
				#print("GAME OVER")
				#print("FINAL SCORE IS %d"%SCORE)

				last_el = Mem[-1]

				#Mem[-1] = [last_el[0], last_el[1], last_el[2], torch.tensor([float(0),float(0),float(0),float(0)]), True]
				Mem.append((last_el[0], last_el[1], last_el[2], torch.tensor([float(0),float(0),float(0),float(0)]), True))

				run = False

			if run == True :
				
				#PYGAME EVENTS
				for event in pygame.event.get():
					if event.type == QUIT or (event.type == KEYDOWN and (event.key == K_ESCAPE )):
						pygame.quit()
						sys.exit()
				for event in pygame.event.get():
					if event.type == KEYDOWN and event.key == K_m :
						pause=1

				if pipe1.x < pipe2.x and pipe1.x > bird.x :
						next_pipe = pipe1

				elif pipe2.x > bird.x :
						next_pipe = pipe2

				else : 
					next_pipe = pipe1

				#CURRENT STATE 
				#state = [bird.y, bird.x -next_pipe.x, abs(int((next_pipe.upperY - next_pipe.lowerY)*0.5)) + next_pipe.upperY]

				# state = torch.tensor([[bird.y],[bird.x -next_pipe.x], [next_pipe.upperY], [next_pipe.lowerY]])
				#state = torch.tensor([bird.y,next_pipe.x,next_pipe.upperY,next_pipe.lowerY])
				state = torch.tensor([bird.y/512, next_pipe.x/388, next_pipe.upperY/512, next_pipe.lowerY/512])

				if cnt % 3 == 0: #an action is only taken every 3 timesteps

					# Make a random action (exploration)
					b_flap = np.random.randint(2)
					b_flap = action[b_flap]

				#print("flapping? ", b_flap)
				#print("counter = ", cnt)

				reward = cnt #pas de probleme de sparse reward en soit

				#move world
				if b_flap == True :
					bird.move("UP")
					moved = True
					

				if moved == False:
					bird.move(None)
				else:
					moved = False

				
				pipe1Pos = pipe1.move()
				if pipe1Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width/2):
					if pipe1.behindBird == 0:
						pipe1.behindBird = 1
						SCORE += 1
						reward += 10
						#print("SCORE IS %d"%SCORE)

				pipe2Pos = pipe2.move()
				if pipe2Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width/2):
					if pipe2.behindBird == 0:
						pipe2.behindBird = 1
						SCORE += 1
						reward += 10
						#print("SCORE IS %d"%SCORE)
				
				# next_state = [bird.y, bird.x -next_pipe.x, abs(int((next_pipe.upperY - next_pipe.lowerY)*0.5)) + next_pipe.upperY]
				# next_state = torch.tensor([[bird.y], [bird.x -next_pipe.x], [next_pipe.upperY], [next_pipe.lowerY]])
				#state = torch.tensor([bird.y/512, next_pipe.x/388, next_pipe.upperY/512, next_pipe.lowerY/512])
				next_state = torch.tensor([float(bird.y/512), float(next_pipe.x/388), float(next_pipe.upperY/512), float(next_pipe.lowerY/512)])

				#print("state: ", state)
				#print("next state: ", next_state)
				#print("x: ", bird.x)
				#print("reward: ", reward)
				
				if pause==0:
					pygame.display.update()

				if cnt % 3 == 0:
					Mem.append((state, b_flap, reward, next_state, False)) #state : Tensor 
																		   #b_flap : Boolean
																		   #reward : Integer
																		   #next state : Tensor 
																		   #Terminal : Boolean

				FPSCLOCK.tick(FPS)

	print("At the end of the observation state, memory's size is ", len(Mem))

	return(Mem)

def train(model, optimizer, loss_fn, decay_step, Mem):
	

	pygame.init()

	FPSCLOCK = pygame.time.Clock()
	DISPLAY  = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
	pygame.display.set_caption('Flappy Bird')

	global SCORE

	bird = Bird(DISPLAY)
	pipe1 = Pipe(DISPLAY, SCREENWIDTH+100)
	pipe2 = Pipe(DISPLAY, SCREENWIDTH+100+(SCREENWIDTH/2))

	pipeGroup = pygame.sprite.Group()
	pipeGroup.add(pipe1.upperBlock)
	pipeGroup.add(pipe2.upperBlock)
	pipeGroup.add(pipe1.lowerBlock)
	pipeGroup.add(pipe2.lowerBlock)

	# birdGroup = pygame.sprite.Group()
	# birdGroup.add(bird1)
	
	b_flap = False #flap boolean value
	moved = False
	pause =0
	action = [False, True]
	reward = 0
	cnt = 0
	run = True

	BATCH_SIZE = 32
	REPLAY_MEMORY_SIZE = 10000

	gamma = 0.99 

	# Exploration parameters for epsilon greedy strategy
	explore_start = 0.1            # exploration probability at start
	explore_stop = 0.0001            # minimum exploration probability 
	decay_rate = 0.0001            # exponential decay rate for exploration prob

	#normalization parameters
	#max_y = 

	while run == True:

		b_flap = False
		moved = False

		cnt +=1 

		DISPLAY.blit(BACKGROUND,(0,0))

		t = pygame.sprite.spritecollideany(bird,pipeGroup)

		if t!=None or (bird.y== 512 - bird.height) or (bird.y == 0):
			#print("GAME OVER")
			#print("FINAL SCORE IS %d"%SCORE)

			last_el = Mem[-1]

			#Mem[-1] = [last_el[0], last_el[1], last_el[2], torch.tensor([float(0),float(0),float(0),float(0)]), True]
			Mem.append((last_el[0], last_el[1], last_el[2], torch.tensor([float(0),float(0),float(0),float(0)]), True))

			run = False

		if run == True :
			
			#PYGAME EVENTS
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYDOWN and (event.key == K_ESCAPE )):
					pygame.quit()
					sys.exit()
			for event in pygame.event.get():
				if event.type == KEYDOWN and event.key == K_m :
					pause=1

			if pipe1.x < pipe2.x and pipe1.x > bird.x :
					next_pipe = pipe1

			elif pipe2.x > bird.x :
					next_pipe = pipe2

			else : 
				next_pipe = pipe1

			#CURRENT STATE 
			#state = [bird.y, bird.x -next_pipe.x, abs(int((next_pipe.upperY - next_pipe.lowerY)*0.5)) + next_pipe.upperY]

			# state = torch.tensor([[bird.y],[bird.x -next_pipe.x], [next_pipe.upperY], [next_pipe.lowerY]])
			state = torch.tensor([bird.y/512, next_pipe.x/388, next_pipe.upperY/512, next_pipe.lowerY/512])


			#print("state ", state)
			#print("next_pipe.x = ", next_pipe.x)
			#print("next_pipe.y = ", next_pipe.upperY)
			#print("              ", next_pipe.lowerY)


			if cnt % 3 == 0: #action choice (frequency 3 FPS)

				#epsilon greedy 
				exp_exp_tradeoff = np.random.rand()
				#print("exp_exp_tradeoff: ", exp_exp_tradeoff)

				# Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
				explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

				#print("explore probability: ", explore_probability)

				if (explore_probability > exp_exp_tradeoff):
					# Make a random action (exploration)
					b_flap = np.random.randint(2)
					b_flap = action[b_flap]

				else:
					# Get action from Q-network (exploitation)
					# Estimate the Qs values state
					Qs = model.forward(state)
					
					# Take the biggest Q value (= the best action)
					b_flap = action[int(Qs.argmax())] #argmax = 1 -> Flap 

				#print(b_flap)

				decay_step += 1

			reward = cnt #pas de problème de sparse reward en soit


			#move world
			if b_flap == True :
				bird.move("UP")
				moved = True
			

			if moved == False:
				bird.move(None)
			else:
				moved = False

			
			pipe1Pos = pipe1.move()
			if pipe1Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width/2):
				if pipe1.behindBird == 0:
					pipe1.behindBird = 1
					SCORE += 1
					reward += 10
					#print("SCORE IS %d"%SCORE)

			pipe2Pos = pipe2.move()
			if pipe2Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width/2):
				if pipe2.behindBird == 0:
					pipe2.behindBird = 1
					SCORE += 1
					reward += 10
					#print("SCORE IS %d"%SCORE)
			
			# next_state = [bird.y, bird.x -next_pipe.x, abs(int((next_pipe.upperY - next_pipe.lowerY)*0.5)) + next_pipe.upperY]
			# next_state = torch.tensor([[bird.y], [bird.x -next_pipe.x], [next_pipe.upperY], [next_pipe.lowerY]])
			#next_state = torch.tensor([float(bird.y), float(next_pipe.x), float(next_pipe.upperY), float(next_pipe.lowerY)])
			next_state = torch.tensor([float(bird.y/512), float(next_pipe.x/388), float(next_pipe.upperY/512), float(next_pipe.lowerY/512)])

			#print("state: ", state)
			#print("next state: ", next_state)
			#print("x: ", bird.x)
			#print("reward: ", reward)
			
			if pause==0:
				pygame.display.update()

			FPSCLOCK.tick(FPS)

			if cnt % 3 == 0: 
				Mem.append((state, b_flap, reward, next_state, False)) #state : Tensor 
																	   #b_flap : Boolean
																	   #reward : Integer
																	   #next state : Tensor 
																	   #Terminal : Boolean

			if len(Mem) > REPLAY_MEMORY_SIZE : 
				Mem.popleft()

			minibatch = random.sample(Mem, BATCH_SIZE)

			s_batch = [d[0] for d in minibatch] #state batch
			f_batch = [d[1] for d in minibatch] #flap batch
			r_batch = [d[2] for d in minibatch] #reward batch 
			n_s_batch = [d[3] for d in minibatch] #next state batch
			t_batch = [d[4] for d in minibatch] #terminal bool batch

			#y_batch = []
			y_batch = torch.zeros(BATCH_SIZE,2)

			#y_target_batch = []
			y_target_batch = torch.zeros(BATCH_SIZE,2)

			for i in range(0, len(minibatch)):

				Qs = model.forward(s_batch[i])

				y_batch[i] = Qs

				next_Qs = model.forward(n_s_batch[i])

				if t_batch[i]:

					#y_target_batch.append(torch.tensor([r_batch[i],r_batch[i]])
					y_target_batch[i][0] = r_batch[i]
					y_target_batch[i][1] = r_batch[i]

				else :

					max_ind = int(next_Qs.argmax())
					min_ind = int(next_Qs.argmin())

					target_Qs = torch.zeros(2)

					new_Qs_max = float(r_batch[i] + gamma*float(max(next_Qs)))
					new_Qs_min = float(r_batch[i] + gamma*float(min(next_Qs)))

					target_Qs[min_ind] = new_Qs_min
					target_Qs[max_ind] = new_Qs_max

					#y_target_batch.append(target_Qs)
					y_target_batch[i][0] = target_Qs[0]
					y_target_batch[i][1] = target_Qs[1]

				#print("size of y_batch: ", y_batch.size())
				#print("size of y_target_batch ", y_target_batch.size())

			#optimize 
			optimizer.zero_grad()
			loss = loss_fn(y_batch, y_target_batch)
			loss.backward()
			optimizer.step()

	return(model, decay_step, optimizer, cnt, Mem)





class DQNetwork(nn.Module):

	def __init__(self):
		super(DQNetwork, self).__init__()

		self.state_size = 4
		self.action_size = 2

		self.fc1_dims = 128
		self.fc2_dims = 64

		self.fc1 = nn.Linear(self.state_size,self.fc1_dims) #state_size = 4
		#self.relu1 = F.relu()

		self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
		#self.relu2 = F.relu()

		self.fc3 = nn.Linear(self.fc2_dims,self.action_size) #action_size = 2 (flap / don't flap)

	def forward(self, x):

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		out = self.fc3(x)

		return(out)




if __name__ == "__main__" : 

	training = True
	n_game = 0

	best_result = 0

	state_size = 4
	action_size = 2

	if args.lr :
		learning_rate = args.lr

	else :
		learning_rate =  0.001

	print("learning rate = ", learning_rate)

	REPLAY_MEMORY_SIZE = 15000
	#BATCH_SIZE = 32

	n_model = 0

	model = DQNetwork()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
	loss_fn = nn.MSELoss()
	decay_step = 0

	# target_model = DQNetwork()

	# print ("init Q nets")

	# for target_param, param in zip(model.parameters(), target_model.parameters()):
	# 	target_param.data.copy_(param)

	# print("copy target param")

	max_game = 10000

	#Memory 
	Mem = deque()

	log_file = open("log/logfile.txt","w") 
	log_model_file = open("log/log_model_file.txt","w")

	Mem = observe(Mem)

	#print("Let's see what's in the Replay Buffer ", Mem)

	print("Done with observation ...")
	print("... start training")

	while training :

		model, decay_step, optimizer, result, Mem = train(model, optimizer, loss_fn, decay_step, Mem)

		if result > best_result: #update best result for log
			best_result = result

		n_game += 1
		#print("agent is playing its ", n_game, "th game")

		if n_game == max_game:
			training = False 

		log_file.write(str(result))
		log_file.write("\n")

		if result > 250: 
			print("saving model number: ", n_model)
			name_model = "./models/model" + str(n_model) + ".ckpt"
			torch.save(model.state_dict(), name_model)

			log_model_file.write(str(result))
			log_model_file.write(" ")
			log_model_file.write(str(n_model))
			log_model_file.write("\n")

			n_model += 1

	print("best result is ", best_result)

	log_file.close()
	log_model_file.close()

	torch.save(model.state_dict(), "./models/latest_model.ckpt")


##notes 

##il est probable qu'il soit necessaire de fine tuner quelques hypers parametres
##resultat grid search learning rate :





