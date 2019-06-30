from itertools import cycle
from numpy.random import randint,choice

import torch 
import torch.nn as nn

import sys

import numpy as np


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



def train(model):

	

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

	gamma = 0.95  

	# Exploration parameters for epsilon greedy strategy
	explore_start = 1.0            # exploration probability at start
	explore_stop = 0.01            # minimum exploration probability 
	decay_rate = 0.0001            # exponential decay rate for exploration prob

	#Memory 
	Mem = []

	while run == True:

		cnt +=1 

		DISPLAY.blit(BACKGROUND,(0,0))

		t = pygame.sprite.spritecollideany(bird,pipeGroup)

		if t!=None or (bird.y== 512 - bird.height) or (bird.y == 0):
			print("GAME OVER")
			print("FINAL SCORE IS %d"%SCORE)

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

			#COMPUTE NEXT WHICH PIPE IS THE NEXT ONE
			if pipe1.x < pipe2.x and pipe1.x > bird.x :
					next_pipe = pipe1

			elif pipe2.x > bird.x :
					next_pipe = pipe2

			else : 
				next_pipe = pipe1


			# state = torch.tensor([[bird.y],[bird.x -next_pipe.x], [next_pipe.upperY], [next_pipe.lowerY]])
			state = torch.tensor([bird.y,bird.x -next_pipe.x,next_pipe.upperY,next_pipe.lowerY])

			#CHOOSE ACTION
			exp_exp_tradeoff = np.random.rand()


			#Forward NN
			output = model.forward(state)
			
			if output > 0.5 :
				b_flap = True
			else :
				b_flap = False

			#print(b_flap)

			fitness = cnt


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
					#print("SCORE IS %d"%SCORE)

			pipe2Pos = pipe2.move()
			if pipe2Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width/2):
				if pipe2.behindBird == 0:
					pipe2.behindBird = 1
					SCORE += 1
					#print("SCORE IS %d"%SCORE)

			
			if pause==0:
				pygame.display.update()


			FPSCLOCK.tick(FPS)


	return(fitness)


class DQNetwork(nn.Module):

	def __init__(self):
		super(DQNetwork, self).__init__()

		self.state_size = 4
		self.action_size = 1


		self.fc1 = nn.Linear(self.state_size,9) #state_size = 4
		torch.nn.init.xavier_uniform_(self.fc1.weight)
		self.Sig1 = nn.Sigmoid()

		self.fc2 = nn.Linear(9,6)
		torch.nn.init.xavier_uniform_(self.fc2.weight)
		self.Sig2 = nn.Sigmoid()

		self.fc3 = nn.Linear(6,self.action_size) #action_size = 2 (flap / don't flap)
		torch.nn.init.xavier_uniform_(self.fc3.weight)
		self.Sig3 = nn.Sigmoid()

	def forward(self, x):

		x = self.Sig1(self.fc1(x))
		x = self.Sig2(self.fc2(x))
		out = self.Sig3(self.fc3(x))

		return(out)

class DQNetwork_custom(nn.Module):

	def __init__(self):
		super(DQNetwork, self,weights1,weights2,weights3).__init__()

		self.weights1 = weights1
		self.weights2 = weights2
		self.weights3 = weights3


		self.fc1 = nn.Linear(self.weights1) #state_size = 4		
		self.Sig1 = nn.Sigmoid()

		self.fc2 = nn.Linear(self.weights2)
		self.Sig2 = nn.Sigmoid()

		self.fc3 = nn.Linear(self.weights3) #action_size = 2 (flap / don't flap)
		self.Sig3 = nn.Sigmoid()

	def forward(self, x):

		x = self.Sig1(self.fc1(x))
		x = self.Sig2(self.fc2(x))
		out = self.Sig3(self.fc3(x))

		return(out)


def model_crossover(model1,model2):

	weights1 = model1.fc2.weight
	weights2 = model2.fc2.weight

	model1.fc2.weight = weights2
	model2.fc2.weight = weights1

	return(model1,model2)


def model_mutation(model):

	#mutate fc1
	weights1 = model.fc1.weight

	for i in range (len(weights1)):

		for j in range (len(weights1[i])):

			if np.random.uniform(0,1) > 0.85:

				change = np.random.uniform(-0.5,0.5)
				weights1[i][j] += change

	#model.fc1.weight = weights1

	#mutate fc2
	weights2 = model.fc2.weight 

	for i in range (len(weights2)):

		for j in range (len(weights2[i])):

			if np.random.uniform(0,1) > 0.85:

				change = np.random.uniform(-0.5,0.5)
				weights2[i][j] += change

	#model.fc2.weight = weights

	#mutate fc3
	weights3 = model.fc3.weight 

	for i in range (len(weights3)):

		for j in range (len(weights3[i])):

			if np.random.uniform(0,1) > 0.85:

				change = np.random.uniform(-0.5,0.5)
				weights3[i][j] += change

	#model.fc3.weight = weights

	new_model = DQNetwork_custom(weights1,weights2,weights3)

	# new_model = nn.Sequential(nn.Linear(weights1),
 #                     nn.Sigmoid(),
 #                     nn.Linear(weights2),
 #                     nn.Sigmoid(),
 #                     nn.Linear(weights3),
 #                     nn.Sigmoid())

	return(new_model)

def model_selection(container, L_fit):

	fit_sum = sum(L_fit)
	L_p = [fit/fit_sum for fit in L_fit]
	L_q = []
	cnt = 0

	for p in L_p :

		cnt = cnt + p
		L_q.append(cnt)

	r_nb = np.random.rand()

	indx = 0
	while L_q[indx] < r_nb:
		indx +=1

	return(container[indx])

def cont_evaluation(container):

	L_fit =[]

	for model in container :
		fitness = train(model)
		L_fit.append(fitness)

	return(L_fit)

def init_container(n_models):

	Ctainer = []

	for i in range(n_models):
		model = DQNetwork()
		Ctainer.append(model)

	return(Ctainer)



if __name__ == '__main__':

	container = init_container(5)

	L_fit = cont_evaluation(container)

	print(max(L_fit))


	parent1 = model_selection(container,L_fit)

	print(parent1)

	offspring1 = model_mutation(parent1)


# if __name__ == "__main__" : 

# 	training = True
# 	generation = 0
# 	n_models = 10

# 	container = init_container(n_models)


# 	log_file = open("log/logfileNEAT.txt","w") 

# 	while training :

# 		L_fit = cont_evaluation(container)

# 		new_container = []

# 		for i in range(int(n_models*0.5)):
# 			parent1 = model_selection(container,L_fit)
# 			parent2 = model_selection(container,L_fit)

# 			parent1,parent2 = model_crossover(parent1,parent2)

# 			offspring1 = model_mutation(parent1)
# 			offspring2 = model_mutation(parent2)

# 			new_container.append(offspring1)
# 			new_container.append(offspring2)

# 		container = new_container

# 		generation += 1
# 		print(generation, "th generation")

# 		if generation == 100 :
# 			training = False 

# 		bestfit = max(L_fit)

# 		log_file.write(str(generation))
# 		log_file.write(" ")
# 		log_file.write(str(bestfit))
# 		log_file.write("\n")

# 	log_file.close()
# 	torch.save(model.state_dict(), "./models/modelNEAT.ckpt")