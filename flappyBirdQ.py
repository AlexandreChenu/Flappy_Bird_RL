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



def train(model, optimizer, loss_fn, decay_step):

	

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

			last_el = Mem[-1]

			Mem[-1] = [last_el[0], last_el[1], last_el[2], torch.tensor([float(0),float(0),float(0),float(0)]), True]

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
			state = torch.tensor([bird.y,bird.x -next_pipe.x,next_pipe.upperY,next_pipe.lowerY])

			#CHOOSE ACTION
			exp_exp_tradeoff = np.random.rand()

			# Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
			explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

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

			# if decay_step % 5 == 0:
			# 	reward = 1

			reward = cnt


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
					reward += 50
					#print("SCORE IS %d"%SCORE)

			pipe2Pos = pipe2.move()
			if pipe2Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width/2):
				if pipe2.behindBird == 0:
					pipe2.behindBird = 1
					SCORE += 1
					reward += 50
					#print("SCORE IS %d"%SCORE)
			
			# next_state = [bird.y, bird.x -next_pipe.x, abs(int((next_pipe.upperY - next_pipe.lowerY)*0.5)) + next_pipe.upperY]
			# next_state = torch.tensor([[bird.y], [bird.x -next_pipe.x], [next_pipe.upperY], [next_pipe.lowerY]])
			next_state = torch.tensor([float(bird.y), float(bird.x -next_pipe.x), float(next_pipe.upperY), float(next_pipe.lowerY)])
			
			if pause==0:
				pygame.display.update()

			Mem.append([state, b_flap, reward, next_state, False]) #state : Tensor 
																   #b_flap : Boolean
																   #reward : Integer
																   #next state : Tensor 
																   #Terminal : Boolean

			FPSCLOCK.tick(FPS)



	#LEARNING PART 
	for el in Mem : 

		[s,f,r,n_s,t] = el #s : state 
						   #f : flap
						   #r : reward
						   #n_s : new state
						   #t : terminal state 

		# print("state : ", s)
		# print("next_state : ", n_s)

		Qs = model.forward(s)
		next_Qs = model.forward(n_s)

		# print("Qs = ", Qs)
		# print("next_Qs = ", next_Qs)  

		if t :
			target_Qs = torch.tensor([float(r),float(r)])

		else :
			max_ind = int(next_Qs.argmax())
			min_ind = int(next_Qs.argmin())

			target_Qs = torch.zeros(2)

			new_Qs_max = float(r + gamma*float(max(next_Qs)))
			new_Qs_min = float(r + gamma*float(min(next_Qs)))

			target_Qs[min_ind] = new_Qs_min
			target_Qs[max_ind] = new_Qs_max

		#optimize 
		optimizer.zero_grad()
		loss = loss_fn(Qs, target_Qs)
		loss.backward()
		optimizer.step()

	return(model, decay_step, optimizer,cnt)


class DQNetwork(nn.Module):

	def __init__(self):
		super(DQNetwork, self).__init__()

		self.state_size = 4
		self.action_size = 2


		self.fc1 = nn.Linear(self.state_size,9) #state_size = 4
		self.Sig1 = nn.Sigmoid()

		self.fc2 = nn.Linear(9,6)
		self.Sig2 = nn.Sigmoid()

		self.fc3 = nn.Linear(6,self.action_size) #action_size = 2 (flap / don't flap)

	def forward(self, x):

		x = self.Sig1(self.fc1(x))
		x = self.Sig2(self.fc2(x))
		out = self.fc3(x)

		return(out)




if __name__ == "__main__" : 

	training = True
	cnt = 0

	state_size = 3
	action_size = 2

	learning_rate =  0.002

	model = DQNetwork()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
	loss_fn = nn.MSELoss()
	decay_step = 0


	log_file = open("log/logfile.txt","w") 

	while training :

		model, decay_step, optimizer, result= train(model, optimizer, loss_fn, decay_step)

		cnt += 1
		print(cnt, "th iteration")

		if cnt == 10000 :
			training = False 

		log_file.write(str(cnt))
		log_file.write(" ")
		log_file.write(str(result))
		log_file.write("\n")

	log_file.close()
	torch.save(model.state_dict(), "./models/model0.ckpt")





