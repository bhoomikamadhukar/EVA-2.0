import numpy as np
import random
import os
import torch
import cv2
import imutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from PIL import Image as PILImage
img1 = PILImage.open("./images/mask_car.png").convert('L')
img2 = np.asarray(img1)/255

class ReplayBuffer(object):

    def __init__(self,max_len=1e6):
        self.storage = []
        self.max_size = max_len
        self.ptr = 0
        print('initialized')

    def add(self,transition):

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr+1)%self.max_size
            # print(transition)
        else:

            self.storage.append(transition)
        # print('transition',transition)

        # self.storage.append(transition)
    def sample(self,batch_size):
        
        ind = np.random.randint(0,len(self.storage),size=batch_size)
        batch_states, batch_rewards, batch_actions, batch_next_states, batch_dones = [], [], [], [], []

        for i in ind:

            state, next_state, action, reward, done = self.storage[i]
            # print('shape of state while fecthing is', len(state))     - nk 27th Apr
            batch_states.append(np.array(state,copy=False))
            batch_next_states.append(np.array(next_state,copy=False))
            batch_actions.append(np.array(action,copy=False))
            batch_rewards.append(np.array(reward,copy=False))
            batch_dones.append(np.array(done,copy=False,dtype=np.uint8))


        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards), np.array(batch_dones)



class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Conv2d(1,8,kernel_size=(3,3)) #28
        self.bn_1 = nn.BatchNorm2d(8)
        self.layer_2 = nn.Conv2d(8,16,kernel_size=(3,3)) #28
        self.bn_2 = nn.BatchNorm2d(16)
        self.layer_3 = nn.Conv2d(16,8,kernel_size=(3,3),stride=2) #14
        self.bn_3 = nn.BatchNorm2d(8)
        self.layer_4 = nn.Conv2d(8,16,kernel_size=(3,3)) #14
        self.bn_4 = nn.BatchNorm2d(16)
        self.layer_5 = nn.Conv2d(16,8,kernel_size=(3,3),stride=2) #14
        self.bn_5 = nn.BatchNorm2d(8)
        self.layer_6 = nn.Conv2d(8,16,kernel_size=(3,3)) #7
        self.bn_6 = nn.BatchNorm2d(16)
        self.layer_7 = nn.Conv2d(16,10,kernel_size=(5,5)) #6
        self.layer_8 =nn.Linear(12,40)
        self.layer_8_1 =nn.Linear(1200,40)
        self.layer_9 = nn.Linear(40,30)
        self.layer_10 = nn.Linear(30,action_dim)

        self.max_action = max_action


    def forward(self,x):
        x_orientation = x[1600:1602]
        x = x[0:1600]
        x = torch.Tensor(x.reshape(-1,1,40,40))
        x_orientation = torch.Tensor(x_orientation)
        x = F.relu(self.layer_1(x))
        x = self.bn_1(x)
        x = F.relu(self.layer_2(x))
        x = self.bn_2(x)
        x = F.relu(self.layer_3(x))
        x = self.bn_3(x)
        x = F.relu(self.layer_4(x))
        x = self.bn_4(x)
        x = F.relu(self.layer_5(x))
        x = self.bn_5(x)
        x = F.relu(self.layer_6(x))
        x = self.bn_6(x)
        x = F.relu(self.layer_7(x))
        x = x.flatten()
        x_orientation = x_orientation.flatten()
        x = torch.cat([x,x_orientation], 0)
        x = F.relu(self.layer_8(x))
        x = F.relu(self.layer_9(x))
        x = self.max_action * torch.tanh(self.layer_10(x))

        return x  
    
    def forward1(self, x):
        
        x_orientation = x[:, 1600:1602]
        x = x[:,0:1600]
        x = torch.Tensor(x.reshape(-1,1,40,40))
        x_orientation = torch.Tensor(x_orientation)
        x = F.relu(self.layer_1(x))
        x = self.bn_1(x)
        x = F.relu(self.layer_2(x))
        x = self.bn_2(x)
        x = F.relu(self.layer_3(x))
        x = self.bn_3(x)
        x = F.relu(self.layer_4(x))
        x = self.bn_4(x)
        x = F.relu(self.layer_5(x))
        x = self.bn_5(x)
        x = F.relu(self.layer_6(x))
        x = self.bn_6(x)
        x = F.relu(self.layer_7(x))
        x = x.flatten()
        x_orientation = x_orientation.flatten()
        x = torch.cat([x,x_orientation], 0)
        x = F.relu(self.layer_8_1(x))
        x = F.relu(self.layer_9(x))
        x = self.max_action * torch.tanh(self.layer_10(x))

        return x  


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        #self.image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
        self.layer_1 = nn.Conv2d(1,8,kernel_size=(3,3)) #28
        self.bn_1 = nn.BatchNorm2d(8)
        self.layer_2 = nn.Conv2d(8,16,kernel_size=(3,3)) #28
        self.bn_2 = nn.BatchNorm2d(16)
        self.layer_3 = nn.Conv2d(16,8,kernel_size=(3,3),stride=2) #14
        self.bn_3 = nn.BatchNorm2d(8)
        self.layer_4 = nn.Conv2d(8,16,kernel_size=(3,3)) #14
        self.bn_4 = nn.BatchNorm2d(16)
        self.layer_5 = nn.Conv2d(16,8,kernel_size=(3,3),stride=2) #14
        self.bn_5 = nn.BatchNorm2d(8)
        self.layer_6 = nn.Conv2d(8,16,kernel_size=(3,3)) #7
        self.bn_6 = nn.BatchNorm2d(16)
        self.layer_7 = nn.Conv2d(16,10,kernel_size=(5,5)) #6
        self.layer_8 = nn.Linear(1300,30)
        self.layer_9 = nn.Linear(30,action_dim)
        self.layer_8_1 = nn.Linear(1201,30)

        self.layer_10 = nn.Conv2d(1,8,kernel_size=(3,3)) #28
        self.bn_10 = nn.BatchNorm2d(8)
        self.layer_11 = nn.Conv2d(8,16,kernel_size=(3,3)) #28
        self.bn_11 = nn.BatchNorm2d(16)
        self.layer_12 = nn.Conv2d(16,8,kernel_size=(3,3),stride=2) #14
        self.bn_12 = nn.BatchNorm2d(8)
        self.layer_13 = nn.Conv2d(8,16,kernel_size=(3,3)) #14
        self.bn_13 = nn.BatchNorm2d(16)
        self.layer_14 = nn.Conv2d(16,8,kernel_size=(3,3),stride=2) #14
        self.bn_14 = nn.BatchNorm2d(8)
        self.layer_15 = nn.Conv2d(8,16,kernel_size=(3,3)) #7
        self.bn_15 = nn.BatchNorm2d(16)
        self.layer_16 = nn.Conv2d(16,10,kernel_size=(5,5)) #6
        self.bn_16 = nn.BatchNorm2d(10)
        self.layer_17 = nn.Linear(1300,30)
        self.layer_18 = nn.Linear(30,action_dim)



    def forward(self, x, u):
        x_orientation = x[:, 1600:1602]
        x = x[:,0:1600]
        x = torch.Tensor(x.reshape(-1,1,40,40))
        x_orientation = torch.Tensor(x_orientation)
        x_orientation = x_orientation.flatten()
        x1 = F.relu(self.layer_1(x))
        x1 = self.bn_1(x1)
        x1 = F.relu(self.layer_2(x1))
        x1 = self.bn_2(x1)
        x1 = F.relu(self.layer_3(x1))
        x1 = self.bn_3(x1)
        x1 = F.relu(self.layer_4(x1))
        x1 = self.bn_4(x1)
        x1 = F.relu(self.layer_5(x1))
        x1 = self.bn_5(x1)
        x1 = F.relu(self.layer_6(x1))
        x1 = self.bn_6(x1)
        x1 = F.relu(self.layer_7(x1))
        x1 = x1.flatten()
        xu_1 = torch.cat([x1, u,x_orientation], 0)
        xu_1 = F.relu(self.layer_8(xu_1))
        xu_1 = self.layer_9(xu_1)


        x2 = F.relu(self.layer_10(x))
        x2 = self.bn_10(x2)
        x2 = F.relu(self.layer_11(x2))
        x2 = self.bn_11(x2)
        x2 = F.relu(self.layer_12(x2))
        x2 = self.bn_12(x2)
        x2 = F.relu(self.layer_13(x2))
        x2 = self.bn_13(x2)
        x2 = F.relu(self.layer_14(x2))
        x2 = self.bn_14(x2)
        x2 = F.relu(self.layer_15(x2))
        x2 = self.bn_15(x2)
        x2 = F.relu(self.layer_16(x2))
        x2 = x2.flatten()
        xu_2 = torch.cat([x2,u,x_orientation], 0)
        xu_2 = F.relu(self.layer_17(xu_2))
        xu_2 = self.layer_18(xu_2)

        return xu_1, xu_2

    def Q1(self, x, u):
        x_orientation = x[:, 1600:1602]
        x = x[:,0:1600]
        x = torch.Tensor(x.reshape(-1,1,40,40))
        x_orientation = torch.Tensor(x_orientation)
        x_orientation = x_orientation.flatten()
        
        x1 = F.relu(self.layer_1(x))
        x1 = self.bn_1(x1)
        x1 = F.relu(self.layer_2(x1))
        x1 = self.bn_2(x1)
        x1 = F.relu(self.layer_3(x1))
        x1 = self.bn_3(x1)
        x1 = F.relu(self.layer_4(x1))
        x1 = self.bn_4(x1)
        x1 = F.relu(self.layer_5(x1))
        x1 = self.bn_5(x1)
        x1 = F.relu(self.layer_6(x1))
        x1 = self.bn_6(x1)
        x1 = F.relu(self.layer_7(x1))
        x1 = x1.flatten()
        xu_1 = torch.cat([x1, u,x_orientation], 0)
        xu_1 = F.relu(self.layer_8_1(xu_1))
        xu_1 = self.layer_9(xu_1)
        return xu_1


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class TD3(object):

	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.max_action = max_action
        
	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    
    # Making a load method to load a pre-trained model
	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


	def select_action(self,state):
		#print('shape of state here is', state.shape)       - nk 27th Apr
		#state = torch.Tensor(state.reshape(-1,140,40)).to(device)
		#return self.actor.forward1(state).cpu().data.numpy().flatten()
		print('='*100)
		#image_1 = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
		#print('image value is',image_1)
		return self.actor.forward(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		count = 0
		for it in range(iterations):
			if iterations - it < 20:
				print('Nihar just 20 more episode steps to be trained - Please be ready to take the video on phone: count down started : ',iterations - it )
		  
			# Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
			batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
			#print('batch state is',batch_states.shape)       - nk 27th Apr
			state = torch.Tensor(batch_states).to(device)
			next_state = torch.Tensor(batch_next_states).to(device)
			action = torch.Tensor(batch_actions.astype(np.float32)).to(device)
			reward = torch.Tensor(batch_rewards).to(device)
			done = torch.Tensor(batch_dones).to(device)
			#next_state_orientation = [next_state[1600],next_state[1601]]
			#next_state =  next_state.view(-1,1,40,40)
			next_action = self.actor_target.forward1(next_state)
			 
			  # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
			noise = torch.Tensor(batch_actions.astype(np.float32)).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
			
			# next_action = next_action.view(-1,1)
			# Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			# print('Target Q1',target_Q1)
			# print('Target Q2',target_Q2.size())  
			  # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
			target_Q = torch.min(target_Q1, target_Q2)
			print('Target Q',target_Q.size(),target_Q.type())  
			  # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
			target_Q = reward[count] + ((1 - done[count]) * discount * target_Q).detach()
			print('Target Q',target_Q.size(),target_Q.type())  
			  # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
			
			#critic_state = state.view(-1,1,40,40)
			# action = action.view(-1,1)
			print('Next state size',next_state.size())
			print('Next Action size',next_action.size())
			print('State size',state.size())
			print('Action size',action.size())
			current_Q1, current_Q2 = self.critic(state, action)
			print('Q1 size',current_Q1.size())
			print('Q2 size',current_Q2.size())
			# print('Target Q',target_Q.size(),target_Q.type())
			  
			 # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			  
			 # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			  
			  # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
			if it % policy_freq == 0:
				#actor_state = state.view(-1,1,40,40) 
				actor_loss = -self.critic.Q1(state, self.actor.forward1(state)).mean()
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()
				
				# Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):

					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
				
				# Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):

					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			count = count + 1

			if count == batch_size - 1:
				count = 0