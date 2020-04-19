import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

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
            batch_states.append(np.array(state,copy=False))
            batch_next_states.append(np.array(next_state,copy=False))
            batch_actions.append(np.array(action,copy=False))
            batch_rewards.append(np.array(reward,copy=False))
            batch_dones.append(np.array(done,copy=False))


        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards), np.array(batch_dones)



class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        

        self.layer_1 = nn.Conv2d(1,16,kernel_size=(5,5)) #24
        self.layer_2 = nn.Conv2d(16,24,kernel_size=(5,5)) #20

        self.layer_3 = nn.MaxPool2d(2,2) #10
        self.layer_4 = nn.Conv2d(24,10,kernel_size=(1,1))

        self.layer_5 = nn.Conv2d(10,32,kernel_size=(5,5)) #6

        self.layer_6 =nn.Conv2d(32,10,kernel_size=(6,6))
        self.layer_6u =nn.Linear(10,300)
        self.layer_7 = nn.Linear(1000,300)
        self.layer_8 = nn.Linear(300,action_dim)




        self.max_action = max_action


    def forward(self, x):
        
        x = F.relu(self.layer_1(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_2(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_3(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_4(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        # x = F.relu(self.layer_7(x))
        # x = F.relu(self.layer_8(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = x.flatten()
        # print('AFTER shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_7(x))

        x = self.max_action * torch.tanh(self.layer_8(x))

        return x  
    
    def forward1(self, x):
        
        x = F.relu(self.layer_1(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_2(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_3(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_4(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        # x = F.relu(self.layer_7(x))
        # x = F.relu(self.layer_8(x))
        # print('BEFORE shape{} and dim{}'.format(x.shape,x.dim()))
        x = x.flatten()
        # print('AFTER shape{} and dim{}'.format(x.shape,x.dim()))
        x = F.relu(self.layer_6u(x))

        x = self.max_action * torch.tanh(self.layer_8(x))

        return x


class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		# Defining the first Critic neural network
		# self.layer_1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3)) #26
		# self.layer_2 = nn.Conv2d(16,32,kernel_size=(3,3)) #24
		# self.layer_3 = nn.Conv2d(32,64,kernel_size=(3,3)) #22

		# self.layer_4 = nn.MaxPool2d(2,2)  # 11
		# self.layer_5 = nn.Conv2d(64,8,kernel_size=(1,1)) # 11

		# self.layer_6 = nn.Conv2d(8,16,kernel_size=(3,3)) # 9
		# self.layer_7 = nn.Conv2d(16,32,kernel_size=(3,3)) # 7
		# self.layer_8 = nn.Conv2d(32,10,kernel_size=(7,7))

		# self.layer_9 = nn.Linear(1100,300)
		# self.layer_10 = nn.Linear(300,action_dim)

		self.layer_1 = nn.Conv2d(1,16,kernel_size=(5,5)) #24
		self.layer_2 = nn.Conv2d(16,24,kernel_size=(5,5)) #20

		self.layer_3 = nn.MaxPool2d(2,2) #10
		self.layer_4 = nn.Conv2d(24,10,kernel_size=(1,1))

		self.layer_5 = nn.Conv2d(10,16,kernel_size=(5,5)) #6

		self.layer_6 =nn.Conv2d(16,10,kernel_size=(6,6))

		self.layer_7 = nn.Linear(1100,300)
		self.layer_8 = nn.Linear(300,action_dim)

		# self.layer_9_q = nn.Linear(1001,300)
		self.layer_7_q = nn.Linear(1001,300)
		# self.layer_10 = nn.Linear(300,action_dim)

		# Defining the second Critic neural network
		# self.layer_11 = nn.Conv2d(1,16,kernel_size=(3,3)) #26
		# self.layer_12 = nn.Conv2d(16,32,kernel_size=(3,3)) #24
		# self.layer_13 = nn.Conv2d(32,64,kernel_size=(3,3)) #22

		# self.layer_14 = nn.MaxPool2d(2,2)  # 11
		# self.layer_15 = nn.Conv2d(64,8,kernel_size=(1,1)) # 11

		# self.layer_16 = nn.Conv2d(8,16,kernel_size=(3,3)) # 9
		# self.layer_17 = nn.Conv2d(16,32,kernel_size=(3,3)) # 7
		# self.layer_18 = nn.Conv2d(32,10,kernel_size=(7,7))

		# self.layer_19 = nn.Linear(1100,300)
		# self.layer_20 = nn.Linear(300,action_dim)

		self.layer_9 = nn.Conv2d(1,16,kernel_size=(5,5)) #24
		self.layer_10 = nn.Conv2d(16,24,kernel_size=(5,5)) #20

		self.layer_11 = nn.MaxPool2d(2,2) #10
		self.layer_12 = nn.Conv2d(24,10,kernel_size=(1,1))

		self.layer_13 = nn.Conv2d(10,16,kernel_size=(5,5)) #6

		self.layer_14 =nn.Conv2d(16,10,kernel_size=(6,6))

		self.layer_15 = nn.Linear(1100,300)
		self.layer_16 = nn.Linear(300,action_dim)



	def forward(self, x, u):
		
		# Forward-Propagation on the first Critic Neural Network
		x1 = F.relu(self.layer_1(x))
		x1 = F.relu(self.layer_2(x1))
		x1 = F.relu(self.layer_3(x1))
		x1 = F.relu(self.layer_4(x1))
		x1 = F.relu(self.layer_5(x1))
		x1 = F.relu(self.layer_6(x1))
		# x1 = F.relu(self.layer_7(x1))
		# x1 = F.relu(self.layer_8(x1))
		
		x1 = x1.flatten()
		# print('Critic x shape{} and dim{}'.format(x1.shape,x1.dim()))
		# print('Critic u shape{} and dim{}'.format(u.shape,u.dim()))
		xu_1 = torch.cat([x1, u], 0)
		# print('Critic xu shape{} and dim{}'.format(xu_1.shape,xu_1.dim()))
		xu_1 = F.relu(self.layer_7(xu_1))
		xu_1 = self.layer_8(xu_1)
		# Forward-Propagation on the second Critic Neural Network
		x2 = F.relu(self.layer_9(x))
		x2 = F.relu(self.layer_10(x2))
		x2 = F.relu(self.layer_11(x2))
		x2 = F.relu(self.layer_12(x2))
		x2 = F.relu(self.layer_13(x2))
		x2 = F.relu(self.layer_14(x2))
		# x2 = F.relu(self.layer_7(x2))
		# x2 = F.relu(self.layer_8(x2))

		x2 = x2.flatten()

		xu_2 = torch.cat([x2,u], 0)

		xu_2 = F.relu(self.layer_15(xu_2))
		xu_2 = self.layer_16(xu_2)

		return xu_1, xu_2

	def Q1(self, x, u):
		# x1 = F.relu(self.layer_1(x))
		# x1 = F.relu(self.layer_2(x1))
		# x1 = F.relu(self.layer_3(x1))
		# x1 = F.relu(self.layer_4(x1))
		# x1 = F.relu(self.layer_5(x1))
		# x1 = F.relu(self.layer_6(x1))
		# x1 = F.relu(self.layer_7(x1))
		# x1 = F.relu(self.layer_8(x1))
		
		# x1 = x1.flatten()

		# xu_1 = torch.cat([x1, u], 0)

		# xu_1 = F.relu(self.layer_9(xu_1))
		# xu_1 = self.layer_10(xu_1)

		# x1 = F.relu(self.layer_1(x))
		# x1 = F.relu(self.layer_2(x1))
		# x1 = F.relu(self.layer_3(x1))
		# x1 = F.relu(self.layer_4(x1))
		# x1 = F.relu(self.layer_5(x1))
		# x1 = F.relu(self.layer_6(x1))
		# x1 = F.relu(self.layer_7(x1))
		# x1 = F.relu(self.layer_8(x1))
		
		# x1 = x1.flatten()
		# print('Critic x shape{} and dim{}'.format(x1.shape,x1.dim()))
		# print('Critic u shape{} and dim{}'.format(u.shape,u.dim()))
		# xu_1 = torch.cat([x1, u], 0)
		# print('Critic xu shape{} and dim{}'.format(xu_1.shape,xu_1.dim()))
		# xu_1 = F.relu(self.layer_9_q(xu_1))
		# xu_1 = self.layer_10(xu_1)

		x1 = F.relu(self.layer_1(x))
		x1 = F.relu(self.layer_2(x1))
		x1 = F.relu(self.layer_3(x1))
		x1 = F.relu(self.layer_4(x1))
		x1 = F.relu(self.layer_5(x1))
		x1 = F.relu(self.layer_6(x1))

		x1 = x1.flatten()

		xu_1 = torch.cat([x1, u], 0)

		xu_1 = F.relu(self.layer_7_q(xu_1))
		xu_1 = self.layer_8(xu_1)

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

	def select_action(self, state):
		state = torch.Tensor(state.reshape(-1,1,28,28)).to(device)
		return self.actor.forward1(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
	
		for it in range(iterations):
		  
			# Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
			batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
			state = torch.Tensor(batch_states).to(device)
			next_state = torch.Tensor(batch_next_states).to(device)
			action = torch.Tensor(batch_actions.astype(np.float32)).to(device)
			reward = torch.Tensor(batch_rewards).to(device)
			done = torch.Tensor(batch_dones).to(device)
			
			# print("next state,dimension",next_state,next_state.dim(),next_state.shape)
			# print("state",state)
			# print('reward',reward)

			# print("---"*100)
			  # Step 5: From the next state s’, the Actor target plays the next action a’

			next_state =  next_state.view(-1,1,28,28)
			next_action = self.actor_target(next_state)
			 
			  # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
			noise = torch.Tensor(batch_actions.astype(np.float32)).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
			
			# next_action = next_action.view(-1,1)
			# Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			  
			  # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
			target_Q = torch.min(target_Q1, target_Q2)
			  
			  # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
			target_Q = reward + ((1 - done) * discount * target_Q).detach()
			  
			  # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
			
			critic_state = state.view(-1,1,28,28)
			# action = action.view(-1,1)
			current_Q1, current_Q2 = self.critic(critic_state, action)
			  
			  # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			  
			  # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			  
			  # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
			if it % policy_freq == 0:
				actor_state = state.view(-1,1,28,28) 
				actor_loss = -self.critic.Q1(actor_state, self.actor(actor_state)).mean()
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()
				
				# Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):

					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
				
				# Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):

					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)