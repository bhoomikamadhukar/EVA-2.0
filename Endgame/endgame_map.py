
# Importing the libraries
import numpy as np
from random import random, randint,randrange
import random
import datetime
import matplotlib.pyplot as plt
import time
import cv2
import os
import imutils
import torch
import torch.nn as nn
# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.properties import BoundedNumericProperty

# Importing the Dqn object from our AI in ai.py
from endgame_ai import TD3, ReplayBuffer

# Adding this line if we don't want the right click to put a red point
Config.set('graphics', 'multisamples', '0')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

#state_dim = (28,28,1)
state_dim = (1600)

action_dim = 1
max_action = 10

replay_buffer = ReplayBuffer()
policy = TD3(state_dim, action_dim, max_action)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
out_file = open("Target.txt", "a")

first_update = True

timestep = 0
max_steps = 1000
episode_timestep = 0
total_timestep = 10
episode_reward = 0 
episode_no = 0
eval_episodes = 10
done = True
episode_step=0
random_positions = {1:[137,385],2:[364,346],3:[582,310],4:[782,292],5:[1081,236],6:[338,179],7:[584,129],8:[671,442],9:[1104,351],10:[172,545],11:[245,101],12:[654,248],13:[804,181],14:[1146,157],15:[807,420],16:[712,440]}  #nk # nk17
car_prev_x = 597
car_prev_y = 71
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    # print('sand values',sand)
    # print('sand shape',sand.shape)
    goal_x = 1090
    goal_y = 283
    first_update = False
    global swap
    swap = 0
    global car_prev_x 
    global car_prev_y 

last_distance = 0

# Creating the car class

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

class Car(Widget):
    
    #angle = NumericProperty(0)
    #rotation = NumericProperty(0)
    angle = BoundedNumericProperty(0.0, min=- 90.0, max=90.0,errorhandler=lambda x: 90.0 if x > 90.0 else 0.0)
    rotation = BoundedNumericProperty(0.0, min=- 10, max=10.0,errorhandler=lambda x: 10.0 if x > 10.0 else 0.0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    

    def move(self, rotation):
        #print("moving by",rotation)             - nk 27th Apr
        #print()
        self.pos = Vector(*self.velocity) + self.pos
        print('position is',self.pos,'type is',type(self.pos))
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        print('the updated angle is', self.angle )              #- nk 27th Apr
        # print(self.pos)



class Game(Widget):

    car = ObjectProperty(None)
    f= open("detail.txt","w+")
    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(2, 0)

    def evaluate_policy(self,policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = 0
            done = False
            while not done:
                print("===================================")
                action = select_action(np.array(obs))
                obs, reward, done, _ = policy.take_step(action)
                avg_reward += reward
            avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

    def reset(self):

        reset_position = random_positions[randint(1,16)]
        self.f.write(str(reset_position))
        #reset_position = [car_prev_x,car_prev_y]
        self.car.x = reset_position[0] #arbitary position
        self.car.y = reset_position[1]
        
        self.car.angle = 0.0
        print('the angle in reset is',self.car.angle)
        print("Car position in reset {} and {}".format(self.car.x,self.car.y))
        img = sand[int(self.car.x)-30:int(self.car.x)+30, int(self.car.y)-30 : int(self.car.y)+30]
        
        height = int(img.shape[0] *(2/3) )
        dimension = (width, height)
        rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
        return rescaled
        #return sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]

    def get_state(self):
        global car_prev_x
        global car_prev_y
        global done
        #send_state = sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        #if send_state.shape == (28, 28):
        if int(self.car.x) > 29 and int(self.car.x) < 1400 and int(self.car.y) > 29 and int(self.car.y) < 631:  
            #print('gone from here up')          - nk 27th Apr
            img = sand[int(self.car.x)-30:int(self.car.x)+30, int(self.car.y)-30 : int(self.car.y)+30]
           
            height = int(img.shape[0] *(2/3) )
            dimension = (width, height)
            rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
            return rescaled
            #return sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        else:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            #print('gone from here up else ===================' ,reset_position)         - nk 27th Apr
            img = sand[int(self.car.x)-30:int(self.car.x)+30, int(self.car.y)-30 : int(self.car.y)+30]
            
            width = int(img.shape[1] *(2/3) )
            height = int(img.shape[0] *(2/3) )
            dimension = (width, height)
            rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
            #done = True 
            return rescaled
            
    def take_step(self,action,last_distance,episode_step):
        global car_prev_x
        global car_prev_y
        global goal_x
        global goal_y
        global swap
        global done
        reward = 0
        #out_file = open("Target.txt", "a")
        
        rotation = action
        #print("=="*100)             - nk 27th Apr
        self.car.move(rotation)
        img = sand[int(self.car.x)-30:int(self.car.x)+30, int(self.car.y)-30 : int(self.car.y)+30]
        
        if int(self.car.x) > 29 and int(self.car.x) < 1400 and int(self.car.y) > 29 and int(self.car.y) < 631:
            width = int(img.shape[1] *(2/3))
            height = int(img.shape[0] *(2/3) )
            dimension = (width, height)
            rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
            new_state = rescaled
        else:
            new_state = img
            print("stuck near border{}".format(new_state))
            reward += -10.0
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
        if distance < 25:
            if swap == 0:
                goal_x = 259
                goal_y = 372
                swap = 1
            else:
                goal_x = 1090
                goal_y = 283
                swap = 0
            out_file.write("Target Achieved \n") 
        #print("Car position in take_step {} and {}".format(self.car.x,self.car.y))          - nk 27th Apr
        if int(self.car.x) > 0 and int(self.car.x) < 1429 and int(self.car.y) > 0 and int(self.car.y) < 660:
            if sand[int(self.car.x),int(self.car.y)] > 0:
                print('Car in sand')
            
            else:
                print("Car on Road")
                car_prev_x = int(self.car.x)
                car_prev_y = int(self.car.y)
                print('x value {}'.format(car_prev_x))
                print('y value {}'.format(car_prev_y))
            
        
            episode_step += 1
            #print('episode step================',episode_step)              - nk 27th Apr
            if int(sand[int(self.car.x),int(self.car.y)]) > 0: 
                reward += -5.0
                
                self.car.velocity = Vector(1,0).rotate(self.car.angle)
                done = True  #nk
            else :
                reward += -0.2
                self.car.velocity = Vector(3,0).rotate(self.car.angle)
                done = False
        #print('just above done condition')              - nk 27th Apr
        if episode_step == 200:  #nk #nk16th again
            # episode_step = 0
            print('greater than 2000')                  
            done = True

        if distance > last_distance:
            print("================== Moving Away from Goal")      
            reward += -5.0
        else:
            print("================== Moving Towards Goal") 
            reward += 2

        if self.car.x < 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            reward += -3
            #print('coming x10')             - nk 27th Apr
            #self.reset()
            done = True
        if self.car.x > self.width - 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            #self.car.x = 580
            #self.car.y = 310
            reward += -3
            #print('coming x-10')                - nk 27th Apr
            done = True
        if self.car.y < 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            #self.car.x = 580
            #self.car.y = 310
            reward += -3
            #print('coming y10')             - nk 27th Apr
            done = True
        if self.car.y > self.height - 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            #self.car.x = 580
            #self.car.y = 310
            reward += -3
            #print('coming y-10')                - nk 27th Apr
            done = True
        last_distance = distance

        return new_state,reward,done,episode_step


    def update(self,dt):

        global policy
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global episode_timestep
        global total_timestep
        global replay_buffer
        global episode_reward 
        global done
        global discount
        global tau
        global policy_noise
        global noise_clip
        global policy_freq
        global timestep
        global episode_step
        global episode_no
        global eval_episodes
       #global self.angle
        temp_state = np.ones((1,1600))
        longueur = self.width
        largeur = self.height

        if first_update:
            init()
        #xx = goal_x - self.car.x
        #yy = goal_y - self.car.y
        #orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        current_state = self.get_state()
        #last_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        file_name = "%s_%s_%s" % ("TD3", 'self_drive', episode_no +1)
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("---------------------------------------")
        #print('timestep',timestep)          - nk 27th Apr
        
        if done:
            
            episode_no += 1
            
            #b = datetime.datetime.now()
            
            print('-------------------EPISODE DONE-------------------')
            if timestep!=0:
                a = datetime.datetime.now()
                print('total episode:{},episode_timestep:{},episode reward:{},timestep:{}'.format(episode_no,episode_timestep,episode_reward,timestep))
                #image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
                #print(image)
                policy.train(replay_buffer,episode_timestep,batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                b = datetime.datetime.now()
                print(' The Time taken to train',timestep, 'timesteps is =',b-a)
            # EVALUATION CONDITION AND CODE
            # print('angle here is', self.car.angle)             - nk 27th Apr
            current_state = self.reset()
            done = False
            episode_reward = 0
            episode_timestep = 0
            episode_step = 0
        file_name = 'TD3' + '_' + str(episode_no)    
        if episode_no % 100 == 0:
        #if episode_no >= 1:        # For Testing removed - nk 27th Apr
           policy.save(file_name, directory="./pytorch_models")
        #     avg_reward = self.evaluate_policy(policy)
        #     np.save("./results/%s" % (file_name))
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        curr_state_orientation = [orientation, -orientation]
        last_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        #current_state = self.get_state()
        # image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar_28/endgame_nihar_28/endgame_nihar/endgame/images/mask_car.png')
        image = cv2.imread('./images/mask_car.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        overlay = imutils.rotate(image,self.car.angle)
        rows,cols = overlay.shape
        #overlay=cv2.addWeighted(img[30:30+rows, 30:30+cols],0.5,overlay,0.5,0)
        current_state[0:0+rows, 20:20+cols ] = overlay

        current_state = list(current_state.ravel())
        current_state .append(curr_state_orientation[0])
        current_state .append(curr_state_orientation[1])
        if timestep < 10000 : 
            action = max_action*random.uniform(-1, 1)
        else:
            action = policy.select_action(np.array(current_state))
        
        new_state,reward,done,episode_step = self.take_step(action,last_distance,episode_step)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        next_state_orientation = [orientation, -orientation]
        #if episode_step == 500:
        #    episode_step = 0

        episode_reward += reward
        
            #print('=========',rows,cols)
            new_state[0:0+rows, 20:20+cols ] = overlay
        new_state = new_state.ravel()
        
        if len(current_state) != 1600:
            
            current_state = np.ones(1600).ravel()
            
        if len(new_state) != 1600:
            
            new_state = np.ones(1600).ravel()
        
        current_state=list(current_state)
        current_state .append(curr_state_orientation[0])
        current_state .append(curr_state_orientation[1])
        #new_state[0:0+rows, 20:20+cols ] = overlay
        new_state = list(new_state)
        new_state .append(next_state_orientation[0])
        new_state .append(next_state_orientation[1])
       
        replay_buffer.add((current_state,new_state,action,reward,done))

        current_state = new_state
        episode_timestep += 1
        timestep += 1

        # Making a save method to save a trained model
    




        


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        # self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        # parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        # parent.update(1.0/60.0)
        return parent

    def clear_canvas(self, obj):
        global sand

        # # self.painter.canvas.clear()
        # sand = np.zeros((longueur,largeur))

        

    def save(self, obj):
        print("saving brain...")
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        #plt.plot(scores)
        #plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()

