
# Importing the libraries
import numpy as np
from random import random, randint,randrange
import random
import matplotlib.pyplot as plt
import time

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

state_dim = (28,28,1)
action_dim = 1
max_action = 10

replay_buffer = ReplayBuffer()
policy = TD3(state_dim, action_dim, max_action)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

first_update = True

timestep = 0
max_steps = 1000
episode_timestep = 0
total_timestep = 10
episode_reward = 0 
done = True
episode_step=0
random_positions = {1:[675,300],2:[656,590],3:[597,71],4:[614,354],5:[614,50],6:[165,278],7:[233,122],8:[937,648],9:[1028,160],10:[110,107],11:[1420,622],12:[766,225],13:[135,323],14:[469,648],15:[552,659],16:[1139,417]}  #nk # nk17
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
    goal_x = 1420
    goal_y = 622
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
    rotation = BoundedNumericProperty(0.0, min=- 90, max=90.0,errorhandler=lambda x: 90.0 if x > 90.0 else 0.0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    

    def move(self, rotation):
        print("moving by",rotation)
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        # print(self.pos)



class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(2, 0)

    def evaluate_policy(policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = 0
            done = False
            while not done:

                action = policy.select_action(np.array(obs))
                obs, reward, done, _ = env.step(action)
                avg_reward += reward
            avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

    def reset(self):

        #reset_position = random_positions[randint(1,16)]
        #reset_position = [car_prev_x,car_prev_y]
        self.car.x = car_prev_x # reset_position[0] #arbitary position
        self.car.y = car_prev_y #reset_position[1]
        print("Car position in reset {} and {}".format(self.car.x,self.car.y))

        return sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]

    def get_state(self):
        global car_prev_x
        global car_prev_y
        #send_state = sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        #if send_state.shape == (28, 28):
        if int(self.car.x) > 15 and int(self.car.x) < 1429 and int(self.car.y) > 15 and int(self.car.y) < 661:  
            return sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        else:
            #value =  random_positions[randint(1,15)]
            #car_prev_x = value[0]
            #car_prev_y = value[1]
            return sand[int(car_prev_x-14)-14:int(car_prev_x-14)+14,int(car_prev_y-14)-14:int(car_prev_y-14)+14]
        #else:
         #   return sand[int(self.car.x)-28:int(self.car.x)-14, int(self.car.y)-28 : int(self.car.y)-14]

    def take_step(self,action,last_distance,episode_step):
        global car_prev_x
        global car_prev_y
        
        rotation = action
        self.car.move(rotation)
        if int(self.car.x) > 15 and int(self.car.x) < 1429 and int(self.car.y) > 15 and int(self.car.y) < 661:
            
            new_state = sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        else:
            value =   random_positions[randint(2,15)]
            car_prev_x = value[0]
            car_prev_y = value[1]
            new_state = sand[int(car_prev_x)-14:int(car_prev_x)+14,int(car_prev_y)-14:int(car_prev_y)+14]
            #self.car.x = int(car_prev_x)
            #self.car.y = int(car_prev_x)
            
            print("stuck near border{}".format(new_state))
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        print("Car position in take_step {} and {}".format(self.car.x,self.car.y))

        if sand[int(self.car.x),int(self.car.y)] > 0:
            print('Car in sand')
           
        else:
            print("Car on Road")
            car_prev_x = int(self.car.x)
            car_prev_y = int(self.car.y)
            print('x value {}'.format(car_prev_x))
            print('y value {}'.format(car_prev_y))
            

        episode_step += 1
        print('episode step',episode_step)
        if int(sand[int(self.car.x),int(self.car.y)]) > 0: 
            reward = -2.0
            self.car.velocity = Vector(2,0).rotate(self.car.angle)
            done = True  #nk
        else :
            reward = -0.2
            self.car.velocity = Vector(5,0).rotate(self.car.angle)
            done = False

        if episode_step == 100:  #nk #nk16th again
            # episode_step = 0
            print('greater than 500')
            done = True

        if int(last_distance) < int(distance):
            reward += -1.0

        if int(last_distance) > int(distance):   
            reward += 0.5

        if self.car.x < 10:
            self.car.x = 580
            self.car.y = 310
            reward += -3
            done = True
        if self.car.x > self.width - 10:
            self.car.x = 580
            self.car.y = 310
            reward += -3
            done = True
        if self.car.y < 10:
            self.car.x = 580
            self.car.y = 310
            reward += -3
            done = True
        if self.car.y > self.height - 10:
            self.car.x = 580
            self.car.y = 310
            reward += -3
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

        longueur = self.width
        largeur = self.height

        
        

        if first_update:
            init()

        
        last_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        print('timestep',timestep)
        if done:
            print('-------------------EPISODE DONE-------------------')
            if timestep!=0:
                print('total timesteps:{},episode_timestep:{},episode reward:{}'.format(total_timestep,episode_timestep,episode_reward))
                policy.train(replay_buffer,episode_timestep,batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

            # EVALUATION CONDITION AND CODE

            current_state = self.reset()
            done = False
            episode_reward = 0
            episode_timestep = 0

        if timestep < 1200 : 
            current_state = self.get_state()
            #action = max_action*randrange(-1,1)
            action = max_action*random.uniform(-1, 1)
        else:
            current_state = self.get_state()
            action = policy.select_action(np.array(current_state))

        new_state,reward,done,episode_step = self.take_step(action,last_distance,episode_step)

        if episode_step == 500:
            episode_step = 0

        episode_reward += reward

        replay_buffer.add((current_state,new_state,action,reward,done))

        current_state = new_state
        episode_timestep += 1
        timestep += 1





        


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
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()

