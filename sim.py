import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import threading
import time
import random
tf.autograph.set_verbosity(1)
sys.setrecursionlimit(10000)

from matplotlib.axes import Axes
from typing import List
from enum import Enum, auto
from collections import deque, namedtuple
from copy import deepcopy

g = 9.81

class CartPendulum():
    def __init__(self):
        self.mc = 1.0
        self.m1 = 0.5
        self.l1 = 1.0
        self.lc1 = 0.5
        self.J = self.m1 * self.lc1**2 * 1/3

        self.x_0    = 0.0
        self.th1_0  = np.pi /2 
        self.dth1_0 = 0.0

        self.q    = np.array([[self.x_0], [self.th1_0]])
        self.dq   = np.array([[0.0], [self.dth1_0]])
        self.ddq  = np.zeros((2,1))

        self.cal_element_poses()
    
    def run(self, dt, ac):
        self.dynamic(ac)
        self.kinematic(dt)
        self.cal_element_poses()

    def dynamic(self, ac:float):
        th1 = self.q[1][0]

        self.ddq[0][0] = ac
        self.ddq[1][0] = self.l1 * self.m1 * (g*np.sin(th1) - np.cos(th1)*ac)/(self.J + self.m1*self.l1**2)
    
    def kinematic(self, dt):
        self.dq  += self.ddq * dt
        self.q   += self.dq * dt
    
    def cal_element_poses(self):
        self.cart_pos = self.cal_cart_pos()
        self.pen1_pos = self.cal_pen1_pos()
        
    def cal_cart_pos(self):
        return np.array([[self.q[0][0]], [0.0]])
    
    def cal_pen1_pos(self):
        th1 = self.q[1][0]
        x = self.q[0][0] + self.l1 * np.sin(th1)
        y = self.l1 * np.cos(th1)
        return np.array([[x], [y]])

PlantStates = namedtuple("PlantStates", ("cart_pos_x", "cart_pos_y", "pen_pos_x", "pen_pos_y"))
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

class ReplayMemory:
    def __init__(self, capacity:int):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class PlantBuffer:
    def __init__(self, max_buffer:int=10):
        self.buffer = deque(maxlen=max_buffer)
    
    def push(self, *args):
        self.buffer.append( PlantStates(*args) )
    
    def fetch(self) -> PlantStates:
        try:
            return self.buffer.popleft()
        except IndexError:
            return None
    
    def __len__(self):
        return len(self.buffer)
    
class DQN(tf.keras.Model):
    def __init__(self, action_size, state_dim, out_units=128):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(out_units, activation='relu', input_shape=(state_dim,))
        self.dense2 = tf.keras.layers.Dense(out_units//2, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x1 = self.dense1(state)
        x  = self.dense2(x1)
        logits = self.policy_logits(x)
        return logits
    
    def action(self, state):
        logits = self.call(state)
        action = np.argmax(logits.numpy())
        return action
    
cart_width = 1.0
cart_height = 0.5
resol = 150
time_width = 10.0
clip_sim_time = 0

plant_buffer = PlantBuffer(max_buffer=10)

class PlayState(Enum):
    END = auto()
    START = auto()

play_state = PlayState.END
play_step = -1
plant_states = None
def update(i, ax:Axes, cart_box:plt.Rectangle):
    global plant_buffer, play_state, play_step, plant_states
    
    if play_state == PlayState.END:
        plant_states = plant_buffer.fetch()
        print(f"new record is fetched!!, remaining buffer is {len(plant_buffer)}")
        if not plant_states is None:
            play_state = PlayState.START
        else:
            play_state = PlayState.END

    elif play_state == PlayState.START:
        if not plant_states is None:
            play_step += 1

            pen1_x, pen1_y = plant_states.pen_pos_x, plant_states.pen_pos_y
            cart_x, cart_y = plant_states.cart_pos_x, plant_states.cart_pos_y
            
            ax.clear()
            ax.set_xlim(-10, 10)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.grid()

            try:
                ax.plot(pen1_x[i], pen1_y[i], "o")
                ax.plot(cart_x[i], cart_y[i], "o")

                ax.plot((cart_x[i], pen1_x[i]),(cart_y[i], pen1_y[i]))

                cart_box.set_xy((cart_x[i] - cart_width/2, -cart_height/2))
                ax.add_patch(cart_box)
            except IndexError:
                play_step = -1
                play_state = PlayState.END
        

def cal_reward_function(cartPen:CartPendulum):
    x, dx, th, dth = cartPen.q[0][0], cartPen.dq[0][0], cartPen.q[1][0], cartPen.dq[1][0]
    reward = 0.0
    reward += 10 * np.cos(th)
    reward -= 1.0 * x**2
    reward -= 0.1 * dx**2
    reward -= 0.1 * dth**2
    return reward

def epsilon_greedy(step, decay=1000, min_v=0.0, init_v=1.0):
    return min_v + (init_v - min_v)*np.exp(-step/decay)
    
def control_law(cartPen:CartPendulum, online_nn:DQN, step:int):
    global ctrl_actions
    eps = epsilon_greedy(step)
    # print(f"current epsilon {eps}")
    if random.random() < eps:
        action_index = random.randint(0, len(ctrl_actions)-1)
    else:
        state = get_system_state(cartPen)
        input_node = tf.reshape(state, (1, -1)) 
        action_index = online_nn.action(input_node)
    action = ctrl_actions[action_index]
    return action, action_index, eps

def get_system_state(cartPen:CartPendulum):
    return np.concatenate([ cartPen.q.flatten() , cartPen.dq.flatten() ])

def is_reach_desired(cartPen:CartPendulum, 
                     cart_pos_thresh:float=0.01,
                     cart_v_thresh:float=0.001,
                     pen_pos_thresh:float=0.01,
                     pen_v_thresh:float=0.001) -> int:
    x, dx, th, dth = cartPen.q[0][0], cartPen.dq[0][0], cartPen.q[1][0], cartPen.dq[1][0]
    condition1 = abs(x) < cart_pos_thresh
    condition2 = abs(dx) < cart_v_thresh
    condition3 = abs(th) < pen_pos_thresh
    condition4 = abs(dth) < pen_v_thresh
    return int(condition1 and condition2 and condition3 and condition4)

replayMemory = ReplayMemory(capacity=10000)

def solve(cartPen:CartPendulum, t_eval:np.ndarray, online_nn:DQN, episode:int):
    global plant_buffer, replayMemory

    t0 = t_eval[0]
    t_prev = t0

    cart_pos = [[], []]
    pen1_pos = [[], []]

    for i in range(1, len(t_eval)):
        t  = t_eval[i]
        dt = t - t_prev

        ac, action_index, eps = control_law(cartPen, online_nn, episode)
        
        state = get_system_state(cartPen)

        cartPen.run(dt=dt, ac=ac)

        next_state = get_system_state(cartPen)
        reward = cal_reward_function(cartPen)
        done = is_reach_desired(cartPen)

        cart_pos[0].append(cartPen.cart_pos[0][0])
        cart_pos[1].append(cartPen.cart_pos[1][0])
        pen1_pos[0].append(cartPen.pen1_pos[0][0])
        pen1_pos[1].append(cartPen.pen1_pos[1][0])

        replayMemory.push(state, action_index, next_state, reward, done)

        t_prev = t
    
    cart_pos[0] = cart_pos[0][clip_sim_time:]
    cart_pos[1] = cart_pos[1][clip_sim_time:]
    pen1_pos[0] = pen1_pos[0][clip_sim_time:]
    pen1_pos[1] = pen1_pos[1][clip_sim_time:]
    
    plant_buffer.push(cart_pos[0], cart_pos[1], pen1_pos[0], pen1_pos[1])
    print(f"add new record!!, buffer size is {len(plant_buffer)}")
    return eps

def trainDQN(online_nn:DQN,target_nn:DQN,optimizer:tf.keras.optimizers,mem:ReplayMemory,training_batch:int,
             gamma:float=0.8):
    batch = mem.sample(training_batch)
    states = np.array([b.state for b in batch]) # (ep, 4)
    next_states = np.array([b.next_state for b in batch]) # (ep, 4)
    actions = np.array([b.action for b in batch]) # (ep,)
    rewards = np.array([b.reward for b in batch]) # (ep,)
    dones = np.array([b.done for b in batch]) # (ep,)
    with tf.GradientTape() as tape:
        q_next_values = target_nn(next_states)
        q_values = online_nn(states)

        action_masks = tf.one_hot(actions, depth=q_values.shape[1])
        q_value = tf.reduce_sum(q_values*action_masks, axis=1)
        
        max_next_q = tf.reduce_max(q_next_values, axis=1)
        target = rewards + gamma * max_next_q * (1 - dones)
        loss = tf.reduce_mean(tf.square(target-q_value))
    
    grads = tape.gradient(loss, online_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, online_nn.trainable_variables))

    print(f"training loss {loss}")
    return loss

min_training_buffer = 500
update_nn_step = 0
update_target_nn_step = 50
training_batch = 10
acc_max = 5.0
# ctrl_actions = [-acc_max, -acc_max/2, 0.0, acc_max/2, acc_max]
ctrl_actions = [-acc_max, acc_max]

dummy_state = get_system_state(CartPendulum())
online_nn = DQN(len(ctrl_actions), len(dummy_state))
target_nn = DQN(len(ctrl_actions), len(dummy_state))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

target_nn.set_weights(online_nn.get_weights())

def roll_out(cartPen:CartPendulum, t_eval:np.ndarray):
    global replayMemory, update_nn_step, update_target_nn_step
    global target_nn, online_nn
    episode = 0
    while True:
        print(f"-------------- episode : {episode} -------------------")
        cartPen = CartPendulum()
        eps = solve(cartPen, t_eval, online_nn, episode)

        if len(replayMemory) >= min_training_buffer:
            trainDQN(online_nn, target_nn, optimizer, replayMemory, training_batch)
            update_nn_step += 1

        if update_nn_step >= update_target_nn_step:
            target_nn.set_weights(online_nn.get_weights())
            update_nn_step = 0

        print(f"current epsilon {eps}")
        episode += 1
        # time.sleep(1)

file_name = "model.weights.h5"

def on_press(event):
    global online_nn
    if event.key == 'o':
        online_nn.save_weights(file_name)

is_load_weights = False
if is_load_weights:
    dummy_input = np.zeros((1, len(dummy_state)), dtype=np.float32)
    online_nn(dummy_input)
    target_nn(dummy_input)
    online_nn.load_weights(file_name)
    target_nn.set_weights(online_nn.get_weights())
    
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8,6))

    t_span = (0, time_width)
    t_eval = np.linspace(*t_span, resol)

    cartPen = CartPendulum()
    cart_box = plt.Rectangle(
        (cartPen.cart_pos[0][0], cartPen.cart_pos[1][0]), 
        cart_width, 
        cart_height, 
        fc='blue'
    )

    roll_out_process = threading.Thread(target=roll_out, daemon=True, args=(cartPen, t_eval,))
    roll_out_process.start()

    ani = animation.FuncAnimation(
        fig, 
        lambda i : update(i, ax, cart_box), 
        frames   = len(t_eval), 
        blit     = False, 
        interval = 20,
    )
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.show()