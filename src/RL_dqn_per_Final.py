# RL Final Project: Navigating complex highway scenarios with advanced RL techniques in Highway-Env
# - Bhaavin Jogeshwar, Chinmayee Prabhakar, Swapneel Wagholikar 


# This is the python file that uses Prioritized Experience Replay (PER) in PyTorch, 
# with the highway-v0 environment.


# first install the necessary libraries 
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import highway_env
import time

# Create a class to use a simple memory buffer replay with priority on experiences.
# This method is used to calculate loss on batches with prioritized experiences and no associated states.
class PER(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.capacity   = capacity
        self.prob_alpha = prob_alpha
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float64)
        self.memory     = []
    
    def sample(self, beta=0.4):
        """Sample batch_size of experiences that have more priority."""
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), BATCH_SIZE, p=probs)
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        transitions = [self.memory[idx] for idx in indices]
        batch       = Transition(*zip(*transitions))
        states      = torch.cat(batch.state)
        actions     = torch.cat(batch.action)
        rewards     = torch.cat(batch.reward)
        next_states = batch.next_state
        dones       = batch.done
        
        return states, actions, rewards, next_states, dones, indices, weights


    def __len__(self):
        return len(self.memory)

    
    def update_priorities(self, batch_indices, batch_priorities):
        """Update the priorities every time we calculate a new loss"""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio[0]


    def push(self, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.pos] = Transition(*args)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    


# DQN algorithm
# The DQN algorithm takes the difference between the current and previous screen patches as input
# and outputs different actions to be taken.
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)


        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 50 episode averages and plot them too
    if len(durations_t) >= 50:
        means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def get_screen():
    
    # Returned screen requested by gym is 600x150x3
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)

  
def select_action(state):
    global steps_done
    sample = random.random()                                # epsilon-greedy is used to chose an action
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)



# This function computes the loss and optimizes the model weights
def optimize_model(beta):
    if len(per_memory) < BATCH_SIZE:
        return
    state, action, reward, next_state, done, indices, weights = per_memory.sample(beta) 

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in next_state
                                                if s is not None])

    q_values      = policy_net(state)
    
    # Compute the Q values
    q_value          = q_values.gather(1, action)
    next_q_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_q_value = (next_q_values * GAMMA) + reward
    
    # Compute the loss and priorities
    loss  = (q_value - expected_q_value.unsqueeze(1)).pow(2)*torch.as_tensor(weights)
    prios = loss.detach() + 1e-5
    loss  = loss.mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    #update priorities
    per_memory.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    return loss        


# this is the main function that is called to run each episode
def run_episode(env, render = False):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    total_reward = 0
    for t in count():
        if render:
            env.render()
        
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _, _ = env.step(action.item())
        total_reward += reward

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Move to the next state
        state = next_state

        if done:
            break
    return total_reward



# This function is called for Training
def training_function():
    num_episodes = 1001
    total_reward = 0
    for i_episode in range(num_episodes):
        
        
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen    
        start_time = time.time()
        for t in count():
            
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            elapsed_time = time.time() - start_time 
            if not done:
                next_state = current_screen - last_screen
                if elapsed_time> 600:
                    break
            else:
                next_state = None
                if elapsed_time > 600:
                    break

            # Store the transition in per_memory
            per_memory.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            beta = beta_by_frame(steps_done)
            optimize_model(beta)
            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break

        total_rewards.append(total_reward)
        avg_score = np.mean(total_rewards[-num_episodes:])
        avg_scores.append(avg_score)
        print(f'episode: {i_episode} | score: {total_reward} | average score: {avg_score} | Elapsed time: {elapsed_time}')

        # if i_episode % 20 == 0:
        #     print(f"Mean episode {i_episode}/250 reward is:{total_reward / 20:.2f}")
        total_reward = 0 
        print("episode number = ", i_episode)   
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 25 == 0:
            plt.figure(3)
            plt.clf()
            plt.plot(total_rewards, label='Score')
            plt.plot(avg_scores, label='Average Score')
            plt.legend()
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.savefig('PER_final.png')
            plt.show()



if __name__ == "__main__":

    # initialize the environment
    env = gym.make('highway-v0').unwrapped

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_CUDA = torch.cuda.is_available()

    # here we define the transition used (state, action, reward, next_state, done)
    # States: screen difference image.
    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state','done'))

    beta_start = 0.4
    beta_frames = 1000 
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    # Input extraction
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(100, interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor()])

    env.reset()
    plt.figure()
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


    # Training Hyperparameters
    learning_rate = 5e-4
    BATCH_SIZE = 32
    GAMMA = 0.8
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 50


    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # initialize the policy and target networks
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = Adam(policy_net.parameters(),lr = learning_rate)
    replay_initial = 10000
    per_memory = PER(10000)

    steps_done = 0

    episode_durations = []

    total_rewards = []
    avg_scores = []

    # Train the model and save it
    # training_function()
    # torch.save(policy_net.state_dict(), "../models/model_per_400")    

    plt.pause(100)



    # In case the training_function is already trained, we load it
    steps_done = 1000000000000
    policy_net.load_state_dict(torch.load("../models/model_per_400"))
    policy_scores = [run_episode(env, True) for _ in range(50)]
    print("Average score of the policy: ", np.mean(policy_scores))
    env.close()
