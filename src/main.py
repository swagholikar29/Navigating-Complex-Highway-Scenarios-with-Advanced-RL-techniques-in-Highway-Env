import gym
import highway_env
import numpy as np
import dqn_agent
import matplotlib.pyplot as plt

config = {
'action': {'type': 'DiscreteMetaAction'},
'centering_position': [0.3, 0.5],
'collision_reward': -30,                         
'controlled_vehicles': 1,
'duration': 30,
'high_speed_reward': 2,
'lane_change_reward': 1,                            
'lanes_count': 4,                                    
'observation': {'absolute': False,
                'features': ['presence',
                            'x',
                            'y',
                            'vx',
                            'vy',
                            'cos_h',
                            'sin_h'],
                'features_range': {'vx': [-10, 10],
                                'vy': [-10, 10],
                                'x': [-200, 200],
                                'y': [-200, 200]},
                'normalize': True,
                'order': 'sorted',
                'type': 'Kinematics',
                'vehicles_count': 50},
'vehicles_count': 50,                                           
'reward_speed_range': [20, 40],                              
'right_lane_reward': 1,                                           
}

env = gym.make('highway-v0')
env.configure(config)
# print(env.config)

agent = dqn_agent.Agent(gamma=0.99, epsilon=0.8, batch_size=32, n_actions=env.action_space.n, eps_end=0.01, input_dims=[350], lr=5e-4)
scores, eps_history = [], []
avg_scores = []
n_games = 2

for i in range(n_games):
    score = 0
    terminated = False
    truncated = False
    observation = env.reset()
    observation=observation[0]
    while not (terminated or truncated):
        env.render()
        action = agent.choose_action(observation.flatten())
        observation_, reward, terminated, truncated, _ = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, (terminated or truncated))
        agent.learn()
        observation = observation_

    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-n_games:])
    avg_scores.append(avg_score)
    print(f'episode: {i} | score: {score} | average score: {avg_score} | epsilon: {agent.epsilon}')

plt.plot(scores, label='Score')
plt.plot(avg_scores, label='Average Score')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Duration')
# plt.pause(0.1)
plt.show()

