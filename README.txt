Setup The Highwayenv - v0 as mentioned here : https://pypi.org/project/highway-env/

Download the repository 'src' files into system.

Perform the following commands:
cd ~
python3 -m venv rl_project
source ~/rl_project/bin/activate

System configuration:
	OS : Ubuntu 20.04 LTS
	Python : 3.9.13
	PyTorch : 1.13.0
	OpenAI Gym : 0.26.2

Change terminal directory to location of src files and run the python files using the command :
python 3 'File_name.py'

The Files to be run are :

1. For DQN (Run only 'main'): dqn_agent
                              main

2. For DQN with Memory Replay:  RL_dqn_MemoryReplay_Final

3. For DQN with Prioritized Memory Replay:   RL_dqn_per_Final
