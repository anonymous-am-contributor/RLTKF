# RLTKF
RLTKF Anonymized code for review.
This repository implements a Reinforcement Learning framework for tuning Linear Kalman Filters under varying observability levels. It provides tools to generate controlled banks of (A, H) matrix pairs, train RL agents, compare them with classical baseline methods, and evaluate performance in terms of RMSE. You can find the following files : 

- generate_bank_AH.py is used to generate banks of (A, H) matrix couples. The user can specify :
	- the number of couples
	- the state dimension
	- the number of measurements
	- and other structural and spectral parameters described in the article, including properties related to observability. 

- main_rltkf_training.py is the main training script. It loads a previously generated bank and launches the training of the RLTKF agent. Many aspects can be configured, including what is included in the state space given to the agent, the action space definition, the reward formulation, the noise parameters, and general training hyperparameters.

- kalman_env.py contains the full definition of the RL environment used for the RLTKF implementation. 

- kf_utils.py provides :
  	-implementation of the linear Kalman filter 
	-function to generate simulation scenarios from a given bank of (A, H) matrices.

- test_functions.py defines all baseline methods used for comparison and provides the evaluation procedures for trained RL agents. It includes functions to compute RMSE and to generate plots that compare the performance of the different methods as a function of observability levels.

The typical workflow consists of generating a bank of systems, training the RL agent using this bank, and then evaluating its performance against the baseline methods. 
