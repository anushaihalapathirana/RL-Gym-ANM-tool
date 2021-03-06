# RL-Gym-ANM-tool

This project contains the python scripts of microgrid generation and simulation using Gym-ANM tool. (https://github.com/robinhenry/gym-anm)

AI techniques with Gym-ANM

* MPC
* PPO
* SAC
* A2C
* TD3

## Code Structure

```
  src - This folder contains the RL algorithms implemented and apply for Gym-ANM simple environment
  
    |
    
    MPC - How to use Gym-ANM built in MPC algorithm
    
    TD3 - Implementation of TD3 algorithm and train, model save and test
          run main.py file
          
    rl_techniques_baseline3 - this folder contains the RL techniques implemented using baseline3 library on simple gym-anm environment
          
              |
              
              a2c
              ppo
              SAC
              
              run main.py file to run all three algorithms at once. train, model save and test
              
   anmEasy6-test-env.py - How to simply access Gym-ANM6-easy environment
   
   customizedEasyenv.py - How to Customize Gym-ANM6 easy environment
   
   mpcPolicy-on-anmeasy.py - MPC policy applied on Gym-ANM6-easy environment. 
                             Run python mpcPolicy-on-anmeasy.py
                             
              
   newEnv.py - Run this file to create new Gym-ANM environment

```

You can find the full doumentation from below links 

1. https://github.com/anushaihalapathirana/RL-Gym-ANM-tool/blob/master/GymANM-tool.pdf
2. https://github.com/anushaihalapathirana/RL-Gym-ANM-tool/blob/master/Gym-ANM-tool-Implementation.pdf
