# Script to create a dataset of interaction with an environment with a random policy 

import os
import gym
import pickle
import argparse
import numpy as np
from wrapper import *



def parse_arguments():
    # Notes for necessary inputs
    # welches environment
    # name dataset
    # how many samples / epochs of experience
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--env", type=str, default="CarRacing-v0", help="Environment name (default: CarRacing-v0)")
    parser.add_argument("--dataset_name", type=str, default="random_dataset", help="Name of the dataset that is created (default: random_dataset)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes of collected data (default: 10)")
    args = parser.parse_args()
    return args

def create_environment(args):
    """ Returns the environment for collecting the experience """
    # check if gym env 
    env = gym.make(args.env)
    env = MaxAndSkipEnv(env, skip=5)
    env = ObservationWrapper(env, image_size=(64,64,3), scale_obs=True)
    env = PytorchWrapper(env)

    return env

def collect_data(args, env:gym.Env):
    """ Collect experience """

    samples = {"states":[], "actions": [], "next_states": [], "rewards": [], "dones":[]}

    for ep in range(args.episodes):
        done=False
        state = env.reset()
        step_count = 0 ## First ~50 steps is zooming to start position. not needed for training
        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            step_count += 1
            
            if step_count > 50:
                # specific for CarRacing env. should be removed later (!)
                samples["states"].append(state)
                samples["next_states"].append(next_state)
                samples["actions"].append(action)
                samples["rewards"].append(reward)
                samples["dones"].append(done)
            state = next_state
            if done:
                print("Ep: {} | Sample size: {}".format(ep, len(samples["states"])))
                break
        env.close()

    return samples

def save_samples(args, samples:dict):
    """ Saves the data samples to a pickle file """

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    with open("./datasets/"+args.dataset_name+".pkl", 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    print("Saved samples to: ./datasets/{}.pkl".format(args.dataset_name))

if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    samples = collect_data(args, env)
    save_samples(args, samples)