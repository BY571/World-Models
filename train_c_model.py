
import os
import gym
import torch
import argparse
import numpy as np 
import torch.nn as nn
from train_vae import VAE
from wrapper import *
from train_mdnrnn import MModel

from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

def parse_arguments()-> argparse.Namespace:
    # Notes for necessary inputs
    # name dataset
    # batch_size
    # training epochs 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--m_model", type=str, default=None, help="Name of the trained m_model if existing (default: None)")
    parser.add_argument("--v_model", type=str, help="Name of the trained v_model.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of hidden nodes in the C-Model (default: 64)")
    parser.add_argument("--env", type=str, default="CarRacing-v0", help="Name of the training environment (default: CarRacing-v0)")
    parser.add_argument("--n_gaussians", type=int, default=5, help="Number of n-gaussians for the Density Mixture Network (default: 5)")
    parser.add_argument("--latent_size", type=int, default=32, help="Number of latent variables (default: 32)")
    parser.add_argument("--rnn_type", type=str, default="LSTM", help="Type of the Recurrent Neural Network (default: lstm)")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device cpu or gpu (default: cuda:0)")
    parser.add_argument("--save_name", type=str, help="Saving name of the trained C-Model state_dicts")
    parser.add_argument("--restart", type=str, default=None, help="Given directory to state dict to restart training the C-Model (default: None)")
    parser.add_argument("--info", type=str, default="C_model", help="Name of the training run (default: C_model")
    parser.add_argument("--tau", type=float, default=1.1, help="Temperature parameter that is used to create hallucinated states by the m-model (default=1.1)")
    args = parser.parse_args()
    return args


def load_state_dict(network: nn.Module, save_dir: str)-> nn.Module:
    """Loads a given state dict to a given model and sets the model into evaluation mode.

    Args:
        network (nn.Module): Network class of the trained model.
        save_dir (str): Path to the state dict of the previously trained network.

    Returns:
        [nn.Module]: Network with the loaded state dict.
    """
    network.load_state_dict(torch.load(save_dir))
    network.eval()
    
    return network


class C_Model(nn.Module):
    def __init__(self, action_size, hidden_size=256, latent_size=32, device="cpu"):
        super(C_Model, self).__init__()
        
        self.device = device
        self.action_size = action_size
        
        self.layer_1 = nn.Linear(latent_size+(2*256), hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, action_size)
        self.to(device)
        
        
    def forward(self, latent_vector:torch.Tensor, hidden_state:torch.Tensor)-> torch.Tensor:
        """  """
        input_ = torch.cat((latent_vector, hidden_state), dim=1)
        x = torch.relu(self.layer_1(input_))
        x = torch.relu(self.layer_2(x)) 
        x = torch.tanh(self.layer_3(x)) 
        # add var to prediction as well
        return x
    
    def get_action(self, latent_vector:torch.Tensor, hidden_state:torch.Tensor)-> torch.Tensor:
        """  """

        mu = self.forward(latent_vector, torch.cat(hidden_state, dim=-1).squeeze(0))
        dist = Normal(mu, torch.ones(1).to(self.device)*0.2)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    

def calc_discounted_rewards(rewards, final_state, gamma = 0.95):
    R = 0
    discounted = []
  
    for idx in reversed(range(len(rewards))):
        R = rewards[idx]+R*gamma *final_state[idx]
        discounted.insert(0,R)
    return discounted
    
def update_cmodel(log_pis:torch.Tensor, rewards:torch.Tensor)-> np.array:
    """ Updating the C-Model in a PG-Style """
    assert log_pis.shape[0] == rewards.shape[0]
    
    
    # Normalize.
    mean = torch.mean(rewards)
    std = torch.std(rewards)
    rewards = (rewards - mean) / (std)
    
    loss = (-log_pis*(rewards-rewards.mean())).sum()
    return loss


def train(args, env, c_model, v_model, m_model, writer):
    
    episodes = args.episodes
    combined_model_rewards = []
    high_action = env.action_space.high
    low_action = env.action_space.low
    c_model_optim = optim.Adam(c_model.parameters(), lr=args.lr)
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        episode_rewards = 0
        
        reward_batch = []
        dones_batch = []
        log_prob_batch = []
        hidden_state = (torch.zeros((1, 1, 256)).to(device),torch.zeros((1, 1,  256)).to(device))
        while True:
            
            with torch.no_grad():
                latent_vector_z = v_model.encode_state(state.to(device))
            
            action, log_prob = c_model.get_action(latent_vector_z, hidden_state)
            action_ = np.asarray(np.clip(action.detach().cpu().numpy(), a_min=low_action, a_max=high_action), dtype=np.float32).squeeze()
            state, reward, done, _ = env.step(action_)

            if m_model != None:
                with torch.no_grad():
                    _, hidden_state = m_model.predict_next_z(latent_vector_z.unsqueeze(0), 
                                                            torch.from_numpy(action_).unsqueeze(0).unsqueeze(0).float().to(device), 
                                                            tau=args.tau, 
                                                            hidden_state=hidden_state)
            
            reward_batch.append(reward)
            dones_batch.append(1-int(done))
            log_prob_batch.append(log_prob)
            
            episode_rewards += reward
            
            if done:
                break
                
        rewards = calc_discounted_rewards(reward_batch, dones_batch)
        
        c_model_optim.zero_grad()
        loss = update_cmodel(torch.cat(log_prob_batch, dim=0),
                            torch.FloatTensor(rewards).unsqueeze(-1).to(device))
        loss.backward()
        c_model_optim.step()
        loss = loss.detach().cpu().numpy()
        
        del(reward_batch)
        del(dones_batch)
        del(log_prob_batch)
               
        print("\rEpoch: {} | Reward: {:.2f} | Loss: {:.2f}".format(ep, episode_rewards, loss), end="", flush=True)
        writer.add_scalar("Reward", episode_rewards, ep)
        writer.add_scalar("Loss", loss, ep)
        if ep % 10 == 0:
            print("\rEpoch: {} | Reward: {:.2f} | Loss: {:.2f}".format(ep, episode_rewards, loss))
            torch.save(c_model.state_dict(), "trained_models/C_Model/"+args.save_name+"_"+str(ep)+"_.pth")
        env.close()
        
    return c_model


if __name__ == "__main__":
    # load args
    args = parse_arguments()
    writer = SummaryWriter("runs/"+args.info)
    
    if args.device == "cuda:0":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("Cuda is not available, using cpu!")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Using: ", device)
    
    # create Training env
    env = gym.make(args.env)
    env = MaxAndSkipEnv(env)
    env = ObservationWrapper(env, image_size=(64,64,3), scale_obs=True)
    env = PytorchWrapper(env)
    state_size = env.observation_space.shape
    action_size = env.action_space.shape[0]

    # create V and M Model
    v_model = VAE(state_size=state_size,
                  latent_size_N=32, 
                  device=args.device)
    v_model.encoder = load_state_dict(network=v_model.encoder,
                              save_dir=args.v_model)
    del(v_model.decoder)
    
    if args.m_model != None:
        m_model = MModel(action_size=action_size,
                        latent_size=32,
                        hidden_size=256,
                        n_gaussians=5,
                        rnn_type="LSTM").to(device)
        m_model = load_state_dict(network=m_model,
                                save_dir=args.m_model)
    else:
        m_model = args.m_model    
    # create C-Model
    c_model = C_Model(action_size=action_size,
                      hidden_size=args.hidden_size,
                      latent_size=32,
                      device=args.device)#.to(device)
    
    # train 
    if not os.path.exists('trained_models/C_Model/'+args.save_name):
        os.makedirs('trained_models/C_Model/'+args.save_name)
        
    trained_model = train(args=args,
                          env=env,
                          c_model=c_model,
                          v_model=v_model,
                          m_model=m_model,
                          writer=writer)
    
    print("\nFinished Training!")
        
   
    # save model
    torch.save(trained_model.state_dict(), "trained_models/C_Model/"+args.save_name+"final.pth")

    print("\nSaved C-Model state dicts to /trained_models/C_Model/{}".format(args.save_name))
