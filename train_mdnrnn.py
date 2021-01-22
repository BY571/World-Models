import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from train_vae import VAE
from typing import Tuple
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader



def parse_arguments()-> argparse.Namespace:
    # Notes for necessary inputs
    # name dataset
    # batch_size
    # training epochs 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", type=str, default="random_dataset", help="Name of the dataset that is created (default: random_dataset)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seq_length", type=int, default=16, help="Sequence length of the input data to the RNN (default: 16)")
    parser.add_argument("--hidden_size", type=int, default=256, help="Number of hidden nodes in the Network (default:256)")
    parser.add_argument("--n_gaussians", type=int, default=5, help="Number of n-gaussians for the Density Mixture Network (default: 5)")
    parser.add_argument("--latent_size", type=int, default=32, help="Number of latent variables (default: 32)")
    parser.add_argument("--rnn_type", type=str, default="LSTM", help="Type of the Recurrent Neural Network (default: lstm)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device cpu or gpu (default: cuda:0)")
    parser.add_argument("--save_name", type=str, help="Name how the trained model shall be saved")
    parser.add_argument("--encoder_dir", type=str, help="Directory of the trained encoder state dict")
    parser.add_argument("--plot", type=int, choices=[0,1], default=1, help="Plot training results if set to 1 (default: 1)")
    args = parser.parse_args()
    return args

def load_samples(name:str)-> dict:
    """Loading the saved data set.

    Args:
        name (str): Name of the pickle file you want to load.

    Returns:
        dict: Returns a dictionary of the collected data 
              with the keys:
              ["states", "actions", "next_states", "rewards", "dones"]
    """
    with open("./datasets/"+name, 'rb') as handle:
        samples = pickle.load(handle)
    return samples

def create_dataset(args: dict)-> DataLoader:
    """Creates a training Dataloader

    Args:
        args (dict): Training specific arguments.

    Returns:
        DataLoader: PyTorch Dataloader
    """
    samples = load_samples(args.dataset_name)

    state_shape = samples["states"][0].shape
    action_shape = samples["actions"][0].shape

    m_actions = samples["actions"]
    m_actions = torch.from_numpy(np.array(m_actions))
    m_states = torch.cat(samples["states"], dim=0)
    m_next_states = torch.cat(samples["next_states"], dim=0)
    m_rewards = torch.FloatTensor(samples["rewards"])
    m_dones = torch.FloatTensor(samples["dones"])

    seq_length = args.seq_length
    batch_size = args.batch_size
    sequence_samples = {"states": [],
                        "actions": [],
                        "next_states": [],
                        "rewards": [],
                        "dones": []}


    samples_ = {"states": m_states,
                "actions": m_actions,
                "next_states": m_next_states,
                "rewards": m_rewards,
                "dones": m_dones}

    for key, elements in samples_.items():
        sequences = []
        #print(len(elements))
        for i in range(len(elements)-seq_length):
            sequences.append(elements[i:i+seq_length].unsqueeze(0))

        sequence_samples[key] = torch.cat(sequences)
        
    dataset = TensorDataset(sequence_samples["states"], 
                            sequence_samples["actions"], 
                            sequence_samples["next_states"], 
                            sequence_samples["rewards"], 
                            sequence_samples["dones"])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader, state_shape, action_shape

def encode_batch_sequence(vae, batch, batch_size=16, sequence_length=10, latent_size=32):
    encoded = []
    for sequence in batch:
        for sample in sequence:
            encoded.append(vae.encode_state(sample.unsqueeze(0).to(device)))
    return torch.cat(encoded).reshape(batch_size, sequence_length, latent_size)


def train(args, vae, mdn_rnn, dataloader):
    # Train M-Model:
    vae.encoder.eval()
    M_optimizer = optim.RMSprop(params=mdn_rnn.parameters(), lr=args.lr, alpha=.9)

    losses = []
    for ep in range(args.episodes):
        ep_losses = []
        for (states, actions, next_states, rewards, dones) in dataloader:
            
            
            with torch.no_grad():
                z = encode_batch_sequence(vae, states, args.batch_size, args.seq_length, args.latent_size)
                targets = encode_batch_sequence(vae, next_states, args.batch_size, args.seq_length, args.latent_size)
            
            (pi, mu, sigma, rs, ds), _ = mdn_rnn(latent_vector=z,
                                    action=actions.to(device),
                                    hidden_state=None)
            
            
            M_optimizer.zero_grad()
            loss = criterion(targets, pi, mu, sigma)
            reward_loss = get_reward_loss(rewards, rs)
            done_loss = get_done_loss(dones, ds)
            # scaling for latent vector size (32) + reward (1) + done (1) value
            loss = (loss+reward_loss+done_loss)/(args.latent_size+2) 
            loss.backward()
            M_optimizer.step()
            ep_losses.append(loss.detach().cpu().numpy())
            
        losses.append(np.mean(ep_losses))    
        print("\rEpisode: {} | Loss: {}".format(ep, np.mean(ep_losses)), end="", flush=True)
        if ep % 25 == 0:
            print("\rEpisode: {} | Loss: {}".format(ep, np.mean(ep_losses)))
            
    return losses, mdn_rnn

def plot_losses(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("MDN-RNN-Training.jpg")
    plt.show()




class MModel(nn.Module):
    def __init__(self, action_size, latent_size=32, hidden_size=256, n_gaussians=5, rnn_type="LSTM"):
        super(MModel, self).__init__()
        
        self.input_shape = action_size+latent_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.n_gaussians = n_gaussians
        
        if rnn_type == "LSTM":
            self.rnn_layer = nn.LSTM(self.input_shape, hidden_size, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn_layer = nn.GRU(self.input_shape, hidden_size, batch_first=True)
            
        self.pi_layer = nn.Linear(hidden_size, latent_size*n_gaussians)
        self.mu_layer = nn.Linear(hidden_size,  latent_size * n_gaussians)
        self.sig_layer = nn.Linear(hidden_size, latent_size * n_gaussians)
        self.reward_layer = nn.Linear(hidden_size, 1)
        self.done_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, latent_vector: torch.Tensor, action:torch.Tensor, hidden_state=None)-> torch.Tensor:
        """ Simple forward pass with the RNN """
        
        assert latent_vector.shape == (latent_vector.shape[0], latent_vector.shape[1], self.latent_size), "Latent vector has the wrong shape!"
        assert action.shape == (action.shape[0], action.shape[1], self.action_size), "Action batch has the wrong shape!"
        
        input_tensor = torch.cat((latent_vector, action),dim=-1)
        assert input_tensor.shape == (action.shape[0], action.shape[1], self.input_shape), "input_tensor has wrong shape!"
        
        output, hidden_state = self.rnn_layer(input_tensor, hidden_state)
        
        (pi, mu, sigma, rs, ds) = self.get_gauss_coeffs(output)
        return (pi, mu, sigma, rs, ds), hidden_state
    
    
    def get_gauss_coeffs(self, y:torch.Tensor):
        
        batch_size = y.shape[0]
        sequence_length = y.shape[1]

        mu = self.mu_layer(y)
        mu = mu.view(batch_size, sequence_length, self.n_gaussians, self.latent_size)
        
        sigma = self.sig_layer(y)
        sigma = sigma.view(batch_size, sequence_length, self.n_gaussians, self.latent_size)
        sigma = torch.exp(sigma)
        
        pi = self.pi_layer(y)
        pi = pi.view(batch_size, sequence_length, self.n_gaussians, self.latent_size)
        pi = F.softmax(pi, dim=-1)
        
        rs = self.reward_layer(y).squeeze()
        ds = F.sigmoid(self.done_layer(y)).squeeze()       
        
        return pi, mu, sigma, rs, ds
    
    def predict_next_z(self,latent_vector: torch.Tensor, action:torch.Tensor, tau: float, hidden_state=None)-> torch.Tensor:
        """ Predicts the next Latent Vector Z """
        values, hidden_state = self.forward(latent_vector, action, hidden_state)
        pi, mu, sigma, rs, ds = values[0], values[1], values[2], values[3], values[4]
        
        dist = Normal(mu, sigma*tau)
        z_ = (pi*dist.sample()).sum(2)
        
        return (z_, rs, ds), hidden_state
        
# M-Model loss calculation
def mdn_loss_fn(y, out_pi, out_mu, out_sigma):
    y = y.view(-1, 16, 1, 32)
    result = Normal(loc=out_mu, scale=out_sigma)
    result = torch.exp(result.log_prob(y))
    result = torch.sum(result * out_pi, dim=2)
    result = -torch.log( result)
    return torch.mean(result)

def criterion(y, pi, mu, sigma):
    #y = y.unsqueeze(-2) 
    return mdn_loss_fn(y, pi, mu, sigma)

def get_reward_loss(reward, pred_reward):
    return F.mse_loss(pred_reward, reward)

def get_done_loss(done, done_pred):
    return F.binary_cross_entropy(done_pred, done)



if __name__ == "__main__":
    args = parse_arguments()
    dataloader, state_shape, action_shape = create_dataset(args)

    if args.device == "cuda:0":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("Cuda is not available, using cpu!")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Using: ", device)
    vae = VAE(state_size=state_shape, device=device)
    vae.encoder.load_state_dict(torch.load(args.encoder_dir))
    del(vae.decoder)
    mdn_rnn = MModel(action_size=action_shape[0], 
                     latent_size=args.latent_size, 
                     hidden_size=args.hidden_size, 
                     n_gaussians=args.n_gaussians, 
                     rnn_type=args.rnn_type)

    training_results, mdn_rnn = train(args, vae, mdn_rnn, dataloader)
    print("\nFinished Training!")
        
    if not os.path.exists('trained_models/MDN_RNNs/'+args.save_name):
        os.makedirs('trained_models/MDN_RNNs/'+args.save_name)
    
    # save model
    torch.save(mdn_rnn.state_dict(), "./trained_models/MDN_RNNs/"+args.save_name+".pth")

    print("\nSaved MDN-RNN state dicts to /trained_models/MDN_RNNs/{}".format(args.save_name))

    if args.plot:
        plot_losses(training_results)