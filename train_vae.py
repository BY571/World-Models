
import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
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
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device cpu or gpu (default: cuda:0)")
    parser.add_argument("--save_name", type=str, help="Name of the trained encoder decoder state_dicts")
    parser.add_argument("--plot", type=int, choices=[0,1], default=1, help="Plot training results if set to 1 (default: 1)")
    args = parser.parse_args()
    return args

def load_samples(name:str)-> dict:
    with open("./datasets/"+name, 'rb') as handle:
        samples = pickle.load(handle)
    return samples

def create_dataset(args)-> DataLoader:

    samples = load_samples(args.dataset_name)
    shape = samples["states"][0].shape
    set = torch.cat(samples["states"], dim=0)
    dataset = TensorDataset(set)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader, shape

def train(args: argparse.Namespace, vae, dataloader: DataLoader)-> dict:

    losses = {"loss1": [], "loss2": [], "loss": []}
    for i in range(args.epochs):
        loss1, loss2 = vae.train(dataloader)
        print("\rEpoch: {} | kl_loss: {:.2f} | l2_loss: {:.2f} | combines loss: {:.2f}".format(i, loss1, loss2, loss1+loss2), end="", flush=True)
        if i % 10 == 0:
            print("\rEpoch: {} | kl_loss: {:.2f} | l2_loss: {:.2f} | combines loss: {:.2f}".format(i, loss1, loss2, loss1+loss2))
        
        losses["loss1"].append(loss1)    
        losses["loss2"].append(loss2)    
        losses["loss"].append(loss1+loss2)

    return losses, vae

def plot_losses(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses["loss1"], label="KL loss")
    plt.plot(losses["loss2"], label="L2 loss")
    plt.plot(losses["loss"], label="Combined loss")
    plt.legend()
    plt.savefig("VAE-Training.jpg")
    plt.show()


class VAE():
    def __init__(self, state_size, latent_size_N=32, device="cpu"):
              
        self.encoder = Encoder(state_size, latent_size_N).to(device)
        self.decoder = Decoder(state_size, latent_size_N).to(device)
        self.device = device
        
        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params=self.params, lr=0.001)
        
    def forward(self, image: torch.Tensor)-> torch.Tensor:
        
        assert image.shape == (image.shape[0] ,3, 64, 64), "Input Image as wrong shape!"
        mu, sig = self.encoder(image)
        encoded = self.sample(mu, sig)
        
        decoded = self.decoder(encoded)
        return decoded, (mu, sig)
    
    def encode_state(self, image: torch.Tensor, return_mu=False)-> torch.Tensor:
        assert image.shape == (image.shape[0], 3, 64, 64), "Input Image as wrong shape!"
        (mu, logsig) = self.encoder(image)
        encoded = self.sample(mu, logsig, return_mu)
        return encoded
    
    def sample(self, mu, logsig, return_mu=False):
            
        #dist = Normal(mu, sig) 
        #latent_vector_z = dist.sample()
        
        # in the paper they had different sample methods this one and mu + sig *N(0,1)
        #latent_vector_z = mu + sig * torch.normal(torch.zeros(1), torch.ones(1)).to(device)
        if return_mu == True:
            return mu
        else:
            sigma = logsig.exp()
            eps = torch.randn_like(sigma)     
            return eps.mul(sigma).add_(mu)
        
    def train(self, dataloader:DataLoader)-> torch.Tensor:
        losses = {"loss1": [], "loss2": []}
        for idx, sample in enumerate(dataloader):
            
            image_batch = sample[0].to(device)
            reconstructed, (mu, sig) = self.forward(image_batch)
            
            loss_1 = kl_loss(mu, sig)
            loss_2 = l2_dist_loss(reconstructed, image_batch)
            
            self.optimizer.zero_grad()
            loss = (loss_1 + loss_2).mean()
            
            loss.backward()
            #clip_grad_norm_(self.params, 10.)
            self.optimizer.step()
            losses["loss1"].append(loss_1.detach().cpu().numpy())
            losses["loss2"].append(loss_2.detach().cpu().numpy())
        return np.mean(losses["loss1"]),np.mean(losses["loss2"]) 
            
            

def kl_loss(mu: torch.Tensor, sig:torch.Tensor)-> torch.Tensor:
    #loss = - 0.5 * (1 + torch.log(sig).pow(2) - sig.pow(2) - mu.pow(2)).sum()
    loss = -0.5 * torch.sum(1 + 2 * sig - mu.pow(2) - (2 * sig).exp())
    return loss

def l2_dist_loss(prediction:torch.Tensor, target:torch.Tensor)-> torch.Tensor:
    #return - torch.dist(target,prediction, p=2)
    return F.mse_loss(prediction, target, size_average=False)
    #return F.binary_cross_entropy(prediction, target, size_average=False)
        
class Encoder(nn.Module):
    def __init__(self, state_size, latent_size_N=32):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.output_mu = nn.Linear(1024, latent_size_N)
        self.output_sig = nn.Linear(1024, latent_size_N)
    
    def forward(self, img):
        
        x = torch.relu(self.conv1(img))
        #print(x.shape)
        x = torch.relu(self.conv2(x))
        #print(x.shape)
        x = torch.relu(self.conv3(x))
        #print(x.shape)
        x = torch.relu(self.conv4(x))
        #print(x.shape)
        
        mu = self.output_mu(x.flatten(1))
        log_sig = self.output_sig(x.flatten(1))
        return (mu, log_sig)

        
class Decoder(nn.Module):
    def __init__(self, state_size, latent_size_N=32):
        super(Decoder, self).__init__()
        
        self.in_linear = nn.Linear(32, 1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2)
        
    def forward(self, latent_vector):
        
        x = torch.relu(self.in_linear(latent_vector)).unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.deconv1(x))
        #print(x.shape)
        x = torch.relu(self.deconv2(x))
        #print(x.shape)
        x = torch.relu(self.deconv3(x))
        #print(x.shape)
        x = torch.sigmoid(self.deconv4(x))
        #print(x.shape)
        return x    



if __name__ == "__main__":
    args = parse_arguments()
    dataloader, shape = create_dataset(args)

    # create VAE
    if args.device == "cuda:0":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("Cuda is not available, using cpu!")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    vae = VAE(state_size=shape, device=device)
    training_results, vae = train(args, vae, dataloader)
    print("Finished Training!")
        
    if not os.path.exists('trained_models/VAEs/'+args.save_name):
        os.makedirs('trained_models/VAEs/'+args.save_name)
    
    # save model
    torch.save(vae.encoder.state_dict(), "./trained_models/VAEs/"+args.save_name+"/Encoder.pth")
    torch.save(vae.decoder.state_dict(), "./trained_models/VAEs/"+args.save_name+"/Decoder.pth")
    print("\nSaved vae state dicts to /trained_models/VAEs/{}".format(args.save_name))

    if args.plot:
        plot_losses(training_results)