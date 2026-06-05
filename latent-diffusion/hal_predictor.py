import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)
   
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, num_timesteps, channels, height, width):
        self.num_timesteps = num_timesteps
        self.channels = channels
        self.height = height
        self.width = width
        
        self.data = [str(f) for f in Path(data_dir).rglob("*") if f.is_file()]
        self.traj_dir = "/data_local1/mbhosale/DiffHaul/Hands/trajectories/"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fname = self.data[idx]
        label = fname.split("/")[-2]
        fname = fname.split("/")[-1].split(".")[0]
        return torch.load(os.path.join(self.traj_dir, fname+'.pt')), int(label == 'hal')

class PredictedImageEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super(PredictedImageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.fc = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        # print(x)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)             # Project to feature vector
        return x

class DenoisingClassifier(nn.Module):
    def __init__(self, in_channels, feature_dim, hidden_dim, num_layers, num_classes, device):
        super(DenoisingClassifier, self).__init__()
        self.encoder = PredictedImageEncoder(in_channels, feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.device = device
    
    def forward(self, x):
        # x: [batch, num_timesteps, height, width, channels]
        x = x.permute(0, 1, 4, 2, 3)
        batch_size, num_timesteps, channels, height, width = x.shape
        
        x = x.contiguous().view(batch_size * num_timesteps, channels, height, width)
        x = x.float()  
        x = x / 255.0
        
        features = self.encoder(x)
        features = features.view(batch_size, num_timesteps, -1)
        lstm_out, (h_n, _) = self.lstm(features)
        final_hidden = h_n[-1]
        logits = self.fc(final_hidden)
        return logits

class Trainer():
    def __init__(self, gpu_id):
        self.num_timesteps = 50           # Timesteps in the reverse/denoising process
        self.channels = 3                 # RGB images
        self.height = 256
        self.width = 256             # Image dimensions
        self.feature_dim = 256            # Feature dimension for each image after CNN encoding
        self.hidden_dim = 128              # Hidden dimension for the LSTM
        self.num_layers = 16               # Number of LSTM layers
        self.num_classes = 2              # Two classes: not hallucinated (0) vs. hallucinated (1)
        self.batch_size = 64
        self.num_epochs = 20
        self.learning_rate = 0.005
        self.train_folder = '/data_local1/mbhosale/DiffHaul/Hands/train/'
        self.val_folder = '/data_local1/mbhosale/DiffHaul/Hands/val/'
        self.save_folder = '/home/csgrad/mbhosale//phd/ddpm_hallucination/latent-diffusion/logs/2025-01-11T12-11-40_hands-ldm-vq-f4/inference/00005580/2025-02-20-13-18-00/checkpoints/'
        self.log_freq = 5
        self.gpu_id = gpu_id

        os.makedirs(self.save_folder, exist_ok=True)
        self.train_dataset = TrajectoryDataset(self.train_folder, num_timesteps=self.num_timesteps, 
                                                channels=self.channels, height=self.height, width=self.width)
        self.val_dataset = TrajectoryDataset(self.val_folder, num_timesteps=self.num_timesteps, 
                                                channels=self.channels, height=self.height, width=self.width)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, 
                                    sampler=DistributedSampler(self.train_dataset))
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                    sampler=DistributedSampler(self.val_dataset, shuffle=False))

        self.model = DenoisingClassifier(in_channels=self.channels, 
                                    feature_dim=self.feature_dim, 
                                    hidden_dim=self.hidden_dim, 
                                    num_layers=self.num_layers, 
                                    num_classes=self.num_classes,
                                    device=self.gpu_id)
        self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _run_batch(self, batch_data, batch_labels):
        self.optimizer.zero_grad()
        outputs = self.model(batch_data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        self.optimizer.step()
        running_loss = loss.item() * batch_data.size(0)
        return running_loss, outputs
    
    def _run_epoch(self, epoch):
        self.model.train()
        self.train_dataloader.sampler.set_epoch(epoch) 
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_data, batch_labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} Training"):
            batch_data = batch_data.to(self.gpu_id)
            batch_labels = batch_labels.to(self.gpu_id)
            loss, outputs = self._run_batch(batch_data, batch_labels)
            predicted = torch.argmax(outputs, dim=1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            running_loss += loss
        epoch_loss = running_loss / len(self.train_dataset)
        train_accuracy = 100.0 * correct / total
        if self.gpu_id == 0:
            print(f"Epoch {epoch+1}/{self.num_epochs} - Training Loss: {epoch_loss:.4f} Training Accuracy: {train_accuracy:.4f}")

    def _eval(self, epoch):
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data, batch_labels in tqdm(self.val_dataloader, desc="Epoch Validation"):
                    batch_data = batch_data.to(self.gpu_id)
                    batch_labels = batch_labels.to(self.gpu_id)
                    outputs = self.model(batch_data)
                    predicted = torch.argmax(outputs, dim=1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            val_accuracy = 100.0 * correct / total
            print(f"Epoch {epoch+1}/{self.num_epochs} - Validation Accuracy: {val_accuracy:.2f}%\n")
            torch.save(self.model.module.state_dict(), os.path.join(self.save_folder, f'hal_predictor_e{epoch+1}.pth'))
    
    def train(self):
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch)
            if epoch % self.log_freq == 0 and self.gpu_id == 0:
                self._eval(epoch)


def main(rank, world_size):
    ddp_setup(rank, world_size)
    trainer = Trainer(rank)
    trainer.train()
    destroy_process_group()
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    