import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

from utils import get_observation_batch, get_observation_and_action_pair, collect_real_data_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WorldModel(nn.Module):
    def __init__(self, observation_encoder, observation_decoder, dynamic_predictor, reward_predictor, latent_dim=2048):
        super().__init__()
        self.observation_encoder = observation_encoder
        self.observation_decoder = observation_decoder
        self.dynamic_predictor = dynamic_predictor
        self.reward_predictor = reward_predictor
        self.latent_dim = latent_dim
        
    def forward(self, observations, observation_action_pairs, latent_state):
        encoded_obs = self.observation_encoder(observations)
        
        next_obs_pred, hidden_latents, _ = self.dynamic_predictor(observation_action_pairs, latent_state)
        
        decoded_obs = self.observation_decoder(encoded_obs)
        
        reward_pred = self.reward_predictor(hidden_latents.detach())
        
        return encoded_obs, next_obs_pred, decoded_obs, reward_pred.squeeze(-1)
    
    def encode(self, observations):
        return self.observation_encoder(observations)
    
    def decode(self, encoded_obs):
        return self.observation_decoder(encoded_obs)
    
    def predict_next_state(self, observation_action_pairs, latent_state):
        return self.dynamic_predictor(observation_action_pairs, latent_state)
    
    def predict_reward(self, hidden_latents):
        return self.reward_predictor(hidden_latents)


class WorldModelTrainer:
    def __init__(self, world_model, lr=1e-4, beta_recon=1.0, beta_dynamics=1.0, 
                 beta_reward=1.0, max_grad_norm=1.0):
        self.world_model = world_model
        self.beta_recon = beta_recon
        self.beta_dynamics = beta_dynamics
        self.beta_reward = beta_reward
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.Adam(
            list(world_model.observation_encoder.parameters()) +
            list(world_model.observation_decoder.parameters()) +
            list(world_model.dynamic_predictor.parameters()) +
            list(world_model.reward_predictor.parameters()),
            lr=lr
        )
        
    def compute_losses(self, observations, next_observations, observation_action_pairs, rewards_real, latent_state):
        observations = observations.float() / 255.0
        next_observations = next_observations.float() / 255.0
        
        encoded_obs = self.world_model.encode(observations)
        encoded_next_obs = self.world_model.encode(next_observations)
        
        decoded_obs = self.world_model.decode(encoded_obs)
        
        next_obs_pred, hidden_latents, _ = self.world_model.predict_next_state(observation_action_pairs, latent_state)
        
        reward_pred = self.world_model.predict_reward(hidden_latents.detach())
        reward_pred = reward_pred.squeeze(-1)
        
        loss_reconstruction = F.mse_loss(decoded_obs, observations)
        
        loss_dynamics_mse = F.mse_loss(next_obs_pred, encoded_next_obs)
        loss_dynamics_cosine = 1 - F.cosine_similarity(next_obs_pred, encoded_next_obs, dim=-1).mean()
        loss_dynamics = 0.5 * loss_dynamics_mse + 0.5 * loss_dynamics_cosine
        
        loss_reward = F.mse_loss(reward_pred, rewards_real)
        
        loss_total = (self.beta_recon * loss_reconstruction + self.beta_dynamics * loss_dynamics + self.beta_reward * loss_reward)
        
        return {
            'total': loss_total,
            'reconstruction': loss_reconstruction,
            'dynamics': loss_dynamics,
            'reward': loss_reward
        }
    
    def update(self, batch, latent_dim=2048, num_action_classes=17):
        observations_batch, next_observations_batch, rewards_real = get_observation_batch(batch)
        rewards_real = rewards_real.to(device)
        observations_batch = observations_batch.to(device)
        next_observations_batch = next_observations_batch.to(device)
        
        latent_state = torch.zeros(1, 1, latent_dim, device=device)
        
        with torch.no_grad():
            encoded_obs = self.world_model.encode(observations_batch.float() / 255.0)
        observation_action_pairs = get_observation_and_action_pair(batch, encoded_obs, num_classes=num_action_classes).to(device).float()
        
        losses = self.compute_losses(observations_batch, next_observations_batch, observation_action_pairs, rewards_real, latent_state)
        
        self.optimizer.zero_grad()
        losses['total'].backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(self.world_model.observation_encoder.parameters()) +
            list(self.world_model.observation_decoder.parameters()) +
            list(self.world_model.dynamic_predictor.parameters()) +
            list(self.world_model.reward_predictor.parameters()),
            max_norm=self.max_grad_norm
        )
        
        self.optimizer.step()
        
        return {
            'total_loss': losses['total'].item(),
            'recon_loss': losses['reconstruction'].item(),
            'dynamics_loss': losses['dynamics'].item(),
            'reward_loss': losses['reward'].item()
        }
    
    def save_checkpoint(self, checkpoint_path, epoch):
        checkpoint = {
            'observation_encoder': self.world_model.observation_encoder.state_dict(),
            'observation_decoder': self.world_model.observation_decoder.state_dict(),
            'dynamic_predictor': self.world_model.dynamic_predictor.state_dict(),
            'reward_predictor': self.world_model.reward_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.world_model.observation_encoder.load_state_dict(checkpoint['observation_encoder'])
        self.world_model.observation_decoder.load_state_dict(checkpoint['observation_decoder'])
        self.world_model.dynamic_predictor.load_state_dict(checkpoint['dynamic_predictor'])
        self.world_model.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get('epoch', 0)


def train_world_model(env, observation_encoder, observation_decoder, dynamic_predictor, reward_predictor, ppo_policy, data_buffer, elite_buffer, valid_buffer, n_episodes, epochs, batch_size, seq_len, iterations_per_epoch, checkpoint_dir, lr=1e-4):

    world_model = WorldModel(observation_encoder=observation_encoder, observation_decoder=observation_decoder, dynamic_predictor=dynamic_predictor, reward_predictor=reward_predictor, latent_dim=2048).to(device)
    
    trainer = WorldModelTrainer(world_model, lr=lr)
    
    print("Collecting initial data...")
    regular_buffer, elite_buffer, threshold = collect_real_data_policy(
        env=env, 
        n_episodes=n_episodes, 
        data_buffer=data_buffer, 
        elite_buffer=elite_buffer,
        obs_encoder=observation_encoder, 
        dynamic_predictor=dynamic_predictor,
        device=device, 
        policy=ppo_policy,
        print_data=True
    )
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        for iteration in range(iterations_per_epoch):
            if len(elite_buffer) > 0 and np.random.rand() < 0.5:
                batch = elite_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)
            else:
                batch = regular_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)
            
            losses = trainer.update(batch)
            
            if iteration % 100 == 0:
                print(f"Iteration [{iteration}/{iterations_per_epoch}] | "
                      f"Recon: {losses['recon_loss']:.4f} | "
                      f"Dynamics: {losses['dynamics_loss']:.4f} | "
                      f"Reward: {losses['reward_loss']:.4f} | "
                      f"Total: {losses['total_loss']:.4f}")
        
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'world_model_epoch_{epoch+1}.pth'
        )
        trainer.save_checkpoint(checkpoint_path, epoch+1)
    
    print("\nTraining complete!")
    return world_model, trainer