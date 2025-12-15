import torch
import csv
import os
import numpy as np
import gym
import crafter

from world_model import WorldModel, WorldModelTrainer
from policy import PPO
from networks import (
    ObservationEncoder, ObservationDecoder, 
    DynamicPredictor, RewardPredictor
)
from utils import (
    RealDataBuffer, ImaginedBuffer, 
    collect_real_data, collect_imagined_data_policy
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlternatingTrainer:
    
    def __init__(self, env, world_model_trainer, ppo_policy, checkpoint_dir):
        self.env = env
        self.world_model_trainer = world_model_trainer
        self.ppo_policy = ppo_policy
        self.checkpoint_dir = checkpoint_dir
        
        self.data_buffer = RealDataBuffer()
        self.elite_data_buffer = RealDataBuffer()
        self.imagined_buffer = ImaginedBuffer()
        
        self.log_file_path = os.path.join(checkpoint_dir, "training_log.csv")
        self._initialize_log()
    
    def _initialize_log(self):
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", 
                    "avg_policy_loss", "avg_actor_loss", 
                    "avg_critic_loss", "avg_entropy_loss",
                    "avg_recon_loss", "avg_dynamics_loss", "avg_reward_loss"
                ])
    
    def _log_metrics(self, epoch, policy_metrics, wm_metrics):
        with open(self.log_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                policy_metrics['policy_loss'],
                policy_metrics['actor_loss'],
                policy_metrics['critic_loss'],
                policy_metrics['entropy_loss'],
                wm_metrics['recon_loss'],
                wm_metrics['dynamics_loss'],
                wm_metrics['reward_loss']
            ])
    
    def train_policy_phase(self, n_imagined_episodes, n_imagined_iterations, batch_size):
        print("\n" + "="*60)
        print("POLICY TRAINING PHASE")
        print("="*60)
        
        print(f"Collecting {n_imagined_episodes} imagined episodes...")
        self.imagined_buffer = collect_imagined_data_policy(
            env=self.env,
            n_episodes=n_imagined_episodes,
            data_buffer=self.imagined_buffer,
            obs_encoder=self.world_model_trainer.world_model.observation_encoder,
            dynamic_predictor=self.world_model_trainer.world_model.dynamic_predictor,
            reward_predictor=self.world_model_trainer.world_model.reward_predictor,
            num_classes=17,
            device=device,
            policy=self.ppo_policy
        )
        
        total_policy_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        
        print(f"Training policy for {n_imagined_iterations} iterations...")
        for iteration in range(n_imagined_iterations):
            # Sample imagined sequence
            imagined_sequence = self.imagined_buffer.sample_sequences(batch_size=batch_size)
            
            policy_loss, actor_loss, critic_loss, entropy_loss = self.ppo_policy.update(
                imagined_rollout=imagined_sequence
            )
            
            total_policy_loss += policy_loss
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy_loss += entropy_loss
            
            if (iteration + 1) % 1000 == 0:
                print(f"Iteration [{iteration+1}/{n_imagined_iterations}] | "
                      f"Policy Loss: {policy_loss:.4f} | "
                      f"Actor: {actor_loss:.4f} | "
                      f"Critic: {critic_loss:.4f} | "
                      f"Entropy: {entropy_loss:.4f}")
        
        self.ppo_policy.update_policy_old()
        
        self.imagined_buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_imagined_iterations,
            'actor_loss': total_actor_loss / n_imagined_iterations,
            'critic_loss': total_critic_loss / n_imagined_iterations,
            'entropy_loss': total_entropy_loss / n_imagined_iterations
        }
    
    def train_world_model_phase(self, n_wm_episodes, n_wm_iterations, batch_size, seq_len):
        print("\n" + "="*60)
        print("WORLD MODEL TRAINING PHASE")
        print("="*60)
        
        print(f"Collecting {n_wm_episodes} real episodes...")
        regular_buffer, elite_buffer = collect_real_data(
            env=self.env,
            n_episodes=n_wm_episodes,
            data_buffer=self.data_buffer,
            elite_buffer=self.elite_data_buffer,
            obs_encoder=self.world_model_trainer.world_model.observation_encoder,
            device=device,
            policy=self.ppo_policy
        )
        
        total_recon_loss = 0.0
        total_dynamics_loss = 0.0
        total_reward_loss = 0.0
        
        print(f"Training world model for {n_wm_iterations} iterations...")
        for iteration in range(n_wm_iterations):
            if len(elite_buffer) > 0 and np.random.rand() < 0.5:
                batch = elite_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)
            else:
                batch = regular_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)
            
            losses = self.world_model_trainer.update(batch)
            
            total_recon_loss += losses['recon_loss']
            total_dynamics_loss += losses['dynamics_loss']
            total_reward_loss += losses['reward_loss']
            
            if (iteration + 1) % 500 == 0:
                print(f"Iteration [{iteration+1}/{n_wm_iterations}] | "
                      f"Recon: {losses['recon_loss']:.4f} | "
                      f"Dynamics: {losses['dynamics_loss']:.4f} | "
                      f"Reward: {losses['reward_loss']:.4f}")
        
        regular_buffer.clear()
        elite_buffer.clear()
        
        return {
            'recon_loss': total_recon_loss / n_wm_iterations,
            'dynamics_loss': total_dynamics_loss / n_wm_iterations,
            'reward_loss': total_reward_loss / n_wm_iterations
        }
    
    def save_checkpoints(self, epoch):
        wm_checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"world_model_{epoch+1}.pth"
        )
        self.world_model_trainer.save_checkpoint(wm_checkpoint_path, epoch+1)
        
        policy_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"ppo_policy_{epoch+1}.pth"
        )
        torch.save({
            'policy_state': self.ppo_policy.policy.state_dict(),
            'policy_old_state': self.ppo_policy.policy_old.state_dict(),
            'optimizer_state': self.ppo_policy.optimizer.state_dict(),
            'epoch': epoch + 1,
        }, policy_checkpoint_path)
        
        print(f"\nCheckpoints saved at epoch {epoch+1}")
    
    def load_checkpoints(self, wm_checkpoint_path=None, policy_checkpoint_path=None):
        if wm_checkpoint_path and os.path.exists(wm_checkpoint_path):
            epoch = self.world_model_trainer.load_checkpoint(wm_checkpoint_path)
            print(f"World model loaded from epoch {epoch}")
        
        if policy_checkpoint_path and os.path.exists(policy_checkpoint_path):
            checkpoint = torch.load(policy_checkpoint_path, map_location=device)
            self.ppo_policy.policy.load_state_dict(checkpoint['policy_state'])
            self.ppo_policy.policy_old.load_state_dict(checkpoint['policy_old_state'])
            self.ppo_policy.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"Policy loaded from epoch {checkpoint.get('epoch', 0)}")
    
    def train(self, n_wm_episodes, n_wm_iterations, n_imagined_episodes, 
              n_imagined_iterations, epochs, batch_size, seq_len, 
              save_every=10, start_epoch=0):
        print("\n" + "="*80)
        print("STARTING ALTERNATING TRAINING")
        print("="*80)
        print(f"Configuration:")
        print(f"  Total Epochs: {epochs}")
        print(f"  Starting from Epoch: {start_epoch + 1}")
        print(f"  World Model Episodes/Epoch: {n_wm_episodes}")
        print(f"  World Model Iterations/Epoch: {n_wm_iterations}")
        print(f"  Imagined Episodes/Epoch: {n_imagined_episodes}")
        print(f"  Imagined Iterations/Epoch: {n_imagined_iterations}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Sequence Length: {seq_len}")
        print(f"  Save Every: {save_every} epochs")
        print("="*80)
        
        for epoch in range(start_epoch, epochs):
            print(f"\n{'#'*80}")
            print(f"EPOCH {epoch+1}/{epochs}")
            print(f"{'#'*80}")
            
            policy_metrics = self.train_policy_phase(
                n_imagined_episodes=n_imagined_episodes,
                n_imagined_iterations=n_imagined_iterations,
                batch_size=batch_size
            )
            
            print("\n" + "-"*60)
            print("Policy Training Summary:")
            print(f"Avg Policy Loss: {policy_metrics['policy_loss']:.4f}")
            print(f"Avg Actor Loss: {policy_metrics['actor_loss']:.4f}")
            print(f"Avg Critic Loss: {policy_metrics['critic_loss']:.4f}")
            print(f"Avg Entropy Loss: {policy_metrics['entropy_loss']:.4f}")
            print("-"*60)
            
            wm_metrics = self.train_world_model_phase(
                n_wm_episodes=n_wm_episodes,
                n_wm_iterations=n_wm_iterations,
                batch_size=batch_size,
                seq_len=seq_len
            )
            
            print("\n" + "-"*60)
            print("World Model Training Summary:")
            print(f"Avg Reconstruction Loss: {wm_metrics['recon_loss']:.4f}")
            print(f"Avg Dynamics Loss: {wm_metrics['dynamics_loss']:.4f}")
            print(f"Avg Reward Loss: {wm_metrics['reward_loss']:.4f}")
            print("-"*60)
            
            self._log_metrics(epoch, policy_metrics, wm_metrics)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoints(epoch)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE - Saving final checkpoints")
        print("="*80)
        self.save_checkpoints(epochs - 1)
        
        print("\nTraining finished successfully!")
        print(f"Logs saved to: {self.log_file_path}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def main():
    env = gym.make('CrafterReward-v1')
    env = crafter.Recorder(
        env, 'logdir',
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    
    checkpoint_dir = "checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    
    observation_encoder = ObservationEncoder().to(device)
    observation_decoder = ObservationDecoder().to(device)
    dynamic_predictor = DynamicPredictor(
        d_model=495, 
        num_heads=8, 
        hidden_layer_dim=1024, 
        num_blocks=4, 
        latent_dim=2048, 
        action_dim=17
    ).to(device)
    reward_predictor = RewardPredictor(
        reward_dim=1, 
        observation_dim=2048, 
        hidden_layer_dim=64
    ).to(device)
    
    world_model = WorldModel(
        observation_encoder=observation_encoder,
        observation_decoder=observation_decoder,
        dynamic_predictor=dynamic_predictor,
        reward_predictor=reward_predictor,
        latent_dim=2048
    )
    
    world_model_trainer = WorldModelTrainer(
        world_model=world_model,
        lr=1e-4,
        beta_recon=1.0,
        beta_dynamics=1.0,
        beta_reward=0.1
    )
    
    ppo_policy = PPO(observation_dim=2543, action_dim=17).to(device)
    
    trainer = AlternatingTrainer(
        env=env,
        world_model_trainer=world_model_trainer,
        ppo_policy=ppo_policy,
        checkpoint_dir=checkpoint_dir
    )

    config = {
        'n_wm_episodes': 200,           # Episodes to collect for WM training
        'n_wm_iterations': 2500,        # Training iterations per epoch for WM
        'n_imagined_episodes': 1000,    # Imagined episodes for policy training
        'n_imagined_iterations': 10000, # Training iterations per epoch for policy
        'epochs': 100,                  # Total training epochs
        'batch_size': 1,                # Batch size
        'seq_len': 50,                  # Sequence length for WM training
        'save_every': 10,               # Save checkpoint every N epochs
        'start_epoch': 0                # Starting epoch (for resuming)
    }
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    trainer.train(**config)


if __name__ == "__main__":
    main()