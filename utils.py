from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F
import random


@dataclass
class RealTransitions:
    observations: torch.Tensor
    action: int
    reward: float
    next_observation: torch.Tensor
    done: bool

@dataclass
class ImaginedTransitions:
    encoded_observations: torch.Tensor
    hidden_state: torch.Tensor
    action: int
    predicted_reward: float
    predicted_next_encoded: torch.Tensor
    predicted_next_hidden: torch.Tensor
    done: bool
    value: float
    log_prob: float

class RealDataBuffer:
    def __init__(self, max_size=10000000):
        self.episodes = []
        self.max_size = max_size
        
    def add_episode(self, episode_transitions):
        self.episodes.append(episode_transitions)
        
        total_transitions = sum(len(ep) for ep in self.episodes)
        while total_transitions > self.max_size and len(self.episodes) > 1:
            removed_ep = self.episodes.pop(0)
            total_transitions -= len(removed_ep)

    def sample_sequences(self, batch_size, seq_len):
        sequences = []
        while len(sequences) != batch_size:
            if not self.episodes:
                break
            idx = np.random.randint(0, len(self.episodes))
            episode = self.episodes[idx]
            
            if len(episode) < seq_len:
                continue
            
            # start_idx = np.random.randint(0, len(episode) - seq_len + 1)
            start_idx = 0
            # print(f"Start index: {start_idx}")
            sequence = episode[start_idx:start_idx + seq_len]
            sequences.append(sequence)
        
        return sequences
    
    def clear(self):
        self.episodes.clear()
    
    def __len__(self):
        return sum(len(ep) for ep in self.episodes)


def collect_real_data_policy(env, n_episodes, data_buffer, elite_buffer,obs_encoder, dynamic_predictor, device, policy=None, elite_percentile=80, latent_dim=2048, print_data=False):
    all_episodes = []
    episode_scores = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_transitions = []

        total_abs_reward = 0

        obs_tensor = torch.tensor(obs).permute(2, 0, 1).contiguous().unsqueeze(0).float().to(device)

        with torch.no_grad():
            encoded_observation = obs_encoder(obs_tensor)

        latent_state = torch.zeros(1, 1, latent_dim, device=device)

        while not done:

            state_and_obs = torch.cat([latent_state.squeeze(0), encoded_observation], dim=-1)

            use_policy = (random.random() < 0.5)

            if use_policy and policy is not None:
                with torch.no_grad():
                    action, log_prob, value = policy.select_action(state_and_obs)
            else:
                action = env.action_space.sample()
                log_prob = None
                value = None

            next_obs, reward, done, info = env.step(action)
            total_abs_reward += abs(reward)

            next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).contiguous().unsqueeze(0).float().to(device)

            with torch.no_grad():
                next_encoded_observation = obs_encoder(next_obs_tensor)

            with torch.no_grad():
                action_onehot = F.one_hot(torch.tensor(action, dtype=torch.long, device=device), num_classes=env.action_space.n)
                obs_act_pair = torch.cat([encoded_observation.squeeze(0), action_onehot], dim=-1).unsqueeze(0)

                _, next_latent_state, _ = dynamic_predictor(obs_act_pair, latent_state)

            transition = RealTransitions(
                observations=obs_tensor.cpu(),
                action=action,
                reward=reward,
                next_observation=next_obs_tensor.cpu(),
                done=done
            )
            episode_transitions.append(transition)

            obs_tensor = next_obs_tensor
            encoded_observation = next_encoded_observation
            latent_state = next_latent_state.unsqueeze(0).detach()

        all_episodes.append(episode_transitions)
        episode_scores.append(total_abs_reward)
        data_buffer.add_episode(episode_transitions)

    threshold = np.percentile(episode_scores, elite_percentile)

    elite_count = 0
    for transitions, score in zip(all_episodes, episode_scores):
        if score >= threshold:
            elite_buffer.add_episode(transitions)
            elite_count += 1
    if print_data:
        print("\nReal Data Policy Collection Summary:")
        print(f"Total episodes collected: {n_episodes}")
        print(f"Elite episodes: {elite_count} (top {100-elite_percentile}% by absolute reward)")
        print(f"Elite threshold: {threshold:.2f}")

    return data_buffer, elite_buffer, threshold



class ImaginedBuffer:
    def __init__(self):
        self.episodes = []
        
    def add_episode(self, episode_transitions):
        self.episodes.append(episode_transitions)
        
    def sample_sequences(self, batch_size):
        sequences = []
        while len(sequences) != batch_size:
            if not self.episodes:
                break
            idx = np.random.randint(0, len(self.episodes))
            episode = self.episodes[idx]
                        
            sequences.append(episode)
        
        return sequences
    
    def clear(self):
        self.episodes.clear()
    
    def __len__(self):
        return sum(len(ep) for ep in self.episodes)


def collect_imagined_data_policy(env, n_episodes, data_buffer, obs_encoder, dynamic_predictor, reward_predictor, num_classes, device, horizon=50, policy=None, return_rewards=False):
    episode_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_transitions = []
        obs_act_pair_list = []
        obs_tensor = torch.Tensor(obs).permute(2, 0, 1).contiguous().unsqueeze(0)
        obs_tensor = obs_tensor.to(device).float() / 255.0
        latent_state = torch.zeros(1, 1, 2048, device=device)
        with torch.no_grad():
            initial_encoded_observation = obs_encoder(obs_tensor)
            initial_encoded_observation = initial_encoded_observation.to(device)
        total_reward = 0.0
        for h in range(horizon):
            if not episode_transitions:
                encoded_observation = initial_encoded_observation
                encoded_observation = encoded_observation.to(device)
            else:
                encoded_observation = next_observation
                encoded_observation = encoded_observation.unsqueeze(0).to(device)

            state_and_obs = torch.cat([latent_state.squeeze(0), encoded_observation], dim=-1)

            with torch.no_grad():
                action, log_prob, value = policy.select_action(state_and_obs)

            encoded_actions = F.one_hot(torch.tensor(action, dtype=torch.long, device=device), num_classes)
            obs_act_pair = torch.cat([encoded_observation.squeeze(0), encoded_actions], dim=0)
            obs_act_pair_list.append(obs_act_pair)
            obs_act_pair_stacked = torch.stack([obs_act for obs_act in obs_act_pair_list], dim=0)
            obs_act_pair_stacked = obs_act_pair_stacked.to(device)
            with torch.no_grad():
                next_observations, next_latent, _ = dynamic_predictor(obs_act_pair_stacked, latent_state, autoreg=True)
                rewards = reward_predictor(next_latent)
            next_observation = next_observations[-1]
            reward = rewards[-1]
            total_reward += reward.item()
            done = 1 if h==horizon-1 else 0
            if encoded_observation.dim() == 2:
                encoded_observation = encoded_observation.squeeze(0)
            transition = ImaginedTransitions(
                encoded_observations=encoded_observation.cpu(),
                hidden_state=latent_state.cpu(),
                action=action,
                predicted_reward=reward.item(),
                predicted_next_encoded=next_observation.cpu(),
                predicted_next_hidden=next_latent.cpu(),
                done=done,
                value=value,
                log_prob=log_prob.cpu()
            )
            latent_state=next_latent.unsqueeze(0).detach()
            episode_transitions.append(transition)
        data_buffer.add_episode(episode_transitions)
        episode_rewards.append(total_reward)
    
    if return_rewards:
        return data_buffer, episode_rewards
    else:
        return data_buffer
    

def get_observation_batch(batch):
    observation_batch = []
    next_observation_batch = []
    reward_batch = []
    observation_batch = torch.stack([b.observations.squeeze(0) for b in batch[0]], dim=0)
    next_observation_batch = torch.stack([b.next_observation.squeeze(0) for b in batch[0]], dim=0)
    reward_batch = torch.tensor([b.reward for b in batch[0]], dtype=torch.float32)

    return observation_batch, next_observation_batch, reward_batch


def get_observation_and_action_pair(batch, encoded_observations, num_classes):
    action_sequence = []
    device = encoded_observations.device
    action_sequence = torch.stack([F.one_hot(torch.tensor(a.action, dtype=torch.long), num_classes) for a in batch[0]], dim=0)
    action_sequence = action_sequence.to(device)
    observation_action_pair = torch.cat([encoded_observations, action_sequence], dim=1)

    return observation_action_pair


def get_imaginary_rollout_data(imagined_sequence):
    encoded_observation_batch = []
    hidden_state_batch = []
    action_batch = []
    predicted_reward_batch = []
    predicted_next_encoded_batch = []
    predicted_next_hidden_batch = []
    done_batch = []
    value_batch = []
    log_prob_batch = []

    encoded_observation_batch = torch.stack([b.encoded_observations for b in imagined_sequence[0]], dim=0)
    hidden_state_batch = torch.stack([b.hidden_state for b in imagined_sequence[0]], dim=0)
    action_batch = torch.tensor([b.action for b in imagined_sequence[0]], dtype=torch.int32)
    predicted_reward_batch = [b.predicted_reward for b in imagined_sequence[0]]
    predicted_next_encoded_batch = torch.stack([b.predicted_next_encoded for b in imagined_sequence[0]], dim=0)
    predicted_next_hidden_batch = torch.stack([b.predicted_next_hidden for b in imagined_sequence[0]], dim=0)
    done_batch = [b.done for b in imagined_sequence[0]]
    value_batch = [b.value for b in imagined_sequence[0]]
    log_prob_batch = torch.tensor([b.log_prob for b in imagined_sequence[0]], dtype=torch.float32)
    return encoded_observation_batch, hidden_state_batch, action_batch, predicted_reward_batch, predicted_next_encoded_batch, predicted_next_hidden_batch, done_batch, value_batch, log_prob_batch
