import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from utils import get_imaginary_rollout_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(observation_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_network_output = self.actor(state)
        action_probabilities = torch.softmax(action_network_output, dim=-1)
        state_value = self.critic(state)
        return action_probabilities, state_value
    
    def act(self, state):
        action_probabilities, state_value = self.forward(state)
        act_prob_distribution = Categorical(action_probabilities)
        action = act_prob_distribution.sample()
        log_prob = act_prob_distribution.log_prob(action)
        return action.item(), log_prob, state_value
    
    def evaluate(self, states, actions):
        action_probabilities, state_values = self.forward(states)
        act_prob_distribution = Categorical(action_probabilities)
        log_probs = act_prob_distribution.log_prob(actions)
        entropy = act_prob_distribution.entropy()
        return log_probs, state_values, entropy


class PPO(nn.Module):
    def __init__(self, observation_dim, action_dim, lr=1e-4, gamma=0.99, epsilon_clip=0.2, k_epochs=4, gae_lambda=0.95, clip_value=False):
        super().__init__()
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.clip_value = clip_value

        self.policy = ActorCritic(observation_dim=observation_dim, action_dim=action_dim, hidden_dim=64)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(observation_dim=observation_dim, action_dim=action_dim, hidden_dim=64)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy_loss = nn.SmoothL1Loss()

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, state_value = self.policy_old.act(state=state)
        return action, log_prob, state_value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0

        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t+1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        return advantages
    
    def update(self, imagined_rollout):
        encoded_observation_batch, hidden_state_batch, action_batch, predicted_reward_batch, predicted_next_encoded_batch, predicted_next_hidden_batch, done_batch, value_batch, log_prob_batch = get_imaginary_rollout_data(imagined_sequence=imagined_rollout)
        last_state = torch.cat([predicted_next_hidden_batch[-1], predicted_next_encoded_batch[-1].unsqueeze(0)], dim=-1)
        with torch.no_grad():
            _, next_value = self.policy_old(last_state.to(device))
            next_value = next_value.item()
        
        advantages = self.compute_gae(rewards=predicted_reward_batch, values=value_batch, dones=done_batch, next_value=next_value)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        returns = advantages + torch.tensor(value_batch, dtype=torch.float32, device=device)

        state_obs_pair = torch.cat([hidden_state_batch.squeeze(1).squeeze(1), encoded_observation_batch], dim=-1)
        log_probs, state_values, entropy = self.policy.evaluate(states=state_obs_pair.to(device), actions=action_batch.to(device))
        state_values = state_values.squeeze()

        ratios = torch.exp(log_probs.to(device) - log_prob_batch.to(device))

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()

        if self.clip_value==False:
            critic_loss = 0.5 * self.policy_loss(state_values, returns)
        
        else:
            value_batch_tensor = torch.tensor(value_batch, dtype=torch.float32, device=device)
        
            value_pred_clipped = value_batch_tensor + torch.clamp(state_values - value_batch_tensor, -self.epsilon_clip, self.epsilon_clip
            )
            
            value_losses = (state_values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            
            critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        entropy_loss = -0.01 * entropy.mean()

        loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()


        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()
    
    def update_policy_old(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
