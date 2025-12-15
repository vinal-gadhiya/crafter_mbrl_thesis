import math

import torch
from torch import nn
from torch.nn import functional as F

class ObservationEncoder(nn.Module):
    def __init__(self, latent_dim=495):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # (B, 128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# (B, 256, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
            # nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class ObservationDecoder(nn.Module):
    def __init__(self, latent_dim=495):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (B, 128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # (B, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # (B, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # (B, 3, 64, 64)
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 256, 4, 4)
        return self.decoder(z)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Handle odd dimensions properly
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]



class TransformerAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_matrix = nn.Linear(d_model, d_model)
        self.key_matrix = nn.Linear(d_model, d_model)
        self.value_matrix = nn.Linear(d_model, d_model)
        self.final_projection = nn.Linear(d_model, d_model)

    def forward(self, sequence, attention_mask=None):
        sequence_length, model_dim = sequence.shape
        q = self.query_matrix(sequence)
        k = self.key_matrix(sequence)
        v = self.value_matrix(sequence)

        q = q.view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)

        q_k = torch.matmul(q, k.transpose(-1, -2))
        q_k = q_k/math.sqrt(self.head_dim)

        if attention_mask is not None:
            q_k = q_k + attention_mask
        attention_score = F.softmax(q_k, dim=-1)
        q_k_v = torch.matmul(attention_score, v)

        q_k_v = q_k_v.transpose(0, 1).contiguous()
        q_k_v = q_k_v.view(sequence_length, self.num_heads*self.head_dim)

        attention_output = self.final_projection(q_k_v)


        return attention_output


class NeuralNet(nn.Module):
    def __init__(self, d_model, hidden_layer_dim):
        super().__init__()
        self.d_model = d_model
        self.hidden_layer_dim = hidden_layer_dim
        self.fc1 = nn.Linear(in_features=self.d_model, out_features=self.hidden_layer_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_layer_dim, out_features=self.d_model)

    def forward(self, attention_output):
        fc1_output = self.fc1(attention_output)
        fc1_output = F.relu(fc1_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_layer_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_layer_dim = hidden_layer_dim

        self.transformer_attention = TransformerAttention(d_model=d_model, num_heads=num_heads)
        self.neural_net = NeuralNet(d_model=d_model, hidden_layer_dim=hidden_layer_dim)

        self.layer_normalization_attention = nn.LayerNorm(self.d_model)
        self.layer_normalization_neuralnet = nn.LayerNorm(self.d_model)

    @staticmethod
    def generate_attention_mask(sequence_length, device):
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def forward(self, sequence):
        sequence_length, _ = sequence.size()
        attention_mask = self.generate_attention_mask(sequence_length, sequence.device)
        attention_mask = attention_mask.unsqueeze(0)
        attention = self.transformer_attention(sequence, attention_mask)
        layer_normed_attention = self.layer_normalization_attention(sequence + attention)
        network_output = self.neural_net(layer_normed_attention)
        layer_normed_network_output = self.layer_normalization_neuralnet(layer_normed_attention + network_output)
        return layer_normed_network_output


class TransformersDecoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_layer_dim, num_blocks):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_layer_dim = hidden_layer_dim
        self.num_blocks = num_blocks

        self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(self.d_model, self.num_heads, self.hidden_layer_dim) for _ in range(self.num_blocks)])

    def forward(self, sequence):
        decoder_block_output = sequence
        for block in self.decoder_blocks:
            decoder_block_output = block(decoder_block_output)
        return decoder_block_output

class DynamicPredictor(nn.Module):
    def __init__(self, d_model, num_heads, hidden_layer_dim, num_blocks, latent_dim, action_dim, reward_dim=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_layer_dim = hidden_layer_dim
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.reward_dim=reward_dim
        
        self.positional_encoding = PositionalEncoding(d_model + action_dim, max_len=1000)
        
        self.decoder_blocks = TransformersDecoder(
            self.d_model + self.action_dim, 
            self.num_heads, 
            self.hidden_layer_dim, 
            self.num_blocks
        )
        
        self.state_update = nn.GRU(self.d_model + self.action_dim, self.latent_dim, batch_first=True)
        
        self.observation_projection = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_layer_dim, out_features=self.d_model),
        )

    def forward(self, past_observations, previous_latent_state, autoreg=False):
        past_observations = self.positional_encoding(past_observations)
        
        transformer_output = self.decoder_blocks(past_observations)
        
        if autoreg==False:
            gru_input = transformer_output.unsqueeze(0)  # (1, seq_len, d_model+action_dim)
        else:
            gru_input = transformer_output[-1:].unsqueeze(0)
        
        hidden_latent, final_latent = self.state_update(gru_input, previous_latent_state)
        
        hidden_latent = hidden_latent.squeeze(0)  # (seq_len, latent_dim)
        next_observation_prediction = self.observation_projection(hidden_latent)
        
        return next_observation_prediction, hidden_latent, final_latent
    

class RewardPredictor(nn.Module):
    def __init__(self, reward_dim, observation_dim, hidden_layer_dim):
        super().__init__()
        self.reward_dim = reward_dim
        self.observation_dim = observation_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.reward_network = nn.Sequential(
            nn.Linear(in_features=self.observation_dim, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=self.reward_dim)
            )

    def forward(self, observation_encoding):
        reward = self.reward_network(observation_encoding)
        return reward
    