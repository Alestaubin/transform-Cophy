import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from derendering.model import DeRendering
import ipdb
import logging
# from transformer import TrajectoryEncoder
import math 

logging.basicConfig(
    level=logging.INFO,
    format='[%(filename)s:%(lineno)d in %(funcName)s()] %(message)s'
)

# Wrapper function (optional, but matches your request)
def log_msg(message):
    logging.info(message)

'''
K : number of objects
T : number of timesteps
B : batch size
H : hidden dimension

'''
"""
shapes / notation:
  B = batch size
  K = number of objects (can be padded/masked)
  T = sequence length (tau)
  D_in = dimensionality of object embedding after GCN
  D_model = transformer model dim (>= D_in)
"""

class PerObjectTemporalEncoder(nn.Module):
    def __init__(self, d_in, H=32,
                 d_model=128, nhead=4, nlayers=3, dim_feedforward=256, dropout=0.1,
                 max_T=100):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.H = H
        # Project input (GCN outputs) to transformer dimension
        self.input_proj = nn.Linear(d_in, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # learnable temporal positional embeddings (for T timesteps)
        self.pos_embed = nn.Parameter(torch.randn(1, max_T + 1, d_model))  

        # learnable CLS token (size d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # map transformer output (CLS token) -> confounder dim H (same as GRU hidden)
        self.out_proj = nn.Linear(d_model, H)

        # optional small readout after projecting to H
        self.readout = nn.Sequential(
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Linear(H, H)
        )
        self._init_weights()
    
    def _init_weights(self):
        print("Initializing weights...")
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, gcn_outputs, valid_mask=None):
        """
        gcn_outputs: (B, T, K, d_in)
            Every element in the batch is a sequence of T timesteps. At each timestep,
            there are K objects, each represented by a d_in-dimensional vector (output of GCN).
        valid_mask: optional (B, K, T) boolean mask where True indicates valid timestep
        Returns: (B, K, H)
        """
        B, T, K, d_in = gcn_outputs.shape
        assert d_in == self.d_in, f"Expected last dim {self.d_in}, got {d_in}"

        # Project inputs to d_model
        # -> (B, T, K, d_model)
        x = self.input_proj(gcn_outputs)

        # Prepare positional embeddings for T timesteps (+ CLS)
        if T + 1 > self.pos_embed.shape[1]:
            raise ValueError(f"Sequence length T+1={T+1} exceeds max pos length {self.pos_embed.shape[1]}")

        # Add temporal positional embeddings to the timestep tokens (not CLS)
        # pos_embed has shape (1, max_T+1, d_model) — we'll take indices 1: T+1 for timesteps
        pos_for_time = self.pos_embed[:, 1:(T+1), :].unsqueeze(2)  # (1, T, 1, d_model)
        # every object at time t gets the same positional embedding
        # similarly, positional embeddings are shared across the batch
        x = x + pos_for_time  # broadcast to (B, T, K, d_model)

        # Move object axis into batch to run transformer over each object's time series in parallel
        # First, reshape to (B*K, T, d_model):
        #   The transformer does not care which object belongs to which demonstration in the batch. It just wants a list of sequences.
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, K, T, d_model)
        x = x.view(B * K, T, self.d_model)      # (B*K, T, d_model)

        # CLS is a learnable parameter of shape (1, 1, d_model)
        # First, duplicate this token so every sequence in the batch gets one
        cls_tokens = self.cls_token.expand(B * K, -1, -1)  # (B*K, 1, d_model)
        # Prepend CLS token for each (B*K) sequence
        x_with_cls = torch.cat([cls_tokens, x], dim=1)     # (B*K, T+1, d_model)

        # Build src_key_padding_mask if valid_mask is provided.
        # Transformer expects src_key_padding_mask shape (batch, seq) with True for positions that should be masked (i.e., PADDING)
        # if valid_mask is not None:
        #     # valid_mask expected shape (B, K, T) (True = valid)
        #     vm = valid_mask.view(B * K, T)                 # (B*K, T)
        #     # we prepended CLS at position 0, which should NOT be masked, so pad a False
        #     cls_pad = torch.zeros(B * K, 1, dtype=torch.bool, device=vm.device)
        #     padding_mask = torch.cat([cls_pad, ~vm], dim=1)  # True where padding (including for time)
        # else:
            
        padding_mask = None

        # Run transformer: input shape (batch, seq, d_model)
        # Note: src_key_padding_mask True values are positions that will be ignored.
        z = self.transformer(x_with_cls, src_key_padding_mask=padding_mask)  # (B*K, T+1, d_model)
        
        # The [CLS] token (index 0) attends to all time steps. 
        # By the end of the layers, the vector at index 0 has aggregated information from the entire video sequence.
        # It now contains a compressed summary of the entire physics interaction, with a notion of time also (via positional embeddings).
        # Take CLS output (index 0)
        cls_out = z[:, 0, :]  # (B*K, d_model)

        # Note: The Gradient Bottleneck: Because we only use [CLS] to predict the Confounders (U), 
        # all the gradients must flow back through the [CLS] token.

        # Project to confounder dimension H and readout
        u = self.out_proj(cls_out)   # (B*K, H)
        u = self.readout(u)          # (B*K, H)

        # Reshape back to (B, K, H)
        u = u.view(B, K, self.H)

        return u


  # End of PerObjectTemporalEncoder    
class CopyC(nn.Module):
  '''
  This is a baseline that just copies the C frame T times
  '''
  def __init__(self, num_objects=5):
    super().__init__()

    self.derendering = DeRendering(num_objects)

  def forward(self, rgb_ab, rgb_c):

    # derender
    presence_ab, presence_c, \
    pose_3d_ab, pose_3d_c = extract_pose_ab_c(self.derendering, rgb_ab, rgb_c)
    T = rgb_ab.shape[1]

    # copy c T times
    pose_3d_d = pose_3d_c.repeat(1,T-1,1,1)

    return pose_3d_d, presence_c, None

def aggreg_E(E, presence):
  list_e = []
  K = E.size(1)
  presence = presence.unsqueeze(-1)
  presence_1 = presence.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
  presence_2 = presence.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
  presence_12 = presence_1 * presence_2
  for k in range(K):
    other_k = [x for x in range(K) if x != k]
    e_k = (E[:,k,other_k] * presence_12[:,k,other_k]).sum(1) # (B,H)
    e_k = e_k /  (0.01+presence_12[:,k,other_k].sum(1))
    list_e.append(e_k)
  E = torch.stack(list_e, 1) # (B,K,H)

  # E_sum
  E_sum = (E * presence).sum(1) / (0.01 + presence.sum(1)) # (B,H)
  E_sum = E_sum.unsqueeze(1).repeat(1,K,1)

  return E, E_sum

class CoPhyNet(nn.Module):
  def __init__(self,
               num_objects=5,
               encoder_type='rnn',
               ):
    super().__init__()
    self.encoder_type = encoder_type
    # CNN
    self.derendering = DeRendering(num_objects)
    self.K = num_objects

    # AB
    H = 32 # Hidden state dimension
    self.H = H
    self.mlp_inter = nn.Sequential(nn.Linear(2*(3),H),
                                   nn.ReLU(),
                                   nn.Linear(H,H),
                                   nn.ReLU(),
                                   nn.Linear(H,H),
                                   nn.ReLU(),
                                   )
    D = H
    self.D = D
    self.mlp_out = nn.Sequential(nn.Linear(3+H+H, H),
                                 nn.ReLU(),
                                 nn.Linear(H, H))

    # RNN
    if self.encoder_type == 'rnn':
      self.rnn = nn.GRU(input_size=D, hidden_size=H, num_layers=1, batch_first=True)
    elif self.encoder_type == 'transformer':
      self.transformer_encoder = PerObjectTemporalEncoder(d_in = D,
                                                          d_model=128,
                                                          H = H,
                                                          nhead=8, 
                                                          nlayers=2, 
                                                          dim_feedforward=512, 
                                                          dropout=0.1)
      
    
    # Stability
    self.mlp_inter_stab = nn.Sequential(nn.Linear(2*(H+3),H),
                                        nn.ReLU(),
                                        nn.Linear(H,H),
                                        nn.ReLU(),
                                        nn.Linear(H,H),
                                        nn.ReLU(),
                                        )
    self.mlp_stab = nn.Sequential(nn.Linear(H+H+H+3, H),
                                  nn.ReLU(),
                                  nn.Linear(H, 1))

    # Next position
    self.mlp_inter_delta = nn.Sequential(nn.Linear(2*(H+3),H),
                                         nn.ReLU(),
                                         nn.Linear(H,H),
                                         nn.ReLU(),
                                         nn.Linear(H,H),
                                         nn.ReLU(),
                                         )
    self.mlp_gcn_delta = nn.Sequential(nn.Linear(H*3 + 3, H),
                                       nn.ReLU(),
                                       nn.Linear(H, H))
    self.rnn_delta = nn.GRU(H,H, num_layers=1, batch_first=True)
    self.fc_delta = nn.Linear(H, 3)


    # args
    self.iterative_stab = True


  def gcn_on_AB(self, pose_ab, presence_ab):
    list_out = []
    K = pose_ab.size(2)
    T = pose_ab.size(1)
    for i in range(T):
      x = pose_ab[:,i,:,:3] # (B,4,3)

      # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
      x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
      x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
      x_12 = torch.cat([x_1, x_2], -1)
      E = self.mlp_inter(x_12) # B,K,K,H
      E, E_sum = aggreg_E(E, presence_ab) # B,K,H

      # next position : o_t+1^1 = f(o_t^1,e_t+1)
      out = self.mlp_out(torch.cat([x, E, E_sum], -1))
      list_out.append(out)

    out = torch.stack(list_out, 1) # (B,T,K,H)

    return out # (B,T,K,3)

  def rnn_on_AB_up(self, seq_o, object_type=None):
    # log_msg(f"object_type: {object_type}")
    # log_msg(f"seq_o shape: {seq_o.shape}") # (B,T,K,H) [32, 30, 9, 32]
    if object_type is not None:
      T = seq_o.shape[1]
      object_type = object_type.unsqueeze(2).repeat(1,1,T,1)

    K = seq_o.size(2)
    list_out = []
    for k in range(K):
      x = seq_o[:,:,k]

      if object_type is not None:
        x = torch.cat([x, object_type[:,k]], -1)

      out, _ = self.rnn(x) # (B,T,H) [32, 30, 32]
      list_out.append(out[:,-1]) # this takes the last hidden state

    # one confounder encoding per object
    # the latent representation of the confounders is the last hidden state of the RNN
    out = torch.stack(list_out, 1) # (B,K,H) [32, 9, 32]
    return out
  
  def transformer_on_AB_up(self, seq_o, object_type=None):
      """
      seq_o: (B, T, K, H_in)  where H_in == self.D (32)
      Currently we do not concatenate object_type here (kept simple).
      Returns: (B, K, H) where H == self.H (32)
      """
      # Optional: if object_type support is needed
      # e.g.
      # if object_type is not None:
      #     # object_type expected shape (B, K, otype_dim)
      #     T = seq_o.shape[1]
      #     ot_rep = object_type.unsqueeze(2).repeat(1,1,T,1)  # (B,K,T,otype_dim)
      #     seq_o = torch.cat([seq_o, ot_rep.permute(0,2,1,3)], dim=-1)  # careful with dims

      u = self.transformer_encoder(seq_o)  # (B, K, H)
      return u
  
  def pred_stab(self, confounders, pose_t, presence):
    """
    Given a timestep - predict the stability per object
    :param confounders: (B,K,D)
    :param pose_t: (B,K,3)
    :param presence: (B,K)
    :return: stab=(B,K,1)
    """
    list_stab = []
    # x = input['pose_cd'][:,0] # (B,4,3)
    x = pose_t # (B,4,3)
    x = torch.cat([confounders, x], -1)
    K = x.size(1)

    # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
    x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
    x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
    x_12 = torch.cat([x_1, x_2], -1)
    E = self.mlp_inter_stab(x_12) # B,K,K,H
    # E, E_sum = aggreg_E(E, input['presence_cd']) # B,K,H
    E, E_sum = aggreg_E(E, presence) # B,K,H

    # stability
    stab = self.mlp_stab(torch.cat([x, E, E_sum], -1))

    return stab # (B,K,1)

  def pred_D(self, confounders, pose_3d_c, presence_c, T=30):
    """

    :param stability: (B,K,1)
    :param confounders: (B,K,D)
    :param input: pose_cd=(B,T,K,3) presence_cd=(B,K)
    :return: out=(B,10,K,3) stability=(B,10,4,1)
    """
    list_pose = []
    list_stability = []
    pose = pose_3d_c # (B,4,3)
    K = pose.size(1)
    list_last_hidden = []
    for i in range(T):
      # Stability prediction
      if i == 0 or self.iterative_stab == 'true':
        stability = self.pred_stab(confounders, pose, presence_c)
      list_stability.append(stability)

      # Cat
      x = torch.cat([pose, confounders], -1) # # .detach() ???

      # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
      x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
      x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
      x_12 = torch.cat([x_1, x_2], -1)
      E = self.mlp_inter_delta(x_12) # B,K,K,H
      E, E_sum = aggreg_E(E, presence_c) # B,K,H

      # next position : o_t+1^1 = f(o_t^1,e_t+1) with RNN on top
      _in = self.mlp_gcn_delta(torch.cat([x, E, E_sum], -1)) # (B,K,H)
      B = _in.size(0)
      list_new_hidden = []
      for k in range(K):
        if i == 0:
          hidden, *_ = self.rnn_delta(_in[:,[k]]) # (B,1,H)
        else:
          hidden, *_ = self.rnn_delta(_in[:,[k]], list_last_hidden[k].reshape(1,B,-1)) # (B,1,H)
        list_new_hidden.append(hidden)
      list_last_hidden = list_new_hidden
      hidden = torch.cat(list_last_hidden, 1) # (B,K,H)

      delta = self.fc_delta(hidden)

      if self.training:
        alpha = 0.01
        delta = delta * (1 - torch.sigmoid(stability/alpha)) # .detach() ???
      else:
        delta = delta * (1-(stability > 0).float())
      pose = pose + delta

      list_pose.append(pose)

    pose = torch.stack(list_pose, 1) # (B,T,K,3)
    stability = torch.stack(list_stability, 1) # (B,T,K,3)

    return pose, stability

  def forward(self, rgb_ab, rgb_c,
    pred_presence_ab=None, pred_pose_3d_ab=None,
    pred_presence_c=None, pred_pose_3d_c=None,
    pred_obj_type_ab=None, pred_obj_type_c=None,
  ):

    if rgb_ab is not None and rgb_c is not None:
      # derender
      presence_ab, presence_c, \
      pose_3d_ab, pose_3d_c = extract_pose_ab_c(self.derendering, rgb_ab, rgb_c)
    else:
      # already precomputed
      presence_ab = pred_presence_ab
      presence_c = pred_presence_c
      pose_3d_ab = pred_pose_3d_ab
      pose_3d_c = pred_pose_3d_c

    # squeeze
    pose_3d_c = pose_3d_c[:,0] # (B,K,3)
    T = pose_3d_ab.shape[1] - 1

    # Run a GCN on AB
    seq_o = self.gcn_on_AB(pose_3d_ab, presence_ab) # (B,T,K,H)

    # Run a RNN on the outputs of GCN
    if self.encoder_type == 'rnn':
      confounders = self.rnn_on_AB_up(seq_o, pred_obj_type_ab) # (B,K,H) [32, 9, 32]
    elif self.encoder_type == 'transformer':
      confounders = self.transformer_on_AB_up(seq_o) # (B,K,H) [32, 9, 32]
    # print("confounders shape:", confounders.shape)
    # pred
    out, stability = self.pred_D(confounders, pose_3d_c, presence_c, T=T)
    # stability = (B,T-1,K,1) 
    # stability is a binary indicator of whether the object is stable (not moving) at time t
    stability = stability.squeeze(-1)

    return out, presence_c, stability


def extract_pose_ab_c(derendering, rgb_ab, rgb_c):
  rgb = torch.cat([rgb_ab, rgb_c], 1)
  B, T, H, W, C = rgb.shape
  rgb = rgb.view(B*T, H, W, C)
  presence, pose_3d, pose_2d = derendering(rgb)
  presence = presence.view(B, T, derendering.num_objects)
  presence = (presence > 0).float()
  presence_ab, presence_c = presence[:, 0], presence[:, -1] # TODO maybe avg over time for AB
  pose_3d = pose_3d.view(B, T, derendering.num_objects, 3)
  pose_3d_ab, pose_3d_c = pose_3d[:, :-1], pose_3d[:, -1:]

  return presence_ab, presence_c, pose_3d_ab, pose_3d_c