import torch
import torch.nn as nn
import torch.nn.functional as F


class FastPathway(nn.Module):
    def __init__(self, obs_dim=160, hidden_dim=32, num_actions=22):
        super().__init__()
        self.shared = nn.Linear(obs_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, current_obs):
        h = F.relu(self.shared(current_obs))
        action_logits = self.action_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        return action_logits, confidence


class Router(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.fc2.bias, -1.0)

    def forward(self, confidence, energy, conflict):
        x = torch.cat([confidence, energy, conflict], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


ENERGY_OBS_INDEX = 36
CONFLICT_OBS_INDEX = 155


def compute_obs_indices(num_limbs=6, num_segments=1, dims=2):
    total_limbs = num_segments * num_limbs
    num_joints = num_segments - 1
    num_actions = total_limbs * 3 + num_joints * 2 + 4
    core_obs_dim = 9 * total_limbs + 42
    extra_dims = dims - 2
    mm_start = core_obs_dim
    pattern_start = mm_start + 4
    proprio_start = pattern_start + 2
    limb_dev_start = proprio_start + 2 + extra_dims
    efference_start = limb_dev_start + total_limbs
    agency_start = efference_start + num_actions
    obj_start = agency_start + 3
    npc_start = obj_start + 6
    segment_start = npc_start + 12
    opt_start = segment_start + 2 * num_joints
    dim3_start = opt_start + 2
    conflict_start = dim3_start + extra_dims
    concept_start = conflict_start + 3
    obs_dim = concept_start + 2
    return {
        'energy': 6 * total_limbs,
        'conflict': conflict_start,
        'core_obs_dim': core_obs_dim,
        'obs_dim': obs_dim,
        'num_actions': num_actions,
        'mm_start': mm_start,
        'pattern_start': pattern_start,
        'agency_start': agency_start,
        'total_limbs': total_limbs,
        'num_segments': num_segments,
        'num_joints': num_joints,
        'dims': dims,
    }


class ArbitrationHead(nn.Module):
    NUM_GROUPS = 5

    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.NUM_GROUPS)

    def forward(self, obs):
        return 0.5 + torch.sigmoid(self.fc2(F.relu(self.fc1(obs))))


def _build_group_indices(obs_dim, num_limbs=6):
    L = num_limbs
    return [
        list(range(0, L)) + list(range(6*L+1, 7*L+1)) + list(range(8*L+42, 9*L+42)),
        list(range(L, 2*L)) + list(range(3*L, 4*L)),
        list(range(obs_dim - 16, obs_dim - 4)),
        [obs_dim - 4, obs_dim - 3],
        list(range(9*L+42, 9*L+48)) + list(range(obs_dim - 16 - 9, obs_dim - 16 - 6)) + list(range(obs_dim - 3, obs_dim)),
    ]


class HierarchicalPolicy(nn.Module):
    def __init__(self, obs_dim=160, d_model=64, nhead=4, num_layers=2,
                 d_ff=128, seq_len=32, num_actions=22, energy_obs_index=36,
                 conflict_obs_index=155, num_pain_channels=6, num_limbs=6):
        super().__init__()
        self.seq_len = seq_len
        self.energy_obs_index = energy_obs_index
        self.conflict_obs_index = conflict_obs_index
        self.num_pain_channels = num_pain_channels
        self.obs_dim = obs_dim

        self.arbitration = ArbitrationHead(obs_dim)
        self.group_indices = _build_group_indices(obs_dim, num_limbs)
        self.fast_pathway = FastPathway(obs_dim, hidden_dim=32, num_actions=num_actions)
        self.router = Router(hidden_dim=8)

        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.slow_head = nn.Linear(d_model, num_actions)
        self.prediction_head = nn.Linear(d_model, num_pain_channels)
        self._causal_mask = None

    def _get_causal_mask(self, sz, device):
        if self._causal_mask is None or self._causal_mask.size(0) != sz:
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(
                sz, device=device
            )
        return self._causal_mask

    def _slow_forward(self, x):
        B, S, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :S, :]
        mask = self._get_causal_mask(S, h.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        last_hidden = h[:, -1, :]
        return self.slow_head(last_hidden), last_hidden

    def forward(self, x):
        current_obs = x[:, -1, :]
        energy = current_obs[:, self.energy_obs_index:self.energy_obs_index + 1]
        conflict = current_obs[:, self.conflict_obs_index:self.conflict_obs_index + 1]

        arb_weights = self.arbitration(current_obs)
        x_weighted = x.clone()
        for g, indices in enumerate(self.group_indices):
            valid = [i for i in indices if i < self.obs_dim]
            if valid:
                x_weighted[:, :, valid] = x[:, :, valid] * arb_weights[:, g:g+1].unsqueeze(1)
        current_weighted = x_weighted[:, -1, :]

        fast_logits, confidence = self.fast_pathway(current_weighted)
        gate = self.router(confidence, energy, conflict)
        slow_logits, slow_hidden = self._slow_forward(x_weighted)
        blended = (1 - gate) * fast_logits + gate * slow_logits
        predicted_next_pain = self.prediction_head(slow_hidden)

        return {
            'blended': blended,
            'fast_logits': fast_logits,
            'slow_logits': slow_logits,
            'gate': gate,
            'confidence': confidence,
            'predicted_next_pain': predicted_next_pain,
            'arb_weights': arb_weights,
        }

    def predict(self, observation_window):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.FloatTensor(observation_window).unsqueeze(0).to(device)
            result = self.forward(x)
            gate_val = result['gate'].item()
            conf_val = result['confidence'].item()
            actions = (torch.sigmoid(result['blended']) > 0.5).int()
            pred_pain = result['predicted_next_pain'].squeeze(0).cpu().numpy()
            arb = result['arb_weights'].squeeze(0).cpu().numpy()

        return actions.squeeze(0).cpu().numpy(), {
            'gate_value': round(gate_val, 4),
            'confidence': round(conf_val, 4),
            'used_slow': gate_val >= 0.3,
            'predicted_next_pain': [round(float(p), 4) for p in pred_pain],
            'arb_weights': [round(float(w), 4) for w in arb],
        }


if __name__ == '__main__':
    model = HierarchicalPolicy()

    fast_params = sum(p.numel() for p in model.fast_pathway.parameters())
    router_params = sum(p.numel() for p in model.router.parameters())
    slow_params = (
        sum(p.numel() for p in model.input_proj.parameters())
        + model.pos_encoding.numel()
        + sum(p.numel() for p in model.transformer.parameters())
        + sum(p.numel() for p in model.slow_head.parameters())
    )
    total = sum(p.numel() for p in model.parameters())
    pred_params = sum(p.numel() for p in model.prediction_head.parameters())
    print(f"Fast pathway:  {fast_params:,} params ({fast_params/total*100:.1f}%)")
    print(f"Router:        {router_params:,} params ({router_params/total*100:.1f}%)")
    print(f"Slow pathway:  {slow_params:,} params ({slow_params/total*100:.1f}%)")
    print(f"Pred head:     {pred_params:,} params ({pred_params/total*100:.1f}%)")
    print(f"Total:         {total:,} params")

    dummy = torch.randn(4, 32, 160)
    result = model(dummy)
    print(f"\nInput:      {dummy.shape}")
    print(f"Blended:    {result['blended'].shape}")
    print(f"Pred pain:  {result['predicted_next_pain'].shape}")
    print(f"Gate:       {result['gate'].shape} mean={result['gate'].mean().item():.3f}")
    print(f"Confidence: {result['confidence'].shape} mean={result['confidence'].mean().item():.3f}")
    print(f"Arb weights: {result['arb_weights'].shape} mean={result['arb_weights'].mean().item():.3f}")
    arb_params = sum(p.numel() for p in model.arbitration.parameters())
    print(f"Arbitration: {arb_params:,} params")
