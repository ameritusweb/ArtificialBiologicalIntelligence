import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastPathway(nn.Module):
    def __init__(self, obs_dim=169, hidden_dim=32, num_actions=22,
                 num_pain_channels=6, num_continuous=0):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_binary = num_actions - num_continuous
        self.shared = nn.Linear(obs_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.pain_pred_head = nn.Linear(hidden_dim, num_pain_channels)

    def forward(self, current_obs):
        h = F.relu(self.shared(current_obs))
        action_logits = self.action_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        pain_pred = self.pain_pred_head(h)
        return action_logits, confidence, pain_pred


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


def compute_obs_indices(num_limbs=6, num_segments=1, dims=2, continuous_actions=False):
    total_limbs = num_segments * num_limbs
    num_joints = num_segments - 1
    if continuous_actions:
        num_continuous = total_limbs * 2
        num_binary_muscle = total_limbs
        num_actions = num_continuous + num_binary_muscle + num_joints * 2 + 4
    else:
        num_continuous = 0
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
    grip_start = concept_start + 2
    thinking_start = grip_start + total_limbs + 3
    NUM_THINKING_CHANNELS = 6
    obs_dim = thinking_start + NUM_THINKING_CHANNELS
    L = total_limbs
    pain_start = 0
    endorphin_start = L
    temperature_start = 2 * L
    chemical_start = 3 * L
    pressure_start = 4 * L
    fatigue_start = 5 * L
    energy_idx = 6 * L
    ta_start = energy_idx + 1
    gain_start = ta_start + L
    pain_memory_start = gain_start + L
    distant_pain_start = pain_memory_start + 25
    distant_endo_start = distant_pain_start + 8
    pe_start = distant_endo_start + 8
    return {
        'pain': (pain_start, pain_start + L),
        'endorphin': (endorphin_start, endorphin_start + L),
        'temperature': (temperature_start, temperature_start + L),
        'chemical': (chemical_start, chemical_start + L),
        'pressure': (pressure_start, pressure_start + L),
        'fatigue': (fatigue_start, fatigue_start + L),
        'energy': energy_idx,
        'ta_start': ta_start,
        'gain_start': gain_start,
        'pain_memory_start': pain_memory_start,
        'distant_pain_start': distant_pain_start,
        'distant_endo_start': distant_endo_start,
        'pe_start': pe_start,
        'conflict': conflict_start,
        'core_obs_dim': core_obs_dim,
        'obs_dim': obs_dim,
        'num_actions': num_actions,
        'num_continuous': num_continuous,
        'mm_start': mm_start,
        'pattern_start': pattern_start,
        'proprio_start': proprio_start,
        'limb_dev_start': limb_dev_start,
        'efference_start': efference_start,
        'agency_start': agency_start,
        'obj_start': obj_start,
        'npc_start': npc_start,
        'opt_start': opt_start,
        'concept_start': concept_start,
        'grip_start': grip_start,
        'thinking_start': thinking_start,
        'num_thinking_channels': NUM_THINKING_CHANNELS,
        'total_limbs': total_limbs,
        'num_segments': num_segments,
        'num_joints': num_joints,
        'dims': dims,
    }


def build_stage_indices(num_limbs=6, num_segments=1, dims=2):
    idx = compute_obs_indices(num_limbs, num_segments, dims)
    L = idx['total_limbs']
    energy_idx = idx['energy']
    proprio_start = idx['pattern_start'] + 2
    extra_dims = dims - 2
    limb_dev_start = proprio_start + 2 + extra_dims
    efference_start = limb_dev_start + L
    agency_start = idx['agency_start']
    obj_start = agency_start + 3
    mm_start = idx['mm_start']
    pattern_start = idx['pattern_start']
    npc_start = obj_start + 6
    segment_start = npc_start + 12
    opt_start = segment_start + 2 * idx['num_joints']
    conflict_start = opt_start + 2 + extra_dims
    concept_start = conflict_start + 3
    grip_start = concept_start + 2

    stage1 = list(range(0, energy_idx + 1)) + list(range(proprio_start, proprio_start + 2 + extra_dims))
    stage2 = list(range(energy_idx + 1, mm_start))
    stage3 = list(range(limb_dev_start, obj_start + 6))
    stage4 = (list(range(mm_start, pattern_start + 2))
              + list(range(npc_start, grip_start + L + 3)))
    return [stage1, stage2, stage3, stage4]


class StagedInputProjection(nn.Module):
    EMA_DECAY = 0.99

    def __init__(self, obs_dim=169, d_model=64, num_limbs=6, num_segments=1, dims=2):
        super().__init__()
        self.d_model = d_model
        self.stage_indices = build_stage_indices(num_limbs, num_segments, dims)
        self.stage_dims = [len(s) for s in self.stage_indices]

        self.encoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        for i, sd in enumerate(self.stage_dims):
            input_dim = sd if i == 0 else sd + d_model
            self.encoders.append(nn.Linear(input_dim, d_model))
            if i < len(self.stage_dims) - 1:
                self.pred_heads.append(nn.Linear(d_model, self.stage_dims[i + 1]))
                self.register_buffer(f'_scale_{i}', torch.ones(self.stage_dims[i + 1]))

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        stages_raw = [flat[:, idx] for idx in self.stage_indices]

        predictions = []
        h = None
        for i, (raw, encoder) in enumerate(zip(stages_raw, self.encoders)):
            if h is None:
                inp = raw
            else:
                inp = torch.cat([raw, h], dim=-1)
            h_new = F.relu(encoder(inp))
            if h is not None:
                h_new = h_new + h
            if i < len(self.pred_heads):
                pred = self.pred_heads[i](h_new)
                actual = stages_raw[i + 1].detach()
                scale_buf = getattr(self, f'_scale_{i}')
                if self.training:
                    batch_scale = actual.abs().mean(dim=0).clamp(min=0.1)
                    scale_buf.mul_(self.EMA_DECAY).add_(batch_scale, alpha=1 - self.EMA_DECAY)
                scale = scale_buf.unsqueeze(0)
                predictions.append((pred / scale, actual / scale))
            h = h_new

        return h.reshape(shape[0], shape[1], self.d_model), predictions


class ArbitrationHead(nn.Module):
    NUM_GROUPS = 5

    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.NUM_GROUPS)

    def forward(self, obs):
        return 0.5 + torch.sigmoid(self.fc2(F.relu(self.fc1(obs))))


def _build_group_indices(num_limbs=6, num_segments=1, dims=2):
    idx = compute_obs_indices(num_limbs, num_segments, dims)
    L = idx['total_limbs']
    energy = idx['energy']
    mm_start = idx['mm_start']
    agency_start = idx['agency_start']
    concept_start = idx['concept_start']
    grip_start = idx['grip_start']
    obs_dim = idx['obs_dim']
    conflict = idx['conflict']
    npc_start = agency_start + 3 + 6
    return [
        list(range(0, L)) + list(range(energy + 1, energy + 1 + L)) + list(range(energy + 1 + 2*L, energy + 1 + 3*L)),
        list(range(L, 2*L)) + list(range(3*L, 4*L)),
        list(range(npc_start, npc_start + 12)),
        list(range(concept_start, concept_start + 2)),
        list(range(mm_start, mm_start + 6)) + list(range(agency_start, agency_start + 3)) + list(range(grip_start, obs_dim)),
    ]


class HierarchicalPolicy(nn.Module):
    def __init__(self, obs_dim=169, d_model=64, nhead=4, num_layers=2,
                 d_ff=128, seq_len=32, num_actions=22, energy_obs_index=36,
                 conflict_obs_index=155, num_pain_channels=6, num_limbs=6,
                 staged=False, num_segments=1, dims=2, num_continuous=0):
        super().__init__()
        self.seq_len = seq_len
        self.energy_obs_index = energy_obs_index
        self.conflict_obs_index = conflict_obs_index
        self.num_pain_channels = num_pain_channels
        self.obs_dim = obs_dim
        self.staged = staged
        self.num_continuous = num_continuous
        self.num_binary = num_actions - num_continuous

        self.arbitration = ArbitrationHead(obs_dim)
        self.group_indices = _build_group_indices(num_limbs, num_segments, dims)
        self.fast_pathway = FastPathway(obs_dim, hidden_dim=32, num_actions=num_actions,
                                       num_pain_channels=num_pain_channels,
                                       num_continuous=num_continuous)
        self.router = Router(hidden_dim=8)

        if staged:
            self.staged_proj = StagedInputProjection(obs_dim, d_model, num_limbs, num_segments, dims)
        else:
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
        if (self._causal_mask is None
                or self._causal_mask.size(0) != sz
                or self._causal_mask.device != device):
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(
                sz, device=device
            )
        return self._causal_mask

    def _slow_forward(self, x):
        B, S, _ = x.shape
        stage_predictions = None
        if self.staged:
            h, stage_predictions = self.staged_proj(x)
        else:
            h = self.input_proj(x)
        h = h + self.pos_encoding[:, :S, :]
        mask = self._get_causal_mask(S, h.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        last_hidden = h[:, -1, :]
        return self.slow_head(last_hidden), last_hidden, stage_predictions

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

        fast_logits, confidence, fast_pain_pred = self.fast_pathway(current_weighted)
        gate = self.router(confidence, energy, conflict)
        slow_logits, slow_hidden, stage_predictions = self._slow_forward(x_weighted)
        blended = (1 - gate) * fast_logits + gate * slow_logits
        slow_pain_pred = self.prediction_head(slow_hidden)
        predicted_next_pain = (1 - gate) * fast_pain_pred + gate * slow_pain_pred

        result = {
            'blended': blended,
            'fast_logits': fast_logits,
            'slow_logits': slow_logits,
            'gate': gate,
            'confidence': confidence,
            'predicted_next_pain': predicted_next_pain,
            'fast_pain_pred': fast_pain_pred,
            'slow_pain_pred': slow_pain_pred,
            'arb_weights': arb_weights,
        }
        if stage_predictions is not None:
            result['stage_predictions'] = stage_predictions
        return result

    def predict(self, observation_window):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.FloatTensor(observation_window).unsqueeze(0).to(device)

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

            fast_logits, confidence, fast_pain_pred = self.fast_pathway(current_weighted)
            gate = self.router(confidence, energy, conflict)
            gate_val = gate.item()
            conf_val = confidence.item()

            if gate_val >= 0.3:
                slow_logits, slow_hidden, stage_preds = self._slow_forward(x_weighted)
                slow_pain_pred = self.prediction_head(slow_hidden)
                blended = (1 - gate) * fast_logits + gate * slow_logits
                pred_pain = ((1 - gate) * fast_pain_pred + gate * slow_pain_pred).squeeze(0).cpu().numpy()
            else:
                blended = fast_logits
                pred_pain = fast_pain_pred.squeeze(0).cpu().numpy()
                stage_preds = None

            nc = self.num_continuous
            if nc > 0:
                continuous = torch.tanh(blended[:, :nc]).squeeze(0).cpu().numpy()
                binary = (torch.sigmoid(blended[:, nc:]) > 0.5).int().squeeze(0).cpu().numpy()
                actions = np.concatenate([continuous, binary])
            else:
                actions = (torch.sigmoid(blended) > 0.5).int().squeeze(0).cpu().numpy()
            arb = arb_weights.squeeze(0).cpu().numpy()

        info = {
            'gate_value': round(gate_val, 4),
            'confidence': round(conf_val, 4),
            'used_slow': gate_val >= 0.3,
            'predicted_next_pain': [round(float(p), 4) for p in pred_pain],
            'arb_weights': [round(float(w), 4) for w in arb],
        }
        if stage_preds is not None:
            info['stage_prediction_errors'] = [
                round(F.mse_loss(p, a).item(), 6)
                for p, a in stage_preds
            ]
        return actions, info


if __name__ == '__main__':
    dummy = torch.randn(4, 32, 169)

    for mode_name, staged in [('FLAT', False), ('STAGED', True)]:
        print(f"\n{'='*50}")
        print(f"  {mode_name} MODE")
        print(f"{'='*50}")
        model = HierarchicalPolicy(staged=staged)
        total = sum(p.numel() for p in model.parameters())

        fast_params = sum(p.numel() for p in model.fast_pathway.parameters())
        router_params = sum(p.numel() for p in model.router.parameters())
        arb_params = sum(p.numel() for p in model.arbitration.parameters())
        pred_params = sum(p.numel() for p in model.prediction_head.parameters())

        if staged:
            proj_params = sum(p.numel() for p in model.staged_proj.parameters())
            print(f"Staged proj:   {proj_params:,} params ({proj_params/total*100:.1f}%)")
            print(f"  Stage dims:  {model.staged_proj.stage_dims}")
        else:
            proj_params = sum(p.numel() for p in model.input_proj.parameters())
            print(f"Input proj:    {proj_params:,} params ({proj_params/total*100:.1f}%)")

        print(f"Fast pathway:  {fast_params:,} params ({fast_params/total*100:.1f}%)")
        print(f"Router:        {router_params:,} params ({router_params/total*100:.1f}%)")
        print(f"Arbitration:   {arb_params:,} params ({arb_params/total*100:.1f}%)")
        print(f"Pred head:     {pred_params:,} params ({pred_params/total*100:.1f}%)")
        print(f"Total:         {total:,} params")

        result = model(dummy)
        print(f"\nInput:      {dummy.shape}")
        print(f"Blended:    {result['blended'].shape}")
        print(f"Pred pain:  {result['predicted_next_pain'].shape}")
        print(f"Gate:       {result['gate'].shape} mean={result['gate'].mean().item():.3f}")
        print(f"Confidence: {result['confidence'].shape} mean={result['confidence'].mean().item():.3f}")
        print(f"Arb weights: {result['arb_weights'].shape} mean={result['arb_weights'].mean().item():.3f}")

        if 'stage_predictions' in result:
            print(f"\nStage predictions ({len(result['stage_predictions'])} boundaries):")
            for i, (pred, actual) in enumerate(result['stage_predictions']):
                mse = F.mse_loss(pred, actual).item()
                print(f"  Stage {i+1}->{i+2}: pred {pred.shape}, actual {actual.shape}, MSE={mse:.6f}")
