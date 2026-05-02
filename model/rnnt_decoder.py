import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional


class PredictionNetwork(nn.Module):
    """RNN-T Prediction Network (LSTM)"""
    hidden_dim: int = 1024
    num_layers: int = 2
    vocab_size: int = 5000
    dropout: float = 0.1

    @nn.compact
    def __call__(self, y, state=None, deterministic: bool = True):
        # y: (batch, seq_len) - target sequence (labels)
        embed = nn.Embed(self.vocab_size, self.hidden_dim)(y)
        hidden = embed
        new_states = []
        for i in range(self.num_layers):
            lstm_cell = nn.LSTMCell(features=self.hidden_dim)
            if state is not None and i < len(state):
                carry = state[i]
                hidden, new_carry = lstm_cell(carry, hidden)
            else:
                hidden, new_carry = lstm_cell(hidden)
            new_states.append(new_carry)
            hidden = nn.Dropout(rate=self.dropout)(hidden, deterministic=deterministic)
        return hidden, new_states


class JointNetwork(nn.Module):
    """RNN-T Joint Network"""
    joint_dim: int = 1024

    @nn.compact
    def __call__(self, encoder_out, pred_out):
        # encoder_out: (batch, time, model_dim)
        # pred_out: (batch, label_len, hidden_dim)
        encoder_proj = nn.Dense(self.joint_dim)(encoder_out)
        pred_proj = nn.Dense(self.joint_dim)(pred_out)
        # Broadcast to combine
        encoder_expanded = encoder_proj[:, :, None, :]  # (batch, time, 1, joint_dim)
        pred_expanded = pred_proj[:, None, :, :]  # (batch, 1, label_len, joint_dim)
        joint = encoder_expanded + pred_expanded
        joint = nn.relu(joint)
        output = nn.Dense(self.vocab_size + 1)(joint)  # +1 for blank
        return output


class RNNTDecoder(nn.Module):
    """RNN-T Decoder complete"""
    config: ModelConfig

    @nn.compact
    def __call__(self, encoder_out, targets, deterministic: bool = True):
        # encoder_out: (batch, time, model_dim)
        # targets: (batch, label_len) - text labels
        pred_net = PredictionNetwork(
            hidden_dim=self.config.decoder_hidden,
            num_layers=self.config.decoder_layers,
            vocab_size=self.config.vocab_size,
            dropout=0.1,
        )
        joint_net = JointNetwork(joint_dim=self.config.decoder_hidden)

        pred_out, _ = pred_net(targets, deterministic=deterministic)
        joint_out = joint_net(encoder_out, pred_out)
        return joint_out  # (batch, time, label_len, vocab_size+1)

    def compute_rnnt_loss(self, logits, targets, input_lengths, target_lengths):
        """Compute RNN-T loss (needs warp-transducer or custom implementation)"""
        # Placeholder: use cross-entropy as approximation
        loss = jax.nn.softmax_cross_entropy_with_logits(
            logits[..., :-1],  # remove blank token
            jax.nn.one_hot(targets, self.config.vocab_size)
        )
        return jnp.mean(loss)
