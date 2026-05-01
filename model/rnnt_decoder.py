import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class PredictionNetwork(nn.Module):
    """RNN-T Prediction Network (LSTM)"""
    hidden_dim: int = 1024
    num_layers: int = 2
    vocab_size: int = 5000
    dropout: float = 0.1

    @nn.compact
    def __call__(self, y, state=None, deterministic: bool = True):
        # y: (batch, seq_len) - target sequence (nhãn)
        # Trả về: (batch, seq_len, hidden_dim), state mới
        embed = nn.Embed(self.vocab_size, self.hidden_dim)(y)
        lstm_states = []
        new_state = []
        for i in range(self.num_layers):
            lstm = nn.LSTMCell(features=self.hidden_dim)
            if state is not None:
                carry, out = lstm(state[i], embed)
            else:
                carry, out = lstm(embed)
            lstm_states.append(out)
            new_state.append(carry)
            embed = out
            embed = nn.Dropout(rate=self.dropout)(embed, deterministic=deterministic)
        return jnp.stack(lstm_states, axis=1), new_state


class JointNetwork(nn.Module):
    """RNN-T Joint Network"""
    joint_dim: int = 1024

    @nn.compact
    def __call__(self, encoder_out, pred_out):
        # encoder_out: (batch, time, model_dim)
        # pred_out: (batch, label_len, hidden_dim)
        # Trả về: (batch, time, label_len, vocab_size+1)
        encoder_out = nn.Dense(self.joint_dim)(encoder_out)
        pred_out = nn.Dense(self.joint_dim)(pred_out)
        # Broadcast để cộng
        encoder_out = encoder_out[:, :, None, :]  # (batch, time, 1, joint_dim)
        pred_out = pred_out[:, None, :, :]  # (batch, 1, label_len, joint_dim)
        joint = encoder_out + pred_out
        joint = nn.relu(joint)
        output = nn.Dense(self.vocab_size + 1)(joint)  # +1 cho blank
        return output


class RNNTDecoder(nn.Module):
    """RNN-T Decoder hoàn chỉnh"""
    config: ModelConfig

    @nn.compact
    def __call__(self, encoder_out, targets, deterministic: bool = True):
        # encoder_out: (batch, time, model_dim)
        # targets: (batch, label_len) - nhãn văn bản
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
        """Tính RNN-T loss (cần thư viện warp-transducer hoặc tự implement)"""
        # Placeholder: dùng cross-entropy đơn giản thay thế
        # Trong thực tế cần dùng jax_triton hoặc thư viện RNN-T chuyên dụng
        loss = jax.nn.softmax_cross_entropy_with_logits(
            logits[..., :-1],  # bỏ blank token
            jax.nn.one_hot(targets, self.config.vocab_size)
        )
        return jnp.mean(loss)
