import torch.nn as nn
import torch.nn.functional as F
import torch as th


from utils.rl_utils import conv2d_size_out


# TODO: Make it optinal to receive a linear embedding layer
class BiGRU(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(BiGRU, self).__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        h = conv2d_size_out(input_shape[0])
        w = conv2d_size_out(input_shape[1])
        self.obs_dim = input_shape[0] * input_shape[1]

        # embedding layer
        # can this handle multiple timesteps?
        self.cnn = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            2 * hidden_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def get_embedding(self, inputs, bs=None, max_t=None):
        if bs is None:
            bs = inputs.shape[0]
        if max_t is None:
            max_t = inputs.shape[1]

        # Foward pass
        new_shape = (bs, max_t) + self.input_shape
        # prune
        inputs = inputs[:bs, :max_t, : self.obs_dim].reshape(new_shape)
        # splits over timesteps
        x = th.tensor_split(inputs, max_t, dim=1)
        # Applies embeddings over time steps preserving batches
        x = th.stack([*map(self.cnn, x)], dim=1)  # [bs, max_t, hidden_dim]
        return x

    def forward(self, inputs, other=None):
        x = self.get_embedding(inputs)

        other = x if other is None else other
        x = th.cat([x, other], dim=-1)

        x, h = self.rnn(x)
        v = self.fc(x)
        return v
