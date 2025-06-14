import torch.nn as nn
import torch.nn.functional as F


from utils.rl_utils import conv2d_size_out


class CNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape

        # TODO: This is a dummy layer -- its here for backwards compatibility
        # self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        # if self.args.use_rnn:
        #     self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        # else:
        #     self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        # self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        h = conv2d_size_out(input_shape[0])
        w = conv2d_size_out(input_shape[1])
        self.cnn = nn.Sequential(
            nn.Conv2d(1, args.hidden_dim, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w * args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions),
            # DiscreteActionValueHead(),
        )

    def init_hidden(self):
        # make hidden states on same device as model
        pass

    def forward(self, inputs, *args):
        new_shape = (inputs.size(0), 1, -1, self.input_shape[-1])
        inputs = inputs.reshape(new_shape)
        inputs = inputs[:, :, : self.input_shape[0], :]
        q = self.cnn(inputs)
        return q
