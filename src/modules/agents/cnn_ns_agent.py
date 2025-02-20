import torch.nn as nn
from modules.agents.cnn_agent import CNNAgent
import torch as th
from IPython.core.debugger import set_trace


class CNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNNSAgent, self).__init__()
        self.args = args
        self.max_shape = max(map(lambda x: x[0] * x[1], input_shape))
        self.n_agents = args.n_agents
        self.agents = th.nn.ModuleList([CNNAgent(sh, args) for sh in input_shape])

    def init_hidden(self):
        # make hidden states on same device as model
        pass

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q = self.agents[i](inputs[i].unsqueeze(0))
                qs.append(q)
            return th.cat(qs)
        else:
            inputs = inputs.view(-1, self.n_agents, self.max_shape)
            for i in range(self.n_agents):
                q = self.agents[i](inputs.select(1, i))
                qs.append(q.unsqueeze(1))
            # return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)
            return th.cat(qs, dim=-1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
