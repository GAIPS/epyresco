from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .cnn_agent import CNNAgent
from .cnn_ns_agent import CNNNSAgent

REGISTRY = {}
REGISTRY["cnn"] = CNNAgent
REGISTRY["cnn_ns"] = CNNNSAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
