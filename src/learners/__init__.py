from .q_learner import QLearner
from .actor_critic_learner import ActorCriticLearner


REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
