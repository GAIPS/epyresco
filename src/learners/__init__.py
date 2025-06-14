from .q_learner import QLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_networked_learner import ActorCriticNetworkedLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["actor_critic_networked_learner"] = ActorCriticNetworkedLearner
