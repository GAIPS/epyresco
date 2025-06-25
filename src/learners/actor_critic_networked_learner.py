import copy
from operator import itemgetter
from functools import partial

import numpy as np
import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class ActorCriticNetworkedLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

        # consensus evaluations
        # TODO: allow only neighbors to be connected.
        def fn(x):
            return th.from_numpy(x.astype(np.float32))

        self.consensus_matrices = args.consensus_matrices
        self.consensus_parameter_names = self._get_critic_parameter_names()
        self.consensus_rounds = self.args.n_consensus_steps
        self.consensus_with_embeddings = False  # BiGRU only

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error(
                "Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env)
            )
            return

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        pi = mac_out
        advantages, critic_train_stats = self.train_critic_sequential(
            self.critic, self.target_critic, batch, rewards, critic_mask
        )
        actions = actions[:, :-1]
        advantages = advantages.detach()
        # Calculate policy grad with mask

        pi[mask == 0] = 1.0

        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)

        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        pg_loss = (
            -(
                (advantages * log_pi_taken + self.args.entropy_coef * entropy) * mask
            ).sum()
            / mask.sum()
        )

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.agent_params, self.args.grad_norm_clip
        )
        self.agent_optimiser.step()

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            if self.consensus_with_embeddings:
                # Get embeddings from network
                embeddings = critic.get_embeddings(batch)

                stacked_weights = self._get_critic_parameters(critic.critics)

                consensus_parameters_step = partial(
                    self._consensus_step_parameters, stacked_weights
                )

                # [b, t, n, e] -> [n, e, b, t]
                embeddings = embeddings.permute((2, 3, 0, 1))
                # Grab the hidden state from GRU
                for cwm in self._get_consensus_matrices():
                    # Consensus on the embeddings.
                    embeddings = th.einsum("nm, mijk-> nijk", cwm, embeddings)

                    # Consensus on the parameters
                    consensus_parameters_step(cwm)

                embeddings = embeddings.permute((2, 3, 0, 1))  # [b, t, n, e]
                self._update_critic_parameters(critic, stacked_weights)

                target_vals = target_critic(batch, embeddings)
            else:
                target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

            if self.args.standardise_returns:
                target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

            target_returns = self.nstep_returns(
                rewards, mask, target_vals, self.args.q_nstep
            )  # [b, t, n]

            # Perform consensus
            target_returns = target_returns.permute((2, 0, 1))
            # Grab the hidden state from GRU
            for cwm in self._get_consensus_matrices():
                target_returns = th.einsum("nm, mij-> nij", cwm, target_returns)
            target_returns = target_returns.permute((1, 2, 0))  # [b, t, n]

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }
        if self.consensus_with_embeddings:
            v = critic(batch, embeddings)[:, :-1].squeeze(3)
        else:
            v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )
        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma**step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma**step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma**step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _get_consensus_matrices(self):
        # Hear communication channels for this timestep
        indices = np.random.randint(
            0, high=len(self.consensus_matrices), size=self.consensus_rounds
        )
        return [self.consensus_matrices[ind] for ind in indices]

    def _get_critic_parameter_names(self):
        # Critics parameters for consensus (minus embeddings layer)
        # FIXME: This is sensive to critic's architecture
        a_critic = self.critic.critics[0]
        return [k for k, v in a_critic.named_parameters() if "rnn" in k or "fc" in k]

    def _get_critic_parameters(self, critics):
        # Collect weights
        weights = []
        for crit in critics:
            weights.append({
                k: v
                for k, v in crit.named_parameters()
                if k in self.consensus_parameter_names
            })
        # Stack weights
        stacked_weights = {}
        for weight_name in self.consensus_parameter_names:
            stacked_weights[weight_name] = th.stack(
                [*map(itemgetter(weight_name), weights)], dim=0
            )
        return stacked_weights

    def _consensus_step_parameters(self, stacked_weights, consensus_weights):
        for name, weight in stacked_weights.items():
            if "weight" in name:
                w = th.einsum("nm, mij-> nij", consensus_weights, weight)
            elif "bias" in name:
                w = th.einsum("nm, mi-> ni", consensus_weights, weight)
            else:
                raise ValueError(f"Unknwon weight type {name}")
            stacked_weights[name] = w

    def _update_critic_parameters(self, critic, stacked_weights):
        # Unstack weights
        weights = [{} for _ in range(self.n_agents)]
        for name, weight in stacked_weights.items():
            for i, w in enumerate(th.tensor_split(weight, self.n_agents, dim=0)):
                weights[i][name] = w.squeeze(0)

        # Assign weights
        for i, crit in enumerate(critic.critics):
            for name, param in crit.named_parameters():
                if name not in self.consensus_parameter_names:
                    continue
                param.data = th.nn.parameter.Parameter(weights[i][name])

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
