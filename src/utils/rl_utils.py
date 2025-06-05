import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

def nstep_returns(rewards, mask, values, nsteps, gamma=0.99, add_value_last_step=False):
    # nstep is a hyperparameter that regulates the number of look aheads
    # example 1: nsteps = 5, t_start = 0
    # R^5_0 = r_0 + (gamma*r_1) + (gamma**2*r_2) + (gamma**3*r_3) + (gamma**4*r_4) + (gamma**5*v_5)
    # example 2: nsteps = 5, t_start = 1
    # R^5_1 = r_1 + (gamma*r_2) + (gamma**2*r_3) + (gamma**3*r_4) + (gamma**4*r_5) + (gamma**5*v_6)
    nstep_values = th.zeros_like(values[:, :-1])
    for t_start in range(rewards.size(1)):
        nstep_return_t = th.zeros_like(values[:, 0])
        for step in range(nsteps + 1):
            t = t_start + step
            if t >= rewards.size(1):
                break
            elif step == nsteps:
                nstep_return_t += gamma ** step * values[:, t] * mask[:, t]
            elif t == rewards.size(1) - 1 and add_value_last_step:
                nstep_return_t += gamma ** step * rewards[:, t] * mask[:, t]
                # BUGFIX: adds last value only if episode wasn't finished
                nstep_return_t += gamma ** (step + 1) * values[:, t+1] * mask[:, t]
            else:
                nstep_return_t += gamma ** step * rewards[:, t] * mask[:, t]
        nstep_values[:, t_start, :] = nstep_return_t
    return nstep_values
