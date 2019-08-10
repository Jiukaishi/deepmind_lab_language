
# coding: utf-8

# In[ ]:


"""Functions to compute V-trace off-policy actor critic targets.
For details and theory see:
"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.
See https://arxiv.org/abs/1802.01561 for the full paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import torch.nn.functional as F


def log_probs_from_logits_and_actions(policy_logits, actions):
    """Computes action log-probs from policy logits and actions.
    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions.
    Args:
      policy_logits: A float32 tensor of shape [T, NUM_ACTIONS, B] with
        un-normalized log-probabilities parameterizing a softmax policy.
      actions: An int32 tensor of shape [T, B] with actions.
    Returns:
      A float32 tensor of shape [T, B] corresponding to the sampling log
      probability of the chosen action w.r.t. the policy.
    """
    # policy_logits = tf.convert_to_tensor(policy_logits, dtype=tf.float32)
    # actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    assert len(policy_logits.shape) == 3
    assert len(actions.shape) == 2

    return -F.cross_entropy(policy_logits, actions, reduction='none')


def from_logits(behaviour_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, bootstrap_value,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
    r"""V-trace for softmax policies.
    Calculates V-trace actor critic targets for softmax polices as described in
    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.
    Target policy refers to the policy we are interested in improving and
    behaviour policy refers to the policy that generated the given
    rewards and actions.
    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions.
    Args:
      behaviour_policy_logits: A float32 tensor of shape [T, NUM_ACTIONS, B] with
        un-normalized log-probabilities parametrizing the softmax behaviour
        policy.
      target_policy_logits: A float32 tensor of shape [T, NUM_ACTIONS, B] with
        un-normalized log-probabilities parametrizing the softmax target policy.
      actions: An int32 tensor of shape [T, B] of actions sampled from the
        behaviour policy.
      discounts: A float32 tensor of shape [T, B] with the discount encountered
        when following the behaviour policy.
      rewards: A float32 tensor of shape [T, B] with the rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
    Returns:
      A `VTraceFromLogitsReturns` namedtuple with the following fields:
        vs: A float32 tensor of shape [T, B]. Can be used as target to train a
            baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float 32 tensor of shape [T, B]. Can be used as an
          estimate of the advantage in the calculation of policy gradients.
        log_rhos: A float32 tensor of shape [T, B] containing the log importance
          sampling weights (log rhos).
        behaviour_action_log_probs: A float32 tensor of shape [T, B] containing
          behaviour policy action log probabilities (log \mu(a_t)).
        target_action_log_probs: A float32 tensor of shape [T, B] containing
          target policy action probabilities (log \pi(a_t)).
    """
    # Make sure tensor ranks are as expected.
    # The rest will be checked by from_action_log_probs.
    assert len(behaviour_policy_logits.shape) == 3
    assert len(target_policy_logits.shape) == 3
    assert len(actions.shape) == 2

    target_action_log_probs = log_probs_from_logits_and_actions(
        target_policy_logits, actions)
    behaviour_action_log_probs = log_probs_from_logits_and_actions(
        behaviour_policy_logits, actions)
    log_rhos = target_action_log_probs - behaviour_action_log_probs
    vs, pg_advantages = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)
    return vs, pg_advantages


def from_importance_weights(
        log_rhos, discounts, rewards, values, bootstrap_value,
        clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
    r"""V-trace from log importance weights.
    Calculates V-trace actor critic targets as described in
    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.
    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions. This code also supports the
    case where all tensors have the same number of additional dimensions, e.g.,
    `rewards` is [T, B, C], `values` is [T, B, C], `bootstrap_value` is [B, C].
    Args:
      log_rhos: A float32 tensor of shape [T, B] representing the log
        importance sampling weights, i.e.
        log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
        on rhos in log-space for numerical stability.
      discounts: A float32 tensor of shape [T, B] with discounts encountered when
        following the behaviour policy.
      rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper. If None, no clipping is applied.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
        None, no clipping is applied.
    Returns:
      A VTraceReturns namedtuple (vs, pg_advantages) where:
        vs: A float32 tensor of shape [T, B]. Can be used as target to
          train a baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
          advantage in the calculation of policy gradients.
    """

    if clip_rho_threshold is not None:
        clip_rho_threshold = torch.tensor(clip_rho_threshold, dtype=torch.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = torch.tensor(clip_pg_rho_threshold, dtype=torch.float32)

    # Make sure tensor ranks are consistent.
    rho_rank = len(log_rhos.shape)  # Usually 2.
    assert len(values.shape) == rho_rank
    assert len(bootstrap_value.shape) == rho_rank - 1
    assert len(discounts.shape) == rho_rank
    assert len(rewards.shape) == rho_rank

    if clip_rho_threshold is not None:
        assert len(clip_rho_threshold.shape) == 0
    if clip_pg_rho_threshold is not None:
        assert len(clip_pg_rho_threshold.shape) == 0

    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.min(clip_rho_threshold, rhos)
        else:
            clipped_rhos = rhos

        cs = torch.min(torch.ones_like(rhos), rhos)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat((values, bootstrap_value.unsqueeze(0)), dim=0)
        # deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        # Note that all sequences are reversed, computation starts from the back.
        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        seq_len = discounts.shape[0]
        vs = []
        for i in range(seq_len):
            v_s = values[i].clone()
            for j in range(i, seq_len):
                v_s += (torch.prod(discounts[i:j], dim=0) * torch.prod(cs[i:j], dim=0) * clipped_rhos[j] *
                        (rewards[j] + discounts[j] * values_t_plus_1[j + 1] - values[j]))
            vs.append(v_s)
        vs = torch.stack(vs, dim=0)
        # Advantage for policy gradient.
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.min(clip_pg_rho_threshold, rhos)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = (
                clipped_pg_rhos * (rewards + discounts * torch.cat(
            (vs[1:], bootstrap_value.unsqueeze(0)), dim=0) - values))

        # Make sure no gradients backpropagated through the returned values.
        return vs, pg_advantages
