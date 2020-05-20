import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb
from .plum_pudding import plumGauss

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_curiosity=False,
                 fwd_model=None,
                 inv_model=None,
                 curiosity_beta=0.2,
                 curiosity_lambda=0.1,
                 use_plum_pudding=False):

        self.actor_critic = actor_critic

        self.use_curiosity = use_curiosity
        self.use_plum_pudding = use_plum_pudding
        if use_curiosity:
            self.fwd_model = fwd_model
            self.inv_model = inv_model
            self.curiosity_beta = curiosity_beta
            self.curiosity_lambda = curiosity_lambda

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        if not self.use_curiosity:
            self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        else:
            self.optimizer = optim.Adam([{'params': actor_critic.parameters()},
                                         {'params': fwd_model.parameters()},
                                         {'params': inv_model.parameters()}],
                                        lr=lr, eps=eps)

    def update(self, rollouts, device):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        fwd_loss_epoch = 0
        inv_loss_epoch = 0


        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent or self.use_curiosity:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, T, N = sample

                # Reshape to do in a single forward pass for all steps
                # values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                #     obs_batch, recurrent_hidden_states_batch, masks_batch,
                #     actions_batch)

                if not self.use_curiosity:
                    values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch,
                        masks_batch, actions_batch)
                else:
                    values, action_log_probs, dist_entropy, states, actor_features = self.actor_critic.evaluate_actions_curiosity(
                        obs_batch, recurrent_hidden_states_batch,
                        masks_batch, actions_batch)


                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                
                if self.use_curiosity:
                    curr_states = actor_features.view(T, N, -1)[:-1].view((T-1)*N, -1)
                    next_states = actor_features.view(T, N, -1)[ 1:].view((T-1)*N, -1)
                    acts = actions_batch.view(T, N, -1)[:-1]
                    acts_one_hot = torch.zeros(T-1, N, self.actor_critic.n_actions).to(device)
                    acts_one_hot.scatter_(2, acts, 1)
                    acts_one_hot = acts_one_hot.view((T-1)*N, -1)
                    acts = acts.view(-1)
                    # Forward prediction loss
                    pred_next_states = self.fwd_model(curr_states.detach(), acts_one_hot)
                    # fwd_loss = 0.5*F.mse_loss(pred_next_states, next_states.detach())
                    fwd_loss = 0.5 * (pred_next_states - next_states.detach()).pow(2).mean()
                    # Inverse prediction loss
                    if self.use_plum_pudding:
                        pred_acts = self.inv_model(curr_states, self.use_plum_pudding)
                        inv_loss = plumGauss(torch.cat([curr_states, pred_acts], dim=1))
                    else:
                        pred_acts = self.inv_model(curr_states, next_states)
                        inv_loss = F.cross_entropy(pred_acts, acts.long())
                
                self.optimizer.zero_grad()
                # (value_loss * self.value_loss_coef + action_loss -
                #  dist_entropy * self.entropy_coef).backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                #                          self.max_grad_norm)
                if not self.use_curiosity:
                    (value_loss * self.value_loss_coef + action_loss -
                     dist_entropy * self.entropy_coef).backward()
                else:
                    pg_term = self.curiosity_lambda * (value_loss * self.value_loss_coef + 
                                                       action_loss - dist_entropy * self.entropy_coef)
                    curiosity_term = self.curiosity_beta*fwd_loss + (1-self.curiosity_beta)*inv_loss
                    overall_loss = pg_term + curiosity_term
                    overall_loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                if self.use_curiosity:
                    nn.utils.clip_grad_norm_(self.fwd_model.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.inv_model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                if self.use_curiosity:
                    fwd_loss_epoch += fwd_loss.item()
                    inv_loss_epoch += inv_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        if self.use_curiosity:
            fwd_loss_epoch /= num_updates
            inv_loss_epoch /= num_updates
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, fwd_loss_epoch, inv_loss_epoch
        # else:
        #     return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

        # return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
