from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.networks import MLPNetwork
from models.ActorNet import ActorNet
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, epsilon = 0.5, discrete_action=True, USE_CUDA = True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = ActorNet((num_in_pol,), num_out_pol)
        self.critic = ActorNet((num_in_critic,), 1)
        self.target_policy = ActorNet((num_in_pol,), num_out_pol)
        self.target_critic = ActorNet((num_in_critic,), 1)

        if USE_CUDA:
            self.critic.cuda()
            self.policy.cuda()
            self.target_critic.cuda()
            self.target_policy.cuda()

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.epsilon = epsilon  # epsilon for eps-greedy

    def step(self, obs):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        return action.cpu().detach().numpy()

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
