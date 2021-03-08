'''This file is extended from pyro.distributions to define distributions used in Revive.'''

import torch
from torch.functional import F
import pyro
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.distributions.kl import register_kl, kl_divergence

class ReviveDistributionMixin:
    '''Define revive distribution API'''

    @property
    def mode(self,):
        '''return the most likely sample of the distributions'''
        raise NotImplementedError 

    @property
    def std(self):
        '''return the standard deviation of the distributions'''
        raise NotImplementedError

    def sample_with_logprob(self, sample_shape=torch.Size()):
        sample = self.rsample(sample_shape) if self.has_rsample else self.sample(sample_shape)
        return sample, self.log_prob(sample)

class ReviveDistribution(pyro.distributions.TorchDistribution, ReviveDistributionMixin):
    pass

class DiagnalNormal(ReviveDistribution):
    def __init__(self, loc, scale, validate_args=None):
        self.base_dist = pyro.distributions.Normal(loc, scale, validate_args)
        batch_shape = torch.Size(loc.shape[:-1])
        event_shape = torch.Size([loc.shape[-1]])
        super(DiagnalNormal, self).__init__(batch_shape, event_shape, validate_args)

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, sample):
        log_prob = self.base_dist.log_prob(sample)
        return torch.sum(log_prob, dim=-1)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return torch.sum(entropy, dim=-1)

    def shift(self, mu_shift):
        '''shift the distribution, useful in local mode transition'''
        return DiagnalNormal(self.base_dist.loc + mu_shift, self.base_dist.scale)

    @property
    def mode(self):
        return self.base_dist.mean

    @property
    def std(self):
        return self.base_dist.scale

class TransformedDistribution(torch.distributions.TransformedDistribution):
    @property
    def mode(self):
        x = self.base_dist.mode
        for transform in self.transforms:
            x = transform(x)
        return x

    @property
    def std(self):
        raise NotImplementedError # TODO: fix this!
    
    def entropy(self, num=torch.Size([100])):
        # use samples to estimate entropy
        samples = self.rsample(num)
        log_prob = self.log_prob(samples)
        entropy = - torch.mean(log_prob, dim=0)
        return entropy

class Onehot(torch.distributions.OneHotCategorical, TorchDistributionMixin, ReviveDistributionMixin):
    """Differentiable Onehot Distribution"""

    has_rsample = True

    def __init__(self, logits):
        """logits -> tensor[*, N]"""
        super(Onehot, self).__init__(logits=logits)

    def rsample(self, sample_shape=torch.Size()):
        # Implement straight-through estimator
        # Bengio et.al. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation 
        sample = self.sample(sample_shape)
        return sample + self.probs - self.probs.detach()
    
    @property
    def mode(self):
        index = torch.argmax(self.logits, dim=-1)
        sample = F.one_hot(index, self.event_shape[0])
        return sample + self.probs - self.probs.detach()

    @property
    def std(self):
        return self.variance

class GaussianMixture(pyro.distributions.MixtureOfDiagNormals, ReviveDistributionMixin):
    @property
    def mode(self):
        # NOTE: this is only an approximate mode
        which = self.categorical.logits.max(dim=-1)[1]
        which = which.unsqueeze(dim=-1).unsqueeze(dim=-1)
        which_expand = which.expand(tuple(which.shape[:-1] + (self.locs.shape[-1],)))
        loc = torch.gather(self.locs, -2, which_expand).squeeze(-2)
        return loc

    @property
    def std(self):
        p = self.categorical.probs
        return torch.sum(self.coord_scale * p.unsqueeze(-1), dim=-2)

    def shift(self, mu_shift):
        '''shift the distribution, useful in local mode transition'''
        return GaussianMixture(self.locs + mu_shift.unsqueeze(dim=-2), self.coord_scale, self.component_logits)

    def entropy(self):
        p = self.categorical.probs
        normal = DiagnalNormal(self.locs, self.coord_scale)
        entropy = normal.entropy()
        return torch.sum(p * entropy, dim=-1)

class MixDistribution(ReviveDistribution):
    """Collection of multiple distributions"""
    
    def __init__(self, dists):
        super().__init__()
        assert len(set([dist.batch_shape for dist in dists])) == 1, "the batch shape of all distributions should be equal"
        assert len(set([len(dist.event_shape) == 1 for dist in dists])) == 1, "the event shape of all distributions should have length 1"
        self.dists = dists
        self.sizes = [dist.event_shape[0] for dist in self.dists]
        batch_shape = self.dists[0].batch_shape
        event_shape = torch.Size((sum(self.sizes),))
        super(MixDistribution, self).__init__(batch_shape, event_shape)

    def sample(self, num=torch.Size()):
        samples = [dist.sample(num) for dist in self.dists]
        return torch.cat(samples, dim=-1)

    def rsample(self, num=torch.Size()): 
        samples = [dist.rsample(num) for dist in self.dists]
        return torch.cat(samples, dim=-1)

    def entropy(self):
        return sum([dist.entropy() for dist in self.dists])  

    def log_prob(self, x):
        if type(x) == list:
            return [self.dists[i].log_prob(x[i]) for i in range(len(x))]
        # manually split the tensor
        x = torch.split(x, self.sizes, dim=-1)
        return sum([self.dists[i].log_prob(x[i]) for i in range(len(x))])

    @property
    def mode(self):
        modes = [dist.mode for dist in self.dists]
        return torch.cat(modes, dim=-1)

    @property
    def std(self):
        stds = [dist.std for dist in self.dists]
        return torch.cat(stds, dim=-1)

    def shift(self, mu_shift):
        '''shift the distribution, useful in local mode transition'''
        assert all([type(dist) in [DiagnalNormal, GaussianMixture] for dist in self.dists]), \
            "all the distributions should have method `shift`"
        return MixDistribution([dist.shift(mu_shift) for dist in self.dists])

@register_kl(DiagnalNormal, DiagnalNormal)
def _kl_diagnalnormal_diagnalnormal(p : DiagnalNormal, q : DiagnalNormal):
    kl = kl_divergence(p.base_dist, q.base_dist)
    kl = torch.sum(kl, dim=-1)
    return kl

@register_kl(Onehot, Onehot)
def _kl_onehot_onehot(p : Onehot, q : Onehot):
    kl = (p.probs * (torch.log(p.probs) - torch.log(q.probs))).sum(dim=-1)
    return kl

@register_kl(GaussianMixture, GaussianMixture)
def _kl_gmm_gmm(p : GaussianMixture, q : GaussianMixture):
    samples = p.rsample()
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)
    return log_p - log_q

@register_kl(MixDistribution, MixDistribution)
def _kl_mix_mix(p : MixDistribution, q : MixDistribution):
    assert all([type(_p) == type(_q) for _p, _q in zip(p.dists, q.dists)])
    kl = 0
    for _p, _q in zip(p.dists, q.dists):
        kl = kl + kl_divergence(_p, _q)
    return kl
     
if __name__ == '__main__':
    print('-' * 50)
    onehot = Onehot(torch.rand(2, 10, requires_grad=True))
    print('onehot batch shape', onehot.batch_shape)
    print('onehot event shape', onehot.event_shape)
    print('onehot sample', onehot.sample())
    print('onehot rsample', onehot.rsample())
    print('onehot log prob', onehot.sample_with_logprob()[1])
    print('onehot mode', onehot.mode)
    print('onehot std', onehot.std)
    print('onehot entropy', onehot.entropy())
    _onehot = Onehot(torch.rand(2, 10, requires_grad=True))
    print('onehot kl', kl_divergence(onehot, _onehot))

    print('-' * 50)
    mixture = GaussianMixture(
        torch.rand(2, 6, 4, requires_grad=True), 
        torch.rand(2, 6, 4, requires_grad=True),
        torch.rand(2, 6, requires_grad=True), 
    )
    print('gmm batch shape', mixture.batch_shape)
    print('gmm event shape', mixture.event_shape)
    print('gmm sample', mixture.sample())
    print('gmm rsample', mixture.rsample())
    print('gmm log prob', mixture.sample_with_logprob()[1])
    print('gmm mode', mixture.mode)
    print('gmm std', mixture.std)
    print('gmm entropy', mixture.entropy())
    _mixture = GaussianMixture(
        torch.rand(2, 6, 4, requires_grad=True), 
        torch.rand(2, 6, 4, requires_grad=True),
        torch.rand(2, 6, requires_grad=True), 
    )
    print('gmm kl', kl_divergence(mixture, _mixture))

    print('-' * 50)
    normal = DiagnalNormal(
        torch.rand(2, 5, requires_grad=True), 
        torch.rand(2, 5, requires_grad=True)
    )
    print('normal batch shape', normal.batch_shape)
    print('normal event shape', normal.event_shape)
    print('normal sample', normal.sample())
    print('normal rsample', normal.rsample())
    print('normal log prob', normal.sample_with_logprob()[1])
    print('normal mode', normal.mode)
    print('normal std', normal.std)
    print('normal entropy', normal.entropy())
    _normal = DiagnalNormal(
        torch.rand(2, 5, requires_grad=True), 
        torch.rand(2, 5, requires_grad=True)
    )
    print('normal kl', kl_divergence(normal, _normal))

    print('-' * 50)
    mix = MixDistribution([onehot, mixture, normal])
    print('mix batch shape', mix.batch_shape)
    print('mix event shape', mix.event_shape)
    print('mix sample', mix.sample())
    print('mix rsample', mix.rsample())
    print('mix log prob', mix.sample_with_logprob()[1])
    print('mix mode', mix.mode)
    print('mix std', mix.std)
    print('mix entropy', mix.entropy())
    _mix = MixDistribution([_onehot, _mixture, _normal])
    print('mix kl', kl_divergence(mix, _mix))
