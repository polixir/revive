from revive.computation.dists import *

def test_dists():
    onehot = Onehot(torch.rand(2, 10, requires_grad=True))
    assert onehot.batch_shape == (2,)
    assert onehot.event_shape == (10,)
    assert onehot.sample().requires_grad == False
    assert onehot.rsample().requires_grad == True
    logprob = onehot.sample_with_logprob()[1]
    assert logprob.shape == (2,)
    assert logprob.requires_grad == True
    assert onehot.mode.shape == (2, 10)
    assert onehot.std.shape == (2, 10)
    assert onehot.entropy().shape == (2,)
    _onehot = Onehot(torch.rand(2, 10, requires_grad=True))
    kl = kl_divergence(onehot, _onehot)
    assert kl.shape == (2,)
    assert kl.requires_grad == True

    mixture = GaussianMixture(
        torch.rand(2, 6, 4, requires_grad=True), 
        torch.rand(2, 6, 4, requires_grad=True),
        torch.rand(2, 6, requires_grad=True), 
    )
    assert mixture.batch_shape == (2,)
    assert mixture.event_shape == (4,)
    assert mixture.sample().requires_grad == False
    assert mixture.rsample().requires_grad == True
    logprob = mixture.sample_with_logprob()[1]
    assert logprob.shape == (2,)
    assert logprob.requires_grad == True
    assert mixture.mode.shape == (2, 4)
    assert mixture.std.shape == (2, 4)
    assert mixture.entropy().shape == (2,)
    _mixture = GaussianMixture(
        torch.rand(2, 6, 4, requires_grad=True), 
        torch.rand(2, 6, 4, requires_grad=True),
        torch.rand(2, 6, requires_grad=True), 
    )
    kl = kl_divergence(mixture, _mixture)
    assert kl.shape == (2,)
    assert kl.requires_grad == True

    normal = DiagnalNormal(
        torch.rand(2, 5, requires_grad=True), 
        torch.rand(2, 5, requires_grad=True)
    )
    assert normal.batch_shape == (2,)
    assert normal.event_shape == (5,)
    assert normal.sample().requires_grad == False
    assert normal.rsample().requires_grad == True
    logprob = normal.sample_with_logprob()[1]
    assert logprob.shape == (2,)
    assert logprob.requires_grad == True
    assert normal.mode.shape == (2, 5)
    assert normal.std.shape == (2, 5)
    assert normal.entropy().shape == (2,)
    _normal = DiagnalNormal(
        torch.rand(2, 5, requires_grad=True), 
        torch.rand(2, 5, requires_grad=True)
    )
    kl = kl_divergence(normal, _normal)
    assert kl.shape == (2,)
    assert kl.requires_grad == True

    discrete_logic = DiscreteLogistic(
        torch.rand(2, 5, requires_grad=True) * 2 - 1, 
        torch.rand(2, 5, requires_grad=True),
        [5, 9, 17, 33, 65],
    )
    assert discrete_logic.batch_shape == (2,)
    assert discrete_logic.event_shape == (5,)
    assert discrete_logic.sample().requires_grad == False
    assert discrete_logic.rsample().requires_grad == True
    logprob = discrete_logic.sample_with_logprob()[1]
    assert logprob.shape == (2,)
    assert logprob.requires_grad == True
    assert normal.mode.shape == (2, 5)
    assert normal.std.shape == (2, 5)
    assert normal.entropy().shape == (2,)
    _discrete_logic = DiscreteLogistic(
        torch.rand(2, 5, requires_grad=True) * 2 - 1, 
        torch.rand(2, 5, requires_grad=True),
        [5, 9, 17, 33, 65],
    )
    kl = kl_divergence(discrete_logic, _discrete_logic)
    assert kl.shape == (2,)
    assert kl.requires_grad == True

    mix = MixDistribution([onehot, mixture, normal, discrete_logic])
    assert mix.batch_shape == (2,)
    assert mix.event_shape == (24,)
    assert mix.sample().requires_grad == False
    assert mix.rsample().requires_grad == True
    logprob = mix.sample_with_logprob()[1]
    assert logprob.shape == (2,)
    assert logprob.requires_grad == True
    assert mix.mode.shape == (2, 24)
    assert mix.std.shape == (2, 24)
    assert mix.entropy().shape == (2,)
    _mix = MixDistribution([_onehot, _mixture, _normal, _discrete_logic])
    kl = kl_divergence(mix, _mix)
    assert kl.shape == (2,)
    assert kl.requires_grad == True    