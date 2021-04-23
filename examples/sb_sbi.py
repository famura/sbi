import torch
from matplotlib import pyplot as plt
from torch.distributions import Bernoulli

import sbi.utils as utils
from sbi.inference import simulate_for_sbi
from sbi.inference.snpe.snpe_a import SNPE_A
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils import posterior_nn
from sbi.utils.user_input_checks import prepare_for_sbi


def simulator(mu):
    # Generate samples from N(mu, sigma=0.5)
    bern = Bernoulli(0.45)
    smaple_from_mu1 = bern.sample()

    if smaple_from_mu1:
        return mu + 0.5 * torch.randn_like(mu)
    else:
        return -mu + 0.5 * torch.randn_like(mu)


if __name__ == "__main__":
    num_sim = 10000

    prior = utils.BoxUniform(
        low=torch.tensor([-5.0, -3.0]), high=torch.tensor([5.0, 3.0])
    )

    # TODO test MVN prior

    gt = torch.tensor([3.0, -1.5])

    # density_estimator = "mdn_snpe_a"
    method = "SNPE_A"
    num_rounds = 2
    if method == "SNPE_A":
        num_components = 2
        density_estimator = "mdn_snpe_a"
        density_estimator = posterior_nn(model=density_estimator, num_components=2)
        snpe = SNPE_A(2, num_rounds, prior, density_estimator)
    else:
        density_estimator = "mdn"
        density_estimator = posterior_nn(model=density_estimator, num_components=2)
        snpe = SNPE_C(prior, density_estimator)

    simulator, prior = prepare_for_sbi(simulator, prior)
    proposal = prior
    
    fig_th, ax_th = plt.subplots(1)

    # multiround training
    for r in range(num_rounds + 1):
        thetas, data_sim = simulate_for_sbi(
            simulator=simulator,
            proposal=proposal,
            num_simulations=500,
            num_workers=1,
        )
        
        ax_th.scatter(x=thetas[:, 0].numpy(), y=thetas[:, 1].numpy(), label=f"round {r}", s=10)
        # if r == 1:
        #     plt.show()

        snpe.append_simulations(thetas, data_sim, proposal)

        if r == num_rounds:
            break
        density_estimator = snpe.train()

        if method == "SNPE_A":
            posterior = snpe.build_posterior(proposal=proposal, density_estimator=density_estimator,
                                             sample_with_mcmc=False
                                             )
            # posterior = snpe.build_posterior(proposal=proposal)  # TODO
        else:
            posterior = snpe.build_posterior(density_estimator=density_estimator)

        posterior.set_default_x(gt)
        proposal = posterior

    # Configure plot
    ax_th.scatter(x=gt[0], y=gt[1], label="gt", marker="*", s=40)
    ax_th.legend()
    ax_th.set_xlim(-5, 5)
    ax_th.set_ylim(-3, 3)

    n_observations = 1
    observation = torch.tensor([3.0, -1.5])[None] + 0.5 * torch.randn(n_observations, 2)

    fig_obs, ax_obs = plt.subplots(1)
    ax_obs.scatter(x=observation[:, 0], y=observation[:, 1], label="obs")
    ax_obs.set_xlabel(r"$x_1$")
    ax_obs.set_ylabel(r"$x_2$")
    ax_obs.set_xlim(-10, 10)
    ax_obs.set_ylim(-10, 10)

    assert isinstance(prior, utils.BoxUniform)
    bounds = [
        prior.support.base_constraint.lower_bound[0].item(),
        prior.support.base_constraint.upper_bound[0].item(),
        prior.support.base_constraint.lower_bound[1].item(),
        prior.support.base_constraint.upper_bound[1].item(),
    ]

    grid_res = 101
    x = torch.linspace(bounds[0], bounds[1], grid_res)  # 1 2 3
    y = torch.linspace(bounds[2], bounds[3], grid_res)  # 4 5 6
    x = x.repeat(grid_res)  # 1 2 3 1 2 3 1 2 3
    y = torch.repeat_interleave(y, grid_res)  # 4 4 4 5 5 5 6 6 6
    grid_x, grid_y = x.view(grid_res, grid_res), y.view(grid_res, grid_res)
    grid = torch.stack([x, y], dim=1)

    if "SNPE" in method:
        log_prob = sum(
            [posterior.log_prob(grid, observation[i], norm_posterior=False) for i in range(len(observation))]
        )
    else:
        log_prob = sum(
            [
                posterior.net(
                    torch.cat(
                        (grid, observation[i].repeat((grid.shape[0])).reshape(-1, 2)),
                        dim=1,
                    )
                )[:, 0]
                + posterior._prior.log_prob(grid)
                for i in range(len(observation))
            ]
        ).detach()

    prob = torch.exp(log_prob - log_prob.max())
    plt.figure(dpi=200)
    plt.contourf(prob.reshape(*grid_x.shape), extent=bounds, origin="lower")
    plt.axis("scaled")
    plt.title(
        "Posterior with learned likelihood\nfrom %d examples of" % (num_sim)
        + r" $\mu_i\in[-5, 5]$"
    )
    plt.xlabel(r"$\mu_x$")
    plt.ylabel(r"$\mu_y$")

    true_like = lambda x: -((x[0] - grid_x) ** 2 + (x[1] - grid_y) ** 2) / (2 * 0.5 ** 2)
    log_prob = sum([true_like(observation[i]) for i in range(len(observation))])
    plt.figure(dpi=200)
    prob = torch.exp(log_prob - log_prob.max())
    plt.contourf(prob.reshape(*grid_x.shape), extent=bounds, origin="lower")
    plt.axis("scaled")
    plt.title("Posterior with\nanalytic likelihood")
    plt.xlabel(r"$\mu_x$")
    plt.ylabel(r"$\mu_y$")

    plt.show()
