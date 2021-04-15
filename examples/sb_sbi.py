import torch

import sbi.utils as utils
from sbi.inference import simulate_for_sbi
from sbi.inference.snpe.snpe_a import SNPE_A
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils.user_input_checks import prepare_for_sbi
from sbi.analysis.plot import pairplot


def simulator(mu):
    # Generate samples from N(mu, sigma=0.5)

    return mu + 0.5 * torch.randn_like(mu)


if __name__ == "__main__":
    num_sim = 200

    prior = utils.BoxUniform(
        low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0])
    )

    # TODO test MVN prior

    x_gt = torch.tensor([3.0, -1.5])

    # density_estimator = "mdn_snpe_a"
    method = "SNPE_A"
    if method == "SNPE_A":
        density_estimator = "mdn_snpe_a"
        snpe = SNPE_A(prior, density_estimator)
    else:
        density_estimator = "mdn"
        snpe = SNPE_C(prior, density_estimator)
    simulator, prior = prepare_for_sbi(simulator, prior)
    proposal = prior

    # multiround training
    num_rounds = 3
    for r in range(num_rounds):
        domain_param, data_sim = simulate_for_sbi(
            simulator=simulator,
            proposal=proposal,
            num_simulations=200,
            num_workers=1,
        )
        snpe.append_simulations(domain_param, data_sim, proposal)
        density_estimator = snpe.train()

        if method == "SNPE_A":
            posterior = snpe.build_posterior(proposal=proposal, density_estimator=density_estimator)
            # posterior = snpe.build_posterior(proposal=proposal)  # TODO
        else:
            posterior = snpe.build_posterior(density_estimator=density_estimator)
        posterior.set_default_x(x_gt)

        print(posterior.net.training)
        posterior.log_prob(torch.tensor([3.0, -1.5]), x=x_gt)

        proposal = posterior


    posterior.log_prob(torch.tensor([3.0, -1.5]), x=x_gt)
    s = posterior.sample((3,), x=x_gt)

    raise SystemExit(0)


    n_observations = 5
    observation = torch.tensor([3.0, -1.5])[None] + 0.5 * torch.randn(n_observations, 2)

    # import seaborn as sns
    from matplotlib import pyplot as plt

    plt.scatter(x=observation[:, 0], y=observation[:, 1])
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()

    samples = posterior.sample((200,), x=observation[0])

    log_probability = posterior.log_prob(samples, x=observation[0])
    out = pairplot(
        samples, limits=[[-5, 5], [-5, 5]], figsize=(6, 6), upper="kde", diag="kde"
    )

    import numpy as np

    bounds = [3 - 1, 3 + 1, -1.5 - 1, -1.5 + 1]

    mu_1, mu_2 = torch.tensor(
        np.mgrid[bounds[0] : bounds[1] : 2 / 50.0, bounds[2] : bounds[3] : 2 / 50.0]
    ).float()

    grids = torch.cat((mu_1.reshape(-1, 1), mu_2.reshape(-1, 1)), dim=1)

    if "SNPE" in method:
        log_prob = sum(
            [posterior.log_prob(grids, observation[i]) for i in range(len(observation))]
        )
    else:
        log_prob = sum(
            [
                posterior.net(
                    torch.cat(
                        (grids, observation[i].repeat((grids.shape[0])).reshape(-1, 2)),
                        dim=1,
                    )
                )[:, 0]
                + posterior._prior.log_prob(grids)
                for i in range(len(observation))
            ]
        ).detach()

    prob = torch.exp(log_prob - log_prob.max())
    plt.figure(dpi=200)
    plt.plot([2, 4], [-1.5, -1.5], color="k")
    plt.plot([3, 3], [-0.5, -2.5], color="k")
    plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin="lower")
    plt.axis("scaled")
    plt.xlim(2 + 0.3, 4 - 0.3)
    plt.ylim(-2.5 + 0.3, -0.5 - 0.3)
    plt.title(
        "Posterior with learned likelihood\nfrom %d examples of" % (num_sim)
        + r" $\mu_i\in[-5, 5]$"
    )
    plt.xlabel(r"$\mu_1$")
    plt.ylabel(r"$\mu_2$")

    true_like = lambda x: -((x[0] - mu_1) ** 2 + (x[1] - mu_2) ** 2) / (2 * 0.5 ** 2)
    log_prob = sum([true_like(observation[i]) for i in range(len(observation))])
    plt.figure(dpi=200)
    prob = torch.exp(log_prob - log_prob.max())
    plt.plot([2, 4], [-1.5, -1.5], color="k")
    plt.plot([3, 3], [-0.5, -2.5], color="k")
    plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin="lower")
    plt.axis("scaled")
    plt.xlim(2 + 0.3, 4 - 0.3)
    plt.ylim(-2.5 + 0.3, -0.5 - 0.3)
    plt.title("Posterior with\nanalytic likelihood")
    plt.xlabel(r"$\mu_1$")
    plt.ylabel(r"$\mu_2$")

    plt.show()
