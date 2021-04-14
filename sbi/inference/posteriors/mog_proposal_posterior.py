import torch
from torch import Tensor, log, nn
from torch.distributions import MultivariateNormal
from typing import Any, Dict, Optional, Union

from sbi.inference.posteriors.direct_posterior import PotentialFunctionProvider, DirectPosterior
from sbi.neural_nets.mog_flow_snpe_a import MoGFlow_SNPE_A
from sbi.types import Shape
from sbi.utils import BoxUniform, del_entries, sample_posterior_within_prior
from sbi.utils.torchutils import batched_first_of_batch


class MoGProposalPosterior(DirectPosterior):
    r"""Proposal posterior $\tilde p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNPE-A.<br/><br/>
    SNPE-A trains a neural network to approximate the proposal posterior distribution.
    However, for bounded priors, the neural network can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `DirectPosterior` class wraps the
    trained network to deal with these cases.<br/><br/>
    Specifically, this class offers the following functionality:<br/>
    - correct the calculation of the log probability such that it compensates for the
      leakage.<br/>
    - reject samples that lie outside of the prior bounds.<br/>
    - alternatively, if leakage is very high (which can happen for multi-round SNPE),
      sample from the posterior with MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.

    [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.
    """

    def __init__(
        self,
        method_family: str,
        proposal: Union[MultivariateNormal, BoxUniform, "MoGProposalPosterior"],
        neural_net: nn.Module,
        prior,
        x_shape: torch.Size,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: bool = True,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            proposal: TODO
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            rejection_sampling_parameters: Dictonary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
            sample_with_mcmc: Whether to sample with MCMC. Will always be `True` for SRE
                and SNL, but can also be set to `True` for SNPE if MCMC is preferred to
                deal with leakage over rejection sampling.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            device: Training device, e.g., cpu or cuda:0
        """

        self._proposal = proposal

        kwargs = del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
                "proposal",
            ),
        )
        super().__init__(**kwargs)

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""
        Returns the log-probability of the posterior $p(\theta|x).$

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `force_update`, `show_progress_bars`, and
                `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.

        """

        assert isinstance(self.net, MoGFlow_SNPE_A)
        self.net.eval()

        theta, x = self._prepare_theta_and_x_for_log_prob_(theta, x)

        with torch.set_grad_enabled(track_gradients):

            theta, x = theta.to(self._device), x.to(self._device)
            # Compute the proposal posterior's log prob for SNPE-A
            # (see [1] Algorithm 2 second to last line, or eq. (3))
            log_prob = self.net.log_prob(theta, x, self._proposal).cpu()

            # Force probability to be zero outside prior support.
            is_prior_finite = torch.isfinite(self._prior.log_prob(theta))

            masked_log_prob = torch.where(
                is_prior_finite,
                log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()  # use defaults
            log_factor = (
                log(self.leakage_correction(x=batched_first_of_batch(x), **leakage_correction_params))
                if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        sample_with_mcmc: Optional[bool] = None,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$.

        Samples are obtained either with rejection sampling or MCMC. Rejection sampling
        will be a lot faster if leakage is rather low. If leakage is high (e.g. over
        99%, which can happen in multi-round SNPE), MCMC can be faster than rejection
        sampling.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with_mcmc: Optional parameter to override `self.sample_with_mcmc`.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
        Returns:
            Samples from posterior.
        """

        x, num_samples, mcmc_method, mcmc_parameters = self._prepare_for_sample(
            x, sample_shape, mcmc_method, mcmc_parameters
        )

        sample_with_mcmc = sample_with_mcmc if sample_with_mcmc is not None else self.sample_with_mcmc

        self.net.eval()

        if sample_with_mcmc:
            potential_fn_provider = PotentialFunctionProvider()
            samples = self._sample_posterior_mcmc(
                num_samples=num_samples,
                potential_fn=potential_fn_provider(self._prior, self.net, x, mcmc_method),
                init_fn=self._build_mcmc_init_fn(
                    self._prior,
                    potential_fn_provider(self._prior, self.net, x, "slice_np"),
                    **mcmc_parameters,
                ),
                mcmc_method=mcmc_method,
                show_progress_bars=show_progress_bars,
                **mcmc_parameters,
            )
        else:
            # Rejection sampling.
            samples, _ = sample_posterior_within_prior(
                self.net,
                self._prior,
                x,
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
                **rejection_sampling_parameters
                if (rejection_sampling_parameters is not None)
                else self.rejection_sampling_parameters,
            )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))
