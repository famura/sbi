# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Union
from warnings import warn

import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows import flows
from pyknos.nflows.transforms import CompositeTransform
from torch import Tensor
from torch.distributions import MultivariateNormal

import sbi.utils as utils


class MoGFlow_SNPE_A(flows.Flow):
    """
    A wrapper for nflow's `Flow` class to enable a different log prob calculation
    sampling strategy for training and testing, tailored to SNPE-A [1]

    [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
        Density Estimation_, Papamakarios et al., NeurIPS 2016,
        https://arxiv.org/abs/1605.06376.
    [2] _Automatic Posterior Transformation for Likelihood-free Inference_,
        Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.
    """

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        # Construct flow
        super().__init__(transform, distribution, embedding_net)

        self.logits_pp, self.m_pp, self.prec_pp = None, None, None
        self._proposal = None
        self.default_x = None

    @property
    def proposal(
        self,
    ) -> Union["utils.BoxUniform", MultivariateNormal, "MoGFlow_SNPE_A"]:
        """ Get the proposal of the previous round. """
        return self._proposal

    def set_proposal(
        self, proposal: Union["utils.BoxUniform", MultivariateNormal, "MoGFlow_SNPE_A"]
    ):
        """ Set the proposal of the previous round. """
        self._proposal = proposal

        # Take care of z-scoring, pre-compute and store prior terms.
        self._set_state_for_mog_proposal()

    def _get_first_prior_from_proposal(self):
        """ Iterate a possible chain of proposals. """
        curr_prior = self._proposal

        while curr_prior:
            if isinstance(curr_prior, (utils.BoxUniform, MultivariateNormal)):
                break
            else:
                curr_prior = curr_prior.proposal

        assert curr_prior is not None
        return curr_prior

    def log_prob(self, inputs, context=None):
        if self._proposal is None:
            # Use Flow.lob_prob() if there has been no previous proposal memorized
            # in this instance. This is the case if we are in the training
            # loop, i.e. this MoGFlow_SNPE_A instance is not an attribute of a
            # DirectPosterior instance.
            return super().log_prob(inputs, context)  # q_phi from eq. (3) in [1]

        else:
            # When we want to compute the approx. posterior, a proposal prior \tilde{p}
            # has already been observed. To analytically calculate the log-prob of the
            # Gaussian, we first need to compute the mixture components.
            return self._log_prob_approx_posterior_mog(
                inputs, context
            )  # \hat{p} from eq. (3) in [1]

    def sample(self, num_samples, context=None, batch_size=None) -> Tensor:
        if self._proposal is None:
            # Use Flow.sample() if there has been no previous proposal memorized
            # in this instance. This is the case if we are in the training
            # loop, i.e. this MoGFlow_SNPE_A instance is not an attribute of a
            # DirectPosterior instance.
            return super().sample(num_samples, context, batch_size)

        else:
            # When we want to sample from the approx. posterior, a proposal prior \tilde{p}
            # has already been observed. To analytically calculate the log-prob of the
            # Gaussian, we first need to compute the mixture components.
            return self._sample_approx_posterior_mog(num_samples, context, batch_size)

    def _sample_approx_posterior_mog(
        self, num_samples, x: Tensor, batch_size: int
    ) -> Tensor:
        """

        Args:
            num_samples: Desired number of samples.
            x: Conditioning context for posterior $p(\theta|x)$.
            batch_size: Batch size for sampling.

        Returns:
            Samples from the approximate mixture of Gaussians posterior.
        """
        # Check if default_x was set previously.
        # if self.default_x is not None and torch.all(x == self.default_x):
        #     # Use the previously computed mixture components of the proposal posterior.
        #     logits_pp, m_pp, prec_pp = self.logits_pp, self.m_pp, self.prec_pp
        # else:
        # Compute the mixture components of the proposal posterior.
        logits_pp, m_pp, prec_pp = self._get_mixture_components(x)

        # Compute the precision factors which represent the upper triangular matrix
        # of the cholesky decomposition of the prec_pp.
        prec_factors_pp = torch.cholesky(prec_pp, upper=True)

        # Only add the default_x if it is a single value and not a batch of data.
        if x.shape[0] == 1:
            self.default_x = x
        self.logits_pp, self.m_pp, self.prec_pp = logits_pp, m_pp, prec_pp

        assert logits_pp.ndim == 2
        assert m_pp.ndim == 3
        assert prec_pp.ndim == 4
        assert prec_factors_pp.ndim == 4

        # Replicate to use batched sampling from pyknos.
        if batch_size is not None and batch_size > 1:
            logits_pp = logits_pp.repeat(batch_size, 1)
            m_pp = m_pp.repeat(batch_size, 1, 1)
            prec_factors_pp = prec_factors_pp.repeat(batch_size, 1, 1, 1)

        # Get (optionally z-scored) MoG samples.
        theta = MultivariateGaussianMDN.sample_mog(
            num_samples, logits_pp, m_pp, prec_factors_pp
        )

        if self.z_score_theta:
            theta, _ = self._transform.inverse(theta)  # 2dn output is the log abs det

        # embedded_context = self._embedding_net(x)
        # if embedded_context is not None:
        #     # Merge the context dimension with sample dimension in order to apply the transform.
        #     noise = torchutils.merge_leading_dims(theta, num_dims=2)
        #     embedded_context = torchutils.repeat_rows(
        #         embedded_context, num_reps=num_samples
        #     )
        #
        #     theta, _ = self._transform.inverse(noise, context=embedded_context)
        #
        #     if embedded_context is not None:
        #         # Split the context dimension from sample dimension.
        #         theta = torchutils.split_leading_dim(theta, shape=[-1, num_samples])

        return theta

    def _log_prob_approx_posterior_mog(self, theta: Tensor, x: Tensor) -> Tensor:
        """
        Return log-probability of the approximate posterior for MoG proposal.

        For MoG proposals and MoG density estimators, this can be done in closed form
        and does not require atomic loss (i.e. there will be no leakage issues).

        Notation:

        m are mean vectors.
        prec are precision matrices.
        cov are covariance matrices.

        _p at the end indicates that it is the proposal.
        _d indicates that it is the density estimator.
        _pp indicates the proposal posterior.

        All tensors will have shapes (batch_dim, num_components, ...)

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.

        Returns:
            Log-probability of the proposal posterior.
        """
        # Check if default_x was set previously.
        # if self.default_x is not None and torch.all(x == self.default_x):
        #     # Use the previously computed mixture components of the proposal posterior.
        #     logits_pp, m_pp, prec_pp = self.logits_pp, self.m_pp, self.prec_pp
        #
        #     # Expand to batch size (e.g., during evaluation)
        #     batch_size = theta.shape[0]
        #     logits_pp = logits_pp.repeat(batch_size, 1)
        #     m_pp = m_pp.repeat(batch_size, 1, 1)
        #     prec_pp = prec_pp.repeat(batch_size, 1, 1, 1)
        # else:
        # Compute the mixture components of the proposal posterior.
        logits_pp, m_pp, prec_pp = self._get_mixture_components(x)

        # Only add the default_x if it is a single value and not a batch of data
        if x.shape[0] == 1:
            self.default_x = x
        self.logits_pp, self.m_pp, self.prec_pp = logits_pp, m_pp, prec_pp

        # z-score theta if it z-scoring had been requested.
        theta = self._maybe_z_score_theta(theta)

        # Compute the log_prob of theta under the product.
        log_prob_proposal_posterior = utils.mog_log_prob(
            theta,
            logits_pp,
            m_pp,
            prec_pp,
        )
        MoGFlow_SNPE_A._assert_all_finite(
            log_prob_proposal_posterior, "proposal posterior eval"
        )
        return log_prob_proposal_posterior

    def _get_mixture_components(self, x: Tensor):
        """
        Compute the mixture components of the posterior given the current density
        estimator and the proposal.

        Args:
            x: Conditioning context for posterior.

        Returns:
            Mixture components of the posterior.
        """
        # Evaluate the density estimator.
        encoded_x = self._embedding_net(x)
        dist = self._distribution  # defined to avoid black formatting.
        logits_d, m_d, prec_d, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_d = logits_d - torch.logsumexp(logits_d, dim=-1, keepdim=True)

        if isinstance(self._proposal, utils.BoxUniform):
            # Uniform prior is uninformative.
            return norm_logits_d, m_d, prec_d

        elif isinstance(self._proposal, MultivariateNormal):
            logits_p = torch.tensor([1])
            m_p = self._proposal.mean
            prec_p = self._proposal.precision_matrix

        else:
            # Recursive ask for the mixture components until the prior is yielded.
            logits_p, m_p, prec_p = self._proposal._get_mixture_components(x)

        # Compute the MoG parameters of the proposal posterior.
        (
            logits_pp,
            m_pp,
            prec_pp,
            cov_pp,
        ) = self._automatic_proposal_posterior_transformation(
            logits_p,
            m_p,
            prec_p,
            norm_logits_d,
            m_d,
            prec_d,
        )
        return logits_pp, m_pp, prec_pp

    @staticmethod
    def _assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
        """
        # TODO move this to sbi (utils), and also do that for NeuralInference
        .. note::
            Hard copy!

        Raise if tensor quantity contains any NaN or Inf element.
        """

        msg = f"NaN/Inf present in {description}."
        assert torch.isfinite(quantity).all(), msg

    def _automatic_proposal_posterior_transformation(
        self,
        logits_p: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""
        Returns the MoG parameters of the proposal posterior.

        The proposal posterior is:
        $pp(\theta|x) = 1/Z * q(\theta|x) * prop(\theta) / p(\theta)$
        In words: proposal posterior = posterior estimate * proposal / prior.

        If the posterior estimate and the proposal are MoG and the prior is either
        Gaussian or uniform, we can solve this in closed-form. The is implemented in
        this function.

        This function implements Appendix A1 from Greenberg et al. 2019.

        We have to build L*K components. How do we do this?
        Example: proposal has two components, density estimator has three components.
        Let's call the two components of the proposal i,j and the three components
        of the density estimator x,y,z. We have to multiply every component of the
        proposal with every component of the density estimator. So, what we do is:
        1) for the proposal, build: i,i,i,j,j,j. Done with torch.repeat_interleave()
        2) for the density estimator, build: x,y,z,x,y,z. Done with torch.repeat()
        3) Multiply them with simple matrix operations.

        Args:
            logits_p: Component weight of each Gaussian of the proposal.
            means_p: Mean of each Gaussian of the proposal.
            precisions_p: Precision matrix of each Gaussian of the proposal.
            logits_d: Component weight for each Gaussian of the density estimator.
            means_d: Mean of each Gaussian of the density estimator.
            precisions_d: Precision matrix of each Gaussian of the density estimator.

        Returns: (Component weight, mean, precision matrix, covariance matrix) of each
            Gaussian of the proposal posterior. Has L*K terms (proposal has L terms,
            density estimator has K terms).
        """

        precisions_pp, covariances_pp = self._precisions_proposal_posterior(
            precisions_p, precisions_d
        )

        means_pp = self._means_proposal_posterior(
            covariances_pp,
            means_p,
            precisions_p,
            means_d,
            precisions_d,
        )

        logits_pp = MoGFlow_SNPE_A._logits_proposal_posterior(
            means_pp,
            precisions_pp,
            covariances_pp,
            logits_p,
            means_p,
            precisions_p,
            logits_d,
            means_d,
            precisions_d,
        )

        return logits_pp, means_pp, precisions_pp, covariances_pp

    def _set_state_for_mog_proposal(self) -> None:
        """
        Set state variables that are used at every training step of non-atomic SNPE-C.

        Three things are computed:
        1) Check if z-scoring was requested. To do so, we check if the `_transform`
            argument of the net had been a `CompositeTransform`. See pyknos mdn.py.
        2) Define a (potentially standardized) prior. It's standardized if z-scoring
            had been requested.
        3) Compute (Precision * mean) for the prior. This quantity is used at every
            training step if the prior is Gaussian.
        """

        self.z_score_theta = isinstance(self._transform, CompositeTransform)

        self._set_maybe_z_scored_prior()

        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            self.prec_m_prod_prior = torch.mv(
                self._maybe_z_scored_prior.precision_matrix,
                self._maybe_z_scored_prior.loc,
            )

    def _set_maybe_z_scored_prior(self) -> None:
        r"""
        Compute and store potentially standardized prior (if z-scoring was requested).

        The proposal posterior is:
        $pp(\theta|x) = 1/Z * q(\theta|x) * p(\theta) / prop(\theta)$

        Let's denote z-scored theta by `a`: a = (theta - mean) / std
        Then $pp'(a|x) = 1/Z_2 * q'(a|x) * p'(a) / prop'(a)$

        The ' indicates that the evaluation occurs in standardized space. The constant
        scaling factor has been absorbed into $Z_2$.
        From the above equation, we see that we need to evaluate the prior **in
        standardized space**. We build the standardized prior in this function.

        The standardize transform that is applied to the samples theta does not use
        the exact prior mean and std (due to implementation issues). Hence, the z-scored
        prior will not be exactly have mean=0 and std=1.
        """
        prior = self._get_first_prior_from_proposal()

        if self.z_score_theta:
            scale = self._transform._transforms[0]._scale
            shift = self._transform._transforms[0]._shift

            # Following the definintion of the linear transform in
            # `standardizing_transform` in `sbiutils.py`:
            # shift=-mean / std
            # scale=1 / std
            # Solving these equations for mean and std:
            estim_prior_std = 1 / scale
            estim_prior_mean = -shift * estim_prior_std

            # Compute the discrepancy of the true prior mean and std and the mean and
            # std that was empirically estimated from samples.
            # N(theta|m,s) = N((theta-m_e)/s_e|(m-m_e)/s_e, s/s_e)
            # Above: m,s are true prior mean and std. m_e,s_e are estimated prior mean
            # and std (estimated from samples and used to build standardize transform).
            almost_zero_mean = (prior.mean - estim_prior_mean) / estim_prior_std
            almost_one_std = torch.sqrt(prior.variance) / estim_prior_std

            if isinstance(prior, MultivariateNormal):
                self._maybe_z_scored_prior = MultivariateNormal(
                    almost_zero_mean,
                    torch.diag(almost_one_std),
                )
            else:
                range_ = torch.sqrt(almost_one_std * 3.0)
                self._maybe_z_scored_prior = utils.BoxUniform(
                    almost_zero_mean - range_, almost_zero_mean + range_
                )
        else:
            self._maybe_z_scored_prior = prior

    def _maybe_z_score_theta(self, theta: Tensor) -> Tensor:
        """Return potentially standardized theta if z-scoring was requested."""

        if self.z_score_theta:
            theta, _ = self._transform(theta)

        return theta

    def _precisions_proposal_posterior(
        self,
        precisions_p: Tensor,
        precisions_d: Tensor,
    ):
        """
        Return the precisions and covariances of the proposal posterior.

        C_ik = (P_i - P_k + P_o)^{-1}.
        (matching the notation of Appendix A.1 in [2])

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.
        [2] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

        Args:
            precisions_p: Precision matrices of the proposal distribution.
            precisions_d: Precision matrices of the density estimator.

        Returns: (Precisions, Covariances) of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Check if precision matrices are positive definite.
        for batches in precisions_p:
            for p in batches:
                eig_p = torch.symeig(p, eigenvectors=False).eigenvalues
                assert (
                    eig_p > 0
                ).all(), (
                    "The precision matrix of the proposal is not positive definite!"
                )
        for batches in precisions_d:
            for d in batches:
                eig_d = torch.symeig(d, eigenvectors=False).eigenvalues
                assert (
                    eig_d > 0
                ).all(), "The precision matrix of the density estimator is not positive definite!"

        precisions_p_rep = precisions_p.repeat_interleave(num_comps_d, dim=1)
        precisions_d_rep = precisions_d.repeat(1, num_comps_p, 1, 1)

        precisions_pp = precisions_d_rep - precisions_p_rep  # see Appendix C in [1]
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            precisions_pp += (
                self._maybe_z_scored_prior.precision_matrix
            )  # see Appendix C in [1]

        # Check if positive definite
        for idx_batch, batches in enumerate(precisions_pp):
            for idx_comp, pp in enumerate(batches):
                eig_pp = torch.symeig(pp, eigenvectors=False).eigenvalues
                # assert (
                #     eig_pp > 0
                # ).all(), "The precision matrix of a proposal posterior is not positive definite!"
                if not (eig_pp > 0).all():
                    # Shift the eigenvalues to be at minimum 1e-6
                    precisions_pp[idx_batch, idx_comp] = pp - torch.eye(pp.shape[0]) * (
                        min(eig_pp) - 1e-6
                    )
                    precisions_pp[idx_batch, idx_comp] = torch.clamp(
                        precisions_pp[idx_batch, idx_comp], min=1e-6
                    )
                    warn(
                        "The precision matrix of a proposal posterior is not positive definite"
                    )
                    # print(torch.symeig(precisions_pp[idx_batch, idx_comp],eigenvectors=False).eigenvalues.numpy())
                    # print("-"*5)

        covariances_pp = torch.inverse(precisions_pp)

        return precisions_pp, covariances_pp

    def _means_proposal_posterior(
        self,
        covariances_pp: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        """
        Return the means of the proposal posterior.

        m_ik = C_ik * (-P_k * m_k + P_i * m_i + P_o * m_o).
        (matching the notation of Appendix A.1 in [2])

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.
        [2] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

        Args:
            covariances_pp: Covariance matrices of the proposal posterior.
            means_p: Means of the proposal distribution.
            precisions_p: Precision matrices of the proposal distribution.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Means of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # First, compute the product P_i * m_i and P_j * m_j
        prec_m_prod_p = utils.batched_mixture_mv(precisions_p, means_p)
        prec_m_prod_d = utils.batched_mixture_mv(precisions_d, means_d)

        # Repeat them to allow for matrix operations: same trick as for the precisions.
        prec_m_prod_p_rep = prec_m_prod_p.repeat_interleave(num_comps_d, dim=1)
        prec_m_prod_d_rep = prec_m_prod_d.repeat(1, num_comps_p, 1)

        # Means = C_ik * (-P_k * m_k + P_i * m_i + P_o * m_o).
        summed_cov_m_prod_rep = (
            prec_m_prod_d_rep - prec_m_prod_p_rep
        )  # see Appendix C in [1]
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            summed_cov_m_prod_rep += self.prec_m_prod_prior  # see Appendix C in [1]

        means_pp = utils.batched_mixture_mv(covariances_pp, summed_cov_m_prod_rep)

        return means_pp

    @staticmethod
    def _logits_proposal_posterior(
        means_pp: Tensor,
        precisions_pp: Tensor,
        covariances_pp: Tensor,
        logits_p: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        """
        Return the component weights (i.e. logits) of the proposal posterior.

        mu_i.T * P_i * mu_i

        Args:
            means_pp: Means of the proposal posterior.
            precisions_pp: Precision matrices of the proposal posterior.
            covariances_pp: Covariance matrices of the proposal posterior.
            logits_p: Component weights (i.e. logits) of the proposal distribution.
            means_p: Means of the proposal distribution.
            precisions_p: Precision matrices of the proposal distribution.
            logits_d: Component weights (i.e. logits) of the density estimator.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.

        Returns: Component weights of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Compute log(alpha_i * beta_j)
        logits_p_rep = logits_p.repeat_interleave(num_comps_d, dim=1)
        logits_d_rep = logits_d.repeat(1, num_comps_p)
        logit_factors = logits_d_rep - logits_p_rep  # see Appendix C in [1]

        # Compute sqrt(det()/(det()*det()))
        logdet_covariances_pp = torch.logdet(covariances_pp)
        logdet_covariances_p = -torch.logdet(
            precisions_p
        )  # Sigma^tilde_k in eq. (14) in [2]
        logdet_covariances_d = -torch.logdet(precisions_d)  # Sigma_i in eq. (14) in [2]

        # Repeat the proposal and density estimator terms such that there are LK terms.
        # Same trick as has been used above.
        logdet_covariances_p_rep = logdet_covariances_p.repeat_interleave(
            num_comps_d, dim=1
        )
        logdet_covariances_d_rep = logdet_covariances_d.repeat(1, num_comps_p)

        log_sqrt_det_ratio = 0.5 * (  # eq (26) in [1]
            logdet_covariances_pp + logdet_covariances_p_rep - logdet_covariances_d_rep
        )

        # Compute for proposal, density estimator, and proposal posterior:
        exponent_p = utils.batched_mixture_vmv(
            precisions_p, means_p  # m_0 in eq (26) in [1]
        )
        exponent_d = utils.batched_mixture_vmv(
            precisions_d, means_d  # m_k in eq (26) in [1]
        )
        exponent_pp = utils.batched_mixture_vmv(
            precisions_pp, means_pp  # m^\prime_k in eq (26) in [1]
        )

        # Extend proposal and density estimator exponents to get LK terms.
        exponent_p_rep = exponent_p.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_p)
        exponent = -0.5 * (
            exponent_p_rep - exponent_d_rep - exponent_pp  # eq (26) in [1]
        )

        logits_pp = logit_factors + log_sqrt_det_ratio + exponent

        return logits_pp
