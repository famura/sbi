# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import math
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from pyknos.nflows import flows
from pyknos.nflows.transforms import CompositeTransform

import sbi.utils as utils


class MoGFlow_SNPE_A(flows.Flow):
    """
    A wrapper for nflow's `Flow` class to enable a different log prob calculation
    sampling strategy for training and testing, tailored to SNPE-A
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
        super().__init__(transform, distribution, embedding_net)

    def log_prob(self, inputs, context=None, proposal=None):
        if self.training:
            assert proposal is None, "Must not provide a proposal when computing the log prob for training!"
            return super().log_prob(inputs, context)

        else:
            assert proposal is not None, "Must provide a proposal when computing the log prob for evaluation!"
            if isinstance(proposal, (MultivariateNormal, utils.BoxUniform)):
                # Evaluating after the first round when the proposal prior is the prior
                log_prob = super().log_prob(inputs, context)
            else:
                # Evaluating after the second round or later when the proposal prior has been updated at least once
                log_prob = self._log_prob_proposal_posterior_mog(
                    inputs,
                    context,
                    self._proposal,
                )
            return log_prob

    def sample(self, num_samples, context=None, batch_size=None):
        if self.training:
            return super().log_prob(num_samples, context, batch_size)

        else:
            raise NotImplementedError

    def _log_prob_proposal_posterior_mog(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: "MoGFlow_SNPE_A",
    ) -> Tensor:
        """
        Return log-probability of the proposal posterior for MoG proposal.

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
            proposal: Proposal distribution.

        Returns:
            Log-probability of the proposal posterior.
        """

        # Evaluate the proposal. MDNs do not have functionality to run the embedding_net
        # and then get the mixture_components (**without** calling log_prob()). Hence,
        # we call them separately here.
        encoded_x = proposal._embedding_net(x)
        dist = proposal._distribution  # defined to avoid ugly black formatting.
        logits_p, m_p, prec_p, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_p = logits_p - torch.logsumexp(logits_p, dim=-1, keepdim=True)

        # Evaluate the density estimator.
        encoded_x = self._embedding_net(x)
        dist = self._distribution  # defined to avoid black formatting.
        logits_d, m_d, prec_d, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_d = logits_d - torch.logsumexp(logits_d, dim=-1, keepdim=True)

        # z-score theta if it z-scoring had been requested.
        theta = self._maybe_z_score_theta(theta)

        # Compute the MoG parameters of the proposal posterior.
        logits_pp, m_pp, prec_pp, cov_pp = self._automatic_posterior_transformation(
            norm_logits_p,
            m_p,
            prec_p,
            norm_logits_d,
            m_d,
            prec_d,
        )

        # Compute the log_prob of theta under the product.
        log_prob_proposal_posterior = self._mog_log_prob(
            theta,
            logits_pp,
            m_pp,
            prec_pp,
        )
        MoGFlow_SNPE_A._assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        return log_prob_proposal_posterior

    @staticmethod
    def _assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
        """
        .. note::
            Hard copy!

        Raise if tensor quantity contains any NaN or Inf element.
        """

        msg = f"NaN/Inf present in {description}."
        assert torch.isfinite(quantity).all(), msg

    @staticmethod
    def _mog_log_prob(
        theta: Tensor,
        logits_pp: Tensor,
        means_pp: Tensor,
        precisions_pp: Tensor,
    ) -> Tensor:
        r"""
        .. note::
            Hard copy!

        Returns the log-probability of parameter sets $\theta$ under a mixture of Gaussians.

        Note that the mixture can have different logits, means, covariances for any theta in
        the batch. This is because these values were computed from a batch of $x$ (and the
        $x$ in the batch are not the same).

        This code is similar to the code of mdn.py in pyknos, but it does not use
        log(det(Cov)) = -2*sum(log(diag(L))), L being Cholesky of Precision. Instead, it
        just computes log(det(Cov)). Also, it uses the above-defined helper
        `_batched_vmv()`.

        Args:
            theta: Parameters at which to evaluate the mixture.
            logits_pp: (Unnormalized) mixture components.
            means_pp: Means of all mixture components. Shape
                (batch_dim, num_components, theta_dim).
            precisions_pp: Precisions of all mixtures. Shape
                (batch_dim, num_components, theta_dim, theta_dim).

        Returns: The log-probability.
        """

        _, _, output_dim = means_pp.size()
        theta = theta.view(-1, 1, output_dim)

        # Split up evaluation into parts.
        weights = logits_pp - torch.logsumexp(logits_pp, dim=-1, keepdim=True)
        constant = -(output_dim / 2.0) * torch.log(torch.tensor([2 * math.pi]))
        log_det = 0.5 * torch.log(torch.det(precisions_pp))
        theta_minus_mean = theta.expand_as(means_pp) - means_pp
        exponent = -0.5 * utils.batched_mixture_vmv(precisions_pp, theta_minus_mean)

        return torch.logsumexp(weights + constant + log_det + exponent, dim=-1)

    def _automatic_posterior_transformation(
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

        precisions_pp, covariances_pp = self._precisions_proposal_posterior(precisions_p, precisions_d)

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

        self.z_score_theta = isinstance(self.net._transform, CompositeTransform)

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
        $pp(\theta|x) = 1/Z * q(\theta|x) * prop(\theta) / p(\theta)$

        Let's denote z-scored theta by `a`: a = (theta - mean) / std
        Then pp'(a|x) = 1/Z_2 * q'(a|x) * prop'(a) / p'(a)$

        The ' indicates that the evaluation occurs in standardized space. The constant
        scaling factor has been absorbed into Z_2.
        From the above equation, we see that we need to evaluate the prior **in
        standardized space**. We build the standardized prior in this function.

        The standardize transform that is applied to the samples theta does not use
        the exact prior mean and std (due to implementation issues). Hence, the z-scored
        prior will not be exactly have mean=0 and std=1.
        """

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
            almost_zero_mean = (self._prior.mean - estim_prior_mean) / estim_prior_std
            almost_one_std = torch.sqrt(self._prior.variance) / estim_prior_std

            if isinstance(self._prior, MultivariateNormal):
                self._maybe_z_scored_prior = MultivariateNormal(
                    almost_zero_mean,
                    torch.diag(almost_one_std),
                )
            else:
                range_ = torch.sqrt(almost_one_std * 3.0)
                self._maybe_z_scored_prior = utils.BoxUniform(almost_zero_mean - range_, almost_zero_mean + range_)
        else:
            self._maybe_z_scored_prior = self._prior

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

        See the notation of Appendix A.1 in [2].
        [2] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

        Args:
            precisions_p: Precision matrices of the proposal distribution.
            precisions_d: Precision matrices of the density estimator.

        Returns: (Precisions, Covariances) of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        precisions_p_rep = precisions_p.repeat_interleave(num_comps_d, dim=1)
        precisions_d_rep = precisions_d.repeat(1, num_comps_p, 1, 1)

        precisions_pp = precisions_d_rep - precisions_p_rep  # changed sign
        # assert torch.all(precisions_p_rep < precisions_d_rep)
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            precisions_pp += self._maybe_z_scored_prior.precision_matrix  # changed sign

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

        See the notation of Appendix A.1 in [2].
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
        summed_cov_m_prod_rep = prec_m_prod_d_rep - prec_m_prod_p_rep  # changed sign
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            summed_cov_m_prod_rep += self.prec_m_prod_prior  # changed sign

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
        logit_factors = logits_d_rep - logits_p_rep  # changed sign

        # Compute sqrt(det()/(det()*det()))
        logdet_covariances_pp = torch.logdet(covariances_pp)
        logdet_covariances_p = -torch.logdet(precisions_p)  # Sigma^tilde_k in eq. (14) in [2]
        logdet_covariances_d = -torch.logdet(precisions_d)  # Sigma_i in eq. (14) in [2]

        # Repeat the proposal and density estimator terms such that there are LK terms.
        # Same trick as has been used above.
        logdet_covariances_p_rep = logdet_covariances_p.repeat_interleave(num_comps_d, dim=1)
        logdet_covariances_d_rep = logdet_covariances_d.repeat(1, num_comps_p)

        log_sqrt_det_ratio = 0.5 * (logdet_covariances_pp + logdet_covariances_p_rep - logdet_covariances_d_rep)

        # Compute for proposal, density estimator, and proposal posterior:
        # mu_i.T * P_i * mu_i
        exponent_p = utils.batched_mixture_vmv(precisions_p, means_p)  # m_0 in eq (26) in [1]
        exponent_d = utils.batched_mixture_vmv(precisions_d, means_d)  # m_k in eq (26) in [1]
        exponent_pp = utils.batched_mixture_vmv(precisions_pp, means_pp)  # m^\prime_k in eq (26) in [1]

        # Extend proposal and density estimator exponents to get LK terms.
        exponent_p_rep = exponent_p.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_p)
        exponent = -0.5 * (exponent_p_rep - exponent_d_rep - exponent_pp)  # changed sign

        logits_pp = logit_factors + log_sqrt_det_ratio + exponent

        return logits_pp
