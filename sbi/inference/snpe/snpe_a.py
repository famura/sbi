# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import torch
from copy import deepcopy

from sbi.inference.posteriors.mog_proposal_posterior import MoGProposalPosterior
from typing import Any, Callable, Dict, Optional, Union

from torch import Tensor, eye, ones, optim
from torch.distributions import MultivariateNormal

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.types import TensorboardSummaryWriter, TorchModule
from sbi.utils import (
    del_entries,
    BoxUniform,
)


class SNPE_A(PosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Any] = None,
        density_estimator: Union[str, Callable] = "mdn",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""SNPE-A [1].

        https://github.com/mackelab/sbi/blob/main/sbi/inference/snpe/snpe_c.py
        https://github.com/mackelab/sbi/blob/main/sbi/neural_nets/mdn.py

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: torch device on which to compute, e.g. gpu, cpu.
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
        """
        if density_estimator != "mdn":
            raise NotImplementedError  # TODO

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> DirectPosterior:
        r"""
        Return density estimator that approximates the distribution $p(\theta|x)$.
        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. If None, we
                train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        kwargs = del_entries(locals(), entries=("self", "__class__"))

        self._round = max(self._data_round_index)

        if self._round == 0:
            # Algorithm 2 from [1]
            return super().train(**kwargs)

        else:
            # self._init_posterior_mdn_2nd_round()
            # TODO replicate for num_of components
            # TODO sample_for_sbi
            # TODO append ...

            # Algorithm 2 from [1]

            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            # TODO iterate over Algorithm 2 from [1]

            # Take care of z-scoring, pre-compute and store prior terms.  # TODO also do after 1st round?
            # self._set_state_for_mog_proposal()

            return super().train(**kwargs)

    def build_posterior(
        self,
        proposal: Union[MultivariateNormal, BoxUniform, MoGProposalPosterior],
        density_estimator: Optional[TorchModule] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> MoGProposalPosterior:
        r"""
        Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the TODO
        following functionality over the raw neural density estimator:

        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            proposal: TODO
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
            sample_with_mcmc: Whether to sample with MCMC. MCMC can be used to deal
                with high leakage.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior` will
                draw init locations from prior, whereas `sir` will use
                Sequential-Importance-Resampling using `init_strategy_num_candidates`
                to find init locations.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """

        if density_estimator is None:
            density_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device

        self._posterior = MoGProposalPosterior(
            method_family="snpe",
            proposal=proposal,
            neural_net=density_estimator,
            prior=self._prior,
            x_shape=self._x_shape,
            rejection_sampling_parameters=rejection_sampling_parameters,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            device=device,
        )

        self._posterior._num_trained_rounds = self._round + 1

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))
        self._model_bank[-1].net.eval()

        return deepcopy(self._posterior)

    def _log_prob_proposal_posterior(self, theta: Tensor, x: Tensor, masks: Tensor, proposal: Optional[Any]) -> Tensor:
        """
        Return the log-probability of the proposal posterior.

        .. note::
            This is the same as `self._neural_net.log_prob(theta, x)` in `_loss()` to be found in `snpe_base.py`.

        If the proposal is a MoG, the density estimator is a MoG, and the prior is
        either Gaussian or uniform, we use non-atomic loss. Else, use atomic loss (which
        suffers from leakage).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.

        Returns: Log-probability of the proposal posterior.
        """
        assert isinstance(self._neural_net, MoGProposalPosterior)
        return self._neural_net.log_prob(theta, x)

    # def _proposal_prior_converged(self, epoch: int, stop_after_epochs: int) -> bool:
    #     """Return whether the training converged yet and save best model state so far.
    #
    #     Checks for improvement in validation performance over previous epochs.
    #
    #     Args:
    #         epoch: Current epoch in training.
    #         stop_after_epochs: How many fruitless epochs to let pass before stopping.
    #
    #     Returns:
    #         Whether the training has stopped improving, i.e. has converged.
    #     """
    #     converged = False
    #
    #     proposal_prior = self._proposal_prior
    #
    #     # (Re)-start the epoch count with the first epoch or any improvement.
    #     if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
    #         self._best_val_log_prob = self._val_log_prob
    #         self._epochs_since_last_improvement = 0
    #         self._best_model_state_dict = deepcopy(proposal_prior.state_dict())
    #     else:
    #         self._epochs_since_last_improvement += 1
    #
    #     # If no validation improvement over many epochs, stop training.
    #     if self._epochs_since_last_improvement > stop_after_epochs - 1:
    #         proposal_prior.load_state_dict(self._best_model_state_dict)
    #         converged = True
    #
    #     return converged
    #
    # def _train_single_component(
    #     self,
    #     training_batch_size: int = 50,
    #     learning_rate: float = 5e-4,
    #     validation_fraction: float = 0.1,
    #     stop_after_epochs: int = 20,
    #     max_num_epochs: Optional[int] = None,
    #     clip_max_norm: Optional[float] = 5.0,
    #     exclude_invalid_x: bool = True,
    #     resume_training: bool = False,
    #     show_train_summary: bool = True,
    #     dataloader_kwargs: Optional[dict] = None,
    # ) -> DirectPosterior:
    #     r"""
    #     Return density estimator that approximates the distribution $p(\theta|x)$.
    #
    #     Args:
    #         training_batch_size: Training batch size.
    #         learning_rate: Learning rate for Adam optimizer.
    #         validation_fraction: The fraction of data to use for validation.
    #         stop_after_epochs: The number of epochs to wait for improvement on the
    #             validation set before terminating training.
    #         max_num_epochs: Maximum number of epochs to run. If reached, we stop
    #             training even when the validation loss is still decreasing. If None, we
    #             train until validation loss increases (see also `stop_after_epochs`).
    #         clip_max_norm: Value at which to clip the total gradient norm in order to
    #             prevent exploding gradients. Use None for no clipping.
    #         exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
    #             during training. Expect errors, silent or explicit, when `False`.
    #         resume_training: Can be used in case training time is limited, e.g. on a
    #             cluster. If `True`, the split between train and validation set, the
    #             optimizer, the number of epochs, and the best validation log-prob will
    #             be restored from the last time `.train()` was called.
    #         show_train_summary: Whether to print the number of epochs and validation
    #             loss after the training.
    #         dataloader_kwargs: Additional or updated kwargs to be passed to the training
    #             and validation dataloaders (like, e.g., a collate_fn)
    #
    #     Returns:
    #         Density estimator that approximates the distribution $p^{\tilde}(\theta)$, see [1].
    #     """
    #     max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs
    #
    #     # Starting index for the training set (0 = do not discard round-0 samples).
    #     start_idx = 0
    #
    #     theta, x, prior_masks = self.get_simulations(start_idx, exclude_invalid_x, warn_on_invalid=True)
    #
    #     # Dataset is shared for training and validation loaders.
    #     dataset = data.TensorDataset(
    #         theta,
    #         x,
    #         prior_masks,
    #     )
    #
    #     train_loader, val_loader = self.get_dataloaders(
    #         dataset,
    #         training_batch_size,
    #         validation_fraction,
    #         resume_training,
    #         dataloader_kwargs=dataloader_kwargs,
    #     )
    #
    #     # Create a singe Gaussian from scratch
    #     self._proposal_prior = build_mdn(
    #         batch_x=theta[self.train_indices], batch_y=x[self.train_indices], num_components=1
    #     )
    #     # self._proposal_prior = self._build_neural_net(
    #     #     theta[self.train_indices], x[self.train_indices]
    #     # )
    #     test_posterior_net_for_multi_d_x(self._proposal_prior, theta, x)
    #     self._x_shape = x_shape_from_simulation(x)
    #
    #     # Move entire net to device for training.
    #     self._proposal_prior.to(self._device)
    #
    #     if not resume_training:
    #         self.optimizer = optim.Adam(
    #             list(self._proposal_prior.parameters()),
    #             lr=learning_rate,
    #         )
    #     self.epoch, self._val_log_prob = 0, float("-Inf")
    #
    #     while self.epoch <= max_num_epochs and not self._proposal_prior_converged(self.epoch, stop_after_epochs):
    #
    #         # Train for a single epoch.
    #         self._proposal_prior.train()
    #         for batch in train_loader:
    #             self.optimizer.zero_grad()
    #             # Get batches on current device.
    #             theta_batch, x_batch, masks_batch = (
    #                 batch[0].to(self._device),
    #                 batch[1].to(self._device),
    #                 batch[2].to(self._device),
    #             )
    #
    #             batch_loss = torch.mean(-1 * self._proposal_prior.log_prob(theta_batch, x_batch))
    #             batch_loss.backward()
    #             if clip_max_norm is not None:
    #                 clip_grad_norm_(
    #                     self._proposal_prior.parameters(),
    #                     max_norm=clip_max_norm,
    #                 )
    #             self.optimizer.step()
    #
    #         self.epoch += 1
    #
    #         # Calculate validation performance.
    #         self._proposal_prior.eval()
    #         log_prob_sum = 0
    #         with torch.no_grad():
    #             for batch in val_loader:
    #                 theta_batch, x_batch, masks_batch = (
    #                     batch[0].to(self._device),
    #                     batch[1].to(self._device),
    #                     batch[2].to(self._device),
    #                 )
    #                 # Get validation log_prob.
    #                 batch_log_prob = self._proposal_prior.log_prob(theta_batch, x_batch)
    #                 log_prob_sum += batch_log_prob.sum().item()
    #
    #         # Take mean over all validation samples.
    #         self._val_log_prob = log_prob_sum / (len(val_loader) * val_loader.batch_size)
    #         # Log validation log prob for every epoch.
    #         self._summary["validation_log_probs"].append(self._val_log_prob)
    #
    #         self._maybe_show_progress(self._show_progress_bars, self.epoch)
    #
    #     # Replicate self._report_convergence_at_end()
    #     if self._proposal_prior_converged(self.epoch, stop_after_epochs):
    #         print(f"Neural network successfully converged after {self.epoch} epochs.")
    #     elif max_num_epochs == self.epoch:
    #         warn(
    #             "Maximum number of epochs `max_num_epochs={max_num_epochs}` reached,"
    #             "but network has not yet fully converged. Consider increasing it."
    #         )
    #
    #     # Update summary.
    #     self._summary["epochs"].append(self.epoch)
    #     self._summary["best_validation_log_probs"].append(self._best_val_log_prob)
    #
    #     # Update tensorboard and summary dict.
    #     self._summarize(
    #         round_=self._round,
    #         x_o=None,
    #         theta_bank=theta,
    #         x_bank=x,
    #     )
    #
    #     # Update description for progress bar.
    #     if show_train_summary:
    #         print(self._describe_round(self._round, self._summary))
    #
    #     return deepcopy(self._proposal_prior)
    #
    # def _init_posterior_mdn_2nd_round(self):
    #     pass
