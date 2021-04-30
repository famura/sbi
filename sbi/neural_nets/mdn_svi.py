"""
Implementation of models based on
C. M. Bishop, "Mixture Density Networks", NCRG Report (1994)

Taken from http://github.com/conormdurkan/lfi. See there for copyright.
"""
import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from copy import deepcopy

from nflows.utils import torchutils


# This implementation based on Conor M. Durkan's et al. lfi package (2020).
# https://github.com/conormdurkan/lfi/blob/master/src/nn_/nde/mdn.py
class MultivariateGaussianMDNSVI(nn.Module):
    """
    Conditional density mixture of multivariate Gaussians, after Bishop [1].

    A multivariate Gaussian mixture with full (rather than diagonal) covariances

    [1] Bishop, C.: 'Mixture Density Networks', Neural Computing Research Group Report
    1994 https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
    """

    def __init__(
            self,
            features: int,
            context_features: int,
            hidden_features: int,
            num_components: int,
            hidden_net: nn.Module,
            # hidden_layers: int = 5,
            # hidden_act: int = nn.ReLu,
            # hidden_dropout: int = 0.0,
            custom_initialization=False,
            svi=False,
            lam=0.01,
            embedding_net=None,
    ):
        """Mixture of multivariate Gaussians with full diagonal.

        Args:
            features: Dimension of output density.
            context_features: Dimension of inputs.
            hidden_features: Dimension of final layer of `hidden_net`.
            hidden_net: A Module which outputs final hidden representation before
                paramterization layers (i.e logits, means, and log precisions).
            num_components: Number of mixture components.
            custom_initialization: XXX
        """

        super().__init__()

        self._features = features
        self._context_features = context_features
        self._hidden_features = hidden_features
        self._num_components = num_components  # 1
        self._svi = svi
        self._lam = lam
        # self._hidden_act = hidden_act
        # self._hidden_dropout = hidden_dropout

        self._num_upper_params = (features * (features - 1)) // 2

        self._row_ix, self._column_ix = np.triu_indices(features, k=1)
        self._diag_ix = range(features)

        # Modules
        # self._hidden_net = [nn.Linear(context_features, hidden_features)]
        self._hidden_net = hidden_net

        self._logits_layer = nn.Linear(hidden_features, num_components)

        self._means_layer = nn.Linear(hidden_features, num_components * features)

        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )

        self._upper_layer = nn.Linear(
            hidden_features, num_components * self._num_upper_params
        )

        # XXX docstring text
        # embedding_net: NOT IMPLEMENTED
        #         A `nn.Module` which has trainable parameters to encode the
        #         context (conditioning). It is trained jointly with the MDN.
        if embedding_net is not None:
            raise NotImplementedError

        # Constant for numerical stability.
        self._epsilon = 1e-2

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

        if self._svi:
            self._hidden_net_std = deepcopy(hidden_net)
            self._logits_layer_std = deepcopy(nn.Linear(hidden_features, num_components))
            self._means_layer_std = deepcopy(nn.Linear(hidden_features, num_components * features))
            self._unconstrained_diagonal_layer_std = deepcopy(nn.Linear(
                hidden_features, num_components * features
            ))
            self._upper_layer_std = deepcopy(nn.Linear(
                hidden_features, num_components * self._num_upper_params
            ))
            self._svi_initialize()

    def get_mixture_components(
            self, context: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return logits, means, precisions and two additional useful quantities.

        Args:
            context: Input to the MDN, leading dimension is batch dimension.

        Returns:
            A tuple with logits (num_components), means (num_components x output_dim),
            precisions (num_components, output_dim, output_dim), sum log diag of
            precision factors (1), precision factors (upper triangular precision factor
            A such that SIGMA^-1 = A^T A.) All batched.
        """

        if self._svi:
            h_m = self._hidden_net(context)
            h_std = self._hidden_net_std(context)
            h = self._reparameterize(h_m, h_std)

            # Logits and Means are unconstrained and are obtained directly from the
            # output of a linear layer.
            logits_m = self._logits_layer(h)
            logits_std = self._logits_layer_std(h)
            logits = self._reparameterize(logits_m, logits_std)

            means_m = self._means_layer(h).view(-1, self._num_components, self._features)
            means_std = self._means_layer_std(h).view(-1, self._num_components, self._features)
            means = self._reparameterize(means_m, means_std)

            # Unconstrained diagonal and upper triangular quantities are unconstrained.
            unconstrained_diagonal_m = self._unconstrained_diagonal_layer(h)
            unconstrained_diagonal_std = self._unconstrained_diagonal_layer_std(h)
            unconstrained_diagonal = self._reparameterize(unconstrained_diagonal_m, unconstrained_diagonal_std).view(
                -1, self._num_components, self._features
            )

            # one dimensional feature does not involve upper triangular parameters
            if self._features > 1:
                upper_m = self._upper_layer(h)
                upper_std = self._upper_layer_std(h)
                upper = self._reparameterize(upper_m, upper_std).view(
                    -1, self._num_components, self._num_upper_params
                )
        else:
            h = self._hidden_net(context)

            # Logits and Means are unconstrained and are obtained directly from the
            # output of a linear layer.
            logits = self._logits_layer(h)
            means = self._means_layer(h).view(-1, self._num_components, self._features)

            # Unconstrained diagonal and upper triangular quantities are unconstrained.
            unconstrained_diagonal = self._unconstrained_diagonal_layer(h).view(
                -1, self._num_components, self._features
            )

            # one dimensional feature does not involve upper triangular parameters
            if self._features > 1:
                upper = self._upper_layer(h).view(
                    -1, self._num_components, self._num_upper_params
                )

        # Elements of diagonal of precision factor must be positive
        # (recall precision factor A such that SIGMA^-1 = A^T A).
        diagonal = F.softplus(unconstrained_diagonal) + self._epsilon

        # Create empty precision factor matrix, and fill with appropriate quantities.
        precision_factors = torch.zeros(
            means.shape[0],
            self._num_components,
            self._features,
            self._features,
            device=context.device,
        )
        precision_factors[..., self._diag_ix, self._diag_ix] = diagonal

        if self._features > 1:
            precision_factors[..., self._row_ix, self._column_ix] = upper

        # Precisions are given by SIGMA^-1 = A^T A.
        precisions = torch.matmul(
            torch.transpose(precision_factors, 2, 3), precision_factors
        )

        # The sum of the log diagonal of A is used in the likelihood calculation.
        sumlogdiag = torch.sum(torch.log(diagonal), dim=-1)

        return logits, means, precisions, sumlogdiag, precision_factors

    def log_prob(self, inputs: Tensor, context=Optional[Tensor]) -> Tensor:
        """Return log MoG(inputs|context) where MoG is a mixture of Gaussians density.

        The MoG's parameters (mixture coefficients, means, and precisions) are the
        outputs of a neural network.

        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
            context: Conditioning variable, leading dim interpreted as batch dimension.

        Returns:
            Log probability of inputs given context under a MoG model.
        """

        logits, means, precisions, sumlogdiag, _ = self.get_mixture_components(context)
        return self.log_prob_mog(inputs, logits, means, precisions, sumlogdiag)

    @staticmethod
    def log_prob_mog(
            inputs: Tensor,
            logits: Tensor,
            means: Tensor,
            precisions: Tensor,
            sumlogdiag: Tensor,
    ) -> Tensor:
        """
        Return the log-probability of `inputs` under a MoG with specified parameters.

        Unlike the `log_prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG 
        parameters are already known.

        Args:
            inputs: Location at which to evaluate the MoG.
            logits: Log-weights of each component of the MoG. Shape: (batch_size,
                num_components).
            means: Means of each MoG, shape (batch_size, num_components, parameter_dim).
            precisions: Precision matrices of each MoG. Shape:
                (batch_size, num_components, parameter_dim, parameter_dim).
            sumlogdiag: Sum of the logarithm of the diagonal of the precision matrix.
                Shape: (batch_size, num_components).

        Returns:
            Log-probabilities of each input.
        """
        batch_size, n_mixtures, output_dim = means.size()
        inputs = inputs.view(-1, 1, output_dim)

        # Split up evaluation into parts.
        a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        b = -(output_dim / 2.0) * np.log(2 * np.pi)
        c = sumlogdiag
        d1 = (inputs.expand_as(means) - means).view(
            batch_size, n_mixtures, output_dim, 1
        )
        d2 = torch.matmul(precisions, d1)
        d = -0.5 * torch.matmul(torch.transpose(d1, 2, 3), d2).view(
            batch_size, n_mixtures
        )

        return torch.logsumexp(a + b + c + d, dim=-1)

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """
        Return num_samples independent samples from MoG(inputs | context).

        Generates num_samples samples for EACH item in context batch i.e. returns
        (num_samples * batch_size) samples in total.

        Args:
            num_samples: Number of samples to generate.
            context: Conditioning variable, leading dimension is batch dimension.

        Returns:
            Generated samples: (num_samples, output_dim) with leading batch dimension.
        """

        # Get necessary quantities.
        logits, means, _, _, precision_factors = self.get_mixture_components(context)
        return self.sample_mog(num_samples, logits, means, precision_factors)

    @staticmethod
    def sample_mog(
            num_samples: int, logits: Tensor, means: Tensor, precision_factors: Tensor
    ) -> Tensor:
        """
        Return samples of a MoG with specified parameters.

        Unlike the `sample()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG 
        parameters are already known.

        Args:
            num_samples: Number of samples to generate.
            logits: Log-weights of each component of the MoG. Shape: (batch_size,
                num_components).
            means: Means of each MoG. Shape: (batch_size, num_components,
                parameter_dim).
            precision_factors: Cholesky factors of each component of the MoG. Shape:
                (batch_size, num_components, parameter_dim, parameter_dim).

        Returns:
            Tensor: Samples from the MoG.
        """
        batch_size, n_mixtures, output_dim = means.shape

        # We need (batch_size * num_samples) samples in total.
        means, precision_factors = (
            torchutils.repeat_rows(means, num_samples),
            torchutils.repeat_rows(precision_factors, num_samples),
        )

        # Normalize the logits for the coefficients.
        coefficients = F.softmax(logits, dim=-1)  # [batch_size, num_components]

        # Choose num_samples mixture components per example in the batch.
        choices = torch.multinomial(
            coefficients, num_samples=num_samples, replacement=True
        ).view(
            -1
        )  # [batch_size, num_samples]

        # Create dummy index for indexing means and precision factors.
        ix = torchutils.repeat_rows(torch.arange(batch_size), num_samples)

        # Select means and precision factors.
        chosen_means = means[ix, choices, :]
        chosen_precision_factors = precision_factors[ix, choices, :, :]

        # Batch triangular solve to multiply standard normal samples by inverse
        # of upper triangular precision factor.
        zero_mean_samples, _ = torch.triangular_solve(
            torch.randn(
                batch_size * num_samples, output_dim, 1
            ),  # Need dummy final dimension.
            chosen_precision_factors,
        )

        # Mow center samples at chosen means, removing dummy final dimension
        # from triangular solve.
        samples = chosen_means + zero_mean_samples.squeeze(-1)

        return samples.reshape(batch_size, num_samples, output_dim)

    @staticmethod
    def _reparameterize(mean, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mean + sigma * eps

    def _initialize(self) -> None:
        """
        Initialize MDN so that mixture coefficients are approximately uniform,
        and covariances are approximately the identity.
        """

        # Initialize mixture coefficients to near uniform.
        self._logits_layer.weight.data = self._epsilon * torch.randn(
            self._num_components, self._hidden_features
        )
        self._logits_layer.bias.data = self._epsilon * torch.randn(self._num_components)

        # Initialize diagonal of precision factors to inverse of softplus at 1.
        self._unconstrained_diagonal_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._features, self._hidden_features
        )
        self._unconstrained_diagonal_layer.bias.data = torch.log(
            torch.exp(torch.tensor([1 - self._epsilon])) - 1
        ) * torch.ones(
            self._num_components * self._features
        ) + self._epsilon * torch.randn(
            self._num_components * self._features
        )

        # Initialize off-diagonal of precision factors to zero.
        self._upper_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params, self._hidden_features
        )
        self._upper_layer.bias.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params
        )

    def _svi_initialize(self) -> None:
        log_var = math.log(self._lam)

        # Initialize the parameters of the hidden layer
        for param in self._hidden_net.parameters():
            param = torch.zeros_like(param)
        for param in self._hidden_net_std.parameters():
            param = log_var * torch.ones_like(param)

        # initialize the parameters of the logits layers
        for param in self._logits_layer.parameters():
            param = torch.zeros_like(param)
        for param in self._logits_layer_std.parameters():
            param = log_var * torch.ones_like(param)

        # initialize the parameters of the means layers
        for param in self._means_layer.parameters():
            param = torch.zeros_like(param)
        for param in self._means_layer_std.parameters():
            param = log_var * torch.ones_like(param)

        # initialize the parameters of the unconstrained diagonal layers
        for param in self._unconstrained_diagonal_layer.parameters():
            param = torch.zeros_like(param)
        for param in self._unconstrained_diagonal_layer_std.parameters():
            param = log_var * torch.ones_like(param)

        # initialize the parameters of the upper layers
        for param in self._upper_layer.parameters():
            param = torch.zeros_like(param)
        for param in self._upper_layer_std.parameters():
            param = log_var * torch.ones_like(param)

    def kl(self):
        """
        Calculates the KL-divergence of the Gaussian NN-parameters and a N(0, \lambda^{-1}) distributed density.

        Returns: Kl-divergence
        """
        # TODO: Testing
        kl_loss = 0
        sum_log_var = 0
        for name, param in self.named_parameters():
            if "std" in name:
                log_var_sum = sum_log_var + param.sum()
                kl_loss = kl_loss + torch.sum(torch.exp(param ** 2))
            else:
                kl_loss = kl_loss + torch.sum(param ** 2)
        return self._lam / 2 * kl_loss - sum_log_var


# XXX This -> tests
def main():
    # probs = torch.Tensor([[1, 0], [0, 1]])
    # samples = torch.multinomial(probs, num_samples=5, replacement=True)
    # print(samples)
    # quit()
    mdn = MultivariateGaussianMDNSVI(
        features=2,
        context_features=3,
        hidden_features=5,
        hidden_net=nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
        ),
        num_components=4,
        svi=True,
    )
    inputs = torch.randn(1, 3)
    samples = mdn.sample(9, inputs)
    print(samples.shape)
    print(mdn.kl())


if __name__ == "__main__":
    main()
