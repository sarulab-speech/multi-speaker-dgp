import torch
import torch.nn as nn
from torchgp.kernels import ArcCos
from torchgp.layers import SVGP
from torchgp.likelihoods import Gaussian
from torchgp.parameter_transform import positive


class DGP(nn.Module):
    def __init__(self,
                 input_dim=535,
                 hidden_dim=128,
                 output_dim=187,
                 speaker_code_dim=100,
                 speaker_feature_dim=None,
                 num_hidden_layers=5,
                 feed_layer_indices=[1, 2, 3, 4, 5],
                 hidden_gp_inducing_size=1024,
                 speaker_gp_inducing_size=8,
                 mode='add',
                 data_size=None):
        super().__init__()

        if mode == 'add':
            speaker_feature_dim = hidden_dim
        elif mode == 'concat':
            assert speaker_feature_dim is not None, \
                f'speaker_feature_dim must be specified if mode = concat'
        else:
            raise NotImplementedError(f'mode {mode}')

        self.speaker_code_dim = speaker_code_dim
        self.feed_layer_indices = feed_layer_indices
        self.mode = mode
        self.data_size = data_size  # the number of all training data
        self.hidden_gps = nn.ModuleList()
        self.speaker_gps = nn.ModuleDict()
        self.likelihood = Gaussian(output_dim)

        for i in range(num_hidden_layers + 1):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_hidden_layers else hidden_dim
            initial_q_S_value = 1.0 if i == num_hidden_layers else 1e-6

            if mode == 'concat' and i in feed_layer_indices:
                in_dim += speaker_feature_dim

            kernel = ArcCos(in_dim)
            layer = SVGP(
                kernel=kernel,
                input_dim=in_dim,
                output_dim=out_dim,
                num_inducings=hidden_gp_inducing_size,
                initial_q_S_value=initial_q_S_value,
            )
            self.hidden_gps.append(layer)

            if i in feed_layer_indices:
                kernel = ArcCos(speaker_code_dim)
                layer = SVGP(
                    kernel=kernel,
                    input_dim=speaker_code_dim,
                    output_dim=speaker_feature_dim,
                    num_inducings=speaker_gp_inducing_size,
                    initial_q_S_value=initial_q_S_value,
                )
                self.speaker_gps[str(i)] = layer

    def forward(self, x):
        speaker_code = x[:, -self.speaker_code_dim:]
        f = x[:, :-self.speaker_code_dim]

        for i in range(len(self.hidden_gps)):
            if i in self.feed_layer_indices:
                fs = self.speaker_gps[str(i)](speaker_code)
                if self.mode == 'add':
                    f += fs
                else:
                    f = torch.cat((f, fs), dim=1)

            if i < len(self.hidden_gps) - 1:
                f = self.hidden_gps[i](f)
            else:
                f = self.hidden_gps[i].pred_mean_and_var(f)

        return f

    def calc_bound(self, f_pred, y):
        '''Calculate evidence lower bound (ELBO) of log marginal likelihood.'''
        pe = self.likelihood.predictive_expectation(y, f_pred)
        bound_info = {'predictive_expectation': pe.item(),
                      'kld': dict(),
                      }

        kld_all = 0
        # calculate KL divergence for hidden GPs
        for i, h_layer in enumerate(self.hidden_gps):
            kld = h_layer.kl_divergence()
            bound_info['kld'][f'hidden_gp_{i}'] = kld.item()
            kld_all += kld
        # calculate KL divergence for speaker GPs
        for i, s_layer in self.speaker_gps.items():
            kld = s_layer.kl_divergence()
            bound_info['kld'][f'speaker_gp_{i}'] = kld.item()
            kld_all += kld

        bound = pe - (len(y) / self.data_size) * kld_all
        return bound, bound_info

    def predict(self, x):
        '''Predict mean and variance.'''
        f_pred = self.forward(x)
        y_pred_dict = self.likelihood.predict(f_pred)
        return y_pred_dict


class DGPLVM(nn.Module):
    def __init__(self,
                 input_dim=535,
                 hidden_dim=128,
                 output_dim=187,
                 speaker_code_dim=100,
                 speaker_latent_dim=None,
                 num_hidden_layers=5,
                 feed_layer_indices=[1, 2, 3, 4, 5],
                 hidden_gp_inducing_size=1024,
                 initial_latent_variance=1e-4,
                 mode='concat',
                 data_size=None):
        super().__init__()

        if mode == 'add':
            speaker_latent_dim = hidden_dim
        elif mode == 'concat':
            assert speaker_latent_dim is not None, \
                f'speaker_latent_dim must be specified if mode = concat'
        else:
            raise NotImplementedError(f'mode {mode}')

        self.speaker_code_dim = speaker_code_dim
        self.speaker_latent_dim = speaker_latent_dim
        self.feed_layer_indices = feed_layer_indices
        self.mode = mode
        self.data_size = data_size  # the number of all training data
        self.hidden_gps = nn.ModuleList()
        self.likelihood = Gaussian(output_dim)

        for i in range(num_hidden_layers + 1):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_hidden_layers else hidden_dim
            initial_q_S_value = 1.0 if i == num_hidden_layers else 1e-6

            if mode == 'concat' and i in feed_layer_indices:
                in_dim += speaker_latent_dim

            kernel = ArcCos(in_dim)
            layer = SVGP(
                kernel=kernel,
                input_dim=in_dim,
                output_dim=out_dim,
                num_inducings=hidden_gp_inducing_size,
                initial_q_S_value=initial_q_S_value,
            )
            self.hidden_gps.append(layer)

        # variational distribution of latent speaker representations.
        # k-th row corresponds to k-th speaker.
        self.mu = nn.Parameter(torch.randn(speaker_code_dim, speaker_latent_dim))
        sigma_raw = torch.ones_like(self.mu) * initial_latent_variance
        self.sigma_ = nn.Parameter(positive.backward(sigma_raw))

    @property
    def sigma(self):
        return positive.forward_tensor(self.sigma_)

    def forward(self, x):
        speaker_code = x[:, -self.speaker_code_dim:]
        f = x[:, :-self.speaker_code_dim]

        indices = torch.max(speaker_code, dim=1)[1]
        mu = torch.index_select(self.mu, dim=0, index=indices)
        sigma = torch.index_select(self.sigma, dim=0, index=indices)

        if self.training:
            r = mu + torch.randn_like(mu) * torch.sqrt(sigma)
        else:
            r = mu

        for i in range(len(self.hidden_gps)):
            if i in self.feed_layer_indices:
                if self.mode == 'add':
                    f += r
                else:
                    f = torch.cat((f, r), dim=1)

            if i < len(self.hidden_gps) - 1:
                f = self.hidden_gps[i](f)
            else:
                f = self.hidden_gps[i].pred_mean_and_var(f)

        return f

    def calc_bound(self, f_pred, y):
        '''Calculate evidence lower bound (ELBO) of log marginal likelihood.'''
        pe = self.likelihood.predictive_expectation(y, f_pred)
        bound_info = {'predictive_expectation': pe.item(),
                      'kld': dict(),
                      }

        kld_all = 0
        # calculate KL divergence for hidden GPs
        for i, h_layer in enumerate(self.hidden_gps):
            kld = h_layer.kl_divergence()
            bound_info['kld'][f'hidden_gp_{i}'] = kld.item()
            kld_all += kld
        # calculate KL divergence for distribution of latent speaker representations.
        kld = -0.5 * torch.sum(torch.log(self.sigma))
        kld -= 0.5 * self.speaker_code_dim * self.speaker_latent_dim
        kld += 0.5 * torch.sum(self.sigma)
        kld += 0.5 * torch.sum(self.mu ** 2)
        bound_info["kld"]["spk"] = kld.item()
        kld_all += kld

        bound = pe - (len(y) / self.data_size) * kld_all
        return bound, bound_info

    def predict(self, x):
        '''Predict mean and variance.'''
        f_pred = self.forward(x)
        y_pred_dict = self.likelihood.predict(f_pred)
        return y_pred_dict
