# Copyright (c) 2021 Li Auto Company. All rights reserved.

from math import pi
from tkinter import W

import numpy as np
import torch


class GaussianMixture(torch.nn.Module):
    """
    reference: https://github.com/ldeecke/gmm-torch.git 
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(self,
                 n_components,
                 n_features,
                 covariance_type="full",
                 eps=1.e-6,
                 init_params="kmeans",
                 mu_init=None,
                 var_init=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random", "multipath"]

        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (
                1, self.n_components,
                self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
                    self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (
                    1, self.n_components,
                    self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (
                        self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (
                    1, self.n_components, self.n_features,
                    self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (
                        self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(
                    self.var_init,
                    requires_grad=False,
                )
            else:
                self.var = torch.nn.Parameter(torch.eye(self.n_features, dtype=torch.float64).reshape(
                    1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
                                              requires_grad=False)

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1),
                                     requires_grad=False).fill_(1. / self.n_components)

        self.params_fitted = False

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic

    def fit(self, x, x_var=None, x_p=None, dist_th=1.0, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            x_var:      torch.Tensor (n, 1, d, d) or (n, 1, d)
            x_p:        torch.Tensor (n, 1)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        # warm_start means what? (init by trained paramerter?)
        if not warm_start and self.params_fitted:
            self._init_params()

        # (n, 1, d)
        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            # -> (1, k, d)
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            # Why mu.data? only cache the mu?
            self.mu.data = mu
        elif self.init_params == "multipath" and self.mu_init is None:
            # -> (1, k, d)
            mu = self.get_multipath_mu(x, x_var, x_p, n_component=self.n_components, dist_th=dist_th)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x, x_var)
            # (1)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes inane values, reinitialize model
                self.__init__(self.n_components,
                              self.n_features,
                              covariance_type=self.covariance_type,
                              mu_init=self.mu_init,
                              var_init=self.var_init,
                              eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True

    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        # prob = pi * N(x|mu,sigma)
        # weighted_log_prob = log(N(x|mu,sigma)) + log(pi)
        # (n, k, 1) + (1, k, 1) -> (n, k, 1)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            # (n, k, 1) / (n, 1, 1) -> (n, k, 1) -> (n, k)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            # (n, k, 1) -> (n, 1) -> (n)
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            p_k:        torch.Tensor (n, k)
        """
        return self.predict(x, probs=True)

    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        # e.g when n=4 probs=[0.5, 0.5] -> [3., 1.]
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        # [3., 1.] -> cat([0, 0, 0], [1]) -> [0, 0, 0, 1]
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        for k in range(self.n_components):
            if self.covariance_type == "diag":
                # mu:(1, k, d) var: (1, k, d)
                # randn: (counts[k], d)*(d)+(d) -> [counts[k], d]
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(
                    self.var[0, k])
            elif self.covariance_type == "full":
                # mu:(1, k, d) var: (1, k, d, d)
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                # stack([[d], [d], ...]) -> [counts[k], d]
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            # cat([[counts[k], d], [counts[k], d], ....], dim=0) -> [n, d]
            x = torch.cat((x, x_k), dim=0)
        # x: (n, d) y:(n)
        return x, y

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        # (n,)
        score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that
        samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        # (n, 1, d)
        x = self.check_size(x)

        if self.covariance_type == "full":
            # (1, k, d)
            mu = self.mu
            # (1, k, d, d)
            var = self.var
            # (1, k, d, d)
            precision = torch.inverse(var)
            # (n, 1, d)
            d = x.shape[-1]

            # (1,)
            log_2pi = d * np.log(2. * pi)

            # (k, 1)  log_det(var) = -log_det(precision) ?
            # log_det = - self._calculate_log_det(precision)
            log_det = self._calculate_log_det(var)

            # (n, 1, d)
            x = x.double()
            # (1, k, d)
            mu = mu.double()
            # (n, 1, d) - (1, k, d) -> (n, k, d) -> (n, k, 1, d)
            x_mu_T = (x - mu).unsqueeze(-2)
            # (n, 1, d) - (1, k, d) -> (n, k, d) -> (n, k, d, 1)
            x_mu = (x - mu).unsqueeze(-1)

            # (x-u)^T \eps^-1 (x-u)
            # (n, k, 1, d)
            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            # (n, k, 1, d) * (n, k, d, 1) -> (n, k, 1)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            # (1,) + (k, 1) + (n, k, 1)
            return -.5 * (log_2pi + log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            # (1, k, d)
            mu = self.mu
            # (1, k, d)
            # prec = 1/sqrt(var) reciprocal of the square-root of each of the elements
            # pi = prec^2
            prec = torch.rsqrt(self.var)

            log_2pi = self.n_features * np.log(2. * pi)

            # x=(n, 1, d) mu:(1, k, d)
            # parallel compute (n, k) log_p
            # log(det(A)) = log(prod(di)) = sum(log(di))
            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec**2), dim=2, keepdim=True)
            # (1, k, d) -> (1, k, 1)
            log_det = torch.sum(torch.log(self.var), dim=2, keepdim=True)  # larger number may overflow?
            # detAB=det(A)det(B) -> det(I)=1=det(A)det(A^-1) -> det(A^-1)=det(A)^-1 , det(A)=det(A^-1)^-1
            # -> log_det(A)=-log_det(A^-1)
            # -log_det(A^-1) = -log(det(A^-1))  = -log(prod(pi)) = -sum(log(pi)) -> pi=prec^2 -> -2.*sum(log(prec))
            # log_det = -2.*torch.sum(torch.log(prec), dim=2, keepdim=True)

            # (1,) + (1, k, 1) + (n, k, 1)
            return -.5 * (log_2pi + log_det + log_p)

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        # (k, )
        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()

        # (k,) -> (k, 1)
        return log_det.unsqueeze(-1)

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities)
        that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        # prob = pi * N(x|mu,sigma)
        # weighted_log_prob = log(pi * N(x|mu,sigma)) = log(N(x|mu,sigma)) + log(pi)
        # (n, k, 1) + (1, k, 1) -> (n, k, 1)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        # log(w_ik) = log(pi * N(x|mu,sigma)) - log(sum(pi * N(x|mu,sigma)))
        # log(sum(pi * N(x|mu,sigma))) = log(sum(exp(log(pi * N(x|mu,sigma)))))
        # (n, k, 1) -> (n, 1, 1),  torch.logsumexp(x)_i = \log{ \sum_j{exp(x_ij)} }
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        # log(w_ik) = (n, k, 1) - (n, 1, 1) -> (n, k, 1)
        log_resp = weighted_log_prob - log_prob_norm

        # torch.mean(log_prob_norm):(1,) log_resp:(n, k, 1)
        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, x_var, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood).
        This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            x_var:      torch.Tensor (n, 1, d) or (n, 1, d, d) or None
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        # (n, 1, d)
        x = self.check_size(x)

        n, _, d = x.shape
        k = self.n_components

        # (n, k ,1)
        resp = torch.exp(log_resp)

        # (n,) -> (n, 1, 1) -> (n, k, 1)
        # sample_weight = torch.full((n), 1./n, dtype=torch.float, device=resp.device).type_as(resp)
        sample_weight = torch.full((n,), 1. / n).type_as(resp)
        sample_weight = sample_weight.view(n, 1, 1).expand(-1, k, -1)
        resp = sample_weight * resp

        # (n, k, 1) -> (1, k, 1)
        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        # (n, k, 1) * (n, 1, d) -> (n, k, d) -> (1, k, d) / (1, k, 1) -> (1, k, d)
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            # -: (n, 1, d) - (1, k, d) -> (n, k, d)
            # unsqueeze(-1): (n, k, d) -> (n, k, d, 1)
            # unsqueeze(-2): (n, k, d) -> (n, k, 1, d)
            # matmul: (n, k, d, 1) * (n, k, 1, d) -> (n, k, d, d)
            # *: (n, k, d, d) * {(n, k, 1) -> (n, k, 1, 1)} -> (n, k, d, d) -> (1, k, d, d)
            # torch.sum:(n, k, 1) -> (1, k, 1) -> (1, k, 1, 1)
            # /: (1, k, d, d)  / (1, k, 1, 1) -> (1, k, d, d)
            # +eps: (1, k, d, d) + (d, d)
            if x_var is None:  # points fitting EM
                tmp_var = (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2))
                var = torch.sum(tmp_var * resp.unsqueeze(-1), dim=0, keepdim=True) / torch.sum(
                    resp, dim=0, keepdim=True).unsqueeze(-1) + eps
            else:  # gaussian fitting EM
                # (n, 1, d, d)
                assert x_var.shape == (n, 1, d, d)
                # x_var: (n, 1, d, d) -> (n, k, d, d)
                # +: (n, k, d, d) + (n, k, d, d) -> (n, k, d, d)
                x_var_repeat = x_var.expand(-1, k, -1, -1)
                tmp_var = x_var_repeat + (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2))
                var = torch.sum(tmp_var * resp.unsqueeze(-1), dim=0, keepdim=True) / torch.sum(
                    resp, dim=0, keepdim=True).unsqueeze(-1) + eps
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            if x_var is None:  # points fitting EM
                var = x2 - 2 * xmu + mu2 + self.eps
            else:  # gaussian fitting EM
                # (n, 1, d)
                assert x_var == (n, 1, d)
                # (n, k, 1) * (n, 1, d) -> (n, k, d) -> (1, k, d)
                # (1, k, d) / (1, k, 1) -> (1, k, d)
                tmp_var = (resp * x_var).sum(0, keepdim=True) / pi
                # (1, k, d)
                var = tmp_var + x2 - 2 * xmu + mu2 + self.eps

        # pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x, x_var=None):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
            x_var:      torch.Tensor (n, 1, d, d) or (n, 1, d)
        """
        # torch.mean(log_prob_norm):(1,) log_resp:(n, k, 1)
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, x_var, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        # prob = pi * N(x|mu,sigma) -> weighted_log_prob = log(pi * N(x|mu,sigma)) = log(pi) + log(N(x|mu,sigma))
        # (n, k, 1) + (1, k, 1) -> (n, k, 1)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        # log(sum( pi * N(x|mu,sigma), dim=1 )) = log(sum( exp(log(pi * N(x|mu,sigma))), dim=1 ))
        # (n, k, 1) -> (n, 1),  torch.logsumexp(x)_i = \log{ \sum_j{exp(x_ij)} }
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            # (n, 1) -> (1)
            return per_sample_score.mean()
        else:
            # (n, 1) -> (n,)
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)
                            ], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
                                self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [
                (self.n_components, self.n_features, self.n_features),
                (1, self.n_components, self.n_features, self.n_features)
            ], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (
                self.n_components, self.n_features, self.n_features, self.n_components, self.n_features,
                self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)
                                 ], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
                                     self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [
            (1, self.n_components, 1)
        ], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            # (n, d)
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            # (n, d) -> (k, d)
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            # (n, d) -> (n, k, d)
            # (n, k, d) - (k, d) -> (n, k, d) -> (n, k)
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            # (n, k) -> (n,)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                # (m, d) - (d) -> (m, d) -> (m) -> (1)
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            # (n, d) -> (n, k, d)
            # (n, k, d) - (k, d) -> (n, k, d) -> (n, k)
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            # (n, k) -> (n,)
            l2_cls = torch.argmin(l2_dis, dim=1)
            # (k, d)
            center_old = center.clone()

            for c in range(n_centers):
                # (m, d) -> (d)
                center[c] = x[l2_cls == c].mean(dim=0)

            # (k, d) - (k, d) -> (k, d) -> (k) -> (1)
            delta = torch.norm((center_old - center), dim=1).max()

        # (k, d) -> (1, k, d)
        return (center.unsqueeze(0) * (x_max - x_min) + x_min)

    def get_multipath_mu(self, x_mu, x_var, x_p, n_component, dist_th=1.0):
        '''
        Args:
            x_mu: torch.FloatTensor (n, 1, d)
            x_var: (n, 1, d, d) or (n, 1, d)
            x_p: (n, 1)
            n_componet: int
            dist_th: float
        Return:
            select_x_mu: (1, m, d)
        '''
        assert len(x_mu.shape) == 3
        n, _, d = x_mu.shape

        # (n, 1, d) -> (n, d)
        x_mu = x_mu.squeeze(1)
        assert x_mu.shape == (n, d)
        # (n, 1, d, d) -> (n, d, d)
        # (n, 1, d) -> (n, d)
        # x_var = x_var.squeeze(1)
        # assert x_var.shape in [(n, 1, d), (n, d)]

        # (n, 1)
        assert x_p.shape == (n, 1)

        # (m-1, d)
        select_x_mu = None
        select_num = 0
        while select_num < n_component:
            # make seletct_x_mu (n, m, d) n: try n curr sample,  m = [m-1 old sample] + [1 curr_sample]
            if select_x_mu is None:
                # (n, d) -> (1, d) -> (1, 1, d) -> (n, 1, d)
                # tmp_select_x_mu = x_mu[[0],:].unsqueeze(0).expand(n, -1, -1)
                # (n, d) -> (n, 1, d)
                tmp_select_x_mu = x_mu.unsqueeze(1)

                # (n, m, d) -> (n, 1, m, d) -> (n, n', m, d)
                tmp_select_x_mu_repeat = tmp_select_x_mu.unsqueeze(1).expand(-1, n, -1, -1)
            else:
                # select_x_mu: (m-1, d) -> (1, m-1, d) -> (n, m-1, d)
                # x_mu: (n, d) -> (n, 1, d)
                # cat([(n, m-1, d), (n, 1, d)]) -> (n, m, d)
                tmp1 = select_x_mu.unsqueeze(0).expand(n, -1, -1)
                tmp2 = x_mu.unsqueeze(1)
                tmp_select_x_mu = torch.cat([tmp1, tmp2], dim=1)

                # (n, m, d) -> (n, 1, m, d) -> (n, n', m, d)
                tmp_select_x_mu_repeat = tmp_select_x_mu.unsqueeze(1).expand(-1, n, -1, -1)

            # print("tmp_select_x_mu", tmp_select_x_mu.shape,  tmp_select_x_mu)

            m = tmp_select_x_mu.shape[1]
            # print("m", m)
            # (n, d) - (m, d) = (n, m', d) - (n', m, d)
            # try n times: (n, d) - (n, m, d) = (n', n, m', d) - (n', n, m, d)

            # (n, d) -> (n, 1, d) -> (n, m', d) -> (1, n, m', d) -> (n', n, m', d)
            x_mu_repeat = x_mu.unsqueeze(1).expand(-1, m, -1).unsqueeze(0).expand(n, -1, -1, -1)

            # (n', n, m', d) - (n, n', m, d) -> (n, n, m, d) -> (n, n, m)
            dist = torch.linalg.vector_norm(x_mu_repeat - tmp_select_x_mu_repeat, ord=2, dim=-1)
            # print("x_mu_repeat", x_mu_repeat.shape, x_mu_repeat)
            # print("tmp_select_x_mu_repeat", tmp_select_x_mu_repeat.shape, tmp_select_x_mu_repeat)
            # print("dist", dist.shape, dist)

            # (n, n, m) -> (n, n)
            x_p_sample_mask = torch.any(dist < dist_th, dim=-1).float()
            # print("x_p_sample_mask", x_p_sample_mask.shape, x_p_sample_mask)

            # (n, 1) -> (n,) -> (1, n) -> (n, n)
            x_p_repeat = x_p.squeeze(-1).unsqueeze(0).expand(n, -1)
            # print("x_p_repeat", x_p_repeat.shape, x_p_repeat)

            # (n, n) * (n, n) -> (n, n) -> (n,)
            score = torch.sum(x_p_repeat * x_p_sample_mask, dim=-1)
            # print("score", score.shape, score)

            # (n, ) -> (1, )
            curr_seletc_idx = torch.argmax(score, dim=0)
            # print("curr_seletc_idx", curr_seletc_idx.shape, curr_seletc_idx)
            if select_x_mu is None:
                # (n, 1, d) -> (1, d)
                # select_x_mu = tmp_select_x_mu[0, :, :]
                # (n, d) -> (1, d)
                select_x_mu = x_mu[[curr_seletc_idx], :]
            else:
                # select_x_mu: (m-1, d)
                # x_mu: (n, d) -> (1, d)
                # cat([(m-1, d), (1, d)]) -> (m, d)
                select_x_mu = torch.cat([select_x_mu, x_mu[[curr_seletc_idx], :]], dim=0)
            # print("select_x_mu", select_x_mu)

            select_num = select_x_mu.shape[0]
            # print("select_num:",select_num)

        # (m, d) -> (1, m, d)
        return select_x_mu.unsqueeze(0)


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).double().to(mat_a.device)

    # todo torch.matmul
    for i in range(n_components):
        # (n, k, 1, d) -> (n, 1, d) -> (n, d)
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        # (1, k, d, d) -> (d, d)
        mat_b_i = mat_b[0, i, :, :]
        # (n, d) * (d, d) -> (n, d) -> (n, 1, d)
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    # (n, k, d) * (n, k, d) -> (n, k, d) -> (n, k, 1)
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)
