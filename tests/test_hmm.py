'Test the HMM model.'


# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import numpy as np
from scipy.special import logsumexp
import torch
import beer
from basetest import BaseTest
import unittest


def forward(init_states, trans_mat, llhs):
    init_log_prob = -np.log(len(init_states))
    log_trans_mat = np.log(trans_mat)
    log_alphas = np.zeros_like(llhs) - np.inf
    log_alphas[0, init_states] = llhs[0, init_states] + init_log_prob
    for i in range(1, llhs.shape[0]):
        log_alphas[i] = llhs[i]
        log_alphas[i] += logsumexp(log_alphas[i-1] + log_trans_mat.T, axis=1)
    return log_alphas

def backward(final_states, trans_mat, llhs):
    final_log_prob = -np.log(len(final_states))
    log_trans_mat = np.log(trans_mat)
    log_betas = np.zeros_like(llhs) - np.inf
    log_betas[-1, final_states] = final_log_prob
    for i in reversed(range(llhs.shape[0] - 1)):
        log_betas[i] = logsumexp(log_trans_mat + llhs[i+1] + log_betas[i+1], axis=1)
    return log_betas

def viterbi(init_states, final_states, trans_mat, llhs):
    init_log_prob = -np.log(len(init_states))
    backtrack = np.zeros_like(llhs, dtype=int)
    omega = np.zeros(llhs.shape[1]) - float('inf')
    omega[init_states] = llhs[0, init_states] + init_log_prob
    log_trans_mat = np.log(trans_mat)

    for i in range(1, llhs.shape[0]):
        hypothesis = omega + log_trans_mat.T
        backtrack[i] = np.argmax(hypothesis, axis=1)
        omega = llhs[i] + hypothesis[range(len(log_trans_mat)), backtrack[i]]

    path = [final_states[np.argmax(omega[final_states])]]
    for i in reversed(range(1, len(llhs))):
        path.insert(0, backtrack[i, path[0]])

    return np.asarray(path)


def create_modelset_diag(ncomps, dim, type_t):
    posts = [beer.NormalGammaPrior(torch.zeros(dim).type(type_t),
                                   torch.ones(dim).type(type_t),
                                   1.)
             for _ in range(ncomps)]
    modelset = beer.NormalDiagonalCovarianceSet(
        beer.NormalGammaPrior(torch.zeros(dim).type(type_t),
                              torch.ones(dim).type(type_t),
                              1.),
        posts
    )
    return modelset


def create_modelset_full(ncomps, dim, type_t):
    posts = [beer.NormalWishartPrior(torch.zeros(dim).type(type_t),
                                     torch.eye(dim).type(type_t),
                                     1.)
             for _ in range(ncomps)]
    modelset = beer.NormalFullCovarianceSet(
        beer.NormalWishartPrior(torch.zeros(dim).type(type_t),
                                torch.eye(dim).type(type_t),
                                1.),
        posts
    )
    return modelset

# pylint: disable=R0902
class TestForwardBackwardViterbi(BaseTest):

    def setUp(self):
        self.nstates = int(1 + np.random.randint(100, size=1))
        self.npoints = int(1 + np.random.randint(100, size=1)) + self.nstates

        n_init_states = int(1 + np.random.randint(self.nstates, size=1))
        self.init_states = list(np.random.choice(range(0, self.nstates), 
                                                 n_init_states, replace=False))
        n_final_states = int(1 + np.random.randint(self.nstates, size=1))
        self.final_states = list(np.random.choice(range(0, self.nstates), 
                                                 n_final_states, replace=False))
        tmp_trans = np.random.ranf((self.nstates, self.nstates))
        self.trans_mat = torch.from_numpy(tmp_trans / \
            tmp_trans.sum(axis=1).reshape(-1, 1)).type(self.type)
        tmp_llh = np.random.ranf((self.npoints, self.nstates))
        self.llhs = torch.from_numpy(np.log(tmp_llh / \
            tmp_llh.sum(axis=1).reshape(-1, 1))).type(self.type)
    
    def test_forward(self):
        log_alphas1 = forward(self.init_states, self.trans_mat.numpy(), 
                              self.llhs.numpy())
        log_alphas2 = beer.HMM.baum_welch_forward(self.init_states, self.trans_mat,
                                       self.llhs).numpy()
        self.assertArraysAlmostEqual(log_alphas1, log_alphas2)
    
    def test_backward(self):
        log_betas1 = backward(self.final_states, self.trans_mat.numpy(),
                              self.llhs.numpy())
        log_betas2 = beer.HMM.baum_welch_backward(self.final_states, self.trans_mat,
                                      self.llhs).numpy()
        self.assertArraysAlmostEqual(log_betas1, log_betas2)


    def test_viterbi(self):
        path1 = viterbi(self.init_states, self.final_states, 
                        self.trans_mat.numpy(), 
                        self.llhs.numpy())
        path2 = beer.HMM.viterbi(self.init_states, self.final_states,
                                 self.trans_mat, self.llhs).numpy()
        self.assertArraysAlmostEqual(path1, path2)


# pylint: disable=R0902
class TestHMM(BaseTest):

    def setUp(self):
        self.npoints = int(1 + torch.randint(100, (1, 1)).item())
        self.dim = int(1 + torch.randint(100, (1, 1)).item())
        self.data = torch.randn(self.npoints, self.dim).type(self.type)
        self.means = torch.randn(self.npoints, self.dim).type(self.type)
        self.vars = torch.randn(self.npoints, self.dim).type(self.type) ** 2
        self.prior_count = 1e-2 + 100 * torch.rand(1).item()
        self.nstates = int(1 + torch.randint(100, (1, 1)).item())
        modelsets = [
            beer.NormalDiagonalCovarianceSet.create(
                torch.zeros(self.dim).type(self.type),
                torch.ones(self.dim).type(self.type),
                self.nstates,
                noise_std=0.1
            ),
            beer.NormalFullCovarianceSet.create(
                torch.zeros(self.dim).type(self.type),
                torch.eye(self.dim).type(self.type),
                self.nstates,
                noise_std=0.1
            ),
            beer.NormalSetSharedDiagonalCovariance.create(
                torch.zeros(self.dim).type(self.type),
                torch.ones(self.dim).type(self.type),
                self.nstates,
                noise_std=0.1
            ),
            beer.NormalSetSharedFullCovariance.create(
                torch.zeros(self.dim).type(self.type),
                torch.eye(self.dim).type(self.type),
                self.nstates,
                noise_std=0.1
            )
        ]

        n_init_states = int(1 + torch.randint(self.nstates, (1, 1)).item())
        self.init_states = list(np.random.choice(range(0, self.nstates), 
                                                 n_init_states, 
                                                 replace=False))
        n_final_states = int(1 + torch.randint(self.nstates, (1, 1)).item())
        self.final_states = list(np.random.choice(range(0, self.nstates),
                                                  n_final_states,
                                                  replace=False))
        tmp_mat = torch.rand((self.nstates, self.nstates))
        self.trans_mat = (tmp_mat / tmp_mat.sum(dim=1).view(-1, 1)).type(self.type)

        self.hmms = []
        for modelset in modelsets:
            self.hmms.append(beer.HMM.create(self.init_states, self.final_states,
                             self.trans_mat, modelset))
    
    def test_create(self):
        for i, hmm in enumerate(self.hmms):
            with self.subTest(i=i):
                self.assertEqual(len(hmm.modelset), self.nstates)

    def test_forward(self):
        for i, hmm in enumerate(self.hmms):
            with self.subTest(i=i):
                stats = hmm.sufficient_statistics(self.data)
                pc_exp_llh = hmm.modelset(stats)
                log_alphas = hmm.baum_welch_forward(self.init_states, self.trans_mat,
                                                        pc_exp_llh)
                log_betas = hmm.baum_welch_backward(self.final_states, self.trans_mat,
                                                        pc_exp_llh)
                log_posts = (log_alphas + log_betas).numpy()
                log_norm = logsumexp(log_posts[0])
                exp_llh1 = log_norm
                exp_llh2 = hmm(stats).numpy()
                self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

    def test_exp_llh_labels(self):
        for i, model in enumerate(self.hmms):
            with self.subTest(i=i):
                label_idxs = torch.zeros(self.data.size(0)).long()
                elabels = beer.onehot(label_idxs, len(model.modelset))
                mask = torch.log(elabels).numpy()
                elabels = elabels.numpy()
                stats = model.sufficient_statistics(self.data)
                pc_exp_llh = model.modelset(stats)
                pc_exp_llh = pc_exp_llh.numpy()
                pc_exp_llh += mask
                exp_llh1 = logsumexp(pc_exp_llh, axis=1)
                exp_llh2 = model(stats, label_idxs).numpy()
                self.assertArraysAlmostEqual(exp_llh1, exp_llh2)

__all__ = ['TestHMM', 'TestForwardBackwardViterbi']
