# # learn unified anchor graph for multimodal RS data clustering
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import k_means
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import pairwise_kernels
from sklearn.kernel_approximation import Nystroem
from utils.loss import net, TVLoss


class AMKSC:

    def __init__(self, spatial_size, n_cluster, n_anchor, anchor_type='random', precomputed_anchors=None, lambda_tv=0.1,
                 smooth_coef=0.1, weight_decay=100, max_iter=100, n_kernel_sampling=300, epsilon=1e-6, device='cpu',
                 kernel=None, verbose=True, seed=None):
        """
        :param n_cluster:
        :param n_anchor: if anchor_type='k-means', the final n_anchor is n_anchor*n_class; if anchor_type='random',
                        n_anchor is predefined, e.g., 500; else if anchor_type='precomputed', n_anchor is automatically computed.
        :param anchor_type: one of 'k-means', 'random', and 'precomputed'.
        :param precomputed_anchors: should be specified with 'precomputed' anchor_type
        :param lambda_tv:
        :param smooth_coef:
        :param max_iter:
        :param epsilon:
        :param device:
        :param kernel:
        """
        self.spatial_size = spatial_size
        self.n_cluster = n_cluster
        self.n_anchor = n_anchor
        self.anchor_type = anchor_type
        self.lambda_tv = lambda_tv
        self.smooth_coef = smooth_coef
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.n_kernel_sampling = n_kernel_sampling
        self.epsilon = epsilon  # # error
        self.device = device
        self.kernel = kernel
        self.verbose = verbose
        self.seed = seed
        self.anchor4_kmeans = n_anchor  # used for 'k-means' type
        if anchor_type == 'precomputed':
            assert precomputed_anchors is not None
            self.anchors = precomputed_anchors

    def _generate_anchors_align_(self, x_list):
        """
        generate anchors by concatenating two view along feature dim
        :param x_list:
        :return:
        """
        x = torch.concat(x_list, dim=0)
        self.n_anchor = self.n_cluster * self.anchor4_kmeans
        k_means = KMeans(n_clusters=self.n_anchor, verbose=False, minibatch=10000)
        _ = k_means.fit_predict(x.t())
        anchors = []
        start = 0
        for i in range(len(x_list)):
            end = start + x_list[i].shape[0]
            anchors.append(k_means.centroids.t()[start:end, :])
            start = start + x_list[i].shape[0]
        return anchors

    def _generate_anchors_(self, x, n_anchors, type='random'):
        """
        generate anchors for each modality
        :param x: Tensor: D*N_samples
        :param n_anchors: k * n_clusters
        :param type: one of 'random', 'k-means', and 'precomputed'
        :return:
        """
        n_samples = x[0].shape[1]
        assert type in ['random', 'k-means',  'precomputed']
        assert n_anchors <= n_samples
        anchors = []
        for i, x_i in enumerate(x):
            if type == 'random':
                selected_index = np.random.choice(np.arange(n_samples), size=n_anchors, replace=False)
                anchors.append(x_i[:, selected_index])
            elif type == 'k-means':
                self.n_anchor = self.n_cluster * self.anchor4_kmeans
                k_means = KMeans(n_clusters=self.n_anchor, verbose=False, minibatch=10000)
                _ = k_means.fit_predict(x_i.t())
                anchors.append(k_means.centroids.t())
            elif type == 'precomputed':
                return self.anchors
            else:
                print(f'There is no type named: {type}')
        return anchors

    def _kernel_approc_(self, x_list, anchors):
        x, a = [], []
        for a_, x_, in zip(anchors, x_list):
            # # Nystroem method is used to approximate the explict kernel mapping
            appro = Nystroem(kernel=self.kernel, n_components=self.n_kernel_sampling, random_state=self.seed,
                             # gamma=0.01
                             )  # # 300 for Trt and MUUFL
            x_ = torch.from_numpy(appro.fit_transform(x_.cpu().t().numpy())).t().to(self.device)
            a_ = torch.from_numpy(appro.transform(a_.cpu().t().numpy())).t().to(self.device)
            x.append(x_)
            a.append(a_)
        return x, a

    def train(self, x):
        """
        :param x: list, [x_view1(N*D1), x_view2(N*D2), ..., x_view_v(N*Dv)],
        :return:
        """
        # # for convenience, each view is transposed before optimization
        x_list = []
        for x_i in x:
            if not isinstance(x[0], torch.Tensor):
                x_list.append(torch.from_numpy(x_i).t().to(self.device))
            else:
                x_list.append(x_i.t().to(self.device))
        n_modality = len(x_list)
        n_samples = x_list[0].shape[1]
        if self.anchor_type == 'precomputed':
            anchors = []
            for a_i in self.anchors:
                anchors.append(torch.from_numpy(a_i).t().float().to(self.device))
            self.anchors = anchors
            self.n_anchor = self.anchors[0].shape[1]
        else:
            # self.anchors = self._generate_anchors_(x_list, n_anchors=self.n_anchor, type=self.anchor_type)
            self.anchors = self._generate_anchors_align_(x_list)

        self.mu = torch.ones(n_modality, device=self.device)/n_modality  # view-specific weights
        self.Z = torch.randn((self.n_anchor, n_samples), device=self.device)  # unified anchor graph

        if self.kernel is not None and self.kernel != 'None':
            x_list, self.anchors = self._kernel_approc_(x_list, self.anchors)

        err_history = []
        mu_history = []
        err = self._err_(self.anchors, x_list)
        print(f'step {0}, error {err}')
        err_history.append(err.item())
        mu_history.append(self.mu.cpu().numpy().tolist())

        # # ========  Iterative Optimization =========== # #
        max_epoch = 500
        for iter_ in range(self.max_iter):
            # # ===== update Z, fix mu ===========
            model = net(self.spatial_size, self.n_anchor, kernel=self.kernel, smooth_coef=self.smooth_coef,
                        lambda_tv=self.lambda_tv, init_z=self.Z).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=self.weight_decay)
            for epoch_ in range(max_epoch):
                # model.train()
                optimizer.zero_grad()
                loss_ = model(x_list, self.anchors, self.mu)
                if self.verbose and epoch_ % 10 == 0:
                    print(f'    {epoch_}: {loss_.item()}')
                loss_.backward()
                optimizer.step()
            self.Z = model.Z.detach()
            # # ===== update mu, fix Z ===========
            term_1 = []
            for a_, x_, in zip(self.anchors, x_list):
                term_1.append(0.5 * torch.norm(x_ - a_.matmul(self.Z), p='fro').pow(2))
            term_1 = self.smooth_coef * torch.tensor(term_1, device=self.device)
            for j in range(n_modality):
                self.mu[j] = term_1[j].pow(1./(1. - self.smooth_coef))
            self.mu /= self.mu.sum()
            mu_history.append(self.mu.cpu().numpy().tolist())

            # # estimate error
            error = self._err_(self.anchors, x_list)
            print(f'step {iter_ + 1}, error {error}')
            err_history.append(error.item())

            if error <= self.epsilon:
                print(f'Stop early with an error of {error}')
                return self.Z
        print(err_history)
        print(mu_history)
        return self.Z

    def _err_(self, anchors, x_list):
        term_1 = []
        for a_, x_, in zip(anchors, x_list):
            term_1.append(0.5 * torch.norm(x_ - a_.matmul(self.Z), p='fro').pow(2))
        term_1 = torch.tensor(term_1, device=self.device)
        tv_loss = TVLoss(self.lambda_tv)
        term_2 = 0.5 * tv_loss(self.Z.reshape((1, self.n_anchor, self.spatial_size[0], self.spatial_size[1])))
        # # estimate error
        error = (term_1 * self.mu.pow(self.smooth_coef)).sum() + term_2
        # print(f'step {iter}, error {error}')
        return error

    # def predict(self, anchor_graph=None):
    #     if anchor_graph is None:
    #         anchor_graph = self.Z
    #     # # post-normalize Z: Z.sum(axis=0)=1
    #     anchor_graph = torch.abs(anchor_graph)
    #     norm_anchor_graph = F.softmax(anchor_graph, dim=0)
    #     diag = torch.diag(norm_anchor_graph.sum(dim=1).pow(-0.5))
    #     norm_anchor_graph_ = diag.matmul(norm_anchor_graph)  # m*N
    #     self.doubly_stochastic_Z = norm_anchor_graph_
    #     u, s, vh = torch.linalg.svd(norm_anchor_graph_, full_matrices=False)  # # descending order
    #     # # top-k rows corresponding to singular values
    #     embedding = vh[:self.n_cluster, :].t().cpu().numpy()
    #     self.embedding = embedding
    #     self.singular_values = s.cpu().numpy()
    #     # # perform k-means to obtain labels
    #     cluster_centers, labels, _ = k_means(embedding, n_clusters=self.n_cluster, random_state=None)
    #     return labels

    def predict(self, anchor_graph=None):
        if anchor_graph is None:
            anchor_graph = self.Z
        # # post-normalize Z: Z.sum(axis=0)=1
        anchor_graph = torch.abs(anchor_graph)
        norm_anchor_graph = F.softmax(anchor_graph/5, dim=0)
        # diag = torch.diag(norm_anchor_graph.sum(dim=1).pow(-0.5))
        # norm_anchor_graph_ = diag.matmul(norm_anchor_graph)  # m*N
        self.doubly_stochastic_Z = norm_anchor_graph
        u, s, vh = torch.linalg.svd(norm_anchor_graph, full_matrices=False)  # # descending order
        # # top-k rows corresponding to singular values
        embedding = vh[:self.n_cluster, :].t().cpu().numpy()
        self.embedding = embedding
        self.singular_values = s.cpu().numpy()
        # # perform k-means to obtain labels
        cluster_centers, labels, _ = k_means(embedding, n_clusters=self.n_cluster, random_state=None)
        return labels
