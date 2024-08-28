# # learn unified anchor graph for multimodal RS data clustering
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import k_means
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import pairwise_kernels
from sklearn.kernel_approximation import Nystroem
from utils.loss import net, TVLoss

# #  version of out of samples
class AMKSC_OOS:

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

    def train(self, x_insamples, insample_label, x_outsamples):
        """
        :param x_insamples: list, [x_view1(N*D1), x_view2(N*D2), ..., x_view_v(N*Dv)],
        :return:
        """
        # # for convenience, each view is transposed before optimization
        self.insample_label = insample_label
        x_list_in = []
        for x_i in x_insamples:
            if not isinstance(x_insamples[0], torch.Tensor):
                x_list_in.append(torch.from_numpy(x_i).t().to(self.device))
            else:
                x_list_in.append(x_i.t().to(self.device))

        x_list = []
        for x_i in x_outsamples:
            if not isinstance(x_outsamples[0], torch.Tensor):
                x_list.append(torch.from_numpy(x_i).t().to(self.device))
            else:
                x_list.append(x_i.t().to(self.device))
        n_modality = len(x_list_in)
        self.n_modality = n_modality
        n_samples_out = x_list[0].shape[1]
        # # if in-samples are very large, select several representative samples
        selected_indx = []
        for i in np.unique(insample_label):
            indx = np.nonzero(i == insample_label)[0]
            np.random.shuffle(indx)
            selected_indx += indx[:self.n_anchor].tolist()
        anchors = []
        for x_ in x_list_in:
            anchors.append(x_[:, selected_indx])
        labels = insample_label[selected_indx]
        self.anchors = anchors  # x_list_in
        self.n_anchor = anchors[0].shape[1]
        self.anchors_labels = labels
        self.mu = torch.ones(n_modality, device=self.device)/n_modality  # view-specific weights
        self.Z = torch.randn((self.n_anchor, n_samples_out), device=self.device)  # unified anchor graph

        if self.kernel is not None and self.kernel != 'None':
            x_list, self.anchors = self._kernel_approc_(x_list, self.anchors)
        self.x_outsamples = x_list
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

    def predict(self):
        residual_list = []
        for i in np.unique(self.anchors_labels):
            indx = torch.from_numpy(np.nonzero(self.anchors_labels == i)[0])
            residual_k = torch.zeros(self.x_outsamples[0].shape[1], device=self.device)
            for j in range(self.n_modality):
                residual = self.mu[j] * torch.linalg.norm(torch.matmul(self.anchors[j][:, indx], self.Z[indx, :]) - self.x_outsamples[j], ord=2, dim=0)
                residual_k += residual
            residual_list.append(residual_k.cpu().numpy().tolist())

        y_pred = np.argmin(residual_list, axis=0)
        return y_pred
