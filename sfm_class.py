import numpy as np
import matplotlib.image as mpimg
import torch
from torch.autograd import Function
from torch.nn import Module
import torch.nn as nn
from settings import curr_dir
from sfm import ceresBA
from scipy.sparse import csr_matrix, diags, coo_matrix
from scipy.sparse.linalg import spsolve
import torch.nn.functional as func
from myfunctions import csr_to_coo, coo_to_crs, sigmoid
import matplotlib.pyplot as plt
from myfunctions import angle_to_rotation_matrix, procrustes, cloud_to_cloud_distance

opt_mod = 0
opt_lr = 0.01
batch_size = 4

monitor_Hvariance = []
monitor_Hmean = []
monitor_dwvariance = []
monitor_dwmean = []
monitor_outgradmean = []
monitor_outgradavar = []


class RobustBa(Function):
    @staticmethod
    def forward(ctx, p_t, i_t, c_t, ob, ca, tr, weights, mode, robust=0.001):

        def rho_1(x, val):
            temp = np.where(x < val, 1, 1/x)
            return temp

        def rho_2(x, val):
            temp = np.where(x < val, 0, -1/np.square(x))
            return temp

        # 4 options for mode: 0, 1, 2 -> train subset of parameters (X & P-R, X & P, X & P & I); 4 -> Evaluation only
        v_grad = np.zeros(1000000)
        v_row = np.zeros(1000000).astype(np.int32)
        v_col = np.zeros(1000000).astype(np.int32)
        cost = np.zeros(1)
        residuals = np.zeros(1000000)
        gradients = np.zeros(1000000)
        rotations = np.zeros(10000)

        w = weights.clone().detach().numpy()
        p = p_t.clone().detach().numpy()
        i = i_t.clone().detach().numpy()
        c = c_t.clone().detach().numpy()
        ob = ob.clone().detach().numpy()
        ca = ca.clone().detach().numpy()
        tr = tr.clone().detach().numpy()

        # Running the Bundle adjustment

        if not ceresBA(p, i, c, ob, ca, tr, v_row, v_col, v_grad, w, cost, residuals, gradients, rotations, mode):
            print("Call to CeresBA failed to converge")

        v_c = v_col[2:2 + v_col[0]]
        v_r = v_row[2:2 + v_row[0]]
        v_g = v_grad[1:1 + int(v_grad[0])]
        resid = residuals[1:1 + int(residuals[0])]
        nabla = gradients[1:1 + int(gradients[0])]
        jacob = csr_matrix((v_g, v_c, v_r), shape=(v_row[1], v_col[1]))

        FinalNewton = False
        if FinalNewton:
            r1 = rho_1(resid, 4)
            r2 = rho_2(resid, 4)
            corr = diags(r1 + 2 * r2 * resid * resid)
            H = jacob.T.__matmul__(corr.dot(jacob))
            H += diags(robust * np.ones(H.shape[0]))
            update = spsolve(H, nabla)
            if mode == 0:
                """TODO: fix this to work for any choice of camera"""
                idx = np.reshape(range(len(p-1)), (-1, 3))[1::2].reshape(-1)
                p[idx] -= update[:int(np.size(p) / 2)]
                i -= update[int(np.size(p) / 2): np.size(i) + int(np.size(p) / 2)]
                c -= update[np.size(i) + int(np.size(p) / 2):]
            else:
                p -= update[:np.size(p)]
                i -= update[np.size(p): np.size(i) + np.size(p)]
                c -= update[np.size(i) + np.size(p):]

            v_grad = np.zeros(1000000)
            v_row = np.zeros(1000000).astype(np.int32)
            v_col = np.zeros(1000000).astype(np.int32)
            cost = np.zeros(1)
            residuals = np.zeros(1000000)
            gradients = np.zeros(1000000)
            if not ceresBA(p, i, c, ob, ca, tr, v_row, v_col, v_grad, w, cost, residuals, gradients, rotations, 4):
                print("Call to CeresBA failed to converge")

            v_c = v_col[2:2 + v_col[0]]
            v_r = v_row[2:2 + v_row[0]]
            v_g = v_grad[1:1 + int(v_grad[0])]
            resid = residuals[1:1 + int(residuals[0])]
            nabla = gradients[1:1 + int(gradients[0])]
            jacob = csr_matrix((v_g, v_c, v_r), shape=(v_row[1], v_col[1]))

        w_extended = np.array([w, w]).transpose().reshape(-1)
        r1 = rho_1(resid, 4)
        r2 = rho_2(resid, 4)

        dndp = diags(r1 * resid / w_extended).dot(jacob)
        dndp += diags(r2 * resid * resid / w_extended).dot(jacob)
        corr = diags(r1 + 2 * r2 * resid * resid)
        Hessian = jacob.T.__matmul__(corr.dot(jacob))

        if robust != 0:
            Hessian += diags(robust * np.ones(Hessian.shape[0]))

        # The derivatives and Hessian are stored for later use
        t_Hessian = csr_to_coo(Hessian)
        t_dndp = csr_to_coo(dndp)
        t_nabla = torch.from_numpy(nabla)

        ctx.save_for_backward(t_Hessian, t_dndp, t_nabla)

        P_out = torch.from_numpy(p)
        I_out = torch.from_numpy(i)
        X_out = torch.from_numpy(c)
        return P_out, I_out, X_out

    @staticmethod
    def backward(ctx, grad_p, grad_i, grad_x):
        """
        :param ctx: as usual, it gives the function access to the other class methods
        :param grad_p: the gradient of the output of this function w.r.t. the camera extrinsics
        :param grad_i: the gradient of the output of this function w.r.t. the camera intrinsics
        :param grad_x: the gradient of the output of this function w.r.t. the point cloud
        :return: gradients for each input of the forward function
        """

        # Uncomment the next two line to use pycharm debugger in backward
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # Load Hessian and derivative of the gradient w.r.t. the hyper-parameters of 0f

        t_Hessian, t_dndp, t_nabla = ctx.saved_tensors

        Hessian = coo_to_crs(t_Hessian)
        dndp = coo_to_crs(t_dndp)
        nabla = t_nabla.detach().numpy()

        grad = torch.cat((grad_p, grad_i, grad_x), 0).detach().numpy()

        if grad.size > Hessian.shape[0]:
            temp = np.zeros(Hessian.shape[0])
            idx = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29, 33, 34, 35]
            temp[:18] = grad[idx]
            temp[18:] = grad[36:]
            grad = temp

        compress = spsolve(Hessian, -grad)

        j_p = compress[:len(grad_p)]
        j_i = compress[len(grad_p):len(grad_p)+len(grad_i)]
        j_c = compress[len(grad_p)+len(grad_i):]

        global monitor_Hvariance
        global monitor_Hmean
        global monitor_dwvariance
        global monitor_dwmean
        global monitor_outgradmean
        global monitor_outgradavar

        monitor_Hvariance.append([np.var(np.absolute(j_p)), np.var(np.absolute(j_i)), np.var(np.absolute(j_c))])
        monitor_Hmean.append([np.mean(np.absolute(j_p)), np.mean(np.absolute(j_i)), np.mean(np.absolute(j_c))])

        monitor_outgradmean.append([np.absolute(grad_p.clone().detach().numpy()).mean(), np.absolute(grad_i.clone().detach().numpy()).mean(), np.absolute(grad_x.clone().detach().numpy()).mean()])
        monitor_outgradavar.append([np.absolute(grad_p.clone().detach().numpy()).var(), np.absolute(grad_i.clone().detach().numpy()).var(), np.absolute(grad_x.clone().detach().numpy()).var()])

        update_y = nabla * grad

        update_p = torch.from_numpy(update_y[:grad_p.shape[0]])
        update_i = torch.from_numpy(update_y[grad_p.shape[0]:grad_p.shape[0] + grad_i.shape[0]])
        update_x = torch.from_numpy(update_y[grad_p.shape[0] + grad_i.shape[0]:])
        dldw = dndp.__matmul__(compress)
        dldw = dldw.reshape([-1, 2])
        dldw = dldw.sum(-1)

        monitor_dwmean.append(np.mean(np.absolute(dldw)))
        monitor_dwvariance.append(np.var(np.absolute(dldw)))

        gw = torch.from_numpy(dldw)

        if not (torch.all(torch.isfinite(gw))):
            print(np.max(dldw), " ", np.min(dldw), " ", np.mean(dldw))
            print(np.max(Hessian), " ", np.min(Hessian), " ", np.mean(Hessian))
            assert (torch.all(torch.isfinite(gw)))

        return update_p, update_i, update_x, None, None, None, gw, None, None


class SFMProblem:

    def __init__(self, exp_dir, aspect_size=32):
        self.image_path = exp_dir + "/images/"
        self.data_path = exp_dir + "/results/openMVG_init.bin"
        self.gt_path = exp_dir + "/results/openMVG_gt.bin"
        self.width = aspect_size

        self.evolving_loss = []
        self.evolving_error = []
        self.max_grad = []
        self.min_grad = []
        self.mean_grad = []

        def read_data(path):
            from sfm import importdata
            pose = np.ones(10000)
            inr = np.zeros(1000)
            cloud = np.zeros(10000000)
            obs = np.zeros(10000000)
            c_idx = np.zeros(10000000).astype(np.int32)
            cam = np.zeros(10000000).astype(np.int32)
            tr = np.zeros(10000000).astype(np.int32)

            importdata(path, pose, inr, cloud, c_idx, obs, cam, tr)

            pose = pose[1:6*pose[0].astype(np.int32)+1].reshape(-1, 6)
            cloud = cloud[:c_idx[0]*3]
            obs = obs[:tr[0] * 2].reshape(-1, 2)[:tr[0]*2]
            return pose, inr[1:7], cloud, obs, cam[1:cam[0]+1], tr[1:tr[0]+1]

        self.P, self.I, self.X, self.x, self.c, self.t = read_data(self.data_path)
        self.P_gt, self.I_gt, self.X_gt, _, _, _ = read_data(self.gt_path)

        self.n_cams = len(self.P)
        self.n_obs = len(self.x)
        self.n_tracks = len(self.X)
        self.sparse_tvc = coo_matrix((np.ones(len(self.x), dtype=np.int32), (self.t, self.c)),
                                     shape=(len(self.X), len(self.P)))
        self.sparse_ovc = coo_matrix((np.ones(len(self.x), dtype=np.int32), (range(len(self.x)), self.c)),
                                     shape=(len(self.x), len(self.P)))
        self.slicer = np.array(self.sparse_tvc.sum(1)).reshape(-1)

        def get_aspects(path, halfsize, centers):
            import os
            images_list = os.listdir(path)
            images = []
            for i in images_list:
                pic = mpimg.imread(path + i)
                images.append(pic)

            images = np.array(images)
            img_shape = np.shape(images)
            centers = centers.astype(np.int32)
            corners = centers - [halfsize, halfsize]
            corners_x = corners[:, 0]
            corners_y = corners[:, 1]

            corners_x = np.where(corners_x < 0, 0, corners_x)
            corners_x = np.where(corners_x > img_shape[2]-2*halfsize, img_shape[2]-2*halfsize, corners_x)
            corners_y = np.where(corners_x < 0, 0, corners_y)
            corners_y = np.where(corners_x > img_shape[1]-2*halfsize, img_shape[1]-2*halfsize, corners_y)

            corns = np.array([corners_y, corners_x]).astype(np.int32).transpose()
            stacked = []
            for obs_id, pt in enumerate(corns):
                arg1 = self.c[obs_id]
                arg2 = int(pt[0])
                arg3 = int(pt[1])
                lab = images[arg1][arg2:arg2 + 2 * halfsize, arg3:arg3 + 2 * halfsize]
                stacked.append(lab)
            stack = np.array(stacked).transpose(0, 3, 1, 2)
            return stack

        self.stack = get_aspects(self.image_path, int(aspect_size/2), self.x)

        def diagonaliser(cuts):
            head_at = 0
            idxs = []

            for track_len in cuts:
                for i in range(track_len):
                    for j in range(1+i, track_len):
                        idxs.append([int(i+head_at), int(j+head_at)])
                head_at += track_len

            [row, col] = np.array(idxs).transpose()
            len_c = np.shape(col)[0]
            sparse = coo_matrix((np.ones(len_c), (row, col)), shape=(len_c, len_c))
            return sparse

        self.d_pairs = diagonaliser(self.slicer)

    def return_data(self):
        p = torch.from_numpy(self.P.reshape(-1))
        i = torch.from_numpy(self.I.reshape(-1))
        x = torch.from_numpy(self.X.reshape(-1))
        o = torch.from_numpy(self.x.reshape(-1))
        c = torch.from_numpy(self.c.reshape(-1))
        t = torch.from_numpy(self.t.reshape(-1))
        return p, i, x, o, c, t

    def return_stack(self):
        col = self.d_pairs.col
        row = self.d_pairs.row
        all = np.concatenate((col, row), axis=0)
        stack = np.array(self.stack[all])
        return torch.from_numpy(stack).type('torch.FloatTensor'), int(len(stack)/2)

    def subsample(self, cams):

        sparse_cvc = coo_matrix((np.ones(len(cams), dtype=np.int32), (cams, cams)), shape=(self.n_cams, self.n_cams))

        reduce_tvc = self.sparse_tvc.dot(sparse_cvc).tocoo()
        reduce_ovc = self.sparse_ovc.dot(sparse_cvc).tocoo()

        ro = reduce_tvc.row
        co = reduce_ovc.row

        reduce_ovt = coo_matrix((np.ones(len(ro), dtype=np.int32), (co, ro)), shape=(len(self.x), len(self.X)))

        idx_where = np.array(reduce_tvc.sum(1).transpose(1, 0)).reshape(-1)
        which_tracks = np.array(np.where(idx_where >= 3)).reshape(-1)
        k = coo_matrix((np.ones(len(which_tracks), dtype=np.int32), (which_tracks, which_tracks)), shape=(len(self.X),
                                                                                                          len(self.X)))
        which_obs = reduce_ovt.dot(k).tocoo().row

        id_c = np.zeros(self.n_cams, dtype=np.int32)
        id_t = np.zeros(self.n_tracks, dtype=np.int32)

        id_c[cams] = range(len(cams))
        id_t[which_tracks] = range(len(which_tracks))

        slicer = np.array(reduce_tvc.sum(1)).reshape(-1)
        tlen1 = np.sort(np.unique(self.slicer))
        lens1 = []
        ll1 = []
        for itr, leng in enumerate(tlen1):
            if leng > 2:
                twm = np.argwhere(self.slicer == leng)
                lens1.append(len(twm))
                ll1.append(leng)
        tlen2 = np.sort(np.unique(slicer))
        lens2 = []
        ll2 = []
        for itr, leng in enumerate(tlen2):
            if leng > 2:
                twm = np.argwhere(slicer == leng)
                lens2.append(len(twm))
                ll2.append(leng)

        plt.scatter(ll1, lens1, label="Full Set ("+str(np.sum(lens1))+" tot)")
        plt.scatter(ll2, lens2, label="Sampled Set ("+str(np.sum(lens2))+" tot)")
        plt.xlabel("Track Length")
        plt.ylabel("# Tracks")
        plt.legend(loc='best')
        plt.show()

        P = self.P[cams]
        I = self.I
        X = self.X.reshape(-1, 3)[which_tracks].reshape(-1)
        obs = self.x.reshape(-1, 2)[which_obs].reshape(-1)
        c = self.c[which_obs]
        t = self.t[which_obs]

        c = id_c[c]
        t = id_t[t]

        p = torch.from_numpy(P.reshape(-1))
        i = torch.from_numpy(I.reshape(-1))
        x = torch.from_numpy(X.reshape(-1))
        o = torch.from_numpy(obs.reshape(-1))
        c = torch.from_numpy(c.reshape(-1))
        t = torch.from_numpy(t.reshape(-1))
        return p, i, x, o, c, t, which_obs, which_tracks

    def weight_observations(self, d_ij):

        # still need to fix how to adapt to a smaller problem. subsampling will change the indices and n_obs

        indexes = torch.LongTensor([list(self.d_pairs.col), list(self.d_pairs.row)])
        upper_d = torch.sparse.FloatTensor(indexes, d_ij, torch.Size([self.n_obs, self.n_obs]))
        lower_d = upper_d.transpose(1, 0)
        D_ij = upper_d.add(lower_d)
        w_j = torch.sparse.sum(D_ij, dim=1).to_dense()
        w = torch.exp(-w_j)

        # f_w = torch.from_numpy(phi_w)
        # w = sigmoid(w + f_w + 0.001)
        w = sigmoid(w) + 0.001
        return w

    def loss0(self, p, x, p0, x0, whichobs, cameras):

        x_red = x0.reshape(-1, 3)[whichobs]
        em, x_w_t, tr = procrustes(x_red, x.reshape(-1, 3))
        error = torch.mean(torch.sum((x_w_t - x_red) ** 2, 1))

        [e_r, e_p] = p.reshape(-1, 6).transpose(1, 0).split(3, dim=0)
        [e_r0, e_p0] = p0.reshape(-1, 6).transpose(1, 0).split(3, dim=0)
        rot_T = angle_to_rotation_matrix(e_r.transpose(1, 0)).transpose(2, 1)
        rot0_T = angle_to_rotation_matrix(e_r0.transpose(1, 0)).transpose(2, 1)

        ue = e_p.transpose(1, 0).unsqueeze(-1)
        ue0 = e_p0.transpose(1, 0).unsqueeze(-1)
        p_w = -1 * torch.einsum("abc,acd->abd", (rot_T, ue)).reshape(-1, 3)
        p_w0 = -1 * torch.einsum("abc,acd->abd", (rot0_T, ue0)).reshape(-1, 3)
        p_ref = p_w0[cameras]

        t = tr['rotation']
        b = tr['scale']
        c = tr['translation']
        p_a = b * torch.mm(p_w.reshape(-1, 3), t) + c
        finerr = torch.mean(torch.sum((p_a - p_ref) ** 2, 1))

        return error, finerr

    def loss(self, p, x, p0, x0, whichobjs, cameras):
        error_pose = 20
        error_cloud = 20
        [e_r, e_p] = p.reshape(-1, 6).transpose(1, 0).split(3, dim=0)
        [e_r0, e_p0] = p0.reshape(-1, 6).transpose(1, 0).split(3, dim=0)
        rot_T = angle_to_rotation_matrix(e_r.transpose(1, 0)).transpose(2, 1)
        rot0_T = angle_to_rotation_matrix(e_r0.transpose(1, 0)).transpose(2, 1)

        """print("myversion ", rot_T)"""
        ue = e_p.transpose(1, 0).unsqueeze(-1)
        ue0 = e_p0.transpose(1, 0).unsqueeze(-1)
        p_w = -1*torch.einsum("abc,acd->abd", (rot_T, ue)).reshape(-1, 3)
        p_w0 = -1*torch.einsum("abc,acd->abd", (rot0_T, ue0)).reshape(-1, 3)
        p_ref = p_w0[cameras]

        em, p_w_t, tr = procrustes(p_ref, p_w)
        t = tr['rotation']
        b = tr['scale']
        c = tr['translation']
        # p2_a = b * torch.mm(p_w, t) + c
        x_a = b * torch.mm(x.reshape(-1, 3), t) + c
        error_pose = torch.mean(torch.sum((p_w_t - p_ref) ** 2, 1))
        error_angle = torch.sqrt(torch.mean((rot_T-rot0_T[cameras])**2))
        error_cloud = cloud_to_cloud_distance(x0.reshape(-1, 3), x_a)
        truth = torch.from_numpy(self.P_gt.reshape(-1, 6)[cameras])

        # print("Errors :: ", error_pose.clone().detach().numpy(), " ", error_angle.clone().detach().numpy(), " ",
        #       error_cloud.clone().detach().numpy())

        # loss_np = l_c.clone().detach().numpy()
        # error_np = l_pgt.clone().detach().numpy()
        # print("Error :: ", error_np, "; Loss :: ", loss_np)
        self.evolving_loss.append(error_pose.clone().detach().numpy())
        self.evolving_error.append(error_cloud.clone().detach().numpy())

        return error_pose, error_cloud

    @staticmethod
    def plot_pose(p0, p, c0, c):

        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        [x, y, z] = p.clone().detach().numpy().reshape(-1, 3).T
        [x0, y0, z0] = p0.clone().detach().numpy().reshape(-1, 3).T
        [a, b, c] = c.clone().detach().numpy().reshape(-1, 3).T
        [a0, b0, c0] = c0.clone().detach().numpy().reshape(-1, 3).T

        ax.scatter(x, y, z, marker="o", label='GT cams')
        ax.scatter(x0, y0, z0, marker="^", label='Rec. Cams')
        """ax.scatter(a, b, c, marker="o", s=0.5)
        ax.scatter(a0, b0, c0, marker="^", s=0.5)"""

        """max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        xbs = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5*(x.max()-x.min())
        ybs = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5*(y.max()-y.min())
        zbs = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5*(z.max()-z.min())
        for xb, yb, zb in zip(xbs, ybs, zbs):
            ax.plot([xb], [yb], [zb], 'w')
        plt.grid()"""

        plt.show()

    def plot_heatmap(self, cam, w0, obs0, c0):

        w = w0.clone().detach().numpy()
        obs = obs0.clone().detach().numpy()
        c = c0.clone().detach().numpy()

        cid = np.where(c == cam)
        w_c = w[cid]
        feat = obs.reshape(-1, 2)[cid]
        import os

        def make_blue(alpha):
            inv_alpha = 1 - alpha
            scaled = (255 * inv_alpha)
            colors = [int(color) for color in scaled]
            color_array = 255 * np.ones([len(scaled), 3], dtype=np.int32)
            color_array[:, 0] = colors
            color_array[:, 1] = colors
            return color_array

        image = os.listdir(self.image_path)[cam]
        pic = mpimg.imread(self.image_path + image)
        plt.imshow(pic)
        w_c_n = (w_c - w_c.min()) * (1 / (w_c.max() - w_c.min() + 0.000001))
        cc = make_blue(w_c_n)
        feats = feat.T
        # y_size = np.shape(pic)[0]
        # x_size = np.shape(pic)[1]
        # feats[0] = -1 * feats[0] + x_size
        # feats[1] = -1 * feats[1] + y_size

        plt.scatter(feats[0], feats[1], c=cc / 255.)
        plt.show()


class Model(Module):
    def __init__(self, ba_function):
        super().__init__()
        self.bundle_adjustment = ba_function.apply

    def forward(self, p0, i0, x0, obs, cams, track, weights):

        # p1, i1, x1 = self.bundle_adjustment(p0, i0, x0, obs, cams, track, weights, 1)
        # p2, i2, x2 = self.bundle_adjustment(p1, i1, x1, obs, cams, track, weights, 1)
        # p3, i3, x3 = self.bundle_adjustment(p2, i2, x2, obs, cams, track, weights, 2)
        p3, i3, x3 = self.bundle_adjustment(p0, i0, x0, obs, cams, track, weights, 2)

        return p3, i3, x3


class SiameseNet(Module):
    def __init__(self):
        super().__init__()

        self.convolution_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7)
        self.pool1 = nn.MaxPool2d(2)
        self.convolution_2 = nn.Conv2d(64, 128, 5)
        self.convolution_3 = nn.Conv2d(128, 256, 5)
        self.linear1 = nn.Linear(6400, 512)

    def forward(self, x):
        x = self.convolution_1(x)
        x = func.relu(x)
        x = self.pool1(x)
        x = self.convolution_2(x)
        x = func.relu(x)
        x = self.convolution_3(x)
        x = func.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)

        return x
    # Code for the siamese network
    # stack, n_pairs = my_sfm.return_stack()
    # features = similarity_scoring(stack)
    # f_e = torch.split(features, n_pairs, dim=0)
    # dist2 = tr.dot(f_e)
    # w = my_sfm.weight_observations(dist)


def main():

    data_dir = curr_dir + "/Experiments/Castle"
    import gc
    my_sfm = SFMProblem(data_dir)
    bundle_adjustment = Model(RobustBa())

    # similarity_scoring = SiameseNet().to(device)
    # similarity_scoring.train()

    # Get initial configuration and generate initial weights
    p, i, x, o, c, t = my_sfm.return_data()

    global monitor_Hvariance
    global monitor_Hmean
    global monitor_dwvariance
    global monitor_dwmean
    global monitor_outgradmean
    global monitor_outgradavar

    w0 = torch.nn.Parameter(data=torch.from_numpy(np.random.uniform(low=-2, high=2, size=my_sfm.n_obs).astype(np.float64)
                                                  ), requires_grad=True)

    # Generating a putative g.t. by passing the whole un-weighted set trough the
    wf = torch.ones(my_sfm.n_obs, dtype=torch.float64)
    p_ref, i_ref, x_ref = bundle_adjustment(p.clone().detach(), i.clone().detach(), x.clone().detach(),
                                            o.clone().detach(), c.clone().detach(), t.clone().detach(),
                                            wf.clone().detach())

    p_ref9, i_ref9, x_ref9 = bundle_adjustment(p.clone().detach(), i.clone().detach(), x.clone().detach(),
                                               o.clone().detach(), c.clone().detach(), t.clone().detach(),
                                               sigmoid(w0.clone().detach()) + 0.0001)

    """cc1, cc2 = my_sfm.loss(p_ref, x_ref, p_ref9, x_ref9, range(11))
    refp = cc1.detach().numpy()
    refx = cc2.detach().numpy()
    print("Initial reference errors are ", refp, " and ", refx)"""

    optimizer = torch.optim.ASGD([w0], lr=1, weight_decay=0.001)
    """optimizer = torch.optim.Adam([w0], lr=1, weight_decay=0.0001)"""

    import random
    cameras0 = np.sort(random.sample(range(my_sfm.n_cams), 4))
    ps0, ins0, xs0, os0, cs0, ts0, who0, wht0 = my_sfm.subsample(cameras0)
    cameras1 = np.sort(random.sample(range(my_sfm.n_cams), 4))
    ps1, ins1, xs1, os1, cs1, ts1, who1, wht1 = my_sfm.subsample(cameras1)
    cameras2 = np.sort(random.sample(range(my_sfm.n_cams), 4))
    ps2, ins2, xs2, os2, cs2, ts2, who2, wht2 = my_sfm.subsample(cameras2)
    cameras3 = np.sort(random.sample(range(my_sfm.n_cams), 4))
    ps3, ins3, xs3, os3, cs3, ts3, who3, wht3 = my_sfm.subsample(cameras3)

    print("Camera sets :: ")
    print("    ", cameras0)
    print("    ", cameras1)
    print("    ", cameras2)
    print("    ", cameras3)

    evolving_loss = []
    evolving_error = []
    all_errors = []

    for epoch in range(1201):
        optimizer.zero_grad()
        w = sigmoid(w0) + 0.0001

        p_m0, i_m0, x_m0 = bundle_adjustment(ps0.clone().detach(), ins0.clone().detach(), xs0.clone().detach(),
                                             os0.clone().detach(), cs0.clone().detach(), ts0.clone().detach(), w[who0])
        p_m1, i_m1, x_m1 = bundle_adjustment(ps1.clone().detach(), ins1.clone().detach(), xs1.clone().detach(),
                                             os1.clone().detach(), cs1.clone().detach(), ts1.clone().detach(), w[who1])
        p_m2, i_m2, x_m2 = bundle_adjustment(ps2.clone().detach(), ins2.clone().detach(), xs2.clone().detach(),
                                             os2.clone().detach(), cs2.clone().detach(), ts2.clone().detach(), w[who2])
        p_m3, i_m3, x_m3 = bundle_adjustment(ps3.clone().detach(), ins3.clone().detach(), xs3.clone().detach(),
                                             os3.clone().detach(), cs3.clone().detach(), ts3.clone().detach(), w[who3])

        """
        cost0, error0, rp0, rp_mine0 = my_sfm.loss(p_m0, x_m0, p_ref, x_ref, cameras0)
        cost1, error1, rp1, rp_mine1 = my_sfm.loss(p_m1, x_m1, p_ref, x_ref, cameras1)
        cost2, error2, rp2, rp_mine2 = my_sfm.loss(p_m2, x_m2, p_ref, x_ref, cameras2)
        cost3, error3, rp3, rp_mine3 = my_sfm.loss(p_m3, x_m3, p_ref, x_ref, cameras3)
        """

        cost0, error0 = my_sfm.loss0(p_m0, x_m0, p_ref, x_ref, wht0, cameras0)
        cost1, error1 = my_sfm.loss0(p_m1, x_m1, p_ref, x_ref, wht1, cameras1)
        cost2, error2 = my_sfm.loss0(p_m2, x_m2, p_ref, x_ref, wht2, cameras2)
        cost3, error3 = my_sfm.loss0(p_m3, x_m3, p_ref, x_ref, wht3, cameras3)

        error = error0 + error1 + error2 + error3
        cost = cost0 + cost1 + cost2 + cost3

        all_errors.append([error0.clone().detach().numpy(), error1.clone().detach().numpy(),
                             error2.clone().detach().numpy(), error3.clone().detach().numpy()])

        evolving_loss.append(cost.clone().detach().numpy())
        evolving_error.append(error.clone().detach().numpy())

        if epoch % 50 == 0 and epoch != 0:
            """my_sfm.plot_pose(rp0, rp_mine0, x_ref, x_m0)"""

            plt.plot(evolving_loss, label="Mean Pose Error ")
            plt.xlabel("Training Step")
            plt.ylabel("# Error")
            plt.legend(loc='best')
            plt.show()

            plt.plot(evolving_error, label="Mean Cloud Error ")
            plt.xlabel("Training Step")
            plt.ylabel("Errors")
            plt.legend(loc='best')
            plt.show()

            plt.plot(monitor_dwmean, label="W update mean")
            plt.xlabel("Training Step")
            plt.legend(loc='best')
            plt.show()

            plt.plot(monitor_dwvariance, label="W update variance")
            plt.xlabel("Training Step")
            plt.legend(loc='best')
            plt.show()

            np.save("poserr_all", np.array(evolving_loss))
            np.save("clouderr_all", np.array(evolving_error))
            np.save("clouderrs", np.array(all_errors))

            np.save("mean_H", np.array(monitor_Hmean))
            np.save("var_H", np.array(monitor_Hvariance))
            np.save("mean_dw", np.array(monitor_dwmean))
            np.save("var_dw", np.array(monitor_dwvariance))
            np.save("mean_g", np.array(monitor_outgradmean))
            np.save("var_g", np.array(monitor_outgradavar))

        error.backward()
        optimizer.step()
        gc.collect()


if __name__ == '__main__':
    main()

""" 
TODOS:
- Move the actual network in a self-contained torch module 
- Re-write the data extraction precess based on the the original .bin files 
- Correct the loss
- Fix the no-siamese approach
- Change the siamese network to fix how the cropping size is decided
- Change the gradient to use the huber loss!
"""