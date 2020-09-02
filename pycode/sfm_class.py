import numpy as np
import matplotlib.image as mpimg
import os
import argparse
import torch
from torch.autograd import Function
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as func
from scipy.sparse import csr_matrix, diags, coo_matrix
from scipy.sparse.linalg import spsolve
import sfm

import sys
# sys.path.append('/opt/openmvg/')
sys.path.append('/home/hangryrat/cvssp/WibergianSFM/build_sfm/')


# noinspection PyUnresolvedReferences
def csr_to_coo(sparse_m):
    coo = sparse_m.tocoo()
    values = coo.data
    idx = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(values)
    shape = list(torch.Size(coo.shape))
    return torch.sparse.FloatTensor(i, v, shape)


def coo_to_crs(coo_mat):
    val = coo_mat.coalesce().values().detach().numpy()
    idx = coo_mat.coalesce().indices().detach().numpy()
    shape = list(coo_mat.size())
    sp_coo = coo_matrix((val, idx), shape=shape)
    return sp_coo.tocsr()


def sigmoid(var_x):
    return torch.div(1., torch.abs(var_x) + 1.)


def siglike(var_x, k, d):
    # 1 / (1 + e ^ (-k(x - d)))
    temp = torch.exp(-(k*k)*(var_x - d*d)) + 1.0
    return torch.div(1., temp)


def procrustes(x, y, scaling=True):
    mu_x = x.mean(0)
    mu_y = y.mean(0)

    x0 = x - mu_x
    y0 = y - mu_y

    ss_x = (x0 ** 2.).sum()
    ss_y = (y0 ** 2.).sum()

    # centred Frobenius norm
    norm_x = ss_x.sqrt()
    norm_y = ss_y.sqrt()

    # scale to equal (unit) norm
    x0 = x0 / norm_x
    y0 = y0 / norm_y

    # optimum rotation matrix of Y
    a = x0.t().mm(y0)
    u, s, v = a.svd()

    t = v.mm(u.t())

    trace_ta = s.sum()

    if scaling:

        # optimum scaling of Y
        b = trace_ta * norm_x / norm_y

        # standardised distance between X and b*Y*T + c
        d = -trace_ta ** 2 + 1

        # transformed coords
        z = norm_x * trace_ta * y0.mm(t) + mu_x

    else:
        b = 1
        d = -2 * trace_ta * norm_y / norm_x + 1 + ss_y / ss_x
        z = norm_y * y0.mm(t) + mu_x

    # transformation matrix

    c = mu_x - b * mu_y.reshape([1, 3]).mm(t)

    t_form = {'rotation': t, 'scale': b, 'translation': c}

    # return d, z, t_form, u, s, vt
    return d, z, t_form


class OpenMvgSfm:

    def __init__(self, img_dir, in_path, gt_path, rgt_path, solver, aspect_size=32):

        self.solver = solver.apply


        pose_in, self.i, cloud_in, f, f_c, f_t, views_in, rotation = self.read_data(in_path)

        pose_gt, self.i_gt, cloud_gt, f_gt, f_c_gt, f_t_gt, views_gt, rotation_gt = self.read_data(gt_path)
         
        self.real_gt = np.load(rgt_path)

        # self.rp_gt, _ = self.extrinsics_to_world(torch.from_numpy(pose_gt))
        # self.rp_gt, _ = self.extrinsics_to_world(torch.from_numpy(pose_gt))
        # self.rp_gt = torch.from_numpy(self.real_gt)

        def remove_outliers(poses, cloud, obs, obs_c, obs_t):
            cam_centers, _ = self.extrinsics_to_world(torch.from_numpy(poses))
            cam_centers = cam_centers.clone().numpy()
            mu = cam_centers.mean(0)

            edge = 4 * np.max(np.sum((cam_centers - mu) ** 2, axis=1))
            diffs = np.sum((cloud - mu) ** 2, axis=1)
            keep = np.argwhere(diffs > edge).reshape(-1)[::-1]

            for t_id in reject:
                cloud = np.concatenate([cloud[:3 * t_id], cloud[3 * (1 + t_id):]], axis=0)
                obs_disc = np.argwhere(obs_t == t_id).reshape(-1)
                obs_c = np.concatenate([obs_c[:obs_disc[0]], obs_c[1 + obs_disc[-1]:]], axis=0)
                obs_t = np.concatenate([obs_t[:obs_disc[0]], obs_t[1 + obs_disc[-1]:] - 1], axis=0)
                obs = np.concatenate([obs[:obs_disc[0]], obs[1 + obs_disc[-1]:]], axis=0)

            return poses, cloud, obs, obs_c, obs_t

        cam_centers, _ = self.extrinsics_to_world(torch.from_numpy(pose_in))
        cam_centers = cam_centers.clone().numpy()
        mu = cam_centers.mean(0)

        edge = 4 * np.max(np.sum((cam_centers - mu) ** 2, axis=1))
        diffs = np.sum((cloud_in - mu) ** 2, axis=1)
        sample_track = np.argwhere(diffs < edge).reshape(-1)
        red_obs = []
        for el in sample_track:
            for item in np.argwhere(f_t == el):
                red_obs.append(item[0])

        self.c = (cloud_in[sample_track]).reshape(-1)
        self.f = (f.reshape(-1, 2)[red_obs])
        self.f_c = f_c[red_obs]
        self.f_t = f_t[red_obs]

        for i, ti in enumerate(sample_track):
            self.f_t[np.argwhere(self.f_t == ti)] = i

        # self.p, self.c, self.f, self.f_c, self.f_t = remove_outliers(pose_in, cloud_in, f, f_c, f_t)
        self.p = pose_in
        self.n_cams = len(self.p)
        missing_views = None
        if self.n_cams != np.max(self.f_c) + 1:
            missing_views = np.sort(np.unique(self.f_c))
            temp = np.zeros(np.max(self.f_c) + 1, dtype=np.int32)
            for idc, ci in enumerate(missing_views):
                temp[ci] = idc
            self.f_c = temp[self.f_c]
            self.real_gt = self.real_gt[missing_views]

        #self.c = cloud_in
        #self.f = f
        #self.f_c = f_c
        #self.f_t = f_t
        # self.p_gt, self.c_gt, f_gt, f_c_gt, f_t_gt = remove_outliers(pose_gt, cloud_gt, f_gt, f_c_gt, f_t_gt)
        if os.path.exists(img_dir + "/../p_gt.npy"):
            self.p_gt = torch.from_numpy(np.load(img_dir + "../p_gt.npy"))
            self.c_gt = np.load(img_dir + "../c_gt.npy")
        else:
            w_ij = torch.from_numpy(np.ones(len(self.f), dtype=np.float64))
            self.p_gt, _, self.c_gt = self.solver(pose_in.reshape(-1), torch.from_numpy(self.i).reshape(-1),
                                                  self.c.reshape(-1), self.f.reshape(-1), self.f_c,
                                                  self.f_t, w_ij.clone().detach())
            np.save(img_dir + "/../p_gt", self.p_gt)
            np.save(img_dir + "/../c_gt", self.c_gt)
        self.rp_gt, _ = self.extrinsics_to_world(self.p_gt.reshape(-1,6))

        #self.p_gt = pose_gt
        #self.c_gt = cloud_gt
        #self.f_gt = f_gt
        #self.f_c_gt = f_c_gt
        #self.f_t_gt = f_t_gt

        # rw2, rots = self.extrinsics_to_world(torch.from_numpy(pose_in))
        # self.plot_pose(rw2)
        # self.plot_pose(cloud_in)

        self.n_feats = len(self.f)
        self.n_tracks = len(self.c)

        self.sparse_tvc = coo_matrix((np.ones(self.n_feats, dtype=np.int32), (self.f_t, self.f_c)),
                                     shape=(self.n_tracks, self.n_cams))
        self.sparse_ovc = coo_matrix((np.ones(self.n_feats, dtype=np.int32), (range(self.n_feats), self.f_c)),
                                     shape=(self.n_feats, self.n_cams))
        if missing_views is not None:
            self.camreader = missing_views
        else:
            self.camreader = range(self.n_cams)

        self.img_dir = img_dir
        self.aspect_size = aspect_size
        # self.stack = torch.from_numpy(get_aspects(img_dir, int(aspect_size / 2), self.f)).type('torch.FloatTensor')
        # self.stack = self.get_aspects(img_dir, int(aspect_size / 2), self.f, self.f_c)

        def diagonalise(cuts):
            head_at = 0
            ids = []

            for track_len in cuts:
                for obs in range(track_len):
                    for j in range(1 + obs, track_len):
                        ids.append([int(obs + head_at), int(j + head_at)])
                head_at += track_len

            [row, col] = np.array(ids).transpose()
            sane = np.sum(cuts)
            len_c = np.shape(col)[0]
            sparse = coo_matrix((np.ones(len_c), (row, col)), shape=(sane, sane))
            return sparse

        self.d_pairs = diagonalise(np.array(self.sparse_tvc.sum(1)).reshape(-1))

    @staticmethod
    def angle_to_rotation_matrix(angle_axis):

        ll = len(angle_axis)
        k_one = 1.0
        theta2 = torch.sum(angle_axis*angle_axis, 1)
        theta = torch.sqrt(theta2)

        wx = angle_axis[:, 0] / theta
        wy = angle_axis[:, 1] / theta
        wz = angle_axis[:, 2] / theta

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r = torch.zeros([ll, 3, 3], dtype=torch.float64)

        r[:, 0, 0] = cos_theta + wx*wx*(k_one - cos_theta)
        r[:, 1, 0] = wz*sin_theta + wx*wy*(k_one - cos_theta)
        r[:, 2, 0] = -wy*sin_theta + wx*wz*(k_one - cos_theta)
        r[:, 0, 1] = wx*wy*(k_one - cos_theta) - wz*sin_theta
        r[:, 1, 1] = cos_theta + wy*wy*(k_one - cos_theta)
        r[:, 2, 1] = wx*sin_theta + wy*wz*(k_one - cos_theta)
        r[:, 0, 2] = wy*sin_theta + wx*wz*(k_one - cos_theta)
        r[:, 1, 2] = -wx*sin_theta + wy*wz*(k_one - cos_theta)
        r[:, 2, 2] = cos_theta + wz*wz*(k_one - cos_theta)

        return r

    def get_aspects(self, path, patch_size, centers, which_c):
        import os
        images_list = sorted(os.listdir(path))
        chosencams = np.unique(which_c)
        images_list = np.asanyarray(images_list)[self.camreader][chosencams]
        images = []
        for ima in images_list:
            pic = mpimg.imread(path + ima)
            images.append(pic)

        # self.plot_heatmap(images[0], centers, self.f_c, 0)

        images = np.array(images)
        img_shape = [np.shape(image) for image in images]
        img_shape = np.array(img_shape)
        centers = centers.astype(np.int32)
        # self.plot_heatmap(images[0], centers, self.f_c, 0)
        corners = centers - [patch_size, patch_size]
        corners_x = corners[:, 0]
        corners_y = corners[:, 1]
        # self.plot_heatmap(images[0], corners, self.f_c, 0)

        newcamidx = np.array([int(np.argwhere(chosencams == which_c[obs_id])) for obs_id in range(len(centers))])
        img_shape = img_shape[newcamidx]

        corners_x = np.where(corners_x < 0, 0, corners_x)
        corners_x = np.where(corners_x > img_shape[:, 1] - 2 * patch_size, img_shape[:, 1] - 2 * patch_size, corners_x)
        corners_y = np.where(corners_y < 0, 0, corners_y)
        corners_y = np.where(corners_y > img_shape[:, 0] - 2 * patch_size, img_shape[:, 0] - 2 * patch_size, corners_y)

        corns = np.array([corners_x, corners_y]).astype(np.int32).transpose()
        # self.plot_heatmap(images[0], corns, self.f_c, 0)
        stacked = []
        for obs_id, pt in enumerate(corns):
            arg1 = newcamidx[obs_id]
            arg2 = int(pt[1])
            arg3 = int(pt[0])
            lab = images[arg1]
            lab1 = lab[arg2:arg2 + 2 * patch_size, :]
            lab2 = lab1[:, arg3:arg3 + 2 * patch_size]

            """
            import matplotlib.pyplot as plt
            plt.imshow(lab2)
            plt.show()
            plt.imshow(lab)
            plt.scatter(pt[0], pt[1])
            plt.show()"""
            if np.shape(lab2) != (32, 32, 3):
                print("ERROR: Wrong patch!!! ", np.shape(lab), " ", arg1, " ", arg2, " ", arg3, " ", obs_id,
                      centers[obs_id], " ", corners_x[obs_id], " ", corners_y[obs_id])
            stacked.append(lab2)
        stack = np.array(stacked).transpose([0, 3, 1, 2])
        return stack

    @staticmethod
    def read_data(path):
        from sfm import import_data
        poses = np.ones(20000)
        rot = np.ones(100000)
        views = np.ones(20000).astype(np.int32)
        intrinsic = np.zeros(100000)
        cloud = np.zeros(10000000)
        feature = np.zeros(10000000)
        c_idx = np.zeros(10000000).astype(np.int32)
        cam_id = np.zeros(10000000).astype(np.int32)
        track_id = np.zeros(10000000).astype(np.int32)

        import_data(path, poses, intrinsic, cloud, c_idx, feature, cam_id, track_id, views, rot)
        views = views[:poses[0].astype(np.int32)]
        rot = rot[1:9 * poses[0].astype(np.int32) + 1].reshape(-1, 3, 3)
        poses = poses[1:6 * poses[0].astype(np.int32) + 1].reshape(-1, 6)
        cloud = cloud[:c_idx[0] * 3].reshape(-1, 3)
        feature = feature[:track_id[0] * 2].reshape(-1, 2)
        return poses, intrinsic[1:7], cloud, feature, cam_id[1:cam_id[0] + 1], track_id[
                                                                               1:track_id[0] + 1], views, rot

    def extrinsics_to_world(self, poses):
        [rotation, center] = torch.split(poses, 3, dim=1)
        r_t = self.angle_to_rotation_matrix(rotation).transpose(2, 1)
        p = center.unsqueeze(-1)
        p_rw = -1 * torch.einsum("abc,acd->abd", (r_t, p)).reshape(-1, 3)
        return p_rw, r_t

    def plot_pose(self, p0, label):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        [x0, y0, z0] = p0.reshape(-1, 3).T
        ax.scatter(x0, y0, z0, marker="^", s=10, label=label)
        plt.show()

    #@staticmethod
    def plot_pose_overlapping(self, p0, p1, lab1, lab2):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        [X, Y, Z] = p0.reshape(-1, 3).T
        [x1, y1, z1] = p1.reshape(-1, 3).T

        ax.scatter(X, Y, Z, marker="o", s=10, label=lab1, color="red")
        ax.scatter(x1, y1, z1, marker="^", s=10, label=lab2, color="blue")

        for i in range(0, len(X)):
            ax.plot([X[i], x1[i]], [Y[i], y1[i]], [Z[i], z1[i]], color="grey")

        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        plt.grid()

        plt.show()

    def plot_pose_labelled(self, p0, lab1):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        [x0, y0, z0] = p0.reshape(-1, 3).T

        ax.scatter(x0, y0, z0, marker="^", s=10, label=lab1)

        labs = [str(i) for i in range(len(x0))]
        for lb, x, y, z in zip(labs, x0, y0, z0):
            label = lb
            ax.text(x, y, z, label)

        plt.show()

    #@staticmethod
    def plot_torch_pose(self, p0):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        [x0, y0, z0] = p0.reshape(-1, 3).T.clone().detach().numpy()
        ax.scatter(x0, y0, z0, label='Rec. Cams')
        ax.plot(x0, y0, z0)
        labs = [str(i) for i in range(len(x0))]
        for lb, x, y, z in zip(labs, x0, y0, z0):
            label = lb
            ax.text(x, y, z, label)

        plt.show()

    @staticmethod
    def plot_heatmap(pic, obs, ca, index):
        import matplotlib.pyplot as plt
        plt.imshow(pic)
        cid = np.where(ca == index)
        pts = obs.reshape(-1, 2)[cid]
        plt.scatter(pts.transpose()[0], pts.transpose()[1])
        plt.show()

    def subsample(self, cams):

        sparse_cvc = coo_matrix((np.ones_like(cams), (cams, cams)), shape=(self.n_cams, self.n_cams))
        red_tvc = self.sparse_tvc.dot(sparse_cvc).tocoo()
        red_ovc = self.sparse_ovc.dot(sparse_cvc).tocoo()

        reduce_ovt = coo_matrix((np.ones_like(red_tvc.row), (red_ovc.row, red_tvc.row)), shape=(self.n_feats,
                                                                                                self.n_tracks))

        track_len = np.array(red_tvc.sum(1)).reshape(-1)
        which_tracks = np.array(np.where(track_len >= 3)[0])
        k = coo_matrix((np.ones_like(which_tracks), (which_tracks, which_tracks)), shape=(self.n_tracks, self.n_tracks))
        which_obs = reduce_ovt.dot(k).tocoo().row

        id_c = np.zeros(self.n_cams, dtype=np.int32)
        id_t = np.zeros(self.n_tracks, dtype=np.int32)
        id_c[cams] = range(len(cams))
        id_t[which_tracks] = range(len(which_tracks))

        # p = torch.from_numpy(self.p[cams].reshape(-1))
        # c = torch.from_numpy(self.c.reshape(-1, 3)[which_tracks].reshape(-1))
        # f = torch.from_numpy(self.f.reshape(-1, 2)[which_obs].reshape(-1))
        p = (self.p[cams].reshape(-1))
        c = (self.c.reshape(-1, 3)[which_tracks].reshape(-1))
        f = (self.f.reshape(-1, 2)[which_obs].reshape(-1))
        f_c = self.f_c[which_obs]
        f_t = self.f_t[which_obs]

        # f_c = torch.from_numpy(id_c[f_c])
        # f_t = torch.from_numpy(id_t[f_t])
        f_c = (id_c[f_c])
        f_t = (id_t[f_t])

        return p, torch.from_numpy(self.i), c, f, f_c, f_t, which_obs, which_tracks

    def get_data(self):
        p = self.p.reshape(-1)
        # p = torch.from_numpy(self.p.reshape(-1))
        c = self.c.reshape(-1)
        # c = torch.from_numpy(self.c.reshape(-1))
        f = self.f.reshape(-1)
        # f = torch.from_numpy(self.f.reshape(-1))
        # f_c = torch.from_numpy(self.f_c.reshape(-1))
        # f_t = torch.from_numpy(self.f_t.reshape(-1))
        f_c = self.f_c
        f_t = self.f_t
        return p, torch.from_numpy(self.i), c, f, f_c, f_t

    def return_stack(self, observations=None):
        if observations is None:
            col = self.d_pairs.col
            row = self.d_pairs.row
            stack = self.get_aspects(self.img_dir, int(self.aspect_size / 2), self.f, self.f_c)
            length = self.n_feats
        else:
            reduced = self.d_pairs.tocsr()[list(observations), :][:, list(observations)].tocoo()
            col = reduced.col
            row = reduced.row
            feats = self.f.reshape(-1, 2)[observations].reshape(-1,2)
            stack = self.get_aspects(self.img_dir, int(self.aspect_size / 2), feats, self.f_c[observations])
            length = len(observations)
        return stack, length, [row, col]

    def pose_loss(self, pose, cloud, treshold=4, cams=None, tracks=None):
        if cams is None:
            p_init, _ = self.extrinsics_to_world(pose.reshape(-1, 6))
            p_gt = self.rp_gt
            cloud_gt = torch.from_numpy(self.c_gt.reshape(-1, 3))
        else:
            p_init, _ = self.extrinsics_to_world(pose.reshape(-1, 6))
            p_gt = self.rp_gt[cams]
            cloud_gt = torch.from_numpy(self.c_gt.reshape(-1, 3))[tracks]

        cloud_in = cloud.reshape(-1, 3)

        #self.plot_pose_overlapping(cloud_in.clone().detach().numpy(), cloud_gt.clone().detach().numpy(),
        #                               "aligned", "gt")
        #san_err0 = np.sqrt(np.sum((cloud_in.clone().detach().numpy() - cloud_gt.clone().detach().numpy())**2, axis=1)).mean()
        #print("Cloud-Cloud error :: ", san_err0)
        npposes = self.rp_gt.clone().detach().numpy()
        normalisation = np.mean(np.sum((npposes - np.mean(npposes, 0)) ** 2, 1))
        # np_cloud = cloud_in.clone().detach().numpy()
        np_cloud = cloud_gt.clone().detach().numpy().reshape(-1, 3)
        quad_diff = np.sum((np_cloud - np.median(np_cloud, 0)) ** 2, 1)
        normalisation_struct = np.mean(np.sum((np_cloud - np.mean(np_cloud, 0)) ** 2, 1))
        inliers = np.argwhere(quad_diff < treshold*normalisation_struct).reshape(-1)

        cloud_in = cloud_in[inliers]
        cloud_gt = cloud_gt[inliers]

        np_cloud = cloud_in.clone().detach().numpy().reshape(-1, 3)
        quad_diff = np.sum((np_cloud - np.median(np_cloud, 0)) ** 2, 1)
        normalisation_struct = np.mean(np.sum((np_cloud - np.mean(np_cloud, 0)) ** 2, 1))
        inliers = np.argwhere(quad_diff < treshold*normalisation_struct).reshape(-1)

        cloud_in = cloud_in[inliers]
        cloud_gt = cloud_gt[inliers]

        #self.plot_pose_overlapping(cloud_in.clone().detach().numpy(), cloud_gt.clone().detach().numpy(),
        #                               "aligned", "gt")
        #san_err1 = np.sqrt(np.sum((cloud_in.clone().detach().numpy() - cloud_gt.clone().detach().numpy())**2, axis=1)).mean()
        #print("Cloud-Cloud error - outlier removed :: ", san_err1)
        normalisation = torch.from_numpy(normalisation_struct.reshape(1))

        _, ttt, transform = procrustes(cloud_gt, cloud_in)

        #self.plot_pose_overlapping(ttt.clone().detach().numpy(), cloud_gt.clone().detach().numpy(),
        #                               "aligned", "gt")
        #san_err2 = np.sqrt(np.sum((ttt.clone().detach().numpy() - cloud_gt.clone().detach().numpy())**2, axis=1)).mean()
        #print("Aligned Cloud-Cloud Error :: ", san_err2)
        aligned = transform['scale'].clone().detach() * p_init.mm(transform['rotation'].clone().detach()) +\
                  transform['translation'].clone().detach()

        #self.plot_pose_overlapping(p_init.clone().detach().numpy(), p_gt,
        #                               "aligned", "gt")
        #san_err3 = np.sqrt(np.sum((p_init.clone().detach().numpy() - p_gt.clone().detach().numpy())**2, axis=1)).mean()
        #self.plot_pose_overlapping(aligned.clone().detach().numpy(), p_gt,
        #                              "aligned", "gt")
        #san_err4 = np.sqrt(np.sum((aligned.clone().detach().numpy() - p_gt.clone().detach().numpy())**2, axis=1)).mean()
        #print("Pose-pose error :: ", san_err3, " Aligned pose-pose error :: ", san_err4)
        error = (aligned-p_gt.clone().detach())**2
        error = torch.sum(error, 1)
        #error = torch.sqrt(torch.sum(error, 1))
        euclid = error.clone().detach().numpy()
        #euclid = error.sqrt().clone().detach().numpy()
        error = torch.mean(error)
        return error/normalisation, aligned.reshape(-1), euclid

    def pose_loss_overlap(self, pose1, pose2):

        al_err, al_pose, transform = procrustes(pose2, pose1)

        normalisation = torch.mean(torch.sum((pose2-torch.mean(pose2, 0))**2, 1))

        error = (al_pose-pose2)**2
        error = (torch.sum(error, 1)/normalisation)
        pc_error = error.clone().detach().numpy()
        return pc_error


class RobustBA(Function):
    @staticmethod
    def forward(ctx, p_t, i_t, c_t, ob, ca, tr, weights, robust=0.1, newton=False):
        from sfm import ceresBA

        def rho_1(x, val):
            temp = np.where(x < val, 1, 1 / (x+0.000000001))
            return temp

        def rho_2(x, val):
            temp = np.where(x < val, 0, -1 / np.square(x+0.000000001))
            return temp

        # 4 options for mode: 0, 1, 2 -> train subset of parameters (X & P-R, X & P, X & P & I); 4 -> Evaluation only
        v_grad = np.zeros(5000000)
        v_row = np.zeros(500000).astype(np.int32)
        v_col = np.zeros(500000).astype(np.int32)
        residuals = np.zeros(10*len(ob))
        gradients = np.zeros(10*len(c_t))

        w = 10*weights.clone().detach().numpy().astype(np.float64)
        i = i_t.clone().detach().numpy()
        p = p_t[:]
        c = c_t[:]

        # Running the Bundle adjustment

        if not ceresBA(p, i, c, ob, ca, tr, w, v_row, v_col, v_grad, residuals, gradients, 1):
            print("Call to CeresBA failed to converge")
            return torch.from_numpy(p_t), i_t, torch.from_numpy(c_t)
        if v_col[0] == 0:
            print("Error Found")
            np.save("error_pose", p)
            np.save("error_intrin", i)
            np.save("error_cloud", c)
            np.save("error_obs", ob)
            np.save("error_cams", ca)
            np.save("error_track", tr)
            np.save("error_weights", w)
            
            
        # p_out = torch.from_numpy(p.reshape(-1,6))
        # i_out = torch.from_numpy(i)
        # x_out = torch.from_numpy(c.reshape(-1,3))
        # return p_out, i_out, x_out

        if v_col[0] == 0:
            print("here is the problem")
            print("here is the problem")

        v_c = v_col[2:2 + v_col[0]]

        v_r = v_row[2:2 + v_row[0]]
        v_g = v_grad[1:1 + int(v_grad[0])]
        resid = residuals[1:1 + int(residuals[0])]
        nabla = gradients[1:1 + int(gradients[0])]
        jacob = csr_matrix((v_g, v_c, v_r), shape=(v_row[1], v_col[1]))

        if newton:
            print("Should still implement the final newton step... ")

        w_extended = np.array([w, w]).transpose().reshape(-1)
        r1 = rho_1(resid, 4)
        r2 = rho_2(resid, 4)

        dndp = diags(resid / w_extended).dot(jacob)
        #dndp += diags(r2 * resid * resid / w_extended).dot(jacob)
        #corr = diags(r1 + 2 * r2 * resid * resid)
        hessian = jacob.T.__matmul__(jacob)

        if robust != 0:
            hessian += diags(robust * np.ones(hessian.shape[0]))

        # The derivatives and hessian are stored for later use
        t_hessian = csr_to_coo(hessian)
        t_dndp = csr_to_coo(dndp)
        t_nabla = torch.from_numpy(nabla)

        ctx.save_for_backward(t_hessian, t_dndp, t_nabla)

        p_out = torch.from_numpy(p)
        i_out = torch.from_numpy(i)
        x_out = torch.from_numpy(c)
        return p_out, i_out, x_out

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
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # Load Hessian and derivative of the gradient w.r.t. the hyper-parameters of 0f
        san1 = grad_p.clone().detach().numpy()
        t_hessian, t_dndp, t_nabla = ctx.saved_tensors

        hessian = coo_to_crs(t_hessian)
        dndp = coo_to_crs(t_dndp)

        grad = torch.cat((grad_p, grad_i, grad_x), 0).detach().numpy()

        compress = spsolve(hessian, -grad)
        dl_dw = dndp.__matmul__(compress)
        dl_dw = dl_dw.reshape([-1, 2])
        dl_dw = dl_dw.sum(-1)

        gw = torch.from_numpy(dl_dw).float()

        if not (torch.all(torch.isfinite(gw))):
            print(np.max(dl_dw), " ", np.min(dl_dw), " ", np.mean(dl_dw))
            print(np.max(hessian), " ", np.min(hessian), " ", np.mean(hessian))
            assert (torch.all(torch.isfinite(gw)))

        return None, None, None, None, None, None, gw, None, None


def topy(x):
    return x.clone().detach().numpy()


class Sanity(Module):
    @staticmethod
    def forward(sfm_class, lam, sample_size):
        import random

        while 1:
            cam = np.sort(random.sample(range(sfm_class.n_cams), sample_size))
            ps, ins, xs, obs, cs, ts, wh_o, wh_t = sfm_class.subsample(cam)
            stack, observation, idx = sfm_class.return_stack()
    
            if len(np.array(wh_o).reshape(-1)) > 100:
                break

        index = torch.LongTensor(idx)

        norm = torch.sparse.FloatTensor(index, torch.ones_like(index[0]), torch.Size([sfm_class.n_feats,
                                                                                      sfm_class.n_feats]))
        norm = norm + norm.transpose(1, 0)
        norm = torch.sparse.sum(norm, 0).to_dense()
        norm = norm/torch.max(norm)
        norm = norm[wh_o]

        w_ij = 10 * sigmoid(lam[wh_o] * norm) + 0.00001

        p_m0, i_m0, x_m0 = sfm_class.solver(ps, ins, xs, obs, cs, ts, w_ij)

        error = sfm_class.pose_loss(p_m0, cam)

        return error


def testing_sanity(sfm_class, lam, name):
    with torch.no_grad(): 
        ps, ins, xs, obs, cs, ts = sfm_class.get_data()

        stack, observation, idx = sfm_class.return_stack()

        index = torch.LongTensor(idx)

        norm = torch.sparse.FloatTensor(index, torch.ones_like(index[0]), torch.Size([observation, observation]))
        norm = norm + norm.transpose(1, 0)
        norm = torch.sparse.sum(norm, 0).to_dense()
        norm = norm / torch.max(norm)
        w_ij = 10 * sigmoid(lam * norm) + 0.00001

        p_m0, i_m0, x_m0 = sfm_class.solver(ps.clone().detach(), ins.clone().detach(), xs.clone().detach(),
                                            obs.clone().detach(), cs.clone().detach(), ts.clone().detach(), w_ij)

        error = sfm_class.pose_loss(p_m0)

        torch.save({
            'error': error,
            'weights': w_ij.clone().detach(),
            'poses': p_m0.clone().detach(),
            'intrinsics': i_m0.clone().detach(),
            'cloud': x_m0.clone().detach(),
        }, "evaluation_results" + name + ".pt")

    return error
