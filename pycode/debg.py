import numpy as np
import sfm_class as sfm
import torch
import argparse
import os
import gc
import random

curr_dir = os.path.dirname(os.path.realpath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("-lr", "--learning_rate", required=False, help="learning rate", default=0.01)
ap.add_argument("-name", "--name", required=False, help="label for the check points", default="local_test_statue")
ap.add_argument("-dataset", "--dataset", required=False, help="what dataset to use", default=0)
args = vars(ap.parse_args())


def plot_pose_overlapping(p0, p1, lab1, lab2):
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


def extrinsics_to_world(poses):
    [rotation, center] = torch.split(poses, 3, dim=1)
    r_t = angle_to_rotation_matrix(rotation).transpose(2, 1)
    p = center.unsqueeze(-1)
    p_rw = -1 * torch.einsum("abc,acd->abd", (r_t, p)).reshape(-1, 3)
    return p_rw, r_t


def initialise_model(set_name):

    img_path = curr_dir + "/dataset/" + set_name + "/images/"
    in_path = curr_dir + "/dataset/" + set_name + "/init.bin"
    gt_path = curr_dir + "/dataset/" + set_name + "/final.bin"
    rgt_path = curr_dir + "/dataset/" + set_name + "/gt.npy"

    return sfm.OpenMvgSfm(img_path, in_path, gt_path, rgt_path, sfm.RobustBA)


def mynormalise(var_x, mode):
    if mode == 0:
        temp = torch.div(1., torch.exp(-var_x) + 1.0)
    elif mode == 1:
        temp = torch.div(1., torch.abs(var_x) + 1.0)
    else:
        div = var_x.clone().detach().numpy()
        div1 = (torch.max(var_x) - torch.min(var_x)).clone().detach().numpy()
        div2 = (var_x - torch.min(var_x)).clone().detach().numpy()
        temp = (var_x - torch.min(var_x)+0.0001)/(torch.max(var_x) - torch.min(var_x) + 0.0001)
    return temp


def main():

    set_id = int(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 50
    iterations = 5000
    mini_batch = 25

    # Parameters of the simulation:
    # How many frames to use
    window_size = 8
    # Which of the camera sets to use
    cam_id = 1
    # Weights normalisation: 0 is sigmoid, 1 is 1/1+abs, 2 is (w-min)/(max-min)
    norm_mode = 1
    # Optimiser to use. 0 is SGD, 1 is ADAM
    optm_mode = 0
    # Learing rate
    lr = 1.0

    # We divide the cameras in blocks of "window_size" views.
    if set_id > 4:
        chunk_eval = 0
    else:
        chunk_eval = 1

    save_path = curr_dir + "/sub_siam/" + save_label
    data_sets = ["statue", "empirevase", "jeep", "temple", "dino", "bra", "dante", "erbe"]
    set_name = data_sets[set_id]
    sfm_model = initialise_model(set_name)

    if chunk_eval == 1:
        nn_table = []
        cam_slide = np.array(range(sfm_model.n_cams))
        for i in range(sfm_model.n_cams):
            temp = cam_slide[i:i+window_size]
            _, _, _, _, cs, _, _, _ = sfm_model.subsample(temp)
            if window_size == len(np.unique(cs)):
                nn_table.append(temp)
        nn_table = np.array(nn_table).reshape(-1, window_size)
    else:
        def find_nn(pose_list, w_size):
            n = np.argsort(-pose_list, axis=1)
            k_list = n[:, :w_size:]
            return k_list

        nn_table = find_nn(sfm_model.sparse_tvc.T.dot(sfm_model.sparse_tvc.todense()), window_size)
        nn_table = np.array(nn_table).reshape(sfm_model.n_cams, -1)
    eval_blocks = nn_table

    print("Starting simulation...")

    cams = eval_blocks[cam_id]
    _, _, _, _, cs, _, _, _ = sfm_model.subsample(cams)
    if len(cams) > len(np.unique(cs)):
                    print("Camera set not connected :: ", cams)

    # We load the initial data for the chosen problem, and the ground truth camera poses
    # ps are the extrinsics, ins the intrinsics, xs the cloud points.
    # obs are the 2d observations, cs tells what camera they belong to, ts which track; These indexes refer to the
    # current problem, while wh_o and wh_t are the observation and track indeces in the full problem.

    ps, ins, xs, obs, cs, ts, wh_o, wh_t = sfm_model.subsample(cams)
    gt_path = curr_dir + "/dataset/statue/images/"
    p_gt = torch.from_numpy(np.load(gt_path + "../p_gt.npy")[cams])
    p_gt, _ = extrinsics_to_world(p_gt.reshape(-1, 6))

    for i in range(5):
        prog = []
        with torch.autograd.detect_anomaly():
            # We initialise the weights to 1, one for each observation
            w2 = torch.tensor(np.ones_like(cs), dtype=torch.float64, requires_grad=True)

            if optm_mode == 0:
                optimizer = torch.optim.SGD([w2], lr=lr)
            else:
                optimizer = torch.optim.Adam([w2], lr=lr)
            optimizer.zero_grad()
            for iter in range(250):
                # No matter how the weights are normalise, we add a small positive bias: Ceres will treat W=0 as W=1
                sw = mynormalise(w2, norm_mode) + 0.0001
                # We run BA. ins is a tesor and gets detached later, but ps and xs are still np arrays. If we do not
                # copy them, we replace them with the output of BA
                pose, i_m0, cloud = sfm_model.solver(np.copy(ps), ins, np.copy(xs), obs, cs, ts, sw)
                # We change the extrinsics into real world coordinates
                p_init, _ = extrinsics_to_world(pose.reshape(-1, 6))
                cloud_in = cloud.reshape(-1, 3)
                cloud_gt = torch.from_numpy(np.load(gt_path + "../c_gt.npy")[wh_t])
                # We perform procrustes alignement on the point clouds
                _, ttt, transform = sfm.procrustes(cloud_gt, cloud_in)

                aligned = transform['scale'].clone().detach() * p_init.mm(transform['rotation'].clone().detach()) + \
                          transform['translation'].clone().detach()

                # plot_pose_overlapping(aligned.clone().detach().numpy(), p_gt, "aligned", "gt")
                """guess = mynormalise(w1, norm_mode) + 0.0001
                guess = guess.reshape(-1, 3)
                _, aligned, transform = sfm.procrustes(p_gt, guess)"""
                # plot_pose_overlapping(guess.clone().detach().numpy(), p_gt, "aligned", "gt")
                # We compute the L2 reconstruction error after alignement.
                error = torch.sum((aligned - p_gt.clone().detach()) ** 2, 1)
                error = torch.mean(error)
                print("Error ::", error.clone().detach().numpy(), " -- cams: ", cams)
                prog.append(error.clone().detach().numpy())
                error.backward()
                optimizer.step()
                optimizer.zero_grad()
        import matplotlib.pyplot as plt
        plt.plot(prog)
        plt.xlabel("Step")
        plt.ylabel("L2 error")
        plt.title("Training loss")
        plt.show()


if __name__ == '__main__':
    main()


"""sw = mynormalise(w1, norm_mode) + 0.0001
pose, i_m0, cloud = sfm_model.solver(np.copy(ps), ins, np.copy(xs), obs, cs, ts, sw)
p_init, _ = extrinsics_to_world(pose.reshape(-1, 6))
cloud_in = cloud.reshape(-1, 3)
cloud_gt = torch.from_numpy(np.load(gt_path + "../c_gt.npy")[wh_t])

_, ttt, transform = sfm.procrustes(cloud_gt, cloud_in)

aligned = transform['scale'].clone().detach() * p_init.mm(transform['rotation'].clone().detach()) + \
          transform['translation'].clone().detach()

plot_pose_overlapping(aligned.clone().detach().numpy(), p_gt, "aligned", "gt")
error = torch.sum((aligned - p_gt.clone().detach()) ** 2, 1)
error = torch.mean(error)"""