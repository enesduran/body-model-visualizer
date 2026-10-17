import time
from functools import wraps
import torch
import numpy as np
from tqdm import tqdm
from smplx import SMPL
from loguru import logger


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        val = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'"{func.__name__}" fn took {end - start:.3f} seconds.')
        return val

    return wrapper

@timeit
def simple_ik_solver(model, target, init=None, device='cpu', max_iter=20,
                     mse_threshold=1e-8, transl=None, betas=None):
    if init is None:
        init_pose = torch.zeros(1, 69, device=device, requires_grad=True)
    else:
        init_pose = init.detach().reshape(1, -1).to(device).clone().requires_grad_(True)
    target = target.detach().to(device)
    transl = torch.zeros(1, 3, device=device) if transl is None else transl.detach().to(device)
    betas = None if betas is None else betas.detach().to(device)
    optimizer = torch.optim.Adam([init_pose], lr=0.1)
    last_mse = None
    loss_value = None
    for i in range(max_iter):

        mse = torch.mean(torch.square((
                model(
                    body_pose=init_pose,
                    betas=betas,
                    transl=transl,
                ).joints[0,:22] - target)))
        loss_value = mse.detach().item()
        if loss_value < mse_threshold or (
            last_mse is not None and abs(loss_value - last_mse) < mse_threshold
        ):
            break
        optimizer.zero_grad(set_to_none=True)
        # Only pose is optimized; model and shape parameter gradients stay untouched.
        init_pose.grad, = torch.autograd.grad(mse, init_pose)
        optimizer.step()
        last_mse = loss_value
    if loss_value is not None:
        logger.info(f'IK last evaluated loss {loss_value:.6f}')
    return init_pose.detach()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SMPL(f'data/body_models/smpl').float()
    joints = model().joints[0,:22]
    # joints[22] = joints[22] + 0.3
    # joints[20] = joints[20] + 0.3

    target_joints = joints + torch.rand_like(joints) * 0.1

    opt_params = simple_ik_solver(model, target_joints, max_iter=100)

    opt_joints = model(body_pose=opt_params).joints[0,:22]

    opt_joints = opt_joints.detach().numpy()
    target_joints = target_joints.detach().numpy()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    RADIUS = 1.0
    xroot, yroot, zroot = target_joints[0, 0], target_joints[0, 1], target_joints[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.scatter(target_joints[:, 0], target_joints[:, 1], target_joints[:, 2], c='b', marker='x')
    ax.scatter(opt_joints[:, 0], opt_joints[:, 1], opt_joints[:, 2], c='r', marker='o')
    plt.show()

