"""Public M4Human mesh, regressed-joint and rotation metrics."""

import torch


def axis_angle_to_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Pinned public Rodrigues formula, including its 1e-8 stabilizer."""
    flat = theta.reshape(-1, 3)
    angle = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
    axis = flat / (angle + 1e-08)
    x, y, z = axis.unbind(1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), 1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=theta.device, dtype=theta.dtype).unsqueeze(0)
    angle = angle.unsqueeze(-1)
    matrix = eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * torch.bmm(skew, skew)
    if theta.shape[-1] == 3:
        return matrix.reshape(*theta.shape[:-1], 3, 3)
    if theta.shape[-1] % 3:
        raise ValueError("axis-angle pose width must be divisible by 3")
    return matrix.reshape(*theta.shape[:-1], theta.shape[-1] // 3, 3, 3)


def mean_joint_rotation_error_deg(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
    pred_rot, gt_rot = (axis_angle_to_matrix(pred_pose), axis_angle_to_matrix(gt_pose))
    cosine = ((pred_rot * gt_rot).sum(dim=(-1, -2)) - 1) / 2
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).mean(-1)


def batched_smplx(self, betas, pose, root, trans, gender):
    from sources.Train_and_model_loss import batch_rodrigues

    vertices = torch.empty((betas.shape[0], 10475, 3), device=betas.device, dtype=betas.dtype)
    mask = gender.reshape(-1) > 0.5
    for male in (True, False):
        indices = torch.where(mask if male else ~mask)[0]
        if indices.numel() == 0:
            continue
        layer = self.smplx_layer_male if male else self.smplx_layer_female
        out = layer(
            betas=betas[indices],
            body_pose=batch_rodrigues(pose[indices].reshape(-1, 3)).reshape(-1, 21, 3, 3),
            global_orient=batch_rodrigues(root[indices]).unsqueeze(1),
            transl=trans[indices],
        )
        vertices[indices] = out.vertices
    return vertices * 1000


def batched_joints(model, vertices, gender):
    male = model.smplx_layer_male.J_regressor[:22]
    female = model.smplx_layer_female.J_regressor[:22]
    reg = torch.where((gender.reshape(-1) > 0.5)[:, None, None], male[None], female[None])
    return torch.einsum("bij,bjc->bic", reg, vertices)
