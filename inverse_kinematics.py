"""Pose-only IK shared by the body-model adapters."""

from dataclasses import dataclass
import copy

import torch


@dataclass
class IKResult:
    parameters: dict
    initial_error: float
    final_error: float
    iterations: int


@torch.enable_grad()
def solve_ik(model, input_params, target_joints, shape_params=None, expression=None,
             target_indices=None, max_iter=100, tolerance=0.0005):
    """Fit edited points in model coordinates, preserving shape and global placement.

    Non-target points are soft anchors. Native parameter limits are projected
    after each step; the best finite pose is returned, including for unreachable
    targets. Errors are RMS distances in viewer units (meters).
    """
    params = {key: value.detach().clone() for key, value in input_params.items()}
    shape = shape_params.detach().clone() if torch.is_tensor(shape_params) else copy.deepcopy(shape_params)
    expression = None if expression is None else expression.detach().clone()
    with torch.no_grad():
        initial_joints = model.ik_joints(params, shape, expression).detach()
    target = torch.as_tensor(target_joints, dtype=initial_joints.dtype,
                             device=initial_joints.device).detach()
    if target.shape != initial_joints.shape or not torch.isfinite(target).all():
        raise ValueError('IK targets must be finite and match the displayed joint positions.')
    if not torch.isfinite(initial_joints).all():
        raise ValueError('The initial model pose has non-finite joint positions.')

    if target_indices is None:
        selected = torch.linalg.vector_norm(target - initial_joints, dim=-1) > 1e-6
    else:
        indices = torch.as_tensor(target_indices, dtype=torch.long, device=target.device)
        if indices.ndim != 1 or torch.any(indices < 0) or torch.any(indices >= len(target)):
            raise ValueError('Invalid IK target indices.')
        selected = torch.zeros(len(target), dtype=torch.bool, device=target.device)
        selected[indices] = True
    if not selected.any():
        return IKResult(params, 0., 0., 0)

    masks = model.ik_parameter_masks(params)
    parts, lower, upper, groups = [], [], [], []
    for name, mask in masks.items():
        value = params[name]
        mask = mask.to(device=value.device, dtype=torch.bool)
        if not mask.any():
            continue
        groups.append((name, mask, int(mask.sum())))
        parts.append(value[mask])
        if name in model.parameter_limits:
            bounds = torch.as_tensor(model.parameter_limits[name], dtype=value.dtype, device=value.device)
            lower.append(bounds[..., 0].reshape_as(value)[mask])
            upper.append(bounds[..., 1].reshape_as(value)[mask])
        else:
            lower.append(torch.full_like(value[mask], -torch.inf))
            upper.append(torch.full_like(value[mask], torch.inf))

    initial_mse = ((initial_joints[selected] - target[selected]) ** 2).sum(-1).mean().item()
    initial_error = initial_mse ** 0.5
    if not parts or initial_error <= tolerance or max_iter <= 0:
        return IKResult(params, initial_error, initial_error, 0)

    reference = torch.cat(parts)
    low, high = torch.cat(lower), torch.cat(upper)
    values = reference.clamp(low, high).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([values], lr=0.03)

    def unpack(vector):
        result = dict(params)
        offset = 0
        for name, mask, count in groups:
            value = params[name].clone()
            value[mask] = vector[offset:offset + count]
            result[name] = value
            offset += count
        return result

    best_values = reference.clone()
    best_loss, best_mse = initial_mse, initial_mse
    stale_steps = 0
    iterations = 0
    for step in range(max_iter + 1):
        predicted = model.ik_joints(unpack(values), shape, expression)
        target_loss = ((predicted[selected] - target[selected]) ** 2).sum(-1).mean()
        anchor_loss = (predicted[~selected] - initial_joints[~selected]).square().sum(-1).mean() if (~selected).any() else 0.
        loss = target_loss + 0.01 * anchor_loss + 1e-5 * (values - reference).square().mean()
        if not torch.isfinite(loss):
            break
        score, error = loss.detach().item(), target_loss.detach().item()
        if score < best_loss:
            best_loss, best_mse = score, error
            best_values = values.detach().clone()
            stale_steps = 0
        else:
            stale_steps += 1
        if error ** 0.5 <= tolerance or step == max_iter or stale_steps >= 25:
            break
        optimizer.zero_grad(set_to_none=True)
        gradient, = torch.autograd.grad(loss, values)
        if not torch.isfinite(gradient).all() or gradient.abs().max() < 1e-12:
            break
        values.grad = gradient
        optimizer.step()
        with torch.no_grad():
            values.clamp_(low, high)
        iterations += 1

    result = {key: value.detach().clone() for key, value in unpack(best_values).items()}
    return IKResult(result, initial_error, best_mse ** 0.5, iterations)
