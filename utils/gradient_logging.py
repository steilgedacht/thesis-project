import torch


def gradient_statistics(model):
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {}

    all_gradients = torch.cat(gradients)
    return {
        "grad_mean": all_gradients.mean().item(),
        "grad_max": all_gradients.max().item(),
        "grad_min": all_gradients.min().item(),
        "grad_std": all_gradients.std(unbiased=False).item(),
    }
