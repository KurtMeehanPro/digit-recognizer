"""Optional learning-rate warm-up and regularization helpers."""
# Standard imports
import math


def linear_warmup_rate(target_rate, step, warmup_steps):
    """Rate for a one-based optimizer update; step zero denotes the starting rate.

    With warmup, update 1 uses target/warmup_steps, update warmup_steps
    reaches target, and all remaining updates hold target. No decay is added.
    Zero warmup steps preserves constant-rate training.
    """
    if not math.isfinite(target_rate) or target_rate <= 0:
        raise ValueError("target_rate must be positive and finite")
    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    return target_rate if warmup_steps == 0 else target_rate * min(step / warmup_steps, 1.0)


def regularization_configuration(l2_strength=0.0, decoupled_weight_decay=0.0):
    """Explicit L2 = (lambda/2)*sum(weight**2); biases excluded from both arms."""
    for value in (l2_strength, decoupled_weight_decay):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Regularization strengths must be finite and nonnegative")
    return {"l2_strength": l2_strength, "decoupled_weight_decay": decoupled_weight_decay,
            "l2_convention": "lambda/2 * sum(weight_squared)",
            "parameter_policy": "named weight tensors only; biases excluded",
            "validation_loss": "unpenalized_cross_entropy"}


def regularized_weights(model):
    return [parameter for name, parameter in model.named_parameters()
            if name.endswith("weight") and parameter.requires_grad]


def explicit_l2_penalty(model, strength):
    weights = regularized_weights(model)
    return .5 * strength * sum(parameter.square().sum() for parameter in weights)
