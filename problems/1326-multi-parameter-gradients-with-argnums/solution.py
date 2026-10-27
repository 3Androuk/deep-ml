import jax
import jax.numpy as jnp

def mse_grads(w, b, x, y):
    """Return (dL/dw, dL/db) for L = mean((w*x + b - y)**2) via jax.grad.
    w, b are floats; x, y are 1-D jnp arrays. Returns two floats."""
    def loss(w, b, x, y):
        return jnp.mean((w * x + b - y) ** 2)

    grad_w , grad_b = jax.grad(loss, argnums=(0, 1))(w, b, x, y)

    return [float(grad_w), float(grad_b)]
