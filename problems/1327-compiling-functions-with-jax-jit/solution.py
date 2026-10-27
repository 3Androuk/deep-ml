import jax
import jax.numpy as jnp

@jax.jit
def standardize(x):
    """Return (x - mean(x)) / std(x) for a 1-D array x."""
    mu = jnp.mean(x)
    sig = jnp.std(x)
    return (x - mu) / sig
