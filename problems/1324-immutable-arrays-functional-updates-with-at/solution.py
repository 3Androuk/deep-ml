import jax
import jax.numpy as jnp

def replace_at(x, indices, values):
    """Return a copy of 1-D array x with x[indices] replaced by values.
    Must not modify x."""
    n = len(indices)
    output = x
    for i in range(n):
        output = output.at[indices[i]].set(values[i])
    return output
