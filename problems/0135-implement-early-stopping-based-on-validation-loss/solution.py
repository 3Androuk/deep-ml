from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    
    best_epoch = 0
    current = 0
    for i in range(1, len(val_losses)):
        improvement = val_losses[best_epoch] - val_losses[i]
        if improvement > min_delta:
            best_epoch = i
            current = 0
        else:
            current += 1
        if current >= patience:
            return (i, best_epoch)
    return (i, best_epoch)
    