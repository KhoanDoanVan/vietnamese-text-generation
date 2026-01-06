import time
import torch


def benchmark_forward(model, input_tensor, hidden_fn, runs=100):

    # warmup
    for _ in range(10):
        with torch.no_grad():
            hidden = hidden_fn()
            model(input_tensor, hidden) if hidden else model(input_tensor)


    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start = time.time()

    for _ in range(runs):
        with torch.no_grad():
            hidden = hidden_fn()
            model(input_tensor, hidden) if hidden else model(input_tensor)


    torch.cuda.synchronize() if torch.cuda.is_available() else None

    return (time.time() - start) / runs