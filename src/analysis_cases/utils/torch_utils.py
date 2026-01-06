import torch


def init_hidden_if_needed(model, batch_size, device):
    if not hasattr(model, "init_hidden"):
        return None
    
    hidden = model.init_hidden(batch_size)

    if isinstance(hidden, tuple):
        return tuple(h.to(device) for h in hidden)
    
    return hidden.to(device)