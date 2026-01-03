



def _to_device(hidden, device):
    if isinstance(hidden, tuple):
        return tuple(h.to(device) for h in hidden)
    return hidden.to(device)