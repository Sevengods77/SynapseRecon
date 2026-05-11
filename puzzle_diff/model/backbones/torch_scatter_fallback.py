import torch

def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    """
    A native PyTorch implementation of the scatter operation to bypass 
    binary dependency issues with torch-scatter on Windows.
    """
    if reduce == "sum" or reduce == "add":
        if out is None:
            if dim_size is None:
                dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src) if src.dim() > index.dim() else index, src)
    
    elif reduce == "mean":
        # Sum then divide by counts
        sum_val = scatter(src, index, dim, dim_size=dim_size, reduce="sum")
        
        # Count occurrences
        counts = torch.zeros(sum_val.shape[dim], dtype=src.dtype, device=src.device)
        ones = torch.ones(index.shape, dtype=src.dtype, device=src.device)
        counts.scatter_add_(0, index, ones)
        
        # Reshape counts for broadcasting
        if sum_val.dim() > 1:
            view_shape = [1] * sum_val.dim()
            view_shape[dim] = -1
            counts = counts.view(view_shape)
            
        return sum_val / counts.clamp(min=1)
    
    else:
        raise NotImplementedError(f"Reduction {reduce} not implemented in fallback.")
