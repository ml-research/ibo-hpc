import numpy as np

def create_buckets(y, sort_idx=None, min_samples=20):
    """
        Recursively create buckets based on a linear scheme.
        Start off  with cutting off the first half of the array y,
        then cut off the first half of the second half from before and so on.
    """

    split_idx = y.shape[0] // 2
    l, r = y[:split_idx], y[(split_idx + 1):]
    if sort_idx is not None:
        lidx, ridx = sort_idx[:split_idx], sort_idx[(split_idx + 1):]
    if r.shape[0] // 2 < 20:
        if sort_idx is not None:
            return [l, r], [lidx, ridx]
        return [l, r]
    else:
        if sort_idx is not None:
            bckt, sidx = create_buckets(r, ridx, min_samples)
            return [l] + bckt, [lidx] + sidx 
        return [l] + create_buckets(r, None, min_samples)
    
def create_quantile_buckets(y, q=0.5, inc_factor=0.5, q_max=0.99):
    """
        split an array y into several chunks according to quanitles.
        We start and cut off the first half of y, then proceed and cut off the first half
        of the right half from before and so on.
    """

    inc = q*inc_factor

    buckets = []
    idx_buckets = []
    
    last_quantile = np.quantile(y, q)
    q += inc
    inc = inc * inc_factor
    while q < q_max:
        quantile = np.quantile(y, q)
        q += inc
        inc = inc * inc_factor
        if q >= q_max:
            y_idx = np.argwhere(y > last_quantile).flatten()
        else:
            y_idx = np.argwhere((y > last_quantile) & (y <= quantile)).flatten()
        buckets.append(y[y_idx])
        idx_buckets.append(y_idx)
        last_quantile = quantile

    assigned = np.concatenate(idx_buckets)
    buckets.append(y[~assigned])
    not_assigned = np.array([i for i in range(len(y)) if i not in assigned])
    idx_buckets.append(not_assigned)

    return buckets, idx_buckets

def compute_bin_number(ds_samples, offset=5, m=0.014, as_int=True):
    num_buckets = ds_samples / (offset + (m*ds_samples))
    if as_int:
        return int(num_buckets)
    return num_buckets