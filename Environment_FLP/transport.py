"""
Exact solver for the small transportation problem behind the empty-travel cost.

    minimise   sum_wb c[w][b] * x[w][b]
    subject to sum_b x[w][b] = supply[w],  sum_w x[w][b] = demand[b],  x >= 0

Successive shortest paths (Bellman-Ford on the residual graph). Two implementations of the same
algorithm:
  * `transport_cost`: plain Python, no dependency;
  * a numba-compiled version, used automatically by `solve_transport` when numba is installed
    (about 50 times faster; `pip install numba`).
Both return the same optimum as a general LP solver (checked against scipy's HiGHS in
tests/test_flp_env.py).
"""
import numpy as np

INF = float('inf')
EPS = 1e-9

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:          # numba is optional
    HAVE_NUMBA = False


def transport_cost(c, supply, demand, return_flow=False):
    """
    c: list of lists (len(supply) x len(demand)); totals of supply and demand must be equal.
    Returns the optimal cost, or (cost, flow) with flow[w][b] the optimal shipment when
    `return_flow` is True.
    """
    nw, nb = len(supply), len(demand)
    s, d = list(supply), list(demand)
    flow = [[0.0] * nb for _ in range(nw)]
    total = 0.0
    remaining = sum(s)
    range_w, range_b = range(nw), range(nb)
    while remaining > EPS:
        dist_w = [0.0 if s[w] > EPS else INF for w in range_w]
        pred_w = [-1] * nw                  # delivery reached through a backward arc b -> w
        dist_b, pred_b = [INF] * nb, [-1] * nb
        changed = True
        while changed:
            changed = False
            for w in range_w:
                dw = dist_w[w]
                if dw == INF:
                    continue
                cw = c[w]
                for b in range_b:
                    v = dw + cw[b]
                    if v < dist_b[b] - EPS:
                        dist_b[b], pred_b[b], changed = v, w, True
            for w in range_w:
                fw, cw = flow[w], c[w]
                for b in range_b:
                    if fw[b] > EPS and dist_b[b] != INF:
                        v = dist_b[b] - cw[b]
                        if v < dist_w[w] - EPS:
                            dist_w[w], pred_w[w], changed = v, b, True
        sink, best = -1, INF
        for b in range_b:
            if d[b] > EPS and dist_b[b] < best:
                sink, best = b, dist_b[b]
        if sink < 0:
            raise RuntimeError('transportation problem is infeasible')
        # walk back to a source, collecting the arcs and the bottleneck
        amount, b, forward, backward = d[sink], sink, [], []
        while True:
            w = pred_b[b]
            forward.append((w, b))
            if pred_w[w] < 0:
                amount = min(amount, s[w])
                source = w
                break
            b = pred_w[w]
            backward.append((w, b))
            amount = min(amount, flow[w][b])
        for w, b in forward:
            flow[w][b] += amount
        for w, b in backward:
            flow[w][b] -= amount
        s[source] -= amount
        d[sink] -= amount
        remaining -= amount
        total += amount * best
    return (total, flow) if return_flow else total


def _ssp_arrays(c, supply, demand):
    """Same algorithm on numpy arrays, written with plain loops so that numba can compile it."""
    nw, nb = c.shape
    s = supply.copy()
    d = demand.copy()
    flow = np.zeros((nw, nb))
    dist_w = np.empty(nw)
    dist_b = np.empty(nb)
    pred_w = np.empty(nw, dtype=np.int64)
    pred_b = np.empty(nb, dtype=np.int64)
    total = 0.0
    remaining = s.sum()
    while remaining > EPS:
        for w in range(nw):
            dist_w[w] = 0.0 if s[w] > EPS else np.inf
            pred_w[w] = -1
        for b in range(nb):
            dist_b[b] = np.inf
            pred_b[b] = -1
        changed = True
        while changed:
            changed = False
            for w in range(nw):
                dw = dist_w[w]
                if dw == np.inf:
                    continue
                for b in range(nb):
                    v = dw + c[w, b]
                    if v < dist_b[b] - EPS:
                        dist_b[b] = v
                        pred_b[b] = w
                        changed = True
            for w in range(nw):
                for b in range(nb):
                    if flow[w, b] > EPS and dist_b[b] != np.inf:
                        v = dist_b[b] - c[w, b]
                        if v < dist_w[w] - EPS:
                            dist_w[w] = v
                            pred_w[w] = b
                            changed = True
        sink = -1
        best = np.inf
        for b in range(nb):
            if d[b] > EPS and dist_b[b] < best:
                sink = b
                best = dist_b[b]
        if sink < 0:
            return -1.0, flow                      # infeasible (totals not equal)
        amount = d[sink]
        b = sink
        source = -1
        while True:                                # bottleneck along the path
            w = pred_b[b]
            if pred_w[w] < 0:
                if s[w] < amount:
                    amount = s[w]
                source = w
                break
            b = pred_w[w]
            if flow[w, b] < amount:
                amount = flow[w, b]
        b = sink
        while True:                                # push the flow
            w = pred_b[b]
            flow[w, b] += amount
            if pred_w[w] < 0:
                break
            b = pred_w[w]
            flow[w, b] -= amount
        s[source] -= amount
        d[sink] -= amount
        remaining -= amount
        total += amount * best
    return total, flow


if HAVE_NUMBA:
    _ssp_compiled = njit(cache=True)(_ssp_arrays)


def solve_transport(c, supply, demand):
    """
    c: float array (len(supply) x len(demand)); supply, demand: float arrays with equal totals.
    Returns (optimal cost, optimal flow array). Uses the compiled solver when numba is installed.
    """
    if HAVE_NUMBA:
        total, flow = _ssp_compiled(np.ascontiguousarray(c, dtype=np.float64),
                                    np.ascontiguousarray(supply, dtype=np.float64),
                                    np.ascontiguousarray(demand, dtype=np.float64))
        if total < 0:
            raise RuntimeError('transportation problem is infeasible')
        return float(total), flow
    total, flow = transport_cost(np.asarray(c).tolist(), list(supply), list(demand), return_flow=True)
    return total, np.array(flow)
