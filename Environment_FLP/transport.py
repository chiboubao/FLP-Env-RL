"""
Exact solver for the small transportation problem behind the empty-travel cost.

    minimise   sum_wb c[w][b] * x[w][b]
    subject to sum_b x[w][b] = supply[w],  sum_w x[w][b] = demand[b],  x >= 0

Successive shortest paths (Bellman-Ford on the residual graph), in plain Python: for the tiny
instances of this project (at most N x N arcs) it is several times faster than calling a general
LP solver, and it returns the same optimum (checked against scipy's HiGHS in tests/test_flp_env.py).
"""
INF = float('inf')
EPS = 1e-9


def transport_cost(c, supply, demand):
    """c: list of lists (len(supply) x len(demand)); totals of supply and demand must be equal."""
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
    return total
