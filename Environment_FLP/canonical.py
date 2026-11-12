"""
Layout geometry and canonical-layout mapping.

A layout is described, per workstation m, by a triple (site_index, reference_node, shape).
The pickup/delivery (P/D) coordinates of a workstation are fully determined by that triple,
and the transportation cost depends only on the P/D coordinates (the graph is rebuilt from
positions by `graph_corrector.corrector`).

Any isometry g of the square workshop grid (rotations by 0/90/180/270 degrees and the four
mirrors, i.e. the dihedral group D4) maps sites onto sites and preserves every shortest-path
distance. Because the transportation demand is attached to workstation IDs and not to sites,
layout L and g(L) always have exactly the same cost. They are therefore the same point for the
optimisation, and the environment can always work on one representative of each orbit: the
canonical layout, chosen as the lexicographically smallest encoding among the (up to) 8 images.
"""

SHAPES = ('U_shape', 'U_shape_prime', 'L_shape', 'L_shape_prime', 'I_shape')


def _key(point):
    """Exact integer key for a coordinate (coordinates are multiples of 0.5)."""
    return int(round(point[0] * 2)), int(round(point[1] * 2))


def _toward(p, q, dist):
    """Point at distance `dist` from p in the direction of q (axis-aligned segment)."""
    dx = 0 if q[0] == p[0] else (1 if q[0] > p[0] else -1)
    dy = 0 if q[1] == p[1] else (1 if q[1] > p[1] else -1)
    return p[0] + dx * dist, p[1] + dy * dist


def _mid(p, q):
    return (p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5


class LayoutGeometry:
    def __init__(self, sites, node_pos):
        """
        sites: {site_id: [corner labels in cyclic order, starting at the default reference]}
        node_pos: {node label: (x, y)} for (at least) all grid nodes.
        """
        self.site_ids = list(sites)
        self.site_corners = [[str(c) for c in sites[s]] for s in self.site_ids]
        self.node_pos = {str(n): (float(p[0]), float(p[1])) for n, p in node_pos.items()}
        self.grid_nodes = {n for corners in self.site_corners for n in corners}
        self.node_at = {_key(self.node_pos[n]): n for n in self.grid_nodes}

        xs = [self.node_pos[n][0] for n in self.grid_nodes]
        ys = [self.node_pos[n][1] for n in self.grid_nodes]
        self.xmin, self.xmax, self.ymin, self.ymax = min(xs), max(xs), min(ys), max(ys)

        self.site_of_corners = {frozenset(c): i for i, c in enumerate(self.site_corners)}
        self.adjacent_sites = self._adjacent_sites()
        self.transforms = self._valid_transforms()

        # (site_index, P key, D key) -> (reference node, shape)
        self.config_lookup = {}
        for i, corners in enumerate(self.site_corners):
            for ref in corners:
                for shape in SHAPES:
                    p, d = self.pd_coords(i, ref, shape)
                    self.config_lookup[(i, _key(p), _key(d))] = (ref, shape)

    # ------------------------------------------------------------------ geometry
    def pd_coords(self, site_index, ref, shape):
        """P/D coordinates of a workstation, identical to the geometry in Actions.py."""
        corners = self.site_corners[site_index]
        i = corners.index(str(ref))
        a, b, c, d = [self.node_pos[n] for n in corners[i:] + corners[:i]]
        if shape == 'U_shape':
            return _toward(a, b, 1), _toward(a, b, 2)
        if shape == 'U_shape_prime':
            return _toward(a, b, 2), _toward(a, b, 1)
        if shape == 'L_shape':
            return _mid(a, b), _mid(b, c)
        if shape == 'L_shape_prime':
            return _mid(b, c), _mid(a, b)
        if shape == 'I_shape':
            return _mid(a, b), _mid(c, d)
        raise ValueError(f'Unknown shape {shape}')

    def _adjacent_sites(self):
        adj = {i: set() for i in range(len(self.site_corners))}
        for i, ci in enumerate(self.site_corners):
            for j, cj in enumerate(self.site_corners):
                if i < j and len(set(ci) & set(cj)) == 2:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    # --------------------------------------------------------------- symmetries
    def _candidate_transforms(self):
        cx, cy = (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2
        maps = [
            lambda u, v: (u, v),       # identity
            lambda u, v: (-v, u),      # rotate 90
            lambda u, v: (-u, -v),     # rotate 180
            lambda u, v: (v, -u),      # rotate 270
            lambda u, v: (-u, v),      # mirror vertical axis
            lambda u, v: (u, -v),      # mirror horizontal axis
            lambda u, v: (v, u),       # mirror main diagonal
            lambda u, v: (-v, -u),     # mirror anti-diagonal
        ]

        def make(f):
            def g(p):
                u, v = f(p[0] - cx, p[1] - cy)
                return u + cx, v + cy
            return g

        return [make(f) for f in maps]

    def _valid_transforms(self):
        """Keep only the isometries that map the grid (and every site) onto itself."""
        valid = []
        for g in self._candidate_transforms():
            ok = True
            for corners in self.site_corners:
                image = set()
                for n in corners:
                    k = _key(g(self.node_pos[n]))
                    if k not in self.node_at:
                        ok = False
                        break
                    image.add(self.node_at[k])
                if not ok or frozenset(image) not in self.site_of_corners:
                    ok = False
                    break
            if ok:
                valid.append(g)
        return valid

    # ------------------------------------------------------------ canonical form
    def transform_layout(self, g, layout):
        """layout: list of (site_index, ref, shape) per machine -> transformed layout + key."""
        new_layout, key = [], []
        for site_index, ref, shape in layout:
            p, d = self.pd_coords(site_index, ref, shape)
            gp, gd = _key(g(p)), _key(g(d))
            image = frozenset(self.node_at[_key(g(self.node_pos[n]))]
                              for n in self.site_corners[site_index])
            new_site = self.site_of_corners[image]
            new_ref, new_shape = self.config_lookup[(new_site, gp, gd)]
            new_layout.append((new_site, new_ref, new_shape))
            key.extend((new_site,) + gp + gd)
        return new_layout, tuple(key)

    def canonical_layout(self, layout):
        """Return (canonical layout, index of the transform used)."""
        best, best_key, best_index = None, None, None
        for index, g in enumerate(self.transforms):
            new_layout, key = self.transform_layout(g, layout)
            if best_key is None or key < best_key:
                best, best_key, best_index = new_layout, key, index
        return best, best_index

    def is_canonical(self, layout):
        normalized = [(int(s), str(r), sh) for s, r, sh in layout]
        return self.canonical_layout(normalized)[0] == normalized

    # ------------------------------------------------------------------ helpers
    def pd_positions(self, layout):
        """{'P1': (x, y), 'D1': (x, y), ...} for a layout."""
        pos = {}
        for m, (site_index, ref, shape) in enumerate(layout, start=1):
            p, d = self.pd_coords(site_index, ref, shape)
            pos[f'P{m}'] = p
            pos[f'D{m}'] = d
        return pos

    @staticmethod
    def has_overlap(pd_pos):
        """True when two P/D points (of any workstations) share the same coordinates."""
        keys = [_key(p) for p in pd_pos.values()]
        return len(keys) != len(set(keys))
