#!/usr/bin/env python3
# Sfiral NeuroCore minimal v0.3
# Standalone diagnosis. No numpy, no torch.

import json
import math
from collections import Counter, defaultdict, deque

OVERLAP_TOL = 10.0
NEAR_RATIO = 1.15
ANGLE_FLIP_TOL = 25.0
BASE_STEP = 140.0


def fnum(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def params(n):
    p = n.get("params") or {}
    return p if isinstance(p, dict) else {}


def angles(n):
    a = params(n).get("angles") or [0, 0, 0]
    if not isinstance(a, list):
        return [0.0, 0.0, 0.0]
    out = []
    for i in range(3):
        out.append(fnum(a[i] if i < len(a) else 0.0))
    return out


def pos(n):
    return (
        fnum(n.get("x")),
        fnum(n.get("y")),
        fnum(n.get("z")),
    )


def dist(a, b):
    ax, ay, az = pos(a)
    bx, by, bz = pos(b)
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def eq_stretch(N=5.0):
    N = max(1.0, fnum(N, 5.0))
    return (60.0 + 2.0 * N) / (80.0 + 2.0 * N)


def is_flip_angle(delta):
    d = abs(fnum(delta))
    if abs(d - 180.0) <= ANGLE_FLIP_TOL:
        return True
    if abs(d - 540.0) <= ANGLE_FLIP_TOL:
        return True
    dm = d % 360.0
    return abs(dm - 180.0) <= ANGLE_FLIP_TOL


def is_flip_pair(a, b):
    aa = angles(a)
    bb = angles(b)
    for i in range(3):
        if is_flip_angle(aa[i] - bb[i]):
            return True
    return False


def prep_nodes(nodes):
    out = []
    for i, n in enumerate(nodes or []):
        if not isinstance(n, dict):
            continue
        m = dict(n)
        m["_uid"] = m.get("id", i)
        out.append(m)
    return out


def make_pair(a, b, idx, edge=None):
    rel = "flip180"
    if not is_flip_pair(a, b):
        rel = "identity"

    et = "auto"
    if isinstance(edge, dict):
        et = str(edge.get("type", "auto"))

    return {
        "from_idx": idx[a["_uid"]],
        "to_idx": idx[b["_uid"]],
        "rel_label": rel,
        "edge_type": et,
    }


def nearest_path(nodes):
    if len(nodes) < 2:
        return []

    idx = {}
    for i, n in enumerate(nodes):
        idx[n["_uid"]] = i

    seen = set([0])
    path = [0]

    while len(seen) < len(nodes):
        cur = path[-1]
        best = None
        best_d = 1e99

        for j, n in enumerate(nodes):
            if j in seen:
                continue
            d = dist(nodes[cur], n)
            if d < best_d:
                best_d = d
                best = j

        if best is None:
            break

        path.append(best)
        seen.add(best)

    pairs = []
    for i in range(len(path) - 1):
        a = nodes[path[i]]
        b = nodes[path[i + 1]]
        pairs.append(make_pair(a, b, idx))

    return pairs


def traversal(nodes, edges):
    nodes = prep_nodes(nodes)
    edges = edges or []

    if not nodes:
        return [], "empty"

    idx = {}
    nmap = {}
    for i, n in enumerate(nodes):
        idx[n["_uid"]] = i
        nmap[n["_uid"]] = n

    chain = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("type") == "chain_seam":
            a = nmap.get(e.get("from"))
            b = nmap.get(e.get("to"))
            if a and b:
                chain.append(make_pair(a, b, idx, e))

    if chain:
        return chain, "chain_seam"

    ds = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = dist(nodes[i], nodes[j])
            if d > 1e-6:
                ds.append(d)

    if ds:
        md = min(ds)
        if md >= OVERLAP_TOL:
            lim = md * NEAR_RATIO
            geom = []

            for e in edges:
                if not isinstance(e, dict):
                    continue

                a = nmap.get(e.get("from"))
                b = nmap.get(e.get("to"))

                if not a or not b:
                    continue

                if dist(a, b) <= lim:
                    geom.append(make_pair(a, b, idx, e))

            if geom:
                return geom, "nearest_lattice"

    pairs = nearest_path(nodes)
    if pairs:
        return pairs, "nearest_path"

    return [], "empty"


def sigma_profile(pairs):
    sig = {}
    cur = 1

    for p in pairs:
        i = int(p["from_idx"])
        j = int(p["to_idx"])

        if i not in sig:
            sig[i] = cur

        if str(p.get("rel_label")) == "flip180":
            cur = -cur

        sig[j] = cur

    return sig


def bipartite(pairs, n_nodes):
    if not pairs:
        return True, 0.0, 0

    adj = defaultdict(set)
    parts = set()

    for p in pairs:
        i = int(p["from_idx"])
        j = int(p["to_idx"])
        adj[i].add(j)
        adj[j].add(i)
        parts.add(i)
        parts.add(j)

    denom = float(max(1, n_nodes))
    cov = float(len(parts)) / denom

    color = {}
    ok = True

    for s in parts:
        if s in color:
            continue

        q = deque([s])
        color[s] = 0

        while q:
            u = q.popleft()

            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    ok = False

        if not ok:
            break

    return ok, round(cov, 6), len(parts)


def sens(pairs, n_nodes):
    ok, cov, npart = bipartite(pairs, n_nodes)

    if not pairs:
        return 0.0, "no traversal", ok, cov, npart

    has_flip = False
    for p in pairs:
        if str(p.get("rel_label")) == "flip180":
            has_flip = True
            break

    if not ok:
        return 0.0, "not bipartite", ok, cov, npart

    if not has_flip:
        return 0.0, "contact/no flip", ok, cov, npart

    base = sigma_profile(pairs)

    flip_idx = None
    for i, p in enumerate(pairs):
        if str(p.get("rel_label")) == "flip180":
            flip_idx = i
            break

    pert = {}
    cur = 1

    for i, p in enumerate(pairs):
        a = int(p["from_idx"])
        b = int(p["to_idx"])

        if a not in pert:
            pert[a] = cur

        rf = str(p.get("rel_label")) == "flip180"

        if i == flip_idx:
            rf = not rf

        if rf:
            cur = -cur

        pert[b] = cur

    common = set(base.keys()) & set(pert.keys())

    if not common:
        return 0.0, "profile anomaly", ok, cov, npart

    changed = 0
    for k in common:
        if base[k] != pert[k]:
            changed += 1

    delta = float(changed) / float(len(common))
    return delta, "bipartite+flip", ok, cov, npart


def tau(nodes, pairs):
    if not pairs:
        return "contact"

    xs = []
    ys = []
    zs = []

    for p in pairs:
        a = nodes[int(p["from_idx"])]
        b = nodes[int(p["to_idx"])]

        ax, ay, az = pos(a)
        bx, by, bz = pos(b)

        xs.append(abs(ax - bx))
        ys.append(abs(ay - by))
        zs.append(abs(az - bz))

    mx = sum(xs) / float(len(xs))
    my = sum(ys) / float(len(ys))
    mz = sum(zs) / float(len(zs))

    name = "Axis X"
    val = mx

    if my > val:
        name = "Axis Y"
        val = my

    if mz > val:
        name = "Axis Z"
        val = mz

    if val < 1e-6:
        return "contact"

    return name


def ott(nodes):
    res = {
        "R": 0.0,
        "L": 0.0,
        "S": 0.0,
        "ent": 0.0,
    }

    if not nodes:
        return res

    macro = Counter()
    addr = Counter()
    sc = BASE_STEP * 0.5

    for n in nodes:
        x, y, z = pos(n)

        if abs(x) >= abs(y):
            m = "R" if x > 0 else "L"
        else:
            m = "S"

        macro[m] += 1

        fx = int(math.floor((x / sc) + 2)) % 2
        fy = int(math.floor((y / sc) + 2)) % 2
        fz = int(math.floor((z / sc) + 2)) % 2

        key = "SF-%s-%d%d%d" % (m, fx, fy, fz)
        addr[key] += 1

    total = float(len(nodes))
    ent = 0.0

    for c in addr.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)

    res["R"] = macro.get("R", 0) / total
    res["L"] = macro.get("L", 0) / total
    res["S"] = macro.get("S", 0) / total
    res["ent"] = ent

    return res


def diag(nodes, edges):
    n = len(nodes)
    e = 0

    deg = Counter()
    etypes = Counter()

    for ed in edges:
        if not isinstance(ed, dict):
            continue

        e += 1
        etypes[str(ed.get("type", "unknown"))] += 1

        a = ed.get("from")
        b = ed.get("to")

        if a is not None:
            deg[a] += 1

        if b is not None:
            deg[b] += 1

    maxd = 0
    if deg:
        maxd = max(deg.values())

    dens = 0.0
    if n > 1:
        dens = 2.0 * float(e) / float(n * (n - 1))

    scales = []
    stretches = []

    for nd in nodes:
        p = params(nd)
        scales.append(fnum(p.get("scale"), 1.0))

        N = fnum(p.get("N"), 5.0)
        stretches.append(
            fnum(p.get("stretch"), eq_stretch(N))
        )

    ss = sorted(set([round(s, 6) for s in scales]))

    ladder = False
    need = set([0.25, 0.5, 1.0])

    if need.issubset(set(ss)):
        ladder = True
    else:
        for i in range(1, len(ss)):
            if ss[i - 1] <= 0:
                continue

            r = ss[i] / ss[i - 1]

            if abs(r - 2.0) < 0.02:
                ladder = True
                break

    coh = True
    if stretches:
        coh = (max(stretches) - min(stretches)) < 1e-4

    return {
        "n_nodes": n,
        "n_edges": e,
        "max_degree": maxd,
        "density": round(dens, 6),
        "edge_types": dict(etypes),
        "scale_levels": ss,
        "fractal_ladder": ladder,
        "coherent": coh,
    }


def features(nodes_raw, edges_raw, mode="canon"):
    nodes = prep_nodes(nodes_raw)
    edges = edges_raw or []

    d = diag(nodes, edges)
    pairs, src = traversal(nodes, edges)

    sig = sigma_profile(pairs)
    sv, cls, ok, cov, npart = sens(pairs, len(nodes))

    flips = 0
    for p in pairs:
        if str(p.get("rel_label")) == "flip180":
            flips += 1

    wp = 0
    if pairs:
        wp = 1 if flips % 2 == 0 else -1

    tv = tau(nodes, pairs)

    vals = list(sig.values())

    npos = 0
    nneg = 0
    nzer = 0

    for v in vals:
        if v > 0:
            npos += 1
        elif v < 0:
            nneg += 1
        else:
            nzer += 1

    mean = 0.0
    contrast = 0.0

    if vals:
        mean = sum(vals) / float(len(vals))
        contrast = float(npos + nneg) / float(len(vals))

    ot = ott(nodes)

    sig_out = {}
    for k, v in sig.items():
        sig_out[str(k)] = int(v)

    return {
        "mode": mode,
        "traversal_source": src,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_pairs": len(pairs),
        "coverage": cov,
        "bipartite": ok,
        "has_flip": flips > 0,
        "flip_count": flips,
        "W_parity": wp,
        "tau": tv,
        "sens_delta": sv,
        "sens_class": cls,
        "mean_sigma": mean,
        "contrast": contrast,
        "n_pos": npos,
        "n_neg": nneg,
        "n_zero": nzer,
        "sigma_profile": sig_out,
        "max_degree": d["max_degree"],
        "density": d["density"],
        "edge_types": d["edge_types"],
        "scale_levels": d["scale_levels"],
        "fractal_ladder": d["fractal_ladder"],
        "coherent": d["coherent"],
        "macro_R": ot["R"],
        "macro_L": ot["L"],
        "macro_S": ot["S"],
        "addr_entropy": ot["ent"],
    }


def diagnose(f):
    lines = []

    s = f.get("sens_delta", 0.0)

    if s > 1e-6:
        lines.append("OK: sigma rhythm found")
    elif not f.get("bipartite", False):
        lines.append("WARN: not bipartite, sigma conflict")
    elif not f.get("has_flip", False):
        lines.append("INFO: contact/no flip")
    else:
        lines.append("INFO: rhythm not confirmed")

    t = f.get("tau", "contact")
    w = f.get("W_parity", 0)

    b = "yes" if f.get("bipartite") else "no"
    c = "yes" if f.get("coherent") else "no"
    l = "yes" if f.get("fractal_ladder") else "no"
    src = f.get("traversal_source", "empty")

    lines.append("tau=%s W=%s bip=%s" % (t, w, b))
    lines.append("coh=%s lad=%s src=%s" % (c, l, src))

    p = f.get("n_pos", 0)
    n = f.get("n_neg", 0)
    z = f.get("n_zero", 0)

    cr = f.get("contrast", 0.0)
    se = f.get("sens_delta", 0.0)
    cv = f.get("coverage", 0.0)

    lines.append("sigma: +%d / -%d / 0=%d" % (p, n, z))
    lines.append(
        "contrast=%.3f sens=%.3f cov=%.2f" % (cr, se, cv)
    )

    r = f.get("macro_R", 0.0)
    le = f.get("macro_L", 0.0)
    ss = f.get("macro_S", 0.0)
    en = f.get("addr_entropy", 0.0)

    lines.append("Ottendorf R=%.2f L=%.2f" % (r, le))
    lines.append("Ottendorf S=%.2f ent=%.3f" % (ss, en))

    et = f.get("edge_types") or {}

    if et:
        parts = []
        for k in sorted(et):
            parts.append("%s:%d" % (k, et[k]))

        lines.append("edges: " + ", ".join(parts))

    return "\n".join(lines)


def predict_graph(nodes, edges, mode="canon"):
    f = features(nodes, edges, mode=mode)

    return {
        "ok": True,
        "mode": mode,
        "features": f,
        "diagnosis": diagnose(f),
        "prediction": None,
        "note": "minimal diagnosis mode",
    }


def selftest():
    n1 = {
        "id": 1,
        "x": 0,
        "y": 0,
        "z": 0,
        "params": {
            "N": 5,
            "scale": 1.0,
            "stretch": 0.7778,
            "angles": [0, 0, 0],
        },
    }

    n2 = {
        "id": 2,
        "x": 140,
        "y": 0,
        "z": 0,
        "params": {
            "N": 5,
            "scale": 1.0,
            "stretch": 0.7778,
            "angles": [180, 0, 0],
        },
    }

    e1 = {
        "from": 1,
        "to": 2,
        "type": "chain_seam",
        "weight": 1.0,
    }

    res = predict_graph([n1, n2], [e1])
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    selftest()