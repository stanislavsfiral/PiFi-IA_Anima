#!/usr/bin/env python3
# Sfiral Builder Agent v0.2
# Hill-climbing по награде NeuroCore.
# Стартует с экрана -> ai_memory -> synthetic (цепочка источников).

import copy
import json
import os
import glob
import random
from sfiral_neurocore_torch import predict_graph

STEP_X = 140.0
STEP_Z = 108.9

AI_MEMORY_DIR = "ai_memory"
MODELS_DIR = r"C:\Калмагоров\модели"


def _read_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_graph(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        return (obj, [])
    if not isinstance(obj, dict):
        return None
    if "graph" in obj and isinstance(obj["graph"], dict):
        g = obj["graph"]
        n = g.get("nodes")
        if isinstance(n, list) and n:
            return (n, g.get("edges", []) or [])
    n = obj.get("nodes")
    if isinstance(n, list) and n:
        return (n, obj.get("edges", []) or [])
    return None


def _candidate_files():
    files = []
    files += glob.glob(os.path.join(AI_MEMORY_DIR, "*.json"))
    if os.path.isdir(MODELS_DIR):
        files += glob.glob(os.path.join(MODELS_DIR, "*.json"))
    def key(p):
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0.0
    files.sort(key=key, reverse=True)
    return files


def pick_seed_from_disk():
    for path in _candidate_files():
        g = _extract_graph(_read_json_safe(path))
        if g is not None:
            name = os.path.basename(path)
            return (g[0], g[1], "memory_session:" + name)
    return None


def starter_graph():
    nodes = []
    for i in range(3):
        nodes.append({
            "id": i + 1,
            "x": -i * STEP_X,
            "y": 0.0,
            "z": -i * STEP_Z,
            "params": {
                "N": 5,
                "scale": 1.0,
                "stretch": 0.7778,
                "angles": [0.0, 0.0, 0.0],
            },
        })
    edges = [
        {"from": 1, "to": 2, "type": "chain_seam", "weight": 1.0},
        {"from": 2, "to": 3, "type": "chain_seam", "weight": 1.0},
    ]
    return {"nodes": nodes, "edges": edges}


def reward_of(feat):
    r = float(feat.get("sens_delta", 0.0))
    if feat.get("bipartite"):
        r += 0.20
    r += 0.30 * float(feat.get("coverage", 0.0))
    return round(r, 6)


def sigma_str(feat):
    p = int(feat.get("n_pos", 0))
    n = int(feat.get("n_neg", 0))
    return "+" * p + " " + "-" * n


def score(graph):
    res = predict_graph(graph["nodes"], graph["edges"], mode="canon")
    feat = res.get("features", {})
    return reward_of(feat), feat, res.get("diagnosis", "")


def mutate(graph, rng):
    g = copy.deepcopy(graph)
    nodes = g["nodes"]
    if not nodes:
        return g, "no-op"
    kind = rng.random()
    idx = rng.randrange(len(nodes))
    if not nodes[idx].get("params"):
        nodes[idx]["params"] = {}
    p = nodes[idx]["params"]
    if not p.get("angles"):
        p["angles"] = [0.0, 0.0, 0.0]
    cur = float(p["angles"][0])
    if kind < 0.70:
        new = cur + 180.0
        if new >= 360.0:
            new -= 360.0
        p["angles"][0] = new
        act = "flip q%d X+180" % (idx + 1)
    elif kind < 0.85:
        p["angles"][0] = 0.0
        act = "reset q%d" % (idx + 1)
    else:
        new = cur - 180.0
        if new < 0.0:
            new += 360.0
        p["angles"][0] = new
        act = "flip q%d X-180" % (idx + 1)
    return g, act


def resolve_start(graph_arg):
    nodes = []
    edges = []
    if isinstance(graph_arg, dict):
        nodes = graph_arg.get("nodes") or []
        edges = graph_arg.get("edges") or []
    if len(nodes) >= 2:
        return ({"nodes": nodes, "edges": edges}, "screen_graph")
    seed = pick_seed_from_disk()
    if seed is not None:
        return ({"nodes": seed[0], "edges": seed[1]}, seed[2])
    return (starter_graph(), "synthetic_starter")


def learn_build(graph_arg, max_steps=24, seed=1):
    rng = random.Random(int(seed))
    cur, source = resolve_start(graph_arg)

    best_r, best_feat, best_diag = score(cur)
    best_g = copy.deepcopy(cur)
    trace = []
    cur_r = best_r

    for step in range(1, int(max_steps) + 1):
        cand, act = mutate(cur, rng)
        cand_r, cand_feat, _ = score(cand)
        accept = False
        if cand_r > cur_r:
            accept = True
        elif rng.random() < 0.15:
            accept = True
        if accept:
            cur = cand
            cur_r = cand_r
        if cand_r > best_r:
            best_r = cand_r
            best_feat = cand_feat
            best_g = copy.deepcopy(cand)
            _, _, best_diag = score(best_g)
        trace.append({
            "step": step,
            "action": act,
            "reward": round(cand_r, 4),
            "sigma": sigma_str(cand_feat),
            "W": str(cand_feat.get("W_parity", "0")),
            "bip": "yes" if cand_feat.get("bipartite") else "no",
            "graph": copy.deepcopy(cand),
        })

    return {
        "ok": True,
        "source": source,
        "n_nodes": len(best_g.get("nodes", [])),
        "best_reward": round(best_r, 4),
        "final_diagnosis": best_diag,
        "final_graph": best_g,
        "trace": trace,
        "note": "hill-climbing by NeuroCore reward",
    }


def selftest():
    res = learn_build({}, max_steps=12, seed=7)
    print("source=", res["source"])
    print("n_nodes=", res["n_nodes"])
    print("best_reward=", res["best_reward"])
    print(res["final_diagnosis"])


if __name__ == "__main__":
    selftest()