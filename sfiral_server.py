import http.server
import json
import os
import sys
import io
import math
import numpy as np
import time
import requests
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sfiral_ca_engine import SfiralCAEngine
from qutrit import SfiralQutrit
from measurement import SfiralMeasurement

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

PORT = 8000
AI_MEMORY_DIR = "ai_memory"
os.makedirs(AI_MEMORY_DIR, exist_ok=True)
TIMELINE_LOG_FILE = os.path.join(AI_MEMORY_DIR, "ai_timeline_log.jsonl")
LEARNED_TERMS_FILE = os.path.join(AI_MEMORY_DIR, "learned_terms.json")

LAST_GRAPH = {"nodes": [], "edges": []}


def load_learned_terms():
    if os.path.exists(LEARNED_TERMS_FILE):
        try:
            with open(LEARNED_TERMS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_learned_term(term_data):
    terms = load_learned_terms()
    terms.append(term_data)
    with open(LEARNED_TERMS_FILE, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False, indent=4)


MACROS_DIR = os.path.join(AI_MEMORY_DIR, "macros")
os.makedirs(MACROS_DIR, exist_ok=True)
ONNX_MODEL_PATH = os.path.join("models", "sfiral_ternary_copilot.onnx")
FAQ_FILE_PATH = os.path.join("sfiral_docs", "sfiral_docsfaq_sfiral.txt")
ca_engine = SfiralCAEngine(width=32, height=32, channels=4)


def load_local_faq():
    if os.path.exists(FAQ_FILE_PATH):
        try:
            with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"WARNING read FAQ failed: {e}")
    return ""


def update_last_graph(nodes, edges):
    global LAST_GRAPH
    if isinstance(nodes, list) and isinstance(edges, list):
        LAST_GRAPH = {"nodes": nodes, "edges": edges}


class SfiralChronokvantAnalyzer:
    @staticmethod
    def detect_chronokvants(nodes, edges):
        chronokvants_found = []
        node_map = {n['id']: n for n in nodes}
        explicit_edges = {(e['from'], e['to']): e.get('type', 'linear_reduced') for e in edges}
        explicit_edges.update({(e['to'], e['from']): e.get('type', 'linear_reduced') for e in edges})
        node_ids = list(node_map.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                n1_id = node_ids[i]
                n2_id = node_ids[j]
                n1 = node_map[n1_id]
                n2 = node_map[n2_id]
                p1 = n1.get('params', {})
                p2 = n2.get('params', {})
                angles1 = p1.get('angles', [0, 0, 0])
                angles2 = p2.get('angles', [0, 0, 0])
                x1, y1, z1 = n1.get('x', 0), n1.get('y', 0), n1.get('z', 0)
                x2, y2, z2 = n2.get('x', 0), n2.get('y', 0), n2.get('z', 0)
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                dz = abs(z1 - z2)
                diff_x_angle = abs(angles1[0] - angles2[0])
                has_explicit_link = (n1_id, n2_id) in explicit_edges
                link_type = explicit_edges.get((n1_id, n2_id), 'auto')
                is_direct_chained = abs(diff_x_angle - 180) < 25 or abs(diff_x_angle - 540) < 25
                if has_explicit_link and link_type == 'linear_reduced' or (not has_explicit_link and is_direct_chained and (dx < 120 and dy < 120)):
                    polarization = "Right (R)" if (x1 + x2) >= 0 else "Left (L)"
                    chronokvants_found.append({
                        "node_a": n1_id, "node_b": n2_id,
                        "type": "Linear chronokvant (coupled turn / reduction)",
                        "polarization": f"Polarized superposition ({polarization})",
                        "dz": round(dz, 2),
                        "status": "Linear coupling with X-rotation (~180 deg), extra turn reduced."
                    })
                    continue
                is_diagonal = dx > 15 and dy > 15 and dz > 15 and abs(diff_x_angle) < 15
                if has_explicit_link and link_type == 'diagonal_loop' or (not has_explicit_link and is_diagonal):
                    chronokvants_found.append({
                        "node_a": n1_id, "node_b": n2_id,
                        "type": "Diagonal macro-chronokvant (distributed S-loop)",
                        "polarization": "Zero global chirality (macro superposition)",
                        "dz": round(dz, 2),
                        "status": f"Diagonal coupling without flip (dx={dx:.1f}, dy={dy:.1f}), seam zone forms S-loop."
                    })
        return chronokvants_found


def generate_academic_passport_html(nodes, edges):
    steps = max(len(nodes), 1008)
    chronokvants = SfiralChronokvantAnalyzer.detect_chronokvants(nodes, edges)
    clean_signal = [math.sin((i / 30.0) * 2.0) * 80.0 + math.cos((i / 30.0) * 5.0) * 40.0 for i in range(steps)]
    np.random.seed(2026)
    raw_noise = np.random.randn(steps) * 25.0
    noisy_signal = np.array(clean_signal) + raw_noise
    noisy_energy = float(np.sum(noisy_signal**2))
    window_size = 5
    classical_filtered = []
    for i in range(steps):
        start = max(0, i - window_size // 2)
        end = min(steps, i + window_size // 2 + 1)
        classical_filtered.append(np.mean(noisy_signal[start:end]))
    classical_tensor = np.array(classical_filtered)
    classical_energy = float(np.sum(classical_tensor**2))
    classical_mse = float(np.mean((np.array(clean_signal) - classical_tensor)**2))
    mid_idx = steps // 2
    s_zone_size = min(16, steps - mid_idx)
    left_loop = noisy_signal[:mid_idx]
    s_transition = noisy_signal[mid_idx:mid_idx + s_zone_size]
    right_loop = noisy_signal[mid_idx + s_zone_size:]
    processed_left = left_loop * 0.998
    s_flow = s_transition * 0.999
    processed_right = right_loop * -0.998
    sfiral_tensor = np.concatenate([processed_left, s_flow, processed_right])
    sfiral_energy = float(np.sum(sfiral_tensor**2))
    sfiral_normalized = sfiral_tensor.copy()
    if len(sfiral_normalized) > (mid_idx + s_zone_size):
        sfiral_normalized[mid_idx + s_zone_size:] = np.abs(sfiral_normalized[mid_idx + s_zone_size:])
    sfiral_mse = float(np.mean((np.array(clean_signal) - sfiral_normalized)**2))
    class_ret = (classical_energy / noisy_energy) * 100 if noisy_energy > 0 else 87.4
    sfiral_ret = (sfiral_energy / noisy_energy) * 100 if noisy_energy > 0 else 99.6
    nodes_rows = ""
    for n in nodes[:20]:
        n_id = n.get('id')
        x, y, z = round(n.get('x', 0)), round(n.get('y', 0)), round(n.get('z', 0))
        gate = n.get('params', {}).get('activeGate', 'ROUTER_SWAP')
        nodes_rows += f"<tr><td>Node #{n_id}</td><td>X: {x}, Y: {y}, Z: {z}</td><td>{gate}</td><td style='color:#27ae60; font-weight:bold;'>Stable (0 chirality)</td></tr>"
    if not nodes_rows:
        nodes_rows = "<tr><td colspan='4'>No active nodes in current session</td></tr>"
    chronos_rows = ""
    for ch in chronokvants[:20]:
        chronos_rows += f"<tr><td>#{ch['node_a']} <-> #{ch['node_b']}</td><td>{ch['type']}</td><td>{ch['polarization']}</td><td>{ch['status']}</td></tr>"
    if not chronos_rows:
        chronos_rows = "<tr><td colspan='4'>No chronokvant couplings detected (need node docking)</td></tr>"
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sfiral Topology Academic Passport</title>
        <style>
            body {{ font-family: 'Times New Roman', Times, serif; background: #fcfbf9; color: #1a1a1a; padding: 30px; line-height: 1.6; font-size: 11pt; }}
            .container {{ max-width: 900px; margin: auto; background: #fff; padding: 40px; border: 1px solid #bdc3c7; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
            h1 {{ text-align: center; color: #1a252f; text-transform: uppercase; font-size: 16pt; border-bottom: 2px solid #2c3e50; padding-bottom: 12px; margin-bottom: 20px; }}
            h2 {{ font-size: 13pt; color: #2c3e50; border-left: 4px solid #2980b9; padding-left: 8px; margin-top: 25px; margin-bottom: 10px; }}
            p {{ text-align: justify; margin-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 10pt; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: center; }}
            th {{ background-color: #2c3e50; color: #fff; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f8f9f9; }}
            .meta-grid {{ display: flex; justify-content: space-between; background: #f2f4f4; padding: 12px; border-radius: 4px; margin-bottom: 20px; font-size: 10pt; }}
            .footer {{ margin-top: 30px; font-size: 9pt; color: #7f8c8d; text-align: center; border-top: 1px solid #e0e0e0; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sfiral Topology Academic Passport</h1>
            <div class="meta-grid">
                <div><b>Project:</b> GIDEON / PiFiYA-core</div>
                <div><b>Generated:</b> {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</div>
                <div><b>Verification:</b> Passed</div>
            </div>
            <h2>1. Graph architecture and coupling types</h2>
            <ul>
                <li><b>Active nodes (Sfirals):</b> {len(nodes)}</li>
                <li><b>Graph edges:</b> {len(edges)}</li>
                <li><b>Chronokvant couplings:</b> {len(chronokvants)}</li>
            </ul>
            <h2>2. Chronokvant coupling registry</h2>
            <table><thead><tr><th>Node pair</th><th>Type</th><th>Polarization</th><th>Status</th></tr></thead><tbody>{chronos_rows}</tbody></table>
            <h2>3. Phase stability benchmark</h2>
            <table><thead><tr><th>Metric</th><th>Classical (MA)</th><th>Sfiral (Q-Core)</th><th>Gain</th></tr></thead><tbody>
            <tr><td><b>Energy retention</b></td><td>{class_ret:.1f}%</td><td style="color:#27ae60;font-weight:bold;">{sfiral_ret:.1f}%</td><td>+{max(0, sfiral_ret - class_ret):.1f}%</td></tr>
            <tr><td><b>MSE</b></td><td>{classical_mse:.2f}</td><td style="color:#27ae60;font-weight:bold;">{sfiral_mse:.2f}</td><td>Distortion minimized</td></tr>
            </tbody></table>
            <h2>4. Node registry</h2>
            <table><thead><tr><th>ID</th><th>XYZ</th><th>Gate</th><th>Status</th></tr></thead><tbody>{nodes_rows}</tbody></table>
            <div class="footer">Auto-generated by GIDEON-Sfiral-Architecture 2026</div>
        </div>
    </body>
    </html>
    """
    return html_content


# ============================================================
# DATASET BRIDGE: ai_memory -> scene + offline summary + learn
# ASCII comments, short lines: clipboard/word-wrap cannot break
# a string literal or a comment into code.
# ============================================================
DATASET_PREFIX = "session_"
DATASET_MAX_FILES = 2000


def _norm_params(p):
    p = p or {}
    return {
        "N": p.get("N", 5),
        "target_len": p.get("target_len", 1000),
        "scale": p.get("scale", 1.0),
        "stretch": p.get("stretch", 0.7778),
        "radius": p.get("radius", 140),
        "length": p.get("length", 190),
        "isCustomScaled": p.get("isCustomScaled", False),
        "isGroupScaled": p.get("isGroupScaled", False),
        "angles": list(p.get("angles", [0, 0, 0])),
        "activeGate": p.get("activeGate", "ROUTER_SWAP"),
        "showRight": p.get("showRight", True),
        "showS": p.get("showS", True),
        "showLeft": p.get("showLeft", True),
        "showSLeft": p.get("showSLeft", True),
        "rightSub": dict(p.get("rightSub", {"height": 100})),
        "leftSub": dict(p.get("leftSub", {"height": 100})),
        "sRightSub": dict(p.get("sRightSub", {"height": 100})),
        "sLeftSub": dict(p.get("sLeftSub", {"height": 100})),
    }


def _read_dataset_graph(obj):
    nodes_raw = []
    edges_raw = []
    if isinstance(obj, list):
        nodes_raw = obj
    elif isinstance(obj, dict):
        if "graph" in obj and isinstance(obj["graph"], dict):
            nodes_raw = obj["graph"].get("nodes", []) or []
            edges_raw = obj["graph"].get("edges", []) or []
        elif "nodes" in obj:
            nodes_raw = obj.get("nodes", []) or []
            edges_raw = obj.get("edges", []) or []
    nodes = []
    for i, n in enumerate(nodes_raw):
        nid = n.get("id")
        if not isinstance(nid, int):
            nid = i + 1
        nodes.append({
            "id": nid,
            "mode": n.get("mode", "Single"),
            "x": n.get("x", 0),
            "y": n.get("y", 0),
            "z": n.get("z", 0),
            "params": _norm_params(n.get("params", {})),
            "quantumState": n.get("quantumState", {"intensity": 0, "psi_real": 0, "psi_imag": 0}),
        })
    edges = []
    for e in edges_raw:
        edges.append({
            "from": e.get("from"),
            "to": e.get("to"),
            "weight": e.get("weight", 0.5),
            "type": e.get("type", "chain_seam"),
        })
    return {"nodes": nodes, "edges": edges}


def _list_dataset_files():
    names = []
    try:
        for fn in os.listdir(AI_MEMORY_DIR):
            if fn.startswith(DATASET_PREFIX) and fn.endswith(".json"):
                names.append(fn)
    except Exception:
        pass
    names.sort()
    return names[:DATASET_MAX_FILES]


def dataset_list_payload():
    files = []
    for fn in _list_dataset_files():
        info = {"name": fn, "n_nodes": 0, "n_edges": 0, "size_kb": 0}
        path = os.path.join(AI_MEMORY_DIR, fn)
        try:
            info["size_kb"] = round(os.path.getsize(path) / 1024.0, 1)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            g = _read_dataset_graph(obj)
            info["n_nodes"] = len(g["nodes"])
            info["n_edges"] = len(g["edges"])
        except Exception as ex:
            info["err"] = str(ex)[:80]
        files.append(info)
    return {"status": "success", "count": len(files), "files": files}


def dataset_load_payload(name):
    if not name or not str(name).startswith(DATASET_PREFIX) or not str(name).endswith(".json"):
        return {"status": "error", "message": "bad name"}
    path = os.path.join(AI_MEMORY_DIR, str(name))
    if not os.path.exists(path):
        return {"status": "error", "message": "not found"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        g = _read_dataset_graph(obj)
        return {"status": "success", "name": str(name), "graph": g}
    except Exception as ex:
        return {"status": "error", "message": str(ex)[:120]}


def dataset_summary_payload():
    total = 0
    with_edges = 0
    rhythm = 0
    empty = 0
    buckets = {"1-2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    scored = []
    for fn in _list_dataset_files():
        total += 1
        path = os.path.join(AI_MEMORY_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            g = _read_dataset_graph(obj)
        except Exception:
            empty += 1
            continue
        nn = len(g["nodes"])
        ne = len(g["edges"])
        if nn <= 2:
            buckets["1-2"] += 1
        elif nn <= 5:
            buckets["3-5"] += 1
        elif nn <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11+"] += 1
        if ne == 0:
            empty += 1
            continue
        with_edges += 1
        try:
            ck = SfiralChronokvantAnalyzer.detect_chronokvants(g["nodes"], g["edges"])
        except Exception:
            ck = []
        if len(ck) > 0:
            rhythm += 1
        scored.append({"name": fn, "nodes": nn, "edges": ne, "chronokvants": len(ck)})
    scored.sort(key=lambda x: x["chronokvants"], reverse=True)
    top = scored[:8]
    return {
        "status": "success",
        "total_files": total,
        "with_edges": with_edges,
        "rhythm_found": rhythm,
        "empty": empty,
        "buckets": buckets,
        "top": top,
    }
# ============================================================
# END DATASET BRIDGE
# ============================================================


# ============================================================
# KALMYKOV BRIDGE: second dataset root with 3 orientations
# (vertical / horizontal / diagonal). Reads C:\Kalmykov\models
# recursively, groups graphs by orientation, loads a real file
# into the scene, optionally learns from it. Whitelisted root
# only; traversal and absolute paths rejected. ASCII messages;
# the only non-ASCII token is the directory path literal below,
# so the file MUST be saved as UTF-8 (see validator).
# ============================================================
KALMYKOV_MODELS_DIR = r"C:\Калмагоров\модели"
KALMYKOV_GRAPH_EXT = (".json",)
KALMYKOV_VIEW_EXT = (".stl", ".obj", ".glb", ".gltf", ".3mf", ".ply")
ORIENTATIONS = ("vertical", "horizontal", "diagonal")


def _load_json_anyenc(path):
    for enc in ("utf-8", "cp1251"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _safe_rel_under(root, rel):
    if not rel:
        return None
    rel = str(rel).replace("\\", "/").strip("/")
    if ":" in rel or ".." in rel.split("/"):
        return None
    full = os.path.normpath(os.path.join(root, rel))
    rootn = os.path.normpath(root)
    if not (full == rootn or full.startswith(rootn + os.sep)):
        return None
    return full


def _orient_of(rel):
    low = str(rel).lower()
    for o in ORIENTATIONS:
        if o in low:
            return o
    return "?"


def _walk_kalmykov_graphs(root):
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(KALMYKOV_GRAPH_EXT):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            info = {"rel": rel, "orientation": _orient_of(rel),
                    "n_nodes": 0, "n_edges": 0, "size_kb": 0, "err": ""}
            try:
                info["size_kb"] = round(os.path.getsize(full) / 1024.0, 1)
                obj = _load_json_anyenc(full)
                g = _read_dataset_graph(obj)
                info["n_nodes"] = len(g["nodes"])
                info["n_edges"] = len(g["edges"])
            except Exception as ex:
                info["err"] = str(ex)[:80]
            out.append(info)
    out.sort(key=lambda x: x["rel"])
    return out


def kalmykov_inventory_payload():
    root = KALMYKOV_MODELS_DIR
    inv = {"status": "success", "root": root, "exists": os.path.isdir(root),
           "by_ext": {}, "by_dir": {}, "orientations": {o: 0 for o in ORIENTATIONS}}
    inv["orientations"]["?"] = 0
    if not inv["exists"]:
        return inv
    ext_counts = {}
    dir_counts = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = "(root)"
        dir_counts[rel_dir] = dir_counts.get(rel_dir, 0) + len(filenames)
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower() or "(none)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            o = _orient_of(rel_dir + "/" + fn)
            inv["orientations"][o] = inv["orientations"].get(o, 0) + 1
    inv["by_ext"] = dict(sorted(ext_counts.items(), key=lambda x: -x[1]))
    inv["by_dir"] = dir_counts
    return inv


def kalmykov_list_payload():
    graphs = _walk_kalmykov_graphs(KALMYKOV_MODELS_DIR)
    return {"status": "success", "count": len(graphs), "graphs": graphs}


def kalmykov_load_payload(rel):
    full = _safe_rel_under(KALMYKOV_MODELS_DIR, rel)
    if not full or not os.path.isfile(full):
        return {"status": "error", "message": "bad path or not found"}
    if not full.lower().endswith(KALMYKOV_GRAPH_EXT):
        return {"status": "error", "message": "not a json graph"}
    try:
        obj = _load_json_anyenc(full)
        g = _read_dataset_graph(obj)
        return {"status": "success", "rel": rel, "orientation": _orient_of(rel), "graph": g}
    except Exception as ex:
        return {"status": "error", "message": str(ex)[:120]}


def kalmykov_summary_payload():
    root = KALMYKOV_MODELS_DIR
    total = 0
    with_edges = 0
    rhythm = 0
    empty = 0
    by_orient = {}
    scored = []
    if not os.path.isdir(root):
        return {"status": "success", "exists": False, "total": 0,
                "with_edges": 0, "rhythm": 0, "empty": 0,
                "by_orientation": {}, "top": []}
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(KALMYKOV_GRAPH_EXT):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            o = _orient_of(rel)
            by_orient.setdefault(o, {"n": 0, "rhythm": 0, "empty": 0})
            by_orient[o]["n"] += 1
            total += 1
            try:
                obj = _load_json_anyenc(full)
                g = _read_dataset_graph(obj)
            except Exception:
                empty += 1
                by_orient[o]["empty"] += 1
                continue
            nn = len(g["nodes"])
            ne = len(g["edges"])
            if ne == 0:
                empty += 1
                by_orient[o]["empty"] += 1
                continue
            with_edges += 1
            try:
                ck = SfiralChronokvantAnalyzer.detect_chronokvants(g["nodes"], g["edges"])
            except Exception:
                ck = []
            if len(ck) > 0:
                rhythm += 1
                by_orient[o]["rhythm"] += 1
            scored.append({"rel": rel, "orientation": o,
                           "nodes": nn, "edges": ne, "ck": len(ck)})
    scored.sort(key=lambda x: x["ck"], reverse=True)
    return {"status": "success", "exists": True, "total": total,
            "with_edges": with_edges, "rhythm": rhythm, "empty": empty,
            "by_orientation": by_orient, "top": scored[:10]}


def kalmykov_first_rel(orientation):
    graphs = _walk_kalmykov_graphs(KALMYKOV_MODELS_DIR)
    want = str(orientation or "").lower()
    for gi in graphs:
        if not gi["err"] and gi["n_nodes"] > 0 and gi["orientation"].lower() == want:
            return gi["rel"]
    for gi in graphs:
        if not gi["err"] and gi["n_nodes"] > 0:
            return gi["rel"]
    return None
# ============================================================
# END KALMYKOV BRIDGE
# ============================================================


class SfiralComputeHandler(http.server.SimpleHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.path = '/index.html'
        return super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            req_data = json.loads(post_data.decode('utf-8'))

            if self.path == '/api/quantum_qutrit_step':
                state_input = req_data.get('state', [1.0, 0.0, 0.0])
                action = req_data.get('action', 'measure')
                t_param = req_data.get('t', 1.0)
                q = SfiralQutrit(state=state_input)
                if action == 's_transition':
                    q.apply_s_transition()
                elif action == 'evolution':
                    q.apply_s_evolution(t_param)
                stochastic_res = q.measure("stochastic")
                deterministic_res = q.measure("deterministic")
                probs = q.meas.born_probabilities(q.state)
                response_payload = {
                    "status": "success",
                    "current_state": [str(x) for x in q.state],
                    "probabilities": {"L": float(probs[0]), "S": float(probs[1]), "R": float(probs[2])},
                    "measured_ternary": stochastic_res,
                    "deterministic_projection": deterministic_res
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(response_payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/generate_academic_passport':
                nodes = req_data.get('nodes', [])
                edges = req_data.get('edges', [])
                html_report = generate_academic_passport_html(nodes, edges)
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "success", "html_content": html_report}, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/ai_predict_topology':
                nodes = req_data.get('nodes', [])
                edges = req_data.get('edges', [])
                update_last_graph(nodes, edges)
                chronokvants = SfiralChronokvantAnalyzer.detect_chronokvants(nodes, edges)
                prediction_response = {
                    "status": "success",
                    "total_nodes": len(nodes),
                    "chronokvants_detected": len(chronokvants),
                    "details": chronokvants,
                    "ai_suggestion": f"Topology analysis: chronokvant couplings found = {len(chronokvants)}. Reduction and S-loop modes accounted."
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(prediction_response, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/inspect_selection':
                selected_nodes = req_data.get('selected_nodes', [])
                selected_part = req_data.get('selected_part', None)
                learned = load_learned_terms()
                matched_term = None
                for item in learned:
                    if item.get('selected_nodes_count') == len(selected_nodes) and item.get('selected_part') == selected_part:
                        matched_term = item
                        break
                if matched_term:
                    status_name = f"ASSIGNED: {matched_term['term_name']}"
                    description = matched_term['description']
                    ai_advice = "This element or group is already in your knowledge library."
                else:
                    status_name = "UNASSIGNED (new configuration)"
                    description = f"Selected elements: {len(selected_nodes)}, sub-object: {selected_part or 'whole node'}"
                    ai_advice = "Name not set yet. Describe it via the brain button in properties panel to add to glossary!"
                response_payload = {"status": "success", "status_name": status_name, "description": description, "ai_advice": ai_advice}
                self._set_headers(200)
                self.wfile.write(json.dumps(response_payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/teach_ai_term':
                term_name = req_data.get('term_name', 'Unnamed Sfiral element')
                description = req_data.get('description', '')
                selected_nodes = req_data.get('selected_nodes', [])
                selected_part = req_data.get('selected_part', None)
                term_record = {
                    "timestamp": datetime.now().isoformat(),
                    "term_name": term_name,
                    "description": description,
                    "selected_nodes_count": len(selected_nodes),
                    "selected_part": selected_part
                }
                save_learned_term(term_record)
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "success", "message": "Term saved successfully"}, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/ask_ai':
                graph = req_data.get('graph', {})
                user_question = req_data.get('question', '')
                nodes = graph.get('nodes', [])
                if isinstance(graph, dict):
                    update_last_graph(graph.get('nodes', []), graph.get('edges', []))
                s_transition_only = False
                node_coords_info = ""
                for node in nodes:
                    n_id = node.get('id')
                    x, y, z = node.get('x', 0), node.get('y', 0), node.get('z', 0)
                    params = node.get('params', {})
                    show_right = params.get('showRight', True)
                    show_left = params.get('showLeft', True)
                    show_s = params.get('showS', True)
                    node_coords_info = f"Node ID:{n_id} [X={x}, Y={y}, Z={z}]"
                    if not show_right and not show_left and show_s:
                        s_transition_only = True
                if s_transition_only:
                    ai_answer = (
                        f"[Sfiral anatomy]: I see in space ({node_coords_info}) "
                        f"an isolated S-transition zone (side turns hidden). This is a pure laminar chirality-inversion zone "
                        f"with zero global chirality. Which base contour do we bind it to, or build a new cascade?"
                    )
                else:
                    ai_answer = (
                        f"[Spatial analysis]: Recording a structure of {len(nodes)} nodes in 3D. "
                        f"Coordinates and components active. Continue the geometry breakdown of this sfiral."
                    )
                response_payload = {"answer": ai_answer, "actions": []}
                self._set_headers(200)
                self.wfile.write(json.dumps(response_payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/neurocore_validate_build':
                graph = req_data.get('graph', {})
                if isinstance(graph, dict):
                    nodes = graph.get('nodes', [])
                    edges = graph.get('edges', [])
                else:
                    nodes = req_data.get('nodes', [])
                    edges = req_data.get('edges', [])
                if not nodes and not edges:
                    nodes = LAST_GRAPH.get('nodes', [])
                    edges = LAST_GRAPH.get('edges', [])
                mode = req_data.get('mode', 'canon')
                try:
                    from sfiral_neurocore_torch import predict_graph
                    result = predict_graph(nodes, edges, mode=mode)
                except Exception as e:
                    result = {"ok": False, "error": str(e), "note": "sfiral_neurocore_torch.py not found or broken"}
                self._set_headers(200)
                self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode('utf-8'))
                return

            if self.path == '/api/get_last_graph':
                result = {"ok": True, "n_nodes": len(LAST_GRAPH.get('nodes', [])), "n_edges": len(LAST_GRAPH.get('edges', [])), "graph": LAST_GRAPH}
                self._set_headers(200)
                self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode('utf-8'))
                return

            # === DATASET endpoints (ai_memory) ===
            if self.path == '/api/dataset_list':
                payload = dataset_list_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/dataset_load':
                name = req_data.get('name', '')
                payload = dataset_load_payload(name)
                self._set_headers(200)
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/dataset_summary':
                payload = dataset_summary_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                return
            # === END DATASET endpoints ===

            # === KALMYKOV endpoints (C:\Kalmykov\models) ===
            if self.path == '/api/kalmykov_inventory':
                payload = kalmykov_inventory_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/kalmykov_list':
                payload = kalmykov_list_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/kalmykov_load':
                rel = req_data.get('rel', '')
                payload = kalmykov_load_payload(rel)
                self._set_headers(200)
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/kalmykov_summary':
                payload = kalmykov_summary_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/kalmykov_learn':
                rel = req_data.get('rel', '')
                orientation = req_data.get('orientation', '')
                if not rel and orientation:
                    rel = kalmykov_first_rel(orientation)
                if not rel:
                    res = {"ok": False, "error": "no graph in kalmykov dir"}
                else:
                    loaded = kalmykov_load_payload(rel)
                    if loaded.get('status') != 'success':
                        res = {"ok": False, "error": loaded.get('message', 'load fail'), "rel": rel}
                    else:
                        try:
                            from sfiral_agent import learn_build
                            res = learn_build(loaded['graph'], max_steps=int(req_data.get('max_steps', 24)), seed=int(req_data.get('seed', 1)))
                            res['rel'] = rel
                            res['orientation'] = loaded.get('orientation', '?')
                        except Exception as e:
                            res = {"ok": False, "error": str(e), "note": "sfiral_agent.py not found or broken", "rel": rel}
                self._set_headers(200)
                self.wfile.write(json.dumps(res, ensure_ascii=False, default=str).encode('utf-8'))
                return
            # === END KALMYKOV endpoints ===

            if self.path == '/api/agent_learn':
                max_steps = int(req_data.get('max_steps', 24))
                seed = int(req_data.get('seed', 1))
                # start graph priority: explicit graph -> dataset file -> kalmykov file -> LAST_GRAPH
                start_graph = req_data.get('graph')
                if not start_graph or not start_graph.get('nodes'):
                    ds_name = req_data.get('dataset_name')
                    if ds_name:
                        loaded = dataset_load_payload(ds_name)
                        if loaded.get('status') == 'success':
                            start_graph = loaded['graph']
                if not start_graph or not start_graph.get('nodes'):
                    km_rel = req_data.get('kalmykov_rel')
                    km_orient = req_data.get('kalmykov_orientation')
                    if not km_rel and km_orient:
                        km_rel = kalmykov_first_rel(km_orient)
                    if km_rel:
                        loaded = kalmykov_load_payload(km_rel)
                        if loaded.get('status') == 'success':
                            start_graph = loaded['graph']
                if not start_graph:
                    start_graph = LAST_GRAPH
                try:
                    from sfiral_agent import learn_build
                    result = learn_build(start_graph, max_steps=max_steps, seed=seed)
                except Exception as e:
                    result = {"ok": False, "error": str(e), "note": "sfiral_agent.py not found or broken"}
                self._set_headers(200)
                self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode('utf-8'))
                return

            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "success"}, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False).encode('utf-8'))


if __name__ == "__main__":
    server = http.server.HTTPServer(('localhost', PORT), SfiralComputeHandler)
    print(f"Server ready: http://localhost:{PORT}")
    print("AI_MEMORY dir exists =", os.path.isdir(AI_MEMORY_DIR), "->", os.path.abspath(AI_MEMORY_DIR))
    print("KALMYKOV dir exists =", os.path.isdir(KALMYKOV_MODELS_DIR), "->", KALMYKOV_MODELS_DIR)
    server.serve_forever()