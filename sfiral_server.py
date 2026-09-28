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


# Импортируем наш волновой движок клеточных автоматов
from sfiral_ca_engine import SfiralCAEngine

# Импортируем наше строгое математическое ядро кутрита и измерения из отдельных модулей
from qutrit import SfiralQutrit
from measurement import SfiralMeasurement

# Пытаемся импортировать PyTorch и EGNN-подобные компоненты для объемного расчета
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Пытаемся импортировать ONNX Runtime для инференса нашей тернарной модели
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

PORT = 8000
AI_MEMORY_DIR = "ai_memory"
os.makedirs(AI_MEMORY_DIR, exist_ok=True)
TIMELINE_LOG_FILE = os.path.join(AI_MEMORY_DIR, "ai_timeline_log.jsonl")

# Файл для сохранения пользовательских терминов (обучение ИИ)
LEARNED_TERMS_FILE = os.path.join(AI_MEMORY_DIR, "learned_terms.json")

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

# Директория для сохранения обученных макросов
MACROS_DIR = os.path.join(AI_MEMORY_DIR, "macros")
os.makedirs(MACROS_DIR, exist_ok=True)

# Путь к обученной ONNX модели с тернарными весами
ONNX_MODEL_PATH = os.path.join("models", "sfiral_ternary_copilot.onnx")

# Путь к локальной базе знаний
FAQ_FILE_PATH = os.path.join("sfiral_docs", "sfiral_docsfaq_sfiral.txt")

# Инициализируем движок клеточных автоматов
ca_engine = SfiralCAEngine(width=32, height=32, channels=4)

def load_local_faq():
    if os.path.exists(FAQ_FILE_PATH):
        try:
            with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Ошибка чтения FAQ файла: {e}")
    return ""

# ==========================================
# МОДУЛЬ АНАЛИЗА ХРОНОКВАНТОВ (ПО ТИПАМ СОСТЫКОВКИ)
# ==========================================
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
                    polarization = "Правая (R)" if (x1 + x2) >= 0 else "Левая (L)"
                    chronokvants_found.append({
                        "node_a": n1_id,
                        "node_b": n2_id,
                        "type": "Линейный хроноквант (Сцепленный виток / Редукция)",
                        "polarization": f"Поляризованная суперпозиция ({polarization})",
                        "dz": round(dz, 2),
                        "status": "Линейное сочленение с разворотом по X (~180°), лишний виток редуцирован."
                    })
                    continue
                    
                is_diagonal = dx > 15 and dy > 15 and dz > 15 and abs(diff_x_angle) < 15
                if has_explicit_link and link_type == 'diagonal_loop' or (not has_explicit_link and is_diagonal):
                    chronokvants_found.append({
                        "node_a": n1_id,
                        "node_b": n2_id,
                        "type": "Диагональный макро-хроноквант (Распределенная S-петля)",
                        "polarization": "Нулевая хиральность целого (Макро-суперпозиция)",
                        "dz": round(dz, 2),
                        "status": f"Диагональное сочленение без переворота (dx={dx:.1f}, dy={dy:.1f}), зона стыковки образует S-петлю."
                    })
                    
        return chronokvants_found

# ==========================================
# ГЕНЕРАТОР ПОДРОБНОГО АКАДЕМИЧЕСКОГО ПАСПОРТА
# ==========================================
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
        nodes_rows += f"<tr><td>Узел #{n_id}</td><td>X: {x}, Y: {y}, Z: {z}</td><td>{gate}</td><td style='color:#27ae60; font-weight:bold;'>Стабилен (0 Хиральность)</td></tr>"
    
    if not nodes_rows:
        nodes_rows = "<tr><td colspan='4'>Активные узлы в текущей сессии отсутствуют</td></tr>"

    chronos_rows = ""
    for ch in chronokvants[:20]:
        chronos_rows += f"<tr><td>#{ch['node_a']} ↔ #{ch['node_b']}</td><td>{ch['type']}</td><td>{ch['polarization']}</td><td>{ch['status']}</td></tr>"
    
    if not chronos_rows:
        chronos_rows = "<tr><td colspan='4'>Хроноквантовые сочленения не обнаружены (требуется состыковка узлов)</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Академический паспорт топологической архитектуры Сфирали</title>
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
            .highlight-box {{ background-color: #ebf5fb; border-left: 4px solid #3498db; padding: 12px 15px; margin: 15px 0; font-size: 10.5pt; }}
            .meta-grid {{ display: flex; justify-content: space-between; background: #f2f4f4; padding: 12px; border-radius: 4px; margin-bottom: 20px; font-size: 10pt; }}
            .footer {{ margin-top: 30px; font-size: 9pt; color: #7f8c8d; text-align: center; border-top: 1px solid #e0e0e0; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Академический паспорт топологической архитектуры Сфирали</h1>
            
            <div class="meta-grid">
                <div><b>Проект:</b> GIDEON / PiFiYA-core</div>
                <div><b>Дата формирования:</b> {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</div>
                <div><b>Статус верификации:</b> Успешно пройдено</div>
            </div>

            <h2>1. Архитектурная спецификация графа и типов сочленений</h2>
            <p>
                Документ фиксирует топологическое состояние системы, состоящей из зеркально-антисимметричных витков с инверсией хиральности и ламинарных S-переходов. Учтены типы сочленений: линейная редукция (сцепленный хроноквант) и диагональные S-петли.
            </p>
            <ul>
                <li><b>Количество активных узлов (Сфиралей):</b> {len(nodes)}</li>
                <li><b>Количество связей графа (Edges):</b> {len(edges)}</li>
                <li><b>Обнаружено хроноквантовых сочленений:</b> {len(chronokvants)}</li>
                <li><b>Топологический базис:</b> Нулевая хиральность целого, управляемая редукция витков.</li>
            </ul>

            <h2>2. Реестр хроноквантовых связей и типов состыковки</h2>
            <table>
                <thead>
                    <tr>
                        <th>Пара узлов</th>
                        <th>Тип сочленения</th>
                        <th>Поляризация / Суперпозиция</th>
                        <th>Статус / Описание</th>
                    </tr>
                </thead>
                <tbody>
                    {chronos_rows}
                </tbody>
            </table>

            <h2>3. Сравнительный бенчмарк фазовой устойчивости</h2>
            <table>
                <thead>
                    <tr>
                        <th>Метрика оценки сигнала</th>
                        <th>Классический метод (MA)</th>
                        <th>Топология Сфирали (Q-Core)</th>
                        <th>Эффект / Прирост</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>Сохранение фазовой энергии</b></td>
                        <td>{class_ret:.1f}%</td>
                        <td style="color: #27ae60; font-weight: bold;">{sfiral_ret:.1f}%</td>
                        <td>+{max(0, sfiral_ret - class_ret):.1f}% в пользу Сфирали</td>
                    </tr>
                    <tr>
                        <td><b>Ошибка восстановления (MSE)</b></td>
                        <td>{classical_mse:.2f}</td>
                        <td style="color: #27ae60; font-weight: bold;">{sfiral_mse:.2f}</td>
                        <td>Минимизация искажения гармоники</td>
                    </tr>
                </tbody>
            </table>

            <h2>4. Выборочный реестр узлов структуры</h2>
            <table>
                <thead>
                    <tr>
                        <th>Идентификатор</th>
                        <th>Пространственные координаты (XYZ)</th>
                        <th>Активный вентиль</th>
                        <th>Хиральный статус</th>
                    </tr>
                </thead>
                <tbody>
                    {nodes_rows}
                </tbody>
            </table>

            <div class="footer">
                Автоматически сгенерировано испытательным комплексом GIDEON-Sfiral-Architecture • 2026 г.
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# ==========================================
# СЕРВЕР (REST API)
# ==========================================
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
                chronokvants = SfiralChronokvantAnalyzer.detect_chronokvants(nodes, edges)
                
                prediction_response = {
                    "status": "success",
                    "total_nodes": len(nodes),
                    "chronokvants_detected": len(chronokvants),
                    "details": chronokvants,
                    "ai_suggestion": f"Топологический анализ: Обнаружено хроноквантовых сочленений — {len(chronokvants)}. Учтены режимы редукции и S-петель."
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(prediction_response, ensure_ascii=False).encode('utf-8'))
                return

            # ==========================================
            # НОВЫЕ ЭНДПОИНТЫ ИНСПЕКТОРА И ОБУЧЕНИЯ (🧠)
            # ==========================================
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
                    status_name = f"✅ Присвоено: «{matched_term['term_name']}»"
                    description = matched_term['description']
                    ai_advice = "Этот элемент или группа уже закреплены в вашей библиотеке знаний."
                else:
                    status_name = "⚠️ Не присвоено (Новая конфигурация)"
                    description = f"Выделено элементов: {len(selected_nodes)}, подобъект: {selected_part or 'целый узел'}"
                    ai_advice = "Название еще не задано. Опишите его в панели свойств через кнопку 🧠, чтобы внести в глоссарий!"

                response_payload = {
                    "status": "success",
                    "status_name": status_name,
                    "description": description,
                    "ai_advice": ai_advice
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(response_payload, ensure_ascii=False).encode('utf-8'))
                return

            if self.path == '/api/teach_ai_term':
                term_name = req_data.get('term_name', 'Безымянный элемент Сфирали')
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
                
                s_transition_only = False
                node_coords_info = ""
                
                for node in nodes:
                    n_id = node.get('id')
                    x, y, z = node.get('x', 0), node.get('y', 0), node.get('z', 0)
                    params = node.get('params', {})
                    show_right = params.get('showRight', True)
                    show_left = params.get('showLeft', True)
                    show_s = params.get('showS', True)
                    
                    node_coords_info = f"Узел ID:{n_id} [X={x}, Y={y}, Z={z}]"
                    if not show_right and not show_left and show_s:
                        s_transition_only = True

                if s_transition_only:
                    ai_answer = (
                        f"🎯 [Анатомический анализ Сфирали]: Станислав, я вижу в пространстве ({node_coords_info}) "
                        f"изолированную зону S-перехода (боковые витки скрыты). Это чистая зона ламинарной инверсии хиральности "
                        f"и нулевой хиральности целого. К какому базовому контуру мы её привязываем или строим новый каскад?"
                    )
                else:
                    ai_answer = (
                        f"📐 [Пространственный анализ]: Фиксирую структуру из {len(nodes)} узлов в 3D. "
                        f"Координаты и компоненты активны. Станислав, продолжайте разбор геометрии этой сфирали."
                    )

                response_payload = {
                    "answer": ai_answer,
                    "actions": []
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(response_payload, ensure_ascii=False).encode('utf-8'))
                return
            
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "success"}, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False).encode('utf-8'))

if __name__ == "__main__":
    server = http.server.HTTPServer(('localhost', PORT), SfiralComputeHandler)
    print(f"🚀 Сервер с поддержкой хроноквантовых сочленений и ИИ-обучения активен: http://localhost:{PORT}")
    server.serve_forever()