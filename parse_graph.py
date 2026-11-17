import json
import os
import numpy as np

def analyze_sfiral_graph(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден.")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Поддерживаем разную структуру файлов (прямой граф или обертка сессии)
    nodes = data.get('nodes', data.get('graph', {}).get('nodes', []))
    edges = data.get('edges', data.get('graph', {}).get('edges', []))
    
    print(f"📁 Анализ файла: {file_path}")
    print(f"🔹 Всего узлов (Сфиралей): {len(nodes)}")
    print(f"🔹 Всего связей (Хроноквантов): {len(edges)}")
    
    node_coords = {}
    node_features = []
    
    for node in nodes:
        nid = node.get('id')
        x = node.get('x', 0)
        y = node.get('y', 0)
        z = node.get('z', 0)
        params = node.get('params', {})
        angles = params.get('angles', [0, 0, 0])
        scale = params.get('scale', 1.0)
        stretch = params.get('stretch', 1.0)
        
        node_coords[nid] = np.array([x, y, z])
        
        # Формируем вектор признаков узла для нейросети:
        # [X, Y, Z, RotX, RotY, RotZ, Scale, Stretch]
        features = [x, y, z, angles[0], angles[1], angles[2], scale, stretch]
        node_features.append(features)
        
    # Рассчитываем пространственные расстояния для рёбер (геометрическая связь весов)
    edge_distances = []
    for edge in edges:
        f_id = edge.get('from')
        t_id = edge.get('to')
        if f_id in node_coords and t_id in node_coords:
            dist = np.linalg.norm(node_coords[f_id] - node_coords[t_id])
            edge_distances.append(dist)
            
    if edge_distances:
        print(f"📏 Среднее расстояние между узлами по рёбрам: {np.mean(edge_distances):.2f}")
        print(f"📏 Мин/Макс расстояние: {np.min(edge_distances):.2f} / {np.max(edge_distances):.2f}")
        
    return np.array(node_features), edges

if __name__ == "__main__":
    # Указываем путь к вашему файлу куба
    target_file = os.path.join("ai_memory", "Граф", "куб.json")
    analyze_sfiral_graph(target_file)