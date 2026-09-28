#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GIDEON Hierarchy Processor
Автономный процессор для иерархических квантовых структур на основе геометрии сфиралей.
Распознаёт диполи, гироскопы и кубы, выполняет квантовые вычисления и удаляет лишние витки на стыках.
"""

import json
import numpy as np
from collections import defaultdict
from itertools import combinations

# ============================================================
# 1. Базовые квантовые вычисления (как в sfiral_server.py)
# ============================================================

def compute_node_state(node, phi_ratio=0.5):
    """
    Вычисляет квантовое состояние для одного узла на основе его геометрии.
    Возвращает словарь с полями: psi_real, psi_imag, intensity, signal, superposition_weight.
    """
    params = node.get('params', {})
    show_right = params.get('showRight', True)
    show_left = params.get('showLeft', True)
    angles = params.get('angles', [0, 0, 0])
    x, y, z = node.get('x', 0), node.get('y', 0), node.get('z', 0)
    gate_type = params.get('activeGate', 'H')
    N = params.get('N', 5)
    scale = params.get('scale', 1.0)
    
    # Радиус основного витка и S-витка (пропорция phi)
    main_radius = (60 + N * 2) * scale
    s_radius = main_radius * phi_ratio   # по умолчанию 0.5, можно заменить на 1/phi
    superposition_weight = s_radius / main_radius  # 0.5 или 1/phi
    
    # Определяем сигнал (для метрик)
    if show_right and not show_left:
        signal = 1.0
    elif show_left and not show_right:
        signal = -1.0
    else:
        signal = 0.0   # суперпозиция
    
    # Пространственная фаза
    phi_spatial = np.radians(angles[1] + angles[2]) + (np.sqrt(x**2 + y**2 + z**2) / 1000.0)
    if gate_type == 'S_TRANSITION':
        phi_spatial += superposition_weight * np.pi   # дополнительный сдвиг для S-перехода
    
    # Матрица Адамара
    H_matrix = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    state_vector = np.array([1.0, 0.0], dtype=complex)
    phase_tensor = np.exp(1j * phi_spatial)
    transformed = np.dot(H_matrix, state_vector) * phase_tensor
    
    psi_real = float(np.real(transformed[0]))
    psi_imag = float(np.imag(transformed[1]))
    intensity = float(np.abs(transformed[0])**2 + np.abs(transformed[1])**2)
    
    return {
        'psi_real': round(psi_real, 4),
        'psi_imag': round(psi_imag, 4),
        'intensity': round(intensity, 4),
        'signal': signal,
        'superposition_weight': round(superposition_weight, 4)
    }

# ============================================================
# 2. Распознавание иерархических структур
# ============================================================

def find_dipoles(nodes, tol=1e-6):
    """
    Находит диполи: пары узлов с одинаковыми координатами (в пределах tol),
    повёрнутые друг относительно друга по одной из осей X, Y или Z.
    Возвращает список диполей, каждый содержит center, axis, node1, node2.
    """
    # Группировка по координатам
    groups = defaultdict(list)
    for node in nodes:
        key = (round(node['x']/tol)*tol, round(node['y']/tol)*tol, round(node['z']/tol)*tol)
        groups[key].append(node)
    
    dipoles = []
    for center, group in groups.items():
        if len(group) < 2:
            continue
        # Проверяем все пары в группе
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                n1, n2 = group[i], group[j]
                a1 = n1['params'].get('angles', [0,0,0])
                a2 = n2['params'].get('angles', [0,0,0])
                diff = [(a2[k] - a1[k]) % 360 for k in range(3)]
                # Определяем ось, по которой разница кратна 90°
                axis = None
                for idx, d in enumerate(diff):
                    if abs(d - 90) < 5 or abs(d - 180) < 5 or abs(d - 270) < 5:
                        axis = ['X','Y','Z'][idx]
                        break
                if axis is not None:
                    dipoles.append({
                        'center': center,
                        'axis': axis,
                        'node1': n1,
                        'node2': n2
                    })
    return dipoles

def find_gyroscopes(dipoles, tol=1e-6):
    """
    Группирует диполи по центру и проверяет наличие всех трёх осей (X, Y, Z).
    Возвращает список гироскопов.
    """
    center_map = defaultdict(list)
    for d in dipoles:
        key = (round(d['center'][0]/tol)*tol, round(d['center'][1]/tol)*tol, round(d['center'][2]/tol)*tol)
        center_map[key].append(d)
    
    gyros = []
    for center, ds in center_map.items():
        axes = {d['axis'] for d in ds}
        if axes == {'X','Y','Z'}:
            # Берём по одному диполю на ось (можно выбрать любой)
            d_x = next(d for d in ds if d['axis']=='X')
            d_y = next(d for d in ds if d['axis']=='Y')
            d_z = next(d for d in ds if d['axis']=='Z')
            gyros.append({
                'center': center,
                'dipole_X': d_x,
                'dipole_Y': d_y,
                'dipole_Z': d_z
            })
    return gyros

def build_cubes(gyros, spacing=200, tol=1e-6):
    """
    Строит кубы из гироскопов, предполагая, что они находятся в вершинах кубической решётки
    с шагом spacing. Возвращает список кубов, каждый содержит список из 8 гироскопов
    и рёбра (связи между соседними гироскопами).
    """
    # Группируем гироскопы по координатам, приведённым к целым индексам
    grid = defaultdict(list)
    for g in gyros:
        cx, cy, cz = g['center']
        ix = int(round(cx / spacing))
        iy = int(round(cy / spacing))
        iz = int(round(cz / spacing))
        grid[(ix, iy, iz)].append(g)   # обычно один гироскоп на вершину, но может быть несколько
    
    cubes = []
    # Ищем комбинации из 8 вершин, образующих куб
    all_indices = list(grid.keys())
    for idx in all_indices:
        ix, iy, iz = idx
        # Рассматриваем все возможные смещения (dx,dy,dz) из {-1,0,1}
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    # Вершины куба: (ix, iy, iz), (ix+dx, iy, iz), (ix, iy+dy, iz), ...
                    corners = []
                    for vx in [ix, ix+dx]:
                        for vy in [iy, iy+dy]:
                            for vz in [iz, iz+dz]:
                                if (vx, vy, vz) in grid:
                                    # Берём первый гироскоп из группы (должен быть один)
                                    corners.append(grid[(vx, vy, vz)][0])
                    if len(corners) == 8:
                        # Проверяем, что это действительно куб (все рёбра равны spacing)
                        # Упрощённо: просто добавляем, если все вершины есть.
                        # Можно добавить проверку расстояний между центрами.
                        cubes.append({
                            'vertices': corners,
                            'indices': [(ix,iy,iz), (ix+dx,iy,iz), (ix,iy+dy,iz), (ix+dx,iy+dy,iz),
                                        (ix,iy,iz+dz), (ix+dx,iy,iz+dz), (ix,iy+dy,iz+dz), (ix+dx,iy+dy,iz+dz)]
                        })
    return cubes

# ============================================================
# 3. Удаление лишних витков при стыковке
# ============================================================

def remove_contact_wires_for_cube(cube):
    """
    Для данного куба (список из 8 гироскопов) удаляет лишние витки на гранях.
    Для каждой пары соседних гироскопов по оси X, Y или Z отключает соответствующие порты.
    """
    vertices = cube['vertices']
    # Создаём словарь: индекс (ix,iy,iz) -> гироскоп
    gyro_map = {}
    for g, idx in zip(vertices, cube['indices']):
        gyro_map[idx] = g
    
    # Проверяем все пары вершин, отличающиеся на 1 по одной оси
    for idx1, g1 in gyro_map.items():
        ix, iy, iz = idx1
        # Соседи по оси X
        for dx in [-1, 1]:
            idx2 = (ix+dx, iy, iz)
            if idx2 in gyro_map:
                g2 = gyro_map[idx2]
                # Отключаем витки на стыке по X
                # У левого гироскопа (меньший x) отключаем правые витки
                if idx1[0] < idx2[0]:
                    set_dipole_wires(g1, side='right', enable=False)
                    set_dipole_wires(g2, side='left', enable=False)
                else:
                    set_dipole_wires(g1, side='left', enable=False)
                    set_dipole_wires(g2, side='right', enable=False)
        # Соседи по оси Y
        for dy in [-1, 1]:
            idx2 = (ix, iy+dy, iz)
            if idx2 in gyro_map:
                g2 = gyro_map[idx2]
                if idx1[1] < idx2[1]:
                    set_dipole_wires(g1, side='front', enable=False)   # условно front = +Y
                    set_dipole_wires(g2, side='back', enable=False)
                else:
                    set_dipole_wires(g1, side='back', enable=False)
                    set_dipole_wires(g2, side='front', enable=False)
        # Соседи по оси Z
        for dz in [-1, 1]:
            idx2 = (ix, iy, iz+dz)
            if idx2 in gyro_map:
                g2 = gyro_map[idx2]
                if idx1[2] < idx2[2]:
                    set_dipole_wires(g1, side='top', enable=False)    # условно top = +Z
                    set_dipole_wires(g2, side='bottom', enable=False)
                else:
                    set_dipole_wires(g1, side='bottom', enable=False)
                    set_dipole_wires(g2, side='top', enable=False)

def set_dipole_wires(gyro, side, enable):
    """
    Устанавливает флаги showRight/showLeft (и возможно showS) для узлов в диполях гироскопа
    в зависимости от стороны.
    side: 'right', 'left', 'front', 'back', 'top', 'bottom'
    enable: True/False
    """
    # Определяем, какие флаги менять в зависимости от стороны
    # Для простоты будем использовать showRight и showLeft, но нужно учитывать оси диполей.
    # В реальности потребуется более детальная логика, но здесь упрощённо:
    # Для диполей, ориентированных по X, right/left соответствуют +X/-X
    # Для диполей по Y — аналогично, но с другими осями.
    # Для демонстрации отключаем showRight или showLeft у всех узлов диполя.
    dipoles = [gyro['dipole_X'], gyro['dipole_Y'], gyro['dipole_Z']]
    for d in dipoles:
        for node in [d['node1'], d['node2']]:
            if side in ['right', 'front', 'top']:
                node['params']['showRight'] = enable
            elif side in ['left', 'back', 'bottom']:
                node['params']['showLeft'] = enable
            # Дополнительно можно управлять showS и showSLeft

# ============================================================
# 4. Основной процессор
# ============================================================

def process_gideon_model(model_data, phi_ratio=0.5, detect_hierarchy=True):
    """
    Основная функция:
    - Принимает словарь с полями 'nodes' и 'edges'.
    - Вычисляет квантовые состояния для каждого узла.
    - Если detect_hierarchy=True, распознаёт диполи, гироскопы, кубы,
      удаляет лишние витки и добавляет метаданные в результат.
    - Возвращает обновлённую модель с заполненными quantumState и дополнительной информацией.
    """
    nodes = model_data.get('nodes', [])
    edges = model_data.get('edges', [])
    
    # 1. Вычисляем состояния для всех узлов
    for node in nodes:
        state = compute_node_state(node, phi_ratio)
        node['quantumState'] = {
            'intensity': state['intensity'],
            'psi_real': state['psi_real'],
            'psi_imag': state['psi_imag']
        }
        # Сохраняем сигнал и вес суперпозиции в дополнительные поля (опционально)
        node['_signal'] = state['signal']
        node['_superposition_weight'] = state['superposition_weight']
    
    result = {
        'nodes': nodes,
        'edges': edges,
        'computed': True,
        'hierarchy': {}
    }
    
    if detect_hierarchy and len(nodes) > 0:
        # 2. Распознаём диполи
        dipoles = find_dipoles(nodes)
        result['hierarchy']['dipoles'] = dipoles
        
        # 3. Распознаём гироскопы
        gyroscopes = find_gyroscopes(dipoles)
        result['hierarchy']['gyroscopes'] = gyroscopes
        
        # 4. Распознаём кубы и удаляем лишние витки
        if gyroscopes:
            cubes = build_cubes(gyroscopes)
            result['hierarchy']['cubes'] = cubes
            for cube in cubes:
                remove_contact_wires_for_cube(cube)
            # После удаления витков пересчитываем состояния узлов (т.к. изменились флаги)
            for node in nodes:
                state = compute_node_state(node, phi_ratio)
                node['quantumState'] = {
                    'intensity': state['intensity'],
                    'psi_real': state['psi_real'],
                    'psi_imag': state['psi_imag']
                }
    
    return result

# ============================================================
# 5. Пример использования и утилиты командной строки
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='GIDEON Hierarchy Processor')
    parser.add_argument('input_file', help='Путь к JSON-файлу модели')
    parser.add_argument('-o', '--output', help='Файл для сохранения результата (по умолчанию модель с состояниями)')
    parser.add_argument('--phi', type=float, default=0.5, help='Коэффициент масштаба S-витка (по умолчанию 0.5)')
    parser.add_argument('--no-hierarchy', action='store_true', help='Отключить распознавание иерархии')
    args = parser.parse_args()
    
    # Загрузка модели
    with open(args.input_file, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    # Обработка
    result = process_gideon_model(model, phi_ratio=args.phi, detect_hierarchy=not args.no_hierarchy)
    
    # Сохранение
    output_file = args.output if args.output else args.input_file.replace('.json', '_processed.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Обработка завершена. Результат сохранён в {output_file}")
    print(f"   Узлов: {len(result['nodes'])}")
    if not args.no_hierarchy and 'hierarchy' in result:
        print(f"   Диполей: {len(result['hierarchy'].get('dipoles', []))}")
        print(f"   Гироскопов: {len(result['hierarchy'].get('gyroscopes', []))}")
        print(f"   Кубов: {len(result['hierarchy'].get('cubes', []))}")