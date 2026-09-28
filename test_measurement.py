import sys
import io
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from measurement import SfiralMeasurement

# Инициализируем наш модуль измерения
meas = SfiralMeasurement()

# Пример квантового состояния кутрита |ψ⟩ = α_L|L⟩ + α_S|S⟩ + α_R|R⟩
# Например, упор на левый виток L (-1)
psi_left = np.array([0.9, 0.1, 0.0], dtype=complex)
# Нормируем вектор (сумма квадратов модулей должна быть равна 1)
psi_left = psi_left / np.linalg.norm(psi_left)

print("Вероятности Борна для состояния с уклоном в L:", meas.born_probabilities(psi_left))

# Проверяем стохастическое измерение (коллапс)
results = [meas.measure_stochastic(psi_left) for _ in range(10)]
print("Результаты 10 измерений коллапса:", results)

# Проверяем детерминированную пороговую проекцию
proj = meas.project_deterministic(psi_left)
print("Детерминированная проекция в {-1, 0, +1}:", proj)