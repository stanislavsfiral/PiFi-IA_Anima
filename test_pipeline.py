import sys
import io
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from qutrit import SfiralQutrit

# 1. Создаем кутрит в левом витке (|L⟩, соответствует -1)
q = SfiralQutrit()
print("Начальное состояние (|L⟩):", q.state)
print("Измерение на входе:", q.measure("stochastic"))

# 2. Пропускаем через S-переход U_S(1) (инверсия хиральности в |R⟩)
q.apply_s_transition()
print("Состояние после S-перехода (должно уйти в |R⟩):", q.state)
print("Измерение после перехода:", q.measure("stochastic"))

# 3. Проверяем непрерывную эволюцию на середине дуги (t = 0.5)
q2 = SfiralQutrit() # Снова старт из |L⟩
q2.apply_s_evolution(0.5)
print("Состояние на середине дуги (t=0.5):", q2.state)
print("Вероятности Борна на середине:", q2.meas.born_probabilities(q2.state))