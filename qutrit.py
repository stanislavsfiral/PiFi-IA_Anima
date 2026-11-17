import numpy as np
from measurement import SfiralMeasurement

class SfiralQutrit:
    def __init__(self, state=None):
        # Базисные состояния в C³
        self.L = np.array([1.0, 0.0, 0.0], dtype=complex) # Хиральность -1
        self.S = np.array([0.0, 1.0, 0.0], dtype=complex) # Нейтраль S
        self.R = np.array([0.0, 0.0, 1.0], dtype=complex) # Хиральность +1
        
        # Если состояние не задано, инициализируем как |L>
        if state is not None:
            self.state = np.array(state, dtype=complex)
            self.normalize()
        else:
            self.state = self.L.copy()
            
        # Подключаем наш модуль измерения
        self.meas = SfiralMeasurement()

    def normalize(self):
        """Нормализация вектора состояния кутрита (сохранение нормы ||ψ|| = 1)"""
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm

    def apply_s_transition(self):
        """
        Применение канонического оператора S-перехода U_S(1):
        Инвертирует хиральность (L <-> R с учетом зеркального минуса) и сохраняет S.
        Матрица:
        [[ 0,  0, -1],
         [ 0,  1,  0],
         [ 1,  0,  0]]
        """
        U_S1 = np.array([
            [ 0.0,  0.0, -1.0],
            [ 0.0,  1.0,  0.0],
            [ 1.0,  0.0,  0.0]
        ], dtype=complex)
        
        self.state = U_S1 @ self.state
        self.normalize()

    def apply_s_evolution(self, t: float):
        """
        Непрерывная эволюция вдоль дуги U_S(t) для t ∈ [0, 1]
        """
        # Генератор H_S = (π/2) * i * (|R><L| - |L><R|)
        # Соответствует вращению в плоскости L-R
        theta = (np.pi / 2.0) * t
        c = np.cos(theta)
        s = np.sin(theta)
        
        U_St = np.array([
            [ c,  0.0, -s],
            [ 0.0, 1.0, 0.0],
            [ s,  0.0,  c]
        ], dtype=complex)
        
        self.state = U_St @ self.state
        self.normalize()

    def measure(self, mode="stochastic"):
        """
        Измерение и проекция кутрита в троичное пространство {-1, 0, +1}
        """
        if mode == "stochastic":
            return self.meas.measure_stochastic(self.state)
        else:
            return self.meas.project_deterministic(self.state)