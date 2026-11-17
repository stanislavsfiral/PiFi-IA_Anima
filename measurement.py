import numpy as np

class SfiralMeasurement:
    def __init__(self):
        # Спектр троичного пространства: L -> -1, S -> 0, R -> +1
        self.ternary_spectrum = np.array([-1.0, 0.0, 1.0])

    def born_probabilities(self, psi: np.ndarray) -> np.ndarray:
        """
        Вычисляет вероятности Борна для кутрита |ψ⟩ = [α_L, α_S, α_R].
        P = [|α_L|², |α_S|², |α_R|²]
        """
        return np.abs(psi) ** 2

    def measure_stochastic(self, psi: np.ndarray) -> int:
        """
        Квантовый коллапс: случайный выбор значения из {-1, 0, +1} 
        согласно распределению вероятностей Борна.
        """
        probs = self.born_probabilities(psi)
        # Убедимся, что вероятности нормированы
        probs /= np.sum(probs)
        # Случайный выбор индекса (0, 1, 2) с весами probs
        outcome_idx = np.random.choice([0, 1, 2], p=probs)
        # Возвращаем соответствующее тернарное метку {-1, 0, +1}
        return int(self.ternary_spectrum[outcome_idx])

    def project_deterministic(self, psi: np.ndarray, dead_zone_s: float = 0.33) -> int:
        """
        Детерминированная пороговая проекция (без коллапса):
        Оценивает среднее значение оператора троичного заряда 
        или возвращает дискретное состояние с учетом мертвой зоны для |S>.
        """
        probs = self.born_probabilities(psi)
        # Ожидаемое значение наблюдаемой <Q> = +1*P(R) - 1*P(L)
        expected_q = probs[2] - probs[0]
        
        # Пороговая классификация в {-1, 0, +1}
        if abs(probs[1]) > (1.0 - dead_zone_s) or abs(expected_q) < dead_zone_s:
            return 0  # Нейтраль S
        elif expected_q > 0:
            return 1  # Правосторонняя ориентация R
        else:
            return -1 # Левосторонняя ориентация L