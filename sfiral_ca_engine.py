import numpy as np

class SfiralCAEngine:
    """Движок клеточных автоматов и волновой динамики для топологии Сфирали."""
    def __init__(self, width=32, height=32, channels=4):
        self.width = width
        self.height = height
        self.channels = channels

    def encode_nodes_to_state(self, nodes):
        """Проецирует узлы графа сфиралей в 2D-матрицу состояния (H, W, C) с учетом фаз и стандарта 0.7778."""
        state = np.zeros((self.height, self.width, self.channels), dtype=np.float32)
        
        if not nodes:
            return state

        # Превращаем параметры узлов в скалярный сигнал волны
        signals = []
        for node in nodes:
            p = node.get('params', {})
            angles = p.get('angles', [0, 0, 0])
            stretch = p.get('stretch', 0.7778)
            # Учитываем физику S-перехода и нулевую хиральность через фазовый угол
            phase_val = np.radians(angles[1]) * (stretch / 0.7778)
            signals.append(phase_val)

        x = np.array(signals, dtype=np.float32)
        
        # Используем логику фазового кодирования (синус/косинус дляканалов 0 и 1)
        scalar = float(np.mean(x)) * 2.0 * np.pi if len(x) > 0 else 0.0
        state[..., 0] = np.sin(scalar)
        state[..., 1] = np.cos(scalar)
        
        # Заполняем каналы волновой амплитудой
        sq = int(np.floor(np.sqrt(len(x)))) if len(x) > 0 else 1
        sq = max(sq, 1)
        h_in, w_in = sq, max(1, len(x) // sq)
        if h_in * w_in < len(x):
            w_in += 1
            
        padded = np.zeros(h_in * w_in, dtype=np.float32)
        if len(x) > 0:
            padded[:len(x)] = x
        img_in = padded.reshape(h_in, w_in)
        
        # Простейший nearest-resize в целевую сетку (H, W)
        row_idx = (np.arange(self.height) * h_in / self.height).astype(int)
        col_idx = (np.arange(self.width) * w_in / self.width).astype(int)
        state[..., 2] = img_in[np.ix_(row_idx, col_idx)]
        
        return state

    def step_wave_simulation(self, state, steps=5, c=0.5, damping=0.999, dt=1.0):
        """Прогоняет дискретную волновую динамику (амплитуда=ch0, скорость=ch1) за `steps` итераций."""
        cur_state = state.astype(np.float32)
        
        for _ in range(steps):
            amp = cur_state[..., 0]
            vel = cur_state[..., 1]
            
            # Дискретный 2D лапласиан с периодическими граничными условиями (np.roll)
            lap = (
                np.roll(amp, 1, axis=0) + np.roll(amp, -1, axis=0) +
                np.roll(amp, 1, axis=1) + np.roll(amp, -1, axis=1) -
                4.0 * amp
            )
            
            c2 = c * c
            new_vel = (vel + dt * c2 * lap) * damping
            new_amp = np.clip(amp + dt * new_vel, -1.0, 1.0)
            
            cur_state[..., 0] = new_amp
            cur_state[..., 1] = new_vel
            
        return cur_state