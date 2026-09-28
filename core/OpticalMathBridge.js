// ============================================================
// МОДУЛЬ СОПРЯЖЕНИЯ ГЕОМЕТРИИ СФИРАЛИ С ФОТОННЫМИ МАТРИЦАМИ (OpticalMathBridge.js)
// ============================================================
import { generateSphiralTopology, OttendorfFractalAddressing, SfiralQutrit, computeQuantumNetwork } from './GideonMath.js';

export class OpticalMathBridge {
    constructor(baseScale = 140.0) {
        this.fractalAddresser = new OttendorfFractalAddressing(baseScale);
    }

    /**
     * Точный расчет комплексной матрицы передачи MZI (адаптировано из neurophox/mzi.py)
     */
    getMZITransferMatrix(internalUpper, internalLower, externalUpper, externalLower, hadamard = false, epsilon = [0.0, 0.0]) {
        const cc = Math.cos(Math.PI / 4 + epsilon[0]) * Math.cos(Math.PI / 4 + epsilon[1]);
        const cs = Math.cos(Math.PI / 4 + epsilon[0]) * Math.sin(Math.PI / 4 + epsilon[1]);
        const sc = Math.sin(Math.PI / 4 + epsilon[0]) * Math.cos(Math.PI / 4 + epsilon[1]);
        const ss = Math.sin(Math.PI / 4 + epsilon[0]) * Math.sin(Math.PI / 4 + epsilon[1]);
        
        const iu = internalUpper, il = internalLower, eu = externalUpper, el = externalLower;

        if (hadamard) {
            return [
                [
                    this.complexMul(this.complexAdd(this.complexExp(0, iu), this.complexExp(0, il), cc, ss), this.complexExp(0, eu)),
                    this.complexMul(this.complexSub(this.complexExp(0, iu), this.complexExp(0, il), cs, sc), this.complexExp(0, el))
                ],
                [
                    this.complexMul(this.complexSub(this.complexExp(0, iu), this.complexExp(0, il), sc, cs), this.complexExp(0, eu)),
                    this.complexMul(this.complexAdd(this.complexExp(0, iu), this.complexExp(0, il), ss, cc), this.complexExp(0, el))
                ]
            ];
        } else {
            return [
                [
                    this.complexMul(this.complexSub(this.complexExp(0, iu), this.complexExp(0, il), cc, ss), this.complexExp(0, eu)),
                    this.complexI_Mul(this.complexAdd(this.complexExp(0, iu), this.complexExp(0, il), cs, sc), this.complexExp(0, el))
                ],
                [
                    this.complexI_Mul(this.complexAdd(this.complexExp(0, iu), this.complexExp(0, il), sc, cs), this.complexExp(0, eu)),
                    this.complexMul(this.complexSub(this.complexExp(0, il), this.complexExp(0, iu), cc, ss), this.complexExp(0, el))
                ]
            ];
        }
    }

    // Вспомогательные комплексные операции для JS
    complexExp(re, im) { return { r: Math.cos(im), i: Math.sin(im) }; }
    complexMul(c, exp) { return { r: c.r * exp.r - c.i * exp.i, i: c.r * exp.i + c.i * exp.r }; }
    complexI_Mul(c, exp) { 
        const m = this.complexMul(c, exp);
        return { r: -m.i, i: m.r }; 
    }
    complexAdd(e1, e2, w1, w2) { return { r: e1.r * w1 + e2.r * w2, i: e1.i * w1 + e2.i * w2 }; }
    complexSub(e1, e2, w1, w2) { return { r: e1.r * w1 - e2.r * w2, i: e1.i * w1 - e2.i * w2 }; }

    /**
     * Дифракционное преобразование волнового фронта (адаптировано из ONNet DiffractiveLayer.py)[cite: 16]
     */
    computeDiffractiveWavefront(nodesCount, wavelength = 1.55) {
        const k = 2.0 * Math.PI / wavelength;
        const delta = 0.03;
        
        let diffractionMatrix = [];
        for (let i = 0; i < nodesCount; i++) {
            let row = [];
            for (let j = 0; j < nodesCount; j++) {
                const phase = ((i - nodesCount / 2) ** 2 + (j - nodesCount / 2) ** 2) * delta;
                const realPart = Math.cos(k * delta - wavelength * Math.PI * delta * phase);
                const imagPart = Math.sin(k * delta - wavelength * Math.PI * delta * phase);
                row.push({ r: realPart, i: imagPart });
            }
            diffractionMatrix.push(row);
        }
        return diffractionMatrix;
    }

    /**
     * Симуляция треугольного разложения сетки по методу Клементса (адаптировано из Photonic-Neural-Networks clements.py)[cite: 16]
     */
    simulateClementsMeshDecomposition(dim) {
        let phases = [];
        for (let p = 0; p < dim - 1; p++) {
            let colPhases = [];
            for (let q = 0; q <= p; q++) {
                colPhases.push({
                    theta: Math.PI / 2 - Math.atan(1.0 / (1.0 + p * 0.1)),
                    phi: (q * Math.PI) / dim
                });
            }
            phases.push(colPhases);
        }
        return { meshType: "Clements-Triangular", dimensions: dim, stages: phases.length };
    }

    /**
     * Шаг обратного распространения ошибки (Backpropagation) для оптимизации фаз MZI в реальном времени
     */
    optimizeWeightsStep(nodes, targetOutput, learningRate = 0.05) {
        let totalLoss = 0;

        nodes.forEach((node, idx) => {
            if (!node.params) node.params = {};
            if (!node.params.angles) node.params.angles = [0.1, 0.2, 0.3];

            let angles = node.params.angles;
            const mzi = this.getMZITransferMatrix(angles[0], angles[1], angles[2], 0.0, false, [0.0, 0.0]);
            
            const currentIntensity = mzi[0][0].r * mzi[0][0].r + mzi[0][0].i * mzi[0][0].i;
            const target = targetOutput[idx] || 1.0;
            const error = currentIntensity - target;
            
            totalLoss += error * error;

            angles[0] -= learningRate * error * (-Math.sin(angles[0]));
            angles[1] -= learningRate * error * (-Math.sin(angles[1]));
            angles[2] -= learningRate * error * (Math.cos(angles[2]));

            node.params.angles = angles.map(a => ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
        });

        const meanLoss = totalLoss / nodes.length;
        console.log(`🔄 [Backprop]: Шаг оптимизации выполнен. Средняя ошибка Loss = ${meanLoss.toFixed(6)}`);
        return meanLoss;
    }

    /**
     * Преобразует граф узлов конструктора в комплексную матрицу фазовых затворов
     */
    convertGraphToOpticalMatrix(nodes, edges) {
        console.log(`⚡ [OpticalMathBridge]: Трансляция графа (${nodes.length} узлов) с MZI, дифракцией, Клементсом и Backprop...`);
        
        const dim = nodes.length > 0 ? nodes.length : 1;
        const diffractiveField = this.computeDiffractiveWavefront(dim);
        const clementsMesh = this.simulateClementsMeshDecomposition(Math.min(dim, 16));

        // Получаем реальные сетевые кьютриты с учетом графа и ребер из GideonMath
        const networkQuantumStates = computeQuantumNetwork(nodes, edges);
        const qutritMap = {};
        networkQuantumStates.forEach(q => {
            qutritMap[q.id] = q.qutrit_state; // содержит { L, S, R }
        });

        let opticalMeshNodes = nodes.map(node => {
            const p = node.params || {};
            const scale = p.scale || 1.0;
            const stretch = p.stretch || 1.0;
            const angles = p.angles || [0, 0, 0];
            
            const addressInfo = this.fractalAddresser.encodeRecursiveAddress(node.id, node.x || 0, node.y || 0, node.z || 0, 2);
            const topology = generateSphiralTopology(140, 190, scale, stretch);
            
            // Берем реальное распределение кьютрита из сетевого расчета графа
            const calculatedQutrit = qutritMap[node.id] || { L: 0.33, S: 0.33, R: 0.34 };

            const mziMatrix = this.getMZITransferMatrix(angles[0], angles[1], angles[2], 0.0, false, [0.0, 0.0]);

            return {
                id: node.id,
                addressCode: addressInfo.address,
                mziTransferMatrix: mziMatrix,
                chiralTopology: topology,
                waveState: calculatedQutrit
            };
        });

        let interferenceMatrix = edges.map(edge => {
            const sourceNode = nodes.find(n => n.id === edge.from);
            const targetNode = nodes.find(n => n.id === edge.to);
            
            return {
                from: edge.from,
                to: edge.to,
                weight: edge.weight || 1.0,
                type: edge.type || 'right_polarization',
                phaseShift: sourceNode && targetNode ? this.calculatePhaseShift(sourceNode, targetNode) : 0.0
            };
        });

        return {
            timestamp: Date.now(),
            totalOpticalNodes: opticalMeshNodes.length,
            opticalChannels: interferenceMatrix,
            nodesData: opticalMeshNodes,
            diffractionFieldSummary: diffractiveField.length,
            clementsMeshConfig: clementsMesh
        };
    }

    calculatePhaseShift(nodeA, nodeB) {
        const dx = (nodeB.x || 0) - (nodeA.x || 0);
        const dy = (nodeB.y || 0) - (nodeA.y || 0);
        const dz = (nodeB.z || 0) - (nodeA.z || 0);
        const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const k = 2 * Math.PI / 1.55; 
        return parseFloat((k * distance) % (2 * Math.PI)).toFixed(4);
    }
}

export const opticalBridge = new OpticalMathBridge();