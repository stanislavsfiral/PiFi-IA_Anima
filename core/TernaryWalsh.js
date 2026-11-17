/**
 * Троичный пространственный алгоритм Уолша с двухуровневой суперпозицией.
 * Чистый ES-модуль ядра PiFiYA / GIDEON.
 */
export class TernarySpatialWalshEngine {
    constructor() {
        this.active = false;
    }
    
    quantizeTernary(val) {
        const threshold = 0.3;
        if (Math.abs(val) < threshold) return 0;  
        return val > 0 ? 1 : -1;                  
    }

    // Уровень 1: Оценка внутренней S-петли узла
    evaluateNode(node, time) {
        const pX = Math.sin(time + node.x * 0.02);
        const pY = Math.sin(time + node.y * 0.02 + Math.PI / 3);
        const pZ = Math.sin(time + node.z * 0.02 + Math.PI / 1.5);

        const tensorVal = (pX + pY + pZ) / 3;
        const ternaryState = this.quantizeTernary(tensorVal);

        const angleZ = (node.params && node.params.angles) ? node.params.angles[2] : 90;
        const phaseShift = (angleZ * Math.PI / 180.0) * 0.5;

        if (node.params) {
            if (ternaryState === 1) {
                node.params.showRight = true;
                node.params.showS = true;
                node.params.showLeft = false;
                node.params.showSLeft = false;
            } else if (ternaryState === -1) {
                node.params.showRight = false;
                node.params.showS = false;
                node.params.showLeft = true;
                node.params.showSLeft = true;
            } else {
                // Уровень суперпозиции (S-петля как нулевой хроноквант)
                node.params.showRight = true;
                node.params.showS = true;
                node.params.showLeft = true;
                node.params.showSLeft = true;
            }
        }
        
        node.quantumState = {
            intensity: Math.abs(tensorVal),
            psi_real: pX * Math.cos(phaseShift),
            psi_imag: pY * Math.sin(phaseShift),
            ternary: ternaryState
        };
    }

    // Уровень 2: Оценка суперпозиции между Сфиралями через общий виток (с учетом поворота на 180° по Y и Z)
    evaluateChainBridge(nodeA, nodeB) {
        const dx = nodeA.x - nodeB.x;
        const dy = nodeA.y - nodeB.y;
        const dz = nodeA.z - nodeB.z;
        const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);

        const anglesA = (nodeA.params && nodeA.params.angles) || [180, 0, 90];
        const anglesB = (nodeB.params && nodeB.params.angles) || [180, 0, 90];
        
        const diffY = Math.abs(anglesA[1] - anglesB[1]);
        const diffZ = Math.abs(anglesA[2] - anglesB[2]);
        const isReversed180 = (Math.abs(diffY - 180) < 15 && Math.abs(diffZ - 180) < 15);

        const scaleA = (nodeA.params && nodeA.params.scale) || 1.0;
        const scaleB = (nodeB.params && nodeB.params.scale) || 1.0;
        const thresholdBridge = 120 * ((scaleA + scaleB) / 2);
        
        if (distance <= thresholdBridge) {
            const bridgeChirality = isReversed180 ? -1 : 1;
            const superpositionWeight = 1.0 - (distance / thresholdBridge);

            return {
                isBridge: true,
                chirality: bridgeChirality,
                weight: superpositionWeight,
                sharedState: true
            };
        }

        return { isBridge: false, chirality: 0, weight: 0, sharedState: false };
    }
}