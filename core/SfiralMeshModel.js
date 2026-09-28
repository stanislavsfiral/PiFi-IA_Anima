// ============================================================
// АДАПТИРОВАННАЯ МЕШ-МОДЕЛЬ СФИРАЛИ (SfiralMeshModel.js)
// ============================================================
export class SfiralMeshModel {
    constructor(units, numLayers, scale = 1.0) {
        this.units = units;
        this.numLayers = numLayers;
        this.scale = scale;
    }

    /**
     * Заменяет стандартные плоские перестановки 
     * на S-образные спиральные переходы и нулевую хиральность
     */
    generateSfiralPermutations() {
        let permIndices = [];
        for (let l = 0; l < this.numLayers; l++) {
            let layerIndices = [];
            // Внедряем логику зеркальной антисимметрии Сфирали
            let chiralitySign = (l % 2 === 0) ? 1 : -1;
            for (let i = 0; i < this.units; i++) {
                let mappedIndex = (i + chiralitySign * Math.floor(this.scale * 2)) % this.units;
                layerIndices.push(Math.abs(mappedIndex));
            }
            permIndices.push(layerIndices);
        }
        return permIndices;
    }

    /**
     * Расчет фазовых матриц с учетом фрактального масштаба Оттендорфа
     */
    computePhaseMasks() {
        const baseMatrix = new Array(this.numLayers).fill(0).map(() => new Array(this.units).fill(0));
        return baseMatrix.map((layer, l) => 
            layer.map((_, i) => Math.sin(l * Math.PI / 4) * Math.cos(i * this.scale))
        );
    }
}