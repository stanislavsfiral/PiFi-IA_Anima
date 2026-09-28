import * as THREE from 'three';

// --- ГЕНЕРАЦИЯ ПРАВОГО ВИТКА СФИРАЛИ ---
export function generateRightBranch(R, H, subParams = {}) {
    const mainPts = [];
    const junctionZ = H / 5;
    
    // Учитываем масштаб и высоту под-объекта правой ветви
    const rightSubHeight = subParams.rightSub?.height !== undefined ? subParams.rightSub.height / 100.0 : 1.0;
    
    for (let i = 0; i <= 100; i++) {
        let t = i / 100;
        let angle = 2 * Math.PI * t;
        let x = R * Math.cos(angle);
        let y = R * Math.sin(angle);
        let z = (junctionZ + (H - junctionZ) * (1 - t)) * rightSubHeight;
        mainPts.push(new THREE.Vector3(x, y, z));
    }

    const sPts = [];
    let r_s = R / 2;
    
    // Учитываем масштабирование S-перехода
    const sRightSubHeight = subParams.sRightSub?.height !== undefined ? subParams.sRightSub.height / 100.0 : 1.0;

    for (let i = 0; i <= 60; i++) {
        let t = i / 60;
        let angle = Math.PI * t;
        let x = r_s + r_s * Math.cos(angle);
        let y = r_s * Math.sin(angle);
        let z = (junctionZ * (1 - t)) * sRightSubHeight;
        sPts.push(new THREE.Vector3(x, y, z));
    }
    return { mainPts, sPts };
}

export { TernarySpatialWalshEngine } from '../core/TernaryWalsh.js';