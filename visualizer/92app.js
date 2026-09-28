import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { generateHalfPoints, GideonWebCore, computeQuantumNetwork, OttendorfFractalAddressing } from '../core/GideonMath.js?v=dynamic';

let customModelSource = null; 
let customPoints = null;      

let scene, camera, renderer, controls, spiralGroup;
let chipContainerGroup; 
let core = new GideonWebCore();
let globalNodesData = {}; 
let selectedNodeIds = []; 

let isAnimationActive = false;
let currentWaveAmplitude = 1.0;

const ottendorfCoder = new OttendorfFractalAddressing(140.0);

const R_sphere = 280.0;
let signalSpheres = [];
let laserPulses = []; 
let animClock = 0;
let pulseClock = new THREE.Clock();
let cachedCurvesData = []; 
let lastQuantumResults = []; 

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function init3D() {
    const container = document.getElementById('canvasContainer');
    if (!container) return;
    
    scene = new THREE.Scene();
    
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 8000);
    camera.position.set(600, 450, 700);

    const canvasEl = document.getElementById('renderCanvas');
    if (!canvasEl) return;

    renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0); 
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };

    scene.add(new THREE.AmbientLight(0xffffff, 1.8));
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(400, 800, 600);
    scene.add(dirLight);

    spiralGroup = new THREE.Group();
    scene.add(spiralGroup);

    chipContainerGroup = new THREE.Group();
    chipContainerGroup.name = "GIDEON_Glass_Chip_Housing";
    scene.add(chipContainerGroup);

    window.addEventListener('resize', onWindowResize);
    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('click', onCanvasClick);

    const toggleBreathingBtn = document.getElementById('toggleBreathingAnimBtn');
    if (toggleBreathingBtn) {
        toggleBreathingBtn.innerText = '🌀 Пульсация и Лазеры: ВЫКЛ';
        toggleBreathingBtn.style.background = '#2a1f2d';
        toggleBreathingBtn.style.color = '#ff88ff';
        toggleBreathingBtn.style.borderColor = '#ff00ff';

        toggleBreathingBtn.addEventListener('click', () => {
            isAnimationActive = !isAnimationActive;
            if (isAnimationActive) {
                toggleBreathingBtn.innerText = '🌀 Пульсация и Лазеры: ВКЛ';
                toggleBreathingBtn.style.background = '#1f2d4a';
                toggleBreathingBtn.style.color = '#00ffaa';
                toggleBreathingBtn.style.borderColor = '#00ffaa';
            } else {
                toggleBreathingBtn.innerText = '🌀 Пульсация и Лазеры: ВЫКЛ';
                toggleBreathingBtn.style.background = '#2a1f2d';
                toggleBreathingBtn.style.color = '#ff88ff';
                toggleBreathingBtn.style.borderColor = '#ff00ff';

                if (spiralGroup) {
                    spiralGroup.children.forEach((group) => {
                        if (group.type === 'Group') group.scale.set(1, 1, 1);
                    });
                }
            }
        });
    }

    updateScene();
    animate();
}

window.askAI = async function(question) {
    let logEl = document.getElementById('consoleLog') || document.getElementById('console');
    if (logEl) {
        logEl.style.display = 'block';
        logEl.innerHTML += `<div class="console-line type-sys" style="color:#ffaa00; margin-top:4px;">🧠 [ИИ думает...]: ${question}</div>`;
        logEl.scrollTop = logEl.scrollHeight;
    }

    try {
        const response = await fetch('http://localhost:8000/api/ask_ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ graph: customModelSource || { nodes: [], edges: [] }, question: question })
        });
        const data = await response.json();
        
        if (logEl) {
            logEl.innerHTML += `<div class="console-line" style="color:#00ffaa; margin-top:4px;">🤖 [DeepSeek]: ${data.answer}</div>`;
            logEl.scrollTop = logEl.scrollHeight;
        }
    } catch (err) {
        console.error("Ошибка ИИ:", err);
    }
};

window.computeQuantumState = async function computeQuantumState(nodes, edges) {
    let logEl = document.getElementById('consoleLog') || document.getElementById('console');
    if (!logEl) return;
    
    try {
        const response = await fetch('http://localhost:8000/api/ai_predict_topology', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nodes: nodes, edges: edges || [] })
        });
        const data = await response.json();
        if (data.status === 'success') {
            currentWaveAmplitude = data.wave_simulation_amplitude || 1.0;
        }
        lastQuantumResults = computeQuantumNetwork(nodes, edges || []);
    } catch (error) {
        console.error("❌ Ошибка связи с CA-сервером:", error);
    }
}

// ========================================================
// СТЕКЛЯННЫЙ КУБ И КОННЕКТОРЫ ПО КАСАТЕЛЬНОЙ ВИТКОВ
// ========================================================
function updateGlassChipHousing(endpointsList) {
    chipContainerGroup.clear();
    
    const box = new THREE.Box3().setFromObject(spiralGroup);
    if (box.isEmpty()) return;

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    const padding = 70; // Увеличенный отступ, чтобы куб гарантированно охватывал модель
    const sizeX = size.x + padding * 2;
    const sizeY = size.y + padding * 2;
    const sizeZ = size.z + padding * 2;

    // 1. Внешний стеклянный корпус чипа
    const boxGeo = new THREE.BoxGeometry(sizeX, sizeY, sizeZ);
    const glassHousingMat = new THREE.MeshPhysicalMaterial({
        color: 0x99ddff,
        transparent: true,
        opacity: 0.2,
        roughness: 0.05,
        metalness: 0.1,
        ior: 1.517,
        transmission: 0.95,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const glassBox = new THREE.Mesh(boxGeo, glassHousingMat);
    glassBox.position.copy(center);
    chipContainerGroup.add(glassBox);

    // 2. Удлиненные коннекторы, ориентированные строго по касательной витков
    const connectorLength = 45; // Делаем длиннее, чтобы выходили из куба наружу
    const connectorGeo = new THREE.CylinderGeometry(2.8, 2.8, connectorLength, 16);
    // Сдвигаем геометрию цилиндра так, чтобы его основание стыковалось в точку торца
    connectorGeo.translate(0, connectorLength / 2, 0);

    const connectorMat = new THREE.MeshStandardMaterial({
        color: 0x00ffaa,
        roughness: 0.3,
        metalness: 0.8,
        emissive: 0x004422,
        emissiveIntensity: 0.4
    });

    if (endpointsList && endpointsList.length > 0) {
        endpointsList.forEach(item => {
            // Входной коннектор (по касательной начала витка наружу)
            if (item.entry && item.entryTangent) {
                const entryConn = new THREE.Mesh(connectorGeo, connectorMat);
                entryConn.position.copy(item.entry);
                // Ориентируем цилиндр вдоль касательного вектора (инвертированного для выхода наружу)
                const dir = item.entryTangent.clone().negate().normalize();
                entryConn.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
                chipContainerGroup.add(entryConn);
            }
            // Выходной коннектор (по касательной окончания витка наружу)
            if (item.exit && item.exitTangent) {
                const exitConn = new THREE.Mesh(connectorGeo, connectorMat);
                exitConn.position.copy(item.exit);
                const dir = item.exitTangent.clone().normalize();
                exitConn.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
                chipContainerGroup.add(exitConn);
            }
        });
    }
}

function updateScene() {
    clearGroup(spiralGroup);
    signalSpheres = [];
    laserPulses = [];
    cachedCurvesData = [];
    globalNodesData = {}; 

    const modeEl = document.getElementById('modeSelect');
    const harmAxisEl = document.getElementById('harmAxisSelect');
    const coresInputEl = document.getElementById('coresInput');

    let mode = modeEl ? modeEl.value : 'Single';
    let harmAxis = harmAxisEl ? harmAxisEl.value : 'Harmonic Z';
    let nCores = coresInputEl ? (parseInt(coresInputEl.value) || 1) : 1;
    let angleStep = 360.0 / nCores;

    const statusHeader = document.getElementById('statusHeader');
    const resetModelBtn = document.getElementById('resetModelBtn');

    let endpointsList = [];

    const glassMaterial = new THREE.MeshPhysicalMaterial({
        color: 0x88ccff,
        transparent: true,
        opacity: 0.45,
        roughness: 0.1,
        metalness: 0.1,
        ior: 1.5,
        transmission: 0.9,
        side: THREE.DoubleSide
    });

    function createTubeMesh(pts, colorHex, partName, nodeScale) {
        if (!pts || pts.length < 2) return null;
        const curve = new THREE.CatmullRomCurve3(pts);
        const tubeGeo = new THREE.TubeGeometry(curve, 64, 2.5 * (nodeScale || 1.0), 8, false);
        const mat = glassMaterial.clone();
        mat.emissive.setHex(colorHex);
        mat.emissiveIntensity = 0.25;
        
        const mesh = new THREE.Mesh(tubeGeo, mat);
        mesh.userData = { part: partName, points: pts, curve: curve };
        return mesh;
    }

    if (customModelSource && customModelSource.nodes) {
        if (statusHeader) statusHeader.innerText = `STATUS: ACTIVE • ${customModelSource.nodes.length} NODES`;
        if (resetModelBtn) resetModelBtn.style.display = 'block';

        customModelSource.nodes.forEach((node) => {
            const nodeGroup = new THREE.Group();
            nodeGroup.name = node.id; 
            
            let px = node.x !== undefined ? node.x : 0;
            let py = node.y !== undefined ? node.y : 0;
            let pz = node.z !== undefined ? node.z : 0;
            nodeGroup.position.set(px, py, pz);

            const angles = (node.params && node.params.angles) ? node.params.angles : [0, 0, 0];
            const euler = new THREE.Euler(
                THREE.MathUtils.degToRad(angles[0]),
                THREE.MathUtils.degToRad(angles[1]),
                THREE.MathUtils.degToRad(angles[2])
            );
            nodeGroup.rotation.copy(euler);

            let nodeScale = (node.params && node.params.scale !== undefined) ? node.params.scale : 1.0;
            let nodeStretch = (node.params && node.params.stretch !== undefined) ? node.params.stretch : 1.0;
            let nodeN = (node.params && node.params.N !== undefined) ? node.params.N : 5;

            let baseR = 60 + nodeN * 2; 
            let baseH = 80 + nodeN * 2; 

            const basePtsObj = generateHalfPoints(baseR, baseH, 1.0, 1.0); 
            const transformScale = new THREE.Vector3(nodeScale, nodeScale, nodeScale * nodeStretch);

            let splitIdx = Math.floor(basePtsObj.x.length * 0.62); 
            
            let fullPts = [];
            for(let i=0; i<basePtsObj.x.length; i++) {
                let p = new THREE.Vector3(basePtsObj.x[i], basePtsObj.y[i], basePtsObj.z[i]);
                p.multiply(transformScale); 
                fullPts.push(p);
            }
            let aPts = fullPts.map(p => new THREE.Vector3(-p.x, -p.y, -p.z));

            const rightPts = fullPts.slice(0, splitIdx + 1);
            const leftPts = aPts.slice(0, splitIdx + 1).reverse();
            const sPts = fullPts.slice(splitIdx);
            for(let i = aPts.length - 2; i >= splitIdx; i--) sPts.push(aPts[i]);

            const rightTube = createTubeMesh(rightPts, 0xff3333, 'right', nodeScale);
            const sTube = createTubeMesh(sPts, 0xffea00, 's', nodeScale);
            const leftTube = createTubeMesh(leftPts, 0x00c0ff, 'left', nodeScale);

            if (rightTube) nodeGroup.add(rightTube);
            if (sTube) nodeGroup.add(sTube);
            if (leftTube) nodeGroup.add(leftTube);
            
            let centerMesh = new THREE.Mesh(new THREE.SphereGeometry(3.0 * nodeScale, 16, 16), new THREE.MeshBasicMaterial({ color: 0xffd000 }));
            nodeGroup.add(centerMesh);
            spiralGroup.add(nodeGroup);

            // Вычисляем точные касательные векторе в торцах через кривые Three.js
            if (rightTube && leftTube) {
                const entryTan = rightTube.userData.curve.getTangentAt(0).applyEuler(euler);
                const exitTan = leftTube.userData.curve.getTangentAt(1).applyEuler(euler);

                let entryWorld = rightPts[0].clone().applyEuler(euler).add(new THREE.Vector3(px, py, pz));
                let exitWorld = leftPts[leftPts.length - 1].clone().applyEuler(euler).add(new THREE.Vector3(px, py, pz));
                
                endpointsList.push({ 
                    entry: entryWorld, entryTangent: entryTan,
                    exit: exitWorld, exitTangent: exitTan 
                });
            }
        });

        updateGlassChipHousing(endpointsList);
        window.computeQuantumState(customModelSource.nodes, customModelSource.edges);

    } else {
        if (statusHeader) statusHeader.innerText = "STATUS: ACTIVE • Q-ZERO CHIRALITY";
        if (resetModelBtn) resetModelBtn.style.display = 'none';

        let rawStruct = generateHalfPoints(140, 190);
        let rawX = rawStruct.x, rawY = rawStruct.y, rawZ = rawStruct.z;

        let splitIdx = Math.floor(rawX.length * 0.62);
        if (rawX.length > 0) {
            let r0 = Math.sqrt(rawX[0]**2 + rawY[0]**2);
            for(let i=0; i<rawX.length; i++) {
                let r = Math.sqrt(rawX[i]**2 + rawY[i]**2);
                if (r0 - r > 1.0) { splitIdx = Math.max(0, i - 1); break; }
            }
        }

        for (let k = 0; k < nCores; k++) {
            let angle = k * angleStep;
            let tPoints = [], aPoints = [];
            for (let i = 0; i < rawX.length; i++) {
                let p1 = rotateCoords(rawX[i], rawY[i], rawZ[i], angle, harmAxis);
                tPoints.push(new THREE.Vector3(p1.x, p1.y, p1.z));
                aPoints.push(new THREE.Vector3(-p1.x, -p1.y, -p1.z));
            }

            let rightPts = tPoints.slice(0, splitIdx + 1);
            let leftPts = aPoints.slice(0, splitIdx + 1).reverse();
            let sPts = tPoints.slice(splitIdx);
            for(let i = aPoints.length - 2; i >= splitIdx; i--) sPts.push(aPoints[i]);

            const rightTube = createTubeMesh(rightPts, 0x00f0ff, 'right', 1.0);
            const sTube = createTubeMesh(sPts, 0xffea00, 's', 1.0);
            const leftTube = createTubeMesh(leftPts, 0xff2255, 'left', 1.0);

            if (rightTube) spiralGroup.add(rightTube);
            if (sTube) spiralGroup.add(sTube);
            if (leftTube) spiralGroup.add(leftTube);

            if (rightTube && leftTube) {
                endpointsList.push({ 
                    entry: rightPts[0].clone(), entryTangent: rightTube.userData.curve.getTangentAt(0),
                    exit: leftPts[leftPts.length - 1].clone(), exitTangent: leftTube.userData.curve.getTangentAt(1)
                });
            }

            if (mode !== 'Single') {
                let t2Points = [], a2Points = [];
                for (let i = 0; i < rawX.length; i++) {
                    let p1 = rotateCoords(rawX[i], rawY[i], rawZ[i], angle, harmAxis);
                    let p2, p3;
                    if (mode === 'Axis X') { p2 = { x: p1.x, y: -p1.y, z: -p1.z }; p3 = { x: -p1.x, y: p1.y, z: p1.z }; } 
                    else if (mode === 'Axis Y') { p2 = { x: -p1.x, y: p1.y, z: -p1.z }; p3 = { x: p1.x, y: -p1.y, z: p1.z }; } 
                    else { p2 = { x: -p1.x, y: -p1.y, z: p1.z }; p3 = { x: p1.x, y: p1.y, z: -p1.z }; }
                    t2Points.push(new THREE.Vector3(p2.x, p2.y, p2.z));
                    a2Points.push(new THREE.Vector3(p3.x, p3.y, p3.z));
                }
                
                let rightPts2 = t2Points.slice(0, splitIdx + 1);
                let leftPts2 = a2Points.slice(0, splitIdx + 1).reverse();
                let sPts2 = t2Points.slice(splitIdx);
                for(let i = a2Points.length - 2; i >= splitIdx; i--) sPts2.push(a2Points[i]);

                const rightTube2 = createTubeMesh(rightPts2, 0x00ffaa, 'right2', 1.0);
                const sTube2 = createTubeMesh(sPts2, 0xffbb00, 's2', 1.0);
                const leftTube2 = createTubeMesh(leftPts2, 0xff9900, 'left2', 1.0);

                if (rightTube2) spiralGroup.add(rightTube2);
                if (sTube2) spiralGroup.add(sTube2);
                if (leftTube2) spiralGroup.add(leftTube2);

                if (rightTube2 && leftTube2) {
                    endpointsList.push({ 
                        entry: rightPts2[0].clone(), entryTangent: rightTube2.userData.curve.getTangentAt(0),
                        exit: leftPts2[leftPts2.length - 1].clone(), exitTangent: leftTube2.userData.curve.getTangentAt(1)
                    });
                }
            }
        }

        updateGlassChipHousing(endpointsList);
    }
}

function animate() {
    requestAnimationFrame(animate);
    const delta = pulseClock.getDelta();

    if (isAnimationActive) {
        const quenchInputEl = document.getElementById('quenchInput');
        const animSpeedRangeEl = document.getElementById('animSpeedRange');

        let quenchVal = quenchInputEl ? (parseFloat(quenchInputEl.value) || 1.0) : 1.0;
        let speedMultiplier = animSpeedRangeEl ? (parseFloat(animSpeedRangeEl.value) || 1.0) : 1.0;
        animClock += 0.015 * quenchVal * speedMultiplier;

        let caPulse = 1.0 + Math.sin(animClock * 3.0) * 0.15 * currentWaveAmplitude;
        if (spiralGroup) {
            spiralGroup.children.forEach((group) => {
                if (group.type === 'Group') {
                    group.scale.set(caPulse, caPulse, caPulse);
                }
            });
        }

        if (Math.random() < 0.4) {
            let targetTubes = [];
            spiralGroup.traverse(child => {
                if (child.geometry instanceof THREE.TubeGeometry && child.userData && child.userData.points) {
                    targetTubes.push(child);
                }
            });

            if (targetTubes.length > 0) {
                const targetTube = targetTubes[Math.floor(Math.random() * targetTubes.length)];
                const pulseGeo = new THREE.SphereGeometry(1.8, 8, 8);
                const pulseMat = new THREE.MeshBasicMaterial({ color: 0x00ffcc, transparent: true, opacity: 0.95 });
                const pulse = new THREE.Mesh(pulseGeo, pulseMat);
                pulse.userData = {
                    points: targetTube.userData.points,
                    progress: 0,
                    speed: 0.6 + Math.random() * 0.4
                };
                spiralGroup.add(pulse);
                laserPulses.push({ mesh: pulse, parentGroup: spiralGroup });
            }
        }

        for (let i = laserPulses.length - 1; i >= 0; i--) {
            const pData = laserPulses[i];
            pData.mesh.userData.progress += delta * pData.mesh.userData.speed;
            
            if (pData.mesh.userData.progress >= 1.0) {
                pData.parentGroup.remove(pData.mesh);
                pData.mesh.geometry.dispose();
                pData.mesh.material.dispose();
                laserPulses.splice(i, 1);
            } else {
                const pts = pData.mesh.userData.points;
                const idx = pData.mesh.userData.progress * (pts.length - 1);
                const lowIdx = Math.floor(idx);
                const highIdx = Math.min(lowIdx + 1, pts.length - 1);
                const t = idx - lowIdx;
                
                if (pts[lowIdx] && pts[highIdx]) {
                    const currentPos = new THREE.Vector3().lerpVectors(pts[lowIdx], pts[highIdx], t);
                    pData.mesh.position.copy(currentPos);
                }
            }
        }
    } else {
        laserPulses.forEach(pData => {
            pData.parentGroup.remove(pData.mesh);
            pData.mesh.geometry.dispose();
            pData.mesh.material.dispose();
        });
        laserPulses = [];
    }

    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

function clearGroup(group) {
    while(group.children.length > 0){ group.remove(group.children[0]); }
}

function onWindowResize() {
    const container = document.getElementById('canvasContainer');
    if (!container || !renderer || !camera) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function rotateCoords(x, y, z, angleDeg, axis) {
    let rad = angleDeg * Math.PI / 180.0;
    let c = Math.cos(rad), s = Math.sin(rad);
    if (axis === 'Harmonic X') return { x: x, y: y * c - z * s, z: y * s + z * c };
    if (axis === 'Harmonic Y') return { x: x * c + z * s, y: y, z: -x * s + z * c };
    return { x: x * c - y * s, y: x * s + y * c, z: z };
}

function onMouseMove(event) {}
function onCanvasClick(event) {}

const modeSelectEl = document.getElementById('modeSelect');
if (modeSelectEl) modeSelectEl.addEventListener('change', updateScene);

const harmAxisSelectEl = document.getElementById('harmAxisSelect');
if (harmAxisSelectEl) harmAxisSelectEl.addEventListener('change', updateScene);

const coresInputEl = document.getElementById('coresInput');
if (coresInputEl) coresInputEl.addEventListener('change', updateScene);

const loadModelBtn = document.getElementById('loadModelBtn');
const modelFileInput = document.getElementById('modelFileInput');
if (loadModelBtn && modelFileInput) {
    loadModelBtn.addEventListener('click', () => modelFileInput.click());
    modelFileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                let data = JSON.parse(e.target.result);
                if (data.graph && Array.isArray(data.graph.nodes)) data = data.graph;
                if (data.nodes && Array.isArray(data.nodes)) {
                    customModelSource = data;
                }
                updateScene();
            } catch(err) { alert('Ошибка чтения файла: ' + err.message); }
        };
        reader.readAsText(file); 
        event.target.value = '';
    });
}

const resetModelBtn = document.getElementById('resetModelBtn');
if (resetModelBtn) {
    resetModelBtn.addEventListener('click', () => {
        customModelSource = null; 
        resetModelBtn.style.display = 'none';
        updateScene();
    });
}

window.onload = init3D;