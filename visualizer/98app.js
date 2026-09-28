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

// Анимация лазера выключена по умолчанию при входе
let isAnimationActive = false;
let currentWaveAmplitude = 1.0;

const ottendorfCoder = new OttendorfFractalAddressing(140.0);

const R_sphere = 280.0;
let signalSpheres = [];
let laserPulses = []; // <--- Массив лазерных импульсов
let globalChainPaths = []; 
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

    // Привязываем кнопку включения/выключения лазера в шапке
    const toggleAnimBtn = document.getElementById('toggleAnimMasterBtn');
    if (toggleAnimBtn) {
        toggleAnimBtn.addEventListener('click', () => {
            isAnimationActive = !isAnimationActive;
            if (isAnimationActive) {
                toggleAnimBtn.innerText = '⚡ Лазер: ВКЛ';
                toggleAnimBtn.style.background = '#1f4a38';
                toggleAnimBtn.style.color = '#00ffaa';
                toggleAnimBtn.style.border = '1px solid #00ffaa';
            } else {
                toggleAnimBtn.innerText = '⚡ Лазер: ВЫКЛ';
                toggleAnimBtn.style.background = '#1f2d4a';
                toggleAnimBtn.style.color = '#ff88ff';
                toggleAnimBtn.style.border = '1px solid #ff00ff';
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
// СТЕКЛЯННЫЙ КУБ И КОННЕКТОРЫ ТОЛЬКО НА ВНЕШНИХ КРАЯХ
// ========================================================
function updateGlassChipHousing(endpointsList) {
    chipContainerGroup.clear();
    
    const box = new THREE.Box3().setFromObject(spiralGroup);
    if (box.isEmpty()) return;

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    const padding = 15; 
    const sizeX = size.x + padding * 2;
    const sizeY = size.y + padding * 2;
    const sizeZ = size.z + padding * 2;

    const boxGeo = new THREE.BoxGeometry(sizeX, sizeY, sizeZ);
    const glassHousingMat = new THREE.MeshPhysicalMaterial({
        color: 0x99ddff,
        transparent: true,
        opacity: 0.18,
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

    let allEnds = [];
    if (endpointsList && endpointsList.length > 0) {
        endpointsList.forEach(item => {
            if (item.entry) allEnds.push({ pos: item.entry, tan: item.entryTangent, isEntry: true });
            if (item.exit) allEnds.push({ pos: item.exit, tan: item.exitTangent, isEntry: false });
        });
    }

    const threshold = 18.0; 
    const externalEnds = allEnds.filter((end, idx) => {
        let hasCloseNeighbor = false;
        for (let j = 0; j < allEnds.length; j++) {
            if (idx === j) continue;
            if (end.pos.distanceTo(allEnds[j].pos) < threshold) {
                hasCloseNeighbor = true;
                break;
            }
        }
        return !hasCloseNeighbor; 
    });

    const connectorLength = 65; 
    const connectorGeo = new THREE.CylinderGeometry(2.8, 2.8, connectorLength, 16);
    connectorGeo.translate(0, connectorLength / 2, 0);

    const connectorMat = new THREE.MeshStandardMaterial({
        color: 0x00ffaa,
        roughness: 0.3,
        metalness: 0.8,
        emissive: 0x004422,
        emissiveIntensity: 0.4
    });

    externalEnds.forEach(end => {
        const conn = new THREE.Mesh(connectorGeo, connectorMat);
        conn.position.copy(end.pos);
        const dir = end.isEntry ? end.tan.clone().negate().normalize() : end.tan.clone().normalize();
        conn.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        chipContainerGroup.add(conn);
    });
}

function updateScene() {
    clearGroup(spiralGroup);
    signalSpheres = [];
    laserPulses = [];
    globalChainPaths = [];
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

        let nodePathsMap = {};

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

            let flowRightToLeft = [];
            for(let i=0; i<fullPts.length; i++) { flowRightToLeft.push(fullPts[i].clone()); }
            for(let i=fullPts.length-1; i>=0; i--) { flowRightToLeft.push(new THREE.Vector3(-fullPts[i].x, -fullPts[i].y, -fullPts[i].z)); }
            
            let worldPath = flowRightToLeft.map(p => p.clone().applyEuler(euler).add(new THREE.Vector3(px, py, pz)));
            nodePathsMap[node.id] = worldPath;

            globalNodesData[node.id] = { id: node.id, next: null };

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

        if (customModelSource.edges && customModelSource.edges.length > 0) {
            customModelSource.edges.forEach(edge => {
                if (globalNodesData[edge.from]) {
                    globalNodesData[edge.from].next = edge.to;
                }
            });
        }

        let visited = {};
        customModelSource.nodes.forEach(startNode => {
            if (visited[startNode.id]) return;
            let hasIncoming = customModelSource.edges && customModelSource.edges.some(e => e.to === startNode.id);
            if (hasIncoming) return;

            let currentId = startNode.id;
            let fullChainPoints = [];
            while(currentId && nodePathsMap[currentId]) {
                visited[currentId] = true;
                fullChainPoints = fullChainPoints.concat(nodePathsMap[currentId]);
                let nextNode = globalNodesData[currentId] ? globalNodesData[currentId].next : null;
                currentId = nextNode;
            }
            if (fullChainPoints.length > 1) {
                globalChainPaths.push(fullChainPoints);
            }
        });

        customModelSource.nodes.forEach(node => {
            if (!visited[node.id] && nodePathsMap[node.id]) {
                globalChainPaths.push(nodePathsMap[node.id]);
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

            let fullCorePath = [];
            for(let i=0; i<rightPts.length; i++) fullCorePath.push(rightPts[i]);
            for(let i=1; i<sPts.length; i++) fullCorePath.push(sPts[i]);
            for(let i=1; i<leftPts.length; i++) fullCorePath.push(leftPts[i]);
            globalChainPaths.push(fullCorePath);

            if (rightPts.length > 0 && leftPts.length > 0) {
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

                let fullCorePath2 = [];
                for(let i=0; i<rightPts2.length; i++) fullCorePath2.push(rightPts2[i]);
                for(let i=1; i<sPts2.length; i++) fullCorePath2.push(sPts2[i]);
                for(let i=1; i<leftPts2.length; i++) fullCorePath2.push(leftPts2[i]);
                globalChainPaths.push(fullCorePath2);

                endpointsList.push({ 
                    entry: rightPts2[0].clone(), entryTangent: rightTube2.userData.curve.getTangentAt(0),
                    exit: leftPts2[leftPts2.length - 1].clone(), exitTangent: leftTube2.userData.curve.getTangentAt(1)
                });
            }
        }

        updateGlassChipHousing(endpointsList);
    }
}

function animate() {
    requestAnimationFrame(animate);
    const delta = pulseClock.getDelta();

    if (isAnimationActive) {
        const animSpeedRangeEl = document.getElementById('animSpeedRange');
        let speedMultiplier = animSpeedRangeEl ? (parseFloat(animSpeedRangeEl.value) || 1.0) : 1.0;

        // Спавн вытянутых лазерных лучей-сгустков
        if (Math.random() < 0.45 && globalChainPaths.length > 0) {
            const targetPath = globalChainPaths[Math.floor(Math.random() * globalChainPaths.length)];
            if (targetPath && targetPath.length > 5) {
                const pulseGeo = new THREE.CylinderGeometry(0.8, 1.8, 8.0, 8);
                pulseGeo.rotateX(Math.PI / 2);
                
                const pulseMat = new THREE.MeshBasicMaterial({ 
                    color: 0x00ffcc, 
                    transparent: true, 
                    opacity: 0.9,
                    blending: THREE.AdditiveBlending 
                });
                const pulseMesh = new THREE.Mesh(pulseGeo, pulseMat);

                pulseMesh.userData = {
                    points: targetPath,
                    progress: 0,
                    speed: (0.5 + Math.random() * 0.4) * speedMultiplier
                };
                spiralGroup.add(pulseMesh);
                laserPulses.push({ mesh: pulseMesh, parentGroup: spiralGroup });
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
                    
                    const nextPos = pts[highIdx];
                    if (currentPos.distanceTo(nextPos) > 0.001) {
                        pData.mesh.lookAt(nextPos);
                    }
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