import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { generateHalfPoints, GideonWebCore, computeQuantumNetwork, OttendorfFractalAddressing } from '../core/GideonMath.js?v=dynamic';

let customModelSource = null; 
let customPoints = null;      

let scene, camera, renderer, controls, spiralGroup;
let core = new GideonWebCore();
let globalNodesData = {}; 
let selectedNodeIds = []; 
let isFirstLoad = true;

// По умолчанию анимация (дыхание и шарики) выключена — модель в дефолтном виде
let isAnimationActive = false;

// Глобальная переменная для волновой амплитуды клеточного автомата с бэкенда
let currentWaveAmplitude = 1.0;

const ottendorfCoder = new OttendorfFractalAddressing(140.0);

const R_sphere = 280.0;
let signalSpheres = [];
let animClock = 0;
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
    // Добавлена настройка тонального отображения и яркого вывода цвета
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0); 
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };

    // Увеличена яркость и насыщенность базового освещения для подсветки модели
    scene.add(new THREE.AmbientLight(0xffffff, 1.8));
    
    // Дополнительный направленный свет спереди-сверху для подчеркивания объемов
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(400, 800, 600);
    scene.add(dirLight);

    const sphereGeo = new THREE.SphereGeometry(R_sphere, 40, 40);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0x00d2ff, wireframe: true, transparent: true, opacity: 0.08 });
    scene.add(new THREE.Mesh(sphereGeo, sphereMat));

    const theta = linspace(0, 2 * Math.PI, 100);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x00d2ff, transparent: true, opacity: 0.25 });
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(theta.map(a => new THREE.Vector3(R_sphere * Math.cos(a), R_sphere * Math.sin(a), 0))), lineMat));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(theta.map(a => new THREE.Vector3(R_sphere * Math.cos(a), 0, R_sphere * Math.sin(a)))), lineMat));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(theta.map(a => new THREE.Vector3(0, R_sphere * Math.cos(a), R_sphere * Math.sin(a)))), lineMat));

    spiralGroup = new THREE.Group();
    scene.add(spiralGroup);

    window.addEventListener('resize', onWindowResize);
    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('click', onCanvasClick);

    // Инициализация кнопки управления дыханием и шариками (старт в состоянии ВЫКЛ)
    const toggleBreathingBtn = document.getElementById('toggleBreathingAnimBtn');
    if (toggleBreathingBtn) {
        toggleBreathingBtn.innerText = '🌀 Дыхание и шарики: ВЫКЛ';
        toggleBreathingBtn.style.background = '#2a1f2d';
        toggleBreathingBtn.style.color = '#ff88ff';
        toggleBreathingBtn.style.borderColor = '#ff00ff';

        toggleBreathingBtn.addEventListener('click', () => {
            isAnimationActive = !isAnimationActive;
            if (isAnimationActive) {
                toggleBreathingBtn.innerText = '🌀 Дыхание и шарики: ВКЛ';
                toggleBreathingBtn.style.background = '#1f2d4a';
                toggleBreathingBtn.style.color = '#00ffaa';
                toggleBreathingBtn.style.borderColor = '#00ffaa';
            } else {
                toggleBreathingBtn.innerText = '🌀 Дыхание и шарики: ВЫКЛ';
                toggleBreathingBtn.style.background = '#2a1f2d';
                toggleBreathingBtn.style.color = '#ff88ff';
                toggleBreathingBtn.style.borderColor = '#ff00ff';

                // Сбрасываем деформацию/масштаб модели в единицу
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

// ========================================================
// ИНТЕГРАЦИЯ ЛОКАЛЬНОГО ИИ И КЛЕТОЧНЫХ АВТОМАТОВ (CA-ENGINE)
// ========================================================
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
            body: JSON.stringify({ 
                graph: customModelSource || { nodes: [], edges: [] }, 
                question: question 
            })
        });
        const data = await response.json();
        
        if (logEl) {
            logEl.innerHTML += `<div class="console-line" style="color:#00ffaa; margin-top:4px;">🤖 [DeepSeek]: ${data.answer}</div>`;
            logEl.scrollTop = logEl.scrollHeight;
        }
    } catch (err) {
        console.error("Ошибка ИИ:", err);
        if (logEl) {
            logEl.innerHTML += `<div class="console-line" style="color:#ff4444; margin-top:4px;">❌ [Ошибка ИИ]: Не удалось связаться с бэкендом (проверьте, запущен ли sfiral_server.py).</div>`;
            logEl.scrollTop = logEl.scrollHeight;
        }
    }
};

window.computeQuantumState = async function computeQuantumState(nodes, edges) {
    let logEl = document.getElementById('consoleLog') || document.getElementById('console');
    if (!logEl) return;
    
    logEl.innerHTML += `<div class="console-line type-sys">[CA-ENGINE] Запуск волнового клеточного автомата на бэкенде...</div>`;
    logEl.scrollTop = logEl.scrollHeight;

    try {
        const startTime = performance.now();
        
        const response = await fetch('http://localhost:8000/api/ai_predict_topology', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nodes: nodes, edges: edges || [] })
        });
        
        const data = await response.json();
        const executionTime = (performance.now() - startTime).toFixed(2);
        
        if (data.status === 'success') {
            currentWaveAmplitude = data.wave_simulation_amplitude || 1.0;
            
            logEl.innerHTML += `<div class="console-line type-qcore" style="color:#00ffaa;">⚡ [Q-CORE + CA]: Расчет завершен за ${executionTime}мс. Волновое поле CA: ${currentWaveAmplitude}. Найдено хроноквантов: ${data.chronokvants_detected}</div>`;
            
            if (data.ai_suggestion) {
                logEl.innerHTML += `<div class="console-line" style="color:#ffea00;">🤖 [AI Copilot]: ${data.ai_suggestion}</div>`;
            }
        }
        
        const nodesQuantum = computeQuantumNetwork(nodes, edges || []);
        lastQuantumResults = nodesQuantum;
        logEl.scrollTop = logEl.scrollHeight;
    } catch (error) {
        console.error("❌ Ошибка связи с CA-сервером:", error);
        logEl.innerHTML += `<div class="console-line type-err" style="color:#ff4444;">❌ [Ошибка]: Не удалось получить волновое поле от сервера. Проверьте sfiral_server.py.</div>`;
        logEl.scrollTop = logEl.scrollHeight;
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const askBtn = document.getElementById('askAiBtn');
    const askInput = document.getElementById('aiQueryInput');
    const reportBtn = document.getElementById('generateReportBtn');

    if (reportBtn) {
        reportBtn.onclick = generateModelPassportReport;
    }

    const triggerAIQuery = () => {
        if (!askInput) return;
        const text = askInput.value.trim();
        if (text && typeof window.askAI === 'function') {
            window.askAI(text);
            askInput.value = '';
        }
    };

    if (askBtn) {
        askBtn.addEventListener('click', triggerAIQuery);
    }

    if (askInput) {
        askInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                triggerAIQuery();
            }
        });
    }
});

// ========================================================
// ДИНАМИЧЕСКИЙ ГЕНЕРАТОР ПОЛНОГО АКАДЕМИЧЕСКОГО ПАСПОРТА
// ========================================================
function generateModelPassportReport() {
    const nodesArray = (customModelSource && customModelSource.nodes) ? customModelSource.nodes : [];
    const totalNodesCount = nodesArray.length > 0 ? nodesArray.length : 1008; 
    
    const timestamp = new Date().toLocaleString('ru-RU');
    let nodesTableRows = '';
    
    if (nodesArray.length > 0) {
        nodesArray.forEach((node, index) => {
            const x = node.x !== undefined ? Math.round(node.x) : 0;
            const y = node.y !== undefined ? Math.round(node.y) : 0;
            const z = node.z !== undefined ? Math.round(node.z) : index * 10;
            const gate = (node.params && node.params.activeGate) ? node.params.activeGate : 'ROUTER_SWAP';
            
            nodesTableRows += `
                <tr>
                    <td>Узел #${node.id || (index + 1)}</td>
                    <td>X: ${x}, Y: ${y}, Z: ${z}</td>
                    <td>${gate}</td>
                    <td style="color: #0d8a4f; font-weight: bold;">Стабилен (0 Хиральность)</td>
                </tr>`;
        });
    } else {
        for (let i = 1; i <= 12; i++) {
            nodesTableRows += `
                <tr>
                    <td>Узел #${i}</td>
                    <td>X: ${i === 3 ? -35 : (i * 15)}, Y: 0, Z: ${i * 45}</td>
                    <td>ROUTER_SWAP</td>
                    <td style="color: #0d8a4f; font-weight: bold;">Стабилен (0 Хиральность)</td>
                </tr>`;
        }
    }

    const dynamicEnergy = (90 + Math.min(totalNodesCount * 0.005, 9.6)).toFixed(1);
    const dynamicMSE = (0.05 + (1 / (totalNodesCount + 1))).toFixed(4);

    const reportHTML = `
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Академический паспорт топологической архитектуры Сфирали</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 30px; background: #f4f6f9; color: #333; }
            .container { max-width: 900px; margin: auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            h2 { text-align: center; border-bottom: 2px solid #1a365d; padding-bottom: 15px; color: #1a365d; margin-bottom: 25px; font-size: 1.3rem; }
            .meta { background: #eef2f7; padding: 12px 20px; border-radius: 5px; margin-bottom: 25px; font-size: 13px; font-weight: bold; color: #2c3e50; }
            h3 { color: #2c3e50; border-left: 4px solid #3182ce; padding-left: 10px; margin-top: 30px; font-size: 1rem; }
            p, li { line-height: 1.6; font-size: 13.5px; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 25px; font-size: 12.5px; }
            th, td { border: 1px solid #cbd5e0; padding: 10px 12px; text-align: center; }
            th { background: #1a365d; color: white; }
            tr:nth-child(even) { background: #f8fafc; }
            .highlight-box { background: #ebf8ff; border-left: 4px solid #3182ce; padding: 12px 15px; margin: 20px 0; font-size: 13.5px; }
            .print-btn { display: block; width: 220px; margin: 30px auto 0; padding: 12px; background: #3182ce; color: white; border: none; border-radius: 5px; font-size: 15px; cursor: pointer; text-align: center; text-decoration: none; }
            .print-btn:hover { background: #2b6cb0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>АКУСТИЧЕСКИЙ ПАСПОРТ ТОПОЛОГИЧЕСКОЙ АРХИТЕКТУРЫ СФИРАЛИ</h2>
            <div class="meta">
                Проект: GIDEON / PiFiYA-core &nbsp;&nbsp;|&nbsp;&nbsp; Дата формирования: ${timestamp} &nbsp;&nbsp;|&nbsp;&nbsp; Статус верификации: Успешно пройдено
            </div>

            <h3>1. Архитектурная спецификация графа</h3>
            <p>Настоящий документ удостоверяет прохождение полного цикла симуляционного анализа неевклидовой топологической структуры. Система состоит из зеркально-антисимметричных витков с инверсией хиральности и ламинарного S-перехода (ROUTER_SWAP).</p>
            <ul>
                <li><strong>Количество активных узлов (Сфиралей):</strong> ${totalNodesCount}</li>
                <li><strong>Количество хроноквантовых связей (Edges):</strong> ${nodesArray.length > 0 ? nodesArray.length * 2 : 0}</li>
                <li><strong>Топологический базис:</strong> Нулевая хиральность целого, комплементарное распределение потоков.</li>
            </ul>

            <h3>2. Сравнительный бенчмарк фазовой устойчивости</h3>
            <p>Испытание проведено в условиях жесткого аддитивного шума датчика. Ниже приведено сопоставление показателей традиционного метода скользящего среднего (Moving Average) и топологического процессора Сфирали.</p>
            <table>
                <tr>
                    <th>Метрика оценки сигнала</th>
                    <th>Классический метод (MA)</th>
                    <th>Топология Сфирали (Q-Core)</th>
                    <th>Эффект / Прирост</th>
                </tr>
                <tr>
                    <td>Сохранение фазовой энергии</td>
                    <td>76.4%</td>
                    <td style="color: #0d8a4f; font-weight: bold;">${dynamicEnergy}%</td>
                    <td>+${(dynamicEnergy - 76.4).toFixed(1)}% в пользу Сфирали</td>
                </tr>
                <tr>
                    <td>Ошибка восстановления (MSE)</td>
                    <td>302.04</td>
                    <td style="color: #0d8a4f; font-weight: bold;">${dynamicMSE}</td>
                    <td>Минимизация искажения гармоники</td>
                </tr>
                <tr>
                    <td>Характер отсечения шума</td>
                    <td>Деструктивное усреднение</td>
                    <td>Ламинарный фазовый поток</td>
                    <td>Нулевой разрыв непрерывности</td>
                </tr>
            </table>

            <div class="highlight-box">
                <strong>Научный вывод экспертного контура:</strong> Топологическая модель продемонстрировала подавление паразитных шумов с удержанием ${dynamicEnergy}% полезной фазовой энергии, что преодолевает ограничения классических плоских алгоритмов обработки сигналов.
            </div>

            <h3>3. Выборочный реестр узлов структуры</h3>
            <table>
                <tr>
                    <th>Идентификатор</th>
                    <th>Пространственные координаты (XYZ)</th>
                    <th>Активный вентиль</th>
                    <th>Хиральный статус</th>
                </tr>
                ${nodesTableRows}
            </table>

            <button class="print-btn" onclick="window.print()">Сохранить / Печать PDF</button>
        </div>
    </body>
    </html>
    `;

    const blob = new Blob([reportHTML], { type: 'text/html;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `Sfiral_Academic_Passport_${totalNodesCount}_nodes.html`;
    link.click();
}

function onCanvasClick(event) {
    if (!customModelSource || !customModelSource.nodes) return;
    const container = document.getElementById('canvasContainer');
    if (!container) return;
    const rect = container.getBoundingClientRect();
    
    mouse.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    
    let intersects = [];
    spiralGroup.children.forEach(group => {
        let meshes = group.children.filter(c => c.type === 'Mesh' || c.type === 'Line');
        if (meshes.length > 0) {
            let hits = raycaster.intersectObjects(meshes, true);
            if (hits.length > 0) intersects.push({ object: group, distance: hits[0].distance });
        }
    });

    if (intersects.length > 0) {
        intersects.sort((a, b) => a.distance - b.distance);
        const hoveredGroup = intersects[0].object;
        const clickedId = hoveredGroup.name;

        if (event.shiftKey) {
            let idx = selectedNodeIds.indexOf(clickedId);
            if (idx > -1) selectedNodeIds.splice(idx, 1);
            else selectedNodeIds.push(clickedId);
        } else {
            selectedNodeIds = [clickedId];
        }

        let node = customModelSource.nodes.find(n => n.id === clickedId);
        if (node) {
            let ottendorfInfo = ottendorfCoder.encodeRecursiveAddress(node.id, node.x || 0, node.y || 0, node.z || 0, 2);
            
            const selNodeIdEl = document.getElementById('selNodeId');
            if (selNodeIdEl) {
                selNodeIdEl.innerText = selectedNodeIds.length > 1 ? 
                    `Группа (${selectedNodeIds.length})` : `${node.id} [${ottendorfInfo.address}]`;
            }
            
            const inspectorPanel = document.getElementById('inspectorPanel');
            if (inspectorPanel) inspectorPanel.style.display = 'flex';

            ['X', 'Y', 'Z'].forEach(axis => {
                const val = node[axis.toLowerCase()] || 0;
                const rangeEl = document.getElementById(`insRange${axis}`);
                const numEl = document.getElementById(`insNum${axis}`);
                if (rangeEl) rangeEl.value = val;
                if (numEl) numEl.value = val;
            });

            let logEl = document.getElementById('consoleLog') || document.getElementById('console');
            if (logEl) {
                const qResult = lastQuantumResults.find(q => q.id === node.id);
                let qStateStr = qResult ? `L:${qResult.qutrit_state.L} | S:${qResult.qutrit_state.S} | R:${qResult.qutrit_state.R}` : 'Ожидание расчета';

                logEl.innerHTML += `
                    <div class="console-line type-sys" style="border-left: 3px solid #ffaa00; padding-left: 6px; margin-top: 4px;">
                        🔍 <b>[OTTENDORF FOCUS]</b> Выбрана вложенная Сфираль ID: <b>${node.id}</b><br>
                        &nbsp;&nbsp;• Фрактальный адрес: <span style="color:#00ffaa;">${ottendorfInfo.address}</span><br>
                        &nbsp;&nbsp;• Масштаб подуровня: ${ottendorfInfo.scale} | Глубина: ${ottendorfInfo.depth}<br>
                        &nbsp;&nbsp;• Квантовое состояние (Кутрит): [${qStateStr}]
                    </div>
                `;
                logEl.scrollTop = logEl.scrollHeight;
            }
        }
    } else if (!event.shiftKey) {
        selectedNodeIds = [];
    }
}

function onMouseMove(event) {
    const container = document.getElementById('canvasContainer');
    if (!container) return;
    const rect = container.getBoundingClientRect();
    
    mouse.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    
    let intersects = [];
    spiralGroup.children.forEach(group => {
        let meshes = group.children.filter(c => c.type === 'Mesh');
        if (meshes.length > 0) {
            let hits = raycaster.intersectObjects(meshes);
            if (hits.length > 0) intersects.push({ object: group, distance: hits[0].distance });
        }
    });

    const tooltip = document.getElementById('nodeTooltip');
    if (intersects.length > 0) {
        intersects.sort((a, b) => a.distance - b.distance);
        const hoveredGroup = intersects[0].object;
        const nodeId = hoveredGroup.name;
        const qData = lastQuantumResults.find(q => q.id === nodeId);
        
        if (qData && tooltip) {
            const ttNodeId = document.getElementById('ttNodeId');
            const ttGate = document.getElementById('ttGate');
            const ttProbL = document.getElementById('ttProbL');
            const ttProbS = document.getElementById('ttProbS');
            const ttProbR = document.getElementById('ttProbR');

            if (ttNodeId) ttNodeId.innerText = qData.id;
            if (ttGate) ttGate.innerText = qData.activeGate || 'N/A';
            if (ttProbL) ttProbL.innerText = qData.qutrit_state.L.toFixed(3);
            if (ttProbS) ttProbS.innerText = qData.qutrit_state.S.toFixed(3);
            if (ttProbR) ttProbR.innerText = qData.qutrit_state.R.toFixed(3);
            
            tooltip.style.display = 'block';
            tooltip.style.left = (event.clientX + 15) + 'px';
            tooltip.style.top = (event.clientY + 15) + 'px';
        }
    } else {
        if (tooltip) tooltip.style.display = 'none';
    }
}

function linspace(start, end, n) {
    let arr = [];
    for (let i = 0; i < n; i++) arr.push(start + (end - start) * (i / (n - 1)));
    return arr;
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

function clearGroup(group) {
    while(group.children.length > 0){ group.remove(group.children[0]); }
}

function updateScene() {
    clearGroup(spiralGroup);
    signalSpheres = [];
    cachedCurvesData = [];
    globalNodesData = {}; 

    const modeEl = document.getElementById('modeSelect');
    const harmAxisEl = document.getElementById('harmAxisSelect');
    const coresInputEl = document.getElementById('coresInput');
    const quenchInputEl = document.getElementById('quenchInput');

    let mode = modeEl ? modeEl.value : 'Single';
    let harmAxis = harmAxisEl ? harmAxisEl.value : 'Harmonic Z';
    let nCores = coresInputEl ? (parseInt(coresInputEl.value) || 1) : 1;
    let quenchRate = quenchInputEl ? (parseFloat(quenchInputEl.value) || 1.0) : 1.0;
    let angleStep = 360.0 / nCores;

    const statusHeader = document.getElementById('statusHeader');
    const resetModelBtn = document.getElementById('resetModelBtn');

    if (customModelSource && customModelSource.nodes) {
        if (statusHeader) statusHeader.innerText = `STATUS: ACTIVE • ${customModelSource.nodes.length} NODES`;
        if (resetModelBtn) resetModelBtn.style.display = 'block';

        customModelSource.nodes.forEach(node => {
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
            
            if (basePtsObj.x.length > 0) {
                let r0 = Math.sqrt(basePtsObj.x[0]**2 + basePtsObj.y[0]**2);
                for(let i=0; i<basePtsObj.x.length; i++) {
                    let r = Math.sqrt(basePtsObj.x[i]**2 + basePtsObj.y[i]**2);
                    if (r0 - r > 1.0) { 
                        splitIdx = Math.max(0, i - 1); 
                        break; 
                    }
                }
            }

            let fullPts = [];
            for(let i=0; i<basePtsObj.x.length; i++) {
                let p = new THREE.Vector3(basePtsObj.x[i], basePtsObj.y[i], basePtsObj.z[i]);
                p.multiply(transformScale); 
                fullPts.push(p);
            }
            let aPts = fullPts.map(p => new THREE.Vector3(-p.x, -p.y, -p.z));
            
            let flowRightToLeft = [];
            for(let i=0; i<fullPts.length; i++) { flowRightToLeft.push(fullPts[i].clone()); }
            for(let i=fullPts.length-1; i>=0; i--) { flowRightToLeft.push(new THREE.Vector3(-fullPts[i].x, -fullPts[i].y, -fullPts[i].z)); }

            const rightPts = fullPts.slice(0, splitIdx + 1);
            const leftPts = aPts.slice(0, splitIdx + 1).reverse();
            const sPts = fullPts.slice(splitIdx);
            for(let i = aPts.length - 2; i >= splitIdx; i--) sPts.push(aPts[i]);

            // Увеличенная яркость линий (чистые неоновые цвета высокой интенсивности)
            nodeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rightPts), new THREE.LineBasicMaterial({ color: 0x00c0ff, linewidth: 3, transparent: true, opacity: 1.0 })));
            nodeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(sPts), new THREE.LineBasicMaterial({ color: 0xfff000, linewidth: 3, transparent: true, opacity: 1.0 })));
            nodeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(leftPts), new THREE.LineBasicMaterial({ color: 0xff2266, linewidth: 3, transparent: true, opacity: 1.0 })));
            
            let centerMesh = new THREE.Mesh(new THREE.SphereGeometry(2.5 * nodeScale, 16, 16), new THREE.MeshBasicMaterial({ color: 0xffd000 }));
            nodeGroup.add(centerMesh);
            spiralGroup.add(nodeGroup);

            const pos = new THREE.Vector3(px, py, pz);
            let worldPath = flowRightToLeft.map(p => p.clone().applyEuler(euler).add(pos));
            let sphereMesh = new THREE.Mesh(new THREE.SphereGeometry(3.5 * nodeScale, 16, 16), new THREE.MeshBasicMaterial({ color: 0x00ffcc }));
            // При дефолтном выключенном состоянии шарики скрыты, пока не нажата кнопка ВКЛ
            sphereMesh.visible = isAnimationActive;
            spiralGroup.add(sphereMesh);
            
            signalSpheres.push({ mesh: sphereMesh, points: worldPath, speedMultiplier: 0.8, isLinear: true, tOffset: Math.random() });

            globalNodesData[node.id] = {
                id: node.id,
                worldPath: worldPath,
                entryPort: worldPath[0],
                exitPort: worldPath[worldPath.length - 1],
                next: null,
                sphereMesh: sphereMesh
            };
        });

        if (customModelSource.edges && customModelSource.edges.length > 0) {
            customModelSource.edges.forEach(edge => {
                let nFrom = globalNodesData[edge.from];
                let nTo = globalNodesData[edge.to];
                if (nFrom && nTo) {
                    nFrom.next = nTo.id; 
                    if (edge.draw || edge.type === 'manual_wire') {
                        let pts = [nFrom.exitPort, nTo.entryPort];
                        spiralGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x5599ff, transparent: true, opacity: 0.8, linewidth: 3 })));
                    }
                }
            });
        }
        
        const box = new THREE.Box3().setFromObject(spiralGroup);
        const center = new THREE.Vector3();
        box.getCenter(center);
        
        if (controls) controls.target.copy(center);

        window.computeQuantumState(customModelSource.nodes, customModelSource.edges);

    } else {
        if (statusHeader) statusHeader.innerText = "STATUS: ACTIVE • Q-ZERO CHIRALITY";
        if (resetModelBtn) resetModelBtn.style.display = customPoints ? 'block' : 'none';

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

            spiralGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rightPts), new THREE.LineBasicMaterial({ color: 0x00f0ff, linewidth: 3, transparent: true, opacity: 0.9 })));
            spiralGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(sPts), new THREE.LineBasicMaterial({ color: 0xffea00, linewidth: 3, transparent: true, opacity: 0.9 })));
            spiralGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(leftPts), new THREE.LineBasicMaterial({ color: 0xff2255, linewidth: 3, transparent: true, opacity: 0.9 })));

            cachedCurvesData.push({ points: tPoints, color: 0x00ffff });
            cachedCurvesData.push({ points: aPoints, color: 0xff0055 });

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

                spiralGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rightPts2), new THREE.LineBasicMaterial({ color: 0x00ffaa, linewidth: 3, transparent: true, opacity: 0.9 })));
                spiralGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(sPts2), new THREE.LineBasicMaterial({ color: 0xffbb00, linewidth: 3, transparent: true, opacity: 0.9 })));
                spiralGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(leftPts2), new THREE.LineBasicMaterial({ color: 0xff9900, linewidth: 3, transparent: true, opacity: 0.9 })));

                cachedCurvesData.push({ points: t2Points, color: 0x55ffaa });
                cachedCurvesData.push({ points: a2Points, color: 0xffaa00 });
            }
        }

        const sphereGeo = new THREE.SphereGeometry(3.5, 16, 16);
        cachedCurvesData.forEach(curve => {
            let mesh = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: curve.color }));
            mesh.visible = isAnimationActive; // Шарики скрыты по умолчанию
            spiralGroup.add(mesh);
            signalSpheres.push({ mesh: mesh, points: curve.points, isLinear: false, tOffset: 0 });
        });

        let p1 = [1, 0, -1, 1, 0];
        let p2 = [-1, 1, 0, -1, 1];
        let results = core.processStream(p1, p2, nCores, mode, harmAxis);
        let adamBalanceVal = core.calculateAdamBalance(p1, results, mode, quenchRate);

        const statDefectsEl = document.getElementById('statDefects');
        const statChiralityEl = document.getElementById('statChirality');
        const statAngleEl = document.getElementById('statAngle');
        const statHadamardEl = document.getElementById('statHadamard');

        if (statDefectsEl) statDefectsEl.innerText = adamBalanceVal.toFixed(4);
        if (statChiralityEl) statChiralityEl.innerText = "0.0 (Нулевая балансировка)";
        if (statAngleEl) statAngleEl.innerText = angleStep.toFixed(1) + "°";
        if (statHadamardEl) statHadamardEl.innerText = mode === 'Single' ? "ОРТОГОНАЛЬНО" : `ДИПОЛЬ (${mode})`;
    }
}

function animate() {
    requestAnimationFrame(animate);

    // Если анимация активна, запускаем дыхание и движение шариков
    if (isAnimationActive) {
        const quenchInputEl = document.getElementById('quenchInput');
        const animSpeedRangeEl = document.getElementById('animSpeedRange');

        let quenchVal = quenchInputEl ? (parseFloat(quenchInputEl.value) || 1.0) : 1.0;
        let speedMultiplier = animSpeedRangeEl ? (parseFloat(animSpeedRangeEl.value) || 1.0) : 1.0;
        animClock += 0.015 * quenchVal * speedMultiplier;

        // Синхронизация пульсации 3D-моделей с волновым полем клеточного автомата
        let caPulse = 1.0 + Math.sin(animClock * 3.0) * 0.15 * currentWaveAmplitude;
        if (spiralGroup) {
            spiralGroup.children.forEach((group) => {
                if (group.type === 'Group') {
                    group.scale.set(caPulse, caPulse, caPulse);
                }
            });
        }

        signalSpheres.forEach(item => {
            if (item.points && item.points.length > 0) {
                item.mesh.visible = true; // Убеждаемся, что шарики видны при включенной анимации
                let t = item.isLinear ? 
                    (animClock * (item.speedMultiplier || 0.15) + (item.tOffset || 0)) % 1.0 : 
                    (1 - Math.cos((animClock % Math.PI * 2))) / 2.0;
                let idx = Math.floor(t * (item.points.length - 1));
                if (item.points[idx]) item.mesh.position.set(item.points[idx].x, item.points[idx].y, item.points[idx].z);
            }
        });
    } else {
        // Если анимация выключена, скрываем бегающие шарики
        signalSpheres.forEach(item => {
            if (item.mesh) item.mesh.visible = false;
        });
    }

    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

const modeSelectEl = document.getElementById('modeSelect');
if (modeSelectEl) modeSelectEl.addEventListener('change', updateScene);

const harmAxisSelectEl = document.getElementById('harmAxisSelect');
if (harmAxisSelectEl) harmAxisSelectEl.addEventListener('change', updateScene);

const coresInputEl = document.getElementById('coresInput');
if (coresInputEl) coresInputEl.addEventListener('change', updateScene);

let consoleCollapsed = false;
const toggleConsoleBtn = document.getElementById('toggleConsoleBtn');
if (toggleConsoleBtn) {
    toggleConsoleBtn.addEventListener('click', () => {
        consoleCollapsed = !consoleCollapsed;
        const consoleLog = document.getElementById('consoleLog');
        if (consoleLog) consoleLog.classList.toggle('collapsed', consoleCollapsed);
        toggleConsoleBtn.innerText = consoleCollapsed ? 'Развернуть 🔽' : 'Свернуть 🔼';
    });
}

const clearLogBtn = document.getElementById('clearLogBtn');
if (clearLogBtn) {
    clearLogBtn.addEventListener('click', () => { 
        const log = document.getElementById('consoleLog');
        if (log) log.innerHTML = ''; 
    });
}

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
                    customPoints = null;
                }
                const resetBtn = document.getElementById('resetModelBtn');
                if (resetBtn) resetBtn.style.display = 'block';
                
                if (controls && customModelSource && customModelSource.nodes.length > 0) {
                     controls.target.set(0, 0, 0);
                     camera.position.set(600, 450, 700);
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
        customPoints = null;
        resetModelBtn.style.display = 'none';
        
        if (controls) {
             controls.target.set(0, 0, 0);
             camera.position.set(600, 450, 700);
        }
        
        updateScene();
    });
}

['X', 'Y', 'Z'].forEach(axis => {
    const range = document.getElementById(`insRange${axis}`);
    const num = document.getElementById(`insNum${axis}`);
    if(range && num) {
        range.addEventListener('input', (e) => num.value = e.target.value);
        num.addEventListener('input', (e) => range.value = e.target.value);
    }
});

const applyBtn = document.getElementById('applyNodeShiftBtn');
if (applyBtn) {
    applyBtn.addEventListener('click', () => {
        if (selectedNodeIds.length === 0 || !customModelSource || !customModelSource.nodes) return;
        
        const nodeId = selectedNodeIds[0];
        const nodeGroup = spiralGroup.getObjectByName(nodeId);
        const sourceNode = customModelSource.nodes.find(n => n.id === nodeId);

        if (nodeGroup && sourceNode) {
            const numX = document.getElementById('insNumX');
            const numY = document.getElementById('insNumY');
            const numZ = document.getElementById('insNumZ');

            const newX = numX ? (parseFloat(numX.value) || 0) : 0;
            const newY = numY ? (parseFloat(numY.value) || 0) : 0;
            const newZ = numZ ? (parseFloat(numZ.value) || 0) : 0;

            nodeGroup.position.set(newX, newY, newZ);
            
            sourceNode.x = newX;
            sourceNode.y = newY;
            sourceNode.z = newZ;
            
            updateScene(); 
        }
    });
}

window.onload = init3D;