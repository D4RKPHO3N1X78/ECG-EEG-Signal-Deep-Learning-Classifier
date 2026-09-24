// App State
let currentModality = 'ecg'; // 'ecg' or 'eeg'
let samplesData = { ecg: {}, eeg: {} };
let activeSignal = [];
let lastPrediction = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    setupTabNavigation();
    setupModalityToggle();
    fetchSystemStatus();
    loadSampleDatasets();

    // Event Listeners
    document.getElementById('btn-load-sample').addEventListener('click', loadSelectedSample);
    document.getElementById('btn-run-preprocess').addEventListener('click', runPreprocessing);
    document.getElementById('btn-run-inference').addEventListener('click', runModelInference);
    document.getElementById('btn-start-training').addEventListener('click', startModelTraining);
    document.getElementById('btn-copy-report').addEventListener('click', copyReportToClipboard);
    document.getElementById('select-condition').addEventListener('change', loadSelectedSample);
});

function setupTabNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPages = document.querySelectorAll('.tab-page');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');

            navItems.forEach(n => n.classList.remove('active'));
            tabPages.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            // Title updates
            if (targetTab === 'tab-inspector') {
                document.getElementById('page-title').innerText = 'ECG & EEG Bio-Signal Processing Inspector';
                document.getElementById('page-subtitle').innerText = 'Interactive signal denoising, spectral analysis, and multi-lead wave inspection';
            } else if (targetTab === 'tab-inference') {
                document.getElementById('page-title').innerText = 'Deep Learning Diagnostic Classifier Studio';
                document.getElementById('page-subtitle').innerText = 'Real-time multi-class abnormality prediction & structured clinical reporting';
            } else if (targetTab === 'tab-training') {
                document.getElementById('page-title').innerText = 'PyTorch Model Training & Fine-Tuning Studio';
                document.getElementById('page-subtitle').innerText = 'Train 1D ResNet-BiLSTM & EEGNet models on clinical dataset benchmarks';
            } else if (targetTab === 'tab-metrics') {
                document.getElementById('page-title').innerText = 'Strict Evaluation Metrics Dashboard';
                document.getElementById('page-subtitle').innerText = 'Validate accuracy, sensitivity, specificity, F1-scores, and confusion matrices';
            }
        });
    });
}

function setupModalityToggle() {
    const btnEcg = document.getElementById('btn-modality-ecg');
    const btnEeg = document.getElementById('btn-modality-eeg');

    btnEcg.addEventListener('click', () => {
        if (currentModality !== 'ecg') {
            currentModality = 'ecg';
            btnEcg.classList.add('active');
            btnEeg.classList.remove('active');
            document.getElementById('train-arch-name').value = '1D-ResNet + BiLSTM + Attention';
            updateConditionSelectOptions();
            loadSelectedSample();
        }
    });

    btnEeg.addEventListener('click', () => {
        if (currentModality !== 'eeg') {
            currentModality = 'eeg';
            btnEeg.classList.add('active');
            btnEcg.classList.remove('active');
            document.getElementById('train-arch-name').value = 'EEGNet + Spatial Conv + Transformer';
            updateConditionSelectOptions();
            loadSelectedSample();
        }
    });
}

async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('device-status').innerText = `PyTorch (${data.device.toUpperCase()})`;
    } catch (e) {
        console.error('Failed to fetch status:', e);
    }
}

async function loadSampleDatasets() {
    try {
        const res = await fetch('/api/samples');
        samplesData = await res.json();
        updateConditionSelectOptions();
        loadSelectedSample();
    } catch (e) {
        console.error('Failed to load samples:', e);
    }
}

function updateConditionSelectOptions() {
    const select = document.getElementById('select-condition');
    select.innerHTML = '';
    const pool = samplesData[currentModality] || {};

    Object.keys(pool).forEach(cond => {
        const opt = document.createElement('option');
        opt.value = cond;
        opt.innerText = cond;
        select.appendChild(opt);
    });
}

function loadSelectedSample() {
    const select = document.getElementById('select-condition');
    const cond = select.value;
    const pool = samplesData[currentModality] || {};

    if (pool[cond]) {
        activeSignal = pool[cond];
        runPreprocessing();
    }
}

async function runPreprocessing() {
    if (!activeSignal || activeSignal.length === 0) return;

    const payload = {
        signal: activeSignal,
        type: currentModality,
        fs: 250.0,
        baseline: document.getElementById('chk-baseline').checked,
        bandpass: document.getElementById('chk-bandpass').checked,
        notch: document.getElementById('chk-notch').checked,
        denoise: document.getElementById('chk-denoise').checked
    };

    try {
        const res = await fetch('/api/preprocess', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Single lead signal array for primary chart display
        const rawArr = Array.isArray(data.raw[0]) ? data.raw[0] : data.raw;
        const procArr = Array.isArray(data.processed[0]) ? data.processed[0] : data.processed;

        initWaveformChart('chart-waveform', rawArr, procArr);
        initFFTChart('chart-fft', data.fft_freqs, data.fft_vals);
    } catch (e) {
        console.error('Preprocessing failed:', e);
    }
}

async function runModelInference() {
    if (!activeSignal || activeSignal.length === 0) return;

    const pid = document.getElementById('input-patient-id').value || 'PT-99482';
    const payload = {
        signal: activeSignal,
        type: currentModality,
        fs: 250.0,
        patient_id: pid
    };

    const btn = document.getElementById('btn-run-inference');
    btn.innerText = '⏳ Running PyTorch Inference...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        lastPrediction = data;

        // Update Diagnosis Result Card
        document.getElementById('res-diag-title').innerText = data.diagnosis;
        const confPct = (data.confidence * 100).toFixed(1) + '%';
        document.getElementById('res-confidence-val').innerText = confPct;
        document.getElementById('res-meter-fill').style.width = confPct;

        // Render Probability Chart
        initProbabilitiesChart('chart-probabilities', data.probabilities);

        // Update Report Output Box
        document.getElementById('report-output').innerText = data.report_markdown;

        // Update Features Container
        renderFeaturesContainer(data.features);
    } catch (e) {
        console.error('Prediction failed:', e);
    } finally {
        btn.innerText = '🧠 Run Deep Learning Classification';
        btn.disabled = false;
    }
}

function renderFeaturesContainer(features) {
    const container = document.getElementById('features-container');
    container.innerHTML = '';

    Object.entries(features).forEach(([key, val]) => {
        const box = document.createElement('div');
        box.className = 'feature-box';
        box.innerHTML = `
            <span>${key.replace(/_/g, ' ').toUpperCase()}</span>
            <h4>${val}</h4>
        `;
        container.appendChild(box);
    });
}

async function startModelTraining() {
    const epochs = parseInt(document.getElementById('train-epochs').value) || 10;
    const btn = document.getElementById('btn-start-training');

    btn.innerText = `🔥 Training Model (${epochs} Epochs)...`;
    btn.disabled = true;

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: currentModality, epochs: epochs })
        });
        const data = await res.json();

        // Render Training Curves
        initTrainingChart('chart-training-curves', data.history);

        // Render Strict Evaluation Metrics
        renderEvaluationMetrics(data.evaluation);
    } catch (e) {
        console.error('Training failed:', e);
    } finally {
        btn.innerText = '🔥 Start Deep Learning Training Job';
        btn.disabled = false;
    }
}

function renderEvaluationMetrics(evalData) {
    const overall = evalData.overall;
    document.getElementById('metric-acc').innerText = (overall.accuracy * 100).toFixed(1) + '%';
    document.getElementById('metric-bal-acc').innerText = (overall.balanced_accuracy * 100).toFixed(1) + '%';
    document.getElementById('metric-f1').innerText = overall.macro_f1_score.toFixed(3);
    document.getElementById('metric-sens').innerText = overall.macro_recall_sensitivity.toFixed(3);
    document.getElementById('metric-spec').innerText = overall.macro_specificity.toFixed(3);
    document.getElementById('metric-auc').innerText = overall.roc_auc_score.toFixed(3);

    // Render Confusion Matrix Table
    const cmContainer = document.getElementById('confusion-matrix-container');
    const classNames = evalData.class_names;
    const cmNorm = evalData.confusion_matrix_normalized;

    let tableHtml = '<table class="matrix-table"><thead><tr><th>True \\ Pred</th>';
    classNames.forEach(name => {
        tableHtml += `<th>${name.split(' ')[0]}</th>`;
    });
    tableHtml += '</tr></thead><tbody>';

    cmNorm.forEach((row, i) => {
        tableHtml += `<tr><th>${classNames[i].split(' ')[0]}</th>`;
        row.forEach(val => {
            const intensity = Math.min(1.0, val);
            const bg = `rgba(0, 242, 254, ${intensity * 0.4})`;
            tableHtml += `<td style="background:${bg}">${(val * 100).toFixed(0)}%</td>`;
        });
        tableHtml += '</tr>';
    });
    tableHtml += 'tbody></table>';
    cmContainer.innerHTML = tableHtml;

    // Render Per-Class Table
    const tbody = document.querySelector('#table-per-class tbody');
    tbody.innerHTML = '';

    Object.entries(evalData.per_class).forEach(([cls, metrics]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${cls}</strong></td>
            <td>${(metrics.Precision * 100).toFixed(1)}%</td>
            <td>${(metrics.Sensitivity_Recall * 100).toFixed(1)}%</td>
            <td>${(metrics.Specificity * 100).toFixed(1)}%</td>
            <td>${metrics.F1_Score.toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function copyReportToClipboard() {
    const reportText = document.getElementById('report-output').innerText;
    navigator.clipboard.writeText(reportText);
    const btn = document.getElementById('btn-copy-report');
    btn.innerText = '✅ Copied!';
    setTimeout(() => { btn.innerText = '📋 Copy Report Markdown'; }, 2000);
}
