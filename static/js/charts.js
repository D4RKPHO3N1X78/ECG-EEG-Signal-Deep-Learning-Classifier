// Chart.js helper module for ECG & EEG Deep Learning Studio

let waveformChart = null;
let fftChart = null;
let probChart = null;
let trainingChart = null;

function initWaveformChart(canvasId, rawSignal, processedSignal) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const labels = Array.from({ length: rawSignal.length }, (_, i) => (i / 250).toFixed(2) + 's');
    
    if (waveformChart) {
        waveformChart.destroy();
    }
    
    waveformChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Raw Input Signal',
                    data: rawSignal,
                    borderColor: 'rgba(156, 163, 175, 0.4)',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Preprocessed & Filtered Signal',
                    data: processedSignal,
                    borderColor: '#00f2fe',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            }
        }
    });
}

function initFFTChart(canvasId, freqs, vals) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    if (fftChart) {
        fftChart.destroy();
    }
    
    fftChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: freqs.map(f => f.toFixed(1) + 'Hz'),
            datasets: [{
                label: 'Spectral Power (PSD)',
                data: vals,
                backgroundColor: 'rgba(138, 43, 226, 0.6)',
                borderColor: '#8a2be2',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af', maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function initProbabilitiesChart(canvasId, probDict) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const labels = Object.keys(probDict);
    const dataVals = Object.values(probDict).map(v => (v * 100).toFixed(2));
    
    if (probChart) {
        probChart.destroy();
    }
    
    probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Softmax Probability (%)',
                data: dataVals,
                backgroundColor: [
                    'rgba(0, 242, 254, 0.7)',
                    'rgba(79, 172, 254, 0.7)',
                    'rgba(138, 43, 226, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(255, 75, 92, 0.7)'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', callback: v => v + '%' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f3f4f6', font: { size: 11 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function initTrainingChart(canvasId, history) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.epoch.map(e => 'Epoch ' + e),
            datasets: [
                {
                    label: 'Train Loss',
                    data: history.train_loss,
                    borderColor: '#ff4b5c',
                    borderWidth: 2,
                    tension: 0.2
                },
                {
                    label: 'Val Loss',
                    data: history.val_loss,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    tension: 0.2
                },
                {
                    label: 'Val Accuracy',
                    data: history.val_acc,
                    borderColor: '#10b981',
                    borderWidth: 2,
                    tension: 0.2
                },
                {
                    label: 'Val F1-Score',
                    data: history.val_f1,
                    borderColor: '#00f2fe',
                    borderWidth: 2,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            }
        }
    });
}
