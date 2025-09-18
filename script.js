// --- DOM Element Selection ---
const heatmapSelect = document.getElementById('heatmapSelect');
const heatmapTitle = document.getElementById('heatmapTitle');
const yAxisLabel = document.getElementById('y-axis-label');
const xAxisLabel = document.getElementById('x-axis-label');
const heatmapContainer = document.getElementById('heatmap-container');
const heatmapCanvas = document.getElementById('heatmapCanvas');
const legendCanvas = document.getElementById('legendCanvas');
const heatmapCtx = heatmapCanvas.getContext('2d');
const legendCtx = legendCanvas.getContext('2d');

// --- State and Constants ---
let allHeatmaps = {};
const COLORS = [
    { val: 0, r: 26, g: 35, b: 126 },
    { val: 0.15, r: 63, g: 81, b: 181 },
    { val: 0.3, r: 33, g: 150, b: 243 },
    { val: 0.5, r: 76, g: 175, b: 80 },
    { val: 0.65, r: 255, g: 235, b: 59 },
    { val: 0.8, r: 255, g: 152, b: 0 },
    { val: 0.9, r: 244, g: 67, b: 54 },
    { val: 1.0, r: 211, g: 47, b: 47 }
];

// --- Data Fetching ---
async function fetchData() {
    try {
        const response = await fetch('all_heatmaps.json');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error("Could not fetch heatmap data:", error);
        heatmapTitle.textContent = "Error: Could not load data.";
        return null; // Return null to indicate failure
    }
}

// --- Color Logic ---
function getColor(value, minVal, maxVal) {
    const scale = (value - minVal) / (maxVal - minVal);
    
    for (let i = 0; i < COLORS.length - 1; i++) {
        if (scale >= COLORS[i].val && scale <= COLORS[i + 1].val) {
            const lower = COLORS[i];
            const upper = COLORS[i + 1];
            const t = (scale - lower.val) / (upper.val - lower.val);
            
            const r = Math.round(lower.r + t * (upper.r - lower.r));
            const g = Math.round(lower.g + t * (upper.g - lower.g));
            const b = Math.round(lower.b + t * (upper.b - lower.b));
            
            return `rgb(${r}, ${g}, ${b})`;
        }
    }
    const lastColor = COLORS[COLORS.length - 1];
    return `rgb(${lastColor.r}, ${lastColor.g}, ${lastColor.b})`;
}

// --- Drawing Functions ---
function drawHeatmap(data) {
    if (!data || data.length === 0) return;
    
    const size = heatmapContainer.clientWidth;
    heatmapCanvas.width = size;
    heatmapCanvas.height = size;
    
    const gridSize = data.length;
    const cellWidth = heatmapCanvas.width / gridSize;
    const cellHeight = heatmapCanvas.height / gridSize;
    
    const minVal = Math.min(...data.flat());
    const maxVal = Math.max(...data.flat());
    
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const val = data[y][x];
            heatmapCtx.fillStyle = getColor(val, minVal, maxVal);
            heatmapCtx.fillRect(x * cellWidth, y * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
        }
    }
}

function drawLegend() {
    const width = legendCanvas.width;
    const height = legendCanvas.height;
    
    for (let y = 0; y < height; y++) {
        const value = 1 - (y / height);
        legendCtx.fillStyle = getColor(value, 0, 1);
        legendCtx.fillRect(0, y, width, 1);
    }
}

// --- UI Update Functions ---
function populateDropdown() {
    const keys = Object.keys(allHeatmaps);
    if (keys.length > 0) {
        heatmapSelect.innerHTML = '<option value="">Select a heatmap</option>';
        keys.forEach(key => {
            const [xFeature, yFeature] = key.split('_vs_');
            const option = document.createElement('option');
            option.value = key;
            option.textContent = `${yFeature.replace(/_/g, ' ')} vs. ${xFeature.replace(/_/g, ' ')}`;
            heatmapSelect.appendChild(option);
        });
        heatmapSelect.value = keys[0];
        updateHeatmapDisplay(keys[0]);
    } else {
        heatmapTitle.textContent = "No heatmaps available.";
    }
}

function updateHeatmapDisplay(selectedKey) {
    if (!selectedKey || !allHeatmaps[selectedKey]) return;
    
    const data = allHeatmaps[selectedKey];
    const [xFeature, yFeature] = selectedKey.split('_vs_');
    
    heatmapTitle.textContent = `${yFeature.replace(/_/g, ' ')} vs. ${xFeature.replace(/_/g, ' ')}`;
    yAxisLabel.textContent = `${yFeature.replace(/_/g, ' ')}`;
    xAxisLabel.textContent = `${xFeature.replace(/_/g, ' ')}`;
    
    drawHeatmap(data);
}

// --- Initialization and Event Listeners ---
async function initialize() {
    // 1. Fetch live data
    allHeatmaps = await fetchData();
    
    // 2. Stop if data fetching failed
    if (!allHeatmaps) return;
    
    // 3. Populate UI elements
    populateDropdown();
    drawLegend();
    
    // 4. Set up event listeners
    heatmapSelect.addEventListener('change', (event) => {
        updateHeatmapDisplay(event.target.value);
    });
    
    // 5. Use ResizeObserver for efficient, responsive canvas redrawing
    const resizeObserver = new ResizeObserver(() => {
        const selectedKey = heatmapSelect.value;
        if (selectedKey) {
            updateHeatmapDisplay(selectedKey);
        }
    });
    
    resizeObserver.observe(heatmapContainer);
}

// Start the application
initialize();