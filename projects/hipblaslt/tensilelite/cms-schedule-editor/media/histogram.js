/**
 * CMS Schedule Editor - Histogram Visualization
 * 
 * D3.js-based interactive histogram for visualizing instruction schedules.
 */

// VS Code API for communication with extension
const vscode = acquireVsCodeApi();

// State
let currentSchedule = null;
let currentLatencies = null;
let currentCodepath = 0;
let usingDummyLatencies = false;  // Track if we're using dummy data

// Multi-selection state
let selectedBars = [];  // Array of selected bar data objects
let selectionAnchor = null;  // Anchor bar for shift-click range selection

// View mode state
let showLatencies = true;  // true = show latency data, false = flat view (all same height)

// Aligned layout state (for MFMA vertical alignment across codepaths)
let alignedMfmaPositions = [];  // X positions for each MFMA
let alignedTotalSlots = 0;      // Total number of X slots

/**
 * Generate dummy latency stats for an instruction
 * All values set to a constant to allow UI usage without real latency data
 */
function generateDummyStats(value = 4) {
    return {
        whisker_high: value,
        whisker_low: value,
        q3: value,
        median: value,
        q1: value
    };
}

/**
 * Generate dummy latencies for all instructions in the schedule
 */
function generateDummyLatencies() {
    if (!currentSchedule) {
        logError('generateDummyLatencies: No currentSchedule');
        return null;
    }
    
    const numCodepaths = currentSchedule.num_codepaths || 1;
    const numMfma = currentSchedule.num_mfma || 96;
    const instructionTypes = currentSchedule.instruction_types || [];
    const optSchedule = currentSchedule.opt_schedule || {};
    
    logInfo('generateDummyLatencies: Starting', { 
        numCodepaths, 
        numMfma, 
        numInstructionTypes: instructionTypes.length 
    });
    
    const dummyLatencies = {};
    
    for (let cp = 0; cp < numCodepaths; cp++) {
        const codepathKey = `codepath_${cp}`;
        dummyLatencies[codepathKey] = {};
        
        // Add MFMA latencies
        for (let i = 0; i < numMfma; i++) {
            dummyLatencies[codepathKey][`MFMA[${i}]`] = generateDummyStats(4);
        }
        
        // Add latencies for each instruction type
        for (const instType of instructionTypes) {
            if (instType === 'MFMA' || instType.startsWith('MFMA')) continue;
            
            const schedule = optSchedule[instType];
            if (!Array.isArray(schedule) || schedule.length === 0) continue;
            
            // Get the schedule for this codepath
            const indices = cp < schedule.length ? schedule[cp] : schedule[0];
            if (!Array.isArray(indices)) continue;
            
            // Add latency for each instruction
            for (let i = 0; i < indices.length; i++) {
                dummyLatencies[codepathKey][`${instType}[${i}]`] = generateDummyStats(4);
            }
        }
    }
    
    logInfo('generateDummyLatencies: Complete', { 
        codepaths: Object.keys(dummyLatencies),
        sampleKeys: dummyLatencies['codepath_0'] ? Object.keys(dummyLatencies['codepath_0']).slice(0, 5) : []
    });
    
    return dummyLatencies;
}

// ============================================================================
// Logging helpers - send messages to extension for persistent logging
// ============================================================================

/**
 * Log an error message to both console and extension
 */
function logError(message, details) {
    console.error(message, details);  // Keep for devtools debugging
    vscode.postMessage({ 
        command: 'log', 
        level: 'error', 
        message: message,
        details: details 
    });
}

/**
 * Log a warning message to both console and extension
 */
function logWarn(message, details) {
    console.warn(message, details);
    vscode.postMessage({ 
        command: 'log', 
        level: 'warn', 
        message: message,
        details: details 
    });
}

/**
 * Log an info message to both console and extension
 */
function logInfo(message, details) {
    console.log(message, details);
    vscode.postMessage({ 
        command: 'log', 
        level: 'info', 
        message: message,
        details: details 
    });
}

// Color manipulation helpers
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = Math.round(Math.max(0, Math.min(255, x))).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

/**
 * Adjust color brightness
 * @param {string} hex - Hex color code
 * @param {number} percent - Positive to lighten, negative to darken (-100 to 100)
 */
function adjustBrightness(hex, percent) {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    
    if (percent > 0) {
        // Lighten: move towards white
        return rgbToHex(
            rgb.r + (255 - rgb.r) * (percent / 100),
            rgb.g + (255 - rgb.g) * (percent / 100),
            rgb.b + (255 - rgb.b) * (percent / 100)
        );
    } else {
        // Darken: move towards black
        const factor = 1 + percent / 100;
        return rgbToHex(
            rgb.r * factor,
            rgb.g * factor,
            rgb.b * factor
        );
    }
}

// Color mapping for instruction types
function getColorForType(instType) {
    let baseColor;
    let variant = 'base';  // 'A', 'B', or 'base'
    
    // Detect A/B variant for LR and GR instructions (LDS color)
    if (instType.startsWith('LRA')) {
        baseColor = INSTRUCTION_COLORS.LDS || '#FF7F00';
        variant = 'A';
    } else if (instType.startsWith('LRB')) {
        baseColor = INSTRUCTION_COLORS.LDS || '#FF7F00';
        variant = 'B';
    } 
    // GRA/GRB - VMEM color with A/B variants
    else if (instType.startsWith('GRA')) {
        baseColor = INSTRUCTION_COLORS.VMEM || '#E6D700';
        variant = 'A';
    } else if (instType.startsWith('GRB')) {
        baseColor = INSTRUCTION_COLORS.VMEM || '#E6D700';
        variant = 'B';
    }
    // GRInc - SALU color with A/B variants (GRIncA, GRIncB)
    else if (instType.startsWith('GRIncA')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'A';
    } else if (instType.startsWith('GRIncB')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'B';
    } else if (instType.startsWith('GRInc')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
    }
    // LRS - SALU color with A/B variants (LRSA, LRSB)
    else if (instType.startsWith('LRSA')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'A';
    } else if (instType.startsWith('LRSB')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'B';
    } else if (instType.startsWith('LRS')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
    }
    // LWS - SALU color with A/B variants (LWSA, LWSB)
    else if (instType.startsWith('LWSA')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'A';
    } else if (instType.startsWith('LWSB')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'B';
    } else if (instType.startsWith('LWS')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
    }
    // LWA/LWB (other LW variants)
    else if (instType.startsWith('LWA')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'A';
    } else if (instType.startsWith('LWB')) {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
        variant = 'B';
    } else if (instType.startsWith('LW') || instType === 'LCC') {
        baseColor = INSTRUCTION_COLORS.SALU || '#A6FFD9';
    }
    // MFMA
    else if (instType.startsWith('MFMA')) {
        baseColor = INSTRUCTION_COLORS.MFMA || '#0A5B0C';
    } 
    // Sync instruction types
    else if (instType === 'SYNC' || instType === 'SWaitCnt') {
        // SWaitCnt (default sync) uses WAITCNT color
        baseColor = INSTRUCTION_COLORS.WAITCNT || '#000000';
    } else if (instType === 'SBarrier') {
        // SBarrier uses darker grey to differentiate from SWaitCnt and background
        baseColor = '#303030';
    } else if (instType === 'SNop') {
        // SNop uses a muted color
        baseColor = '#606060';
    } 
    // Pack - VALU color with A/B variants
    else if (instType.startsWith('PackA')) {
        baseColor = INSTRUCTION_COLORS.VALU || '#118C13';
        variant = 'A';
    } else if (instType.startsWith('PackB')) {
        baseColor = INSTRUCTION_COLORS.VALU || '#118C13';
        variant = 'B';
    } else if (instType.startsWith('Pack')) {
        baseColor = INSTRUCTION_COLORS.VALU || '#118C13';
    } else {
        baseColor = INSTRUCTION_COLORS.OTHER || '#A6D854';
    }
    
    // Apply brightness adjustment based on variant
    // A version: lighter (+25%), B version: darker (-25%)
    if (variant === 'A') {
        return adjustBrightness(baseColor, 25);
    } else if (variant === 'B') {
        return adjustBrightness(baseColor, -25);
    }
    return baseColor;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    
    // Notify extension that webview is ready
    vscode.postMessage({ command: 'ready' });
});

// Handle messages from extension
window.addEventListener('message', event => {
    const message = event.data;
    
    switch (message.command) {
        case 'updateSchedule':
            currentSchedule = message.schedule;
            // Preserve existing latencies - they're still valid after moving instructions
            // Latencies will be reset when updateLatencies is called with new data
            // Only reset usingDummyLatencies if we don't have real latencies yet
            if (!currentLatencies || Object.keys(currentLatencies).length === 0) {
                usingDummyLatencies = false;  // Reset dummy flag to allow regeneration
            }
            clearBarSelection(true);  // Clear selection and anchor when schedule updates
            updateScheduleInfo();
            updateInstructionList();
            renderHistogram();
            renderSyncPanel();
            renderMfmaReorderPanel();
            break;
            
        case 'updateLatencies':
            currentLatencies = message.latencies;
            renderHistogram();
            updateStatus('Latencies updated');
            break;
    }
});

function initializeUI() {
    // Run button
    document.getElementById('run-btn').addEventListener('click', () => {
        vscode.postMessage({ command: 'requestRun' });
    });
    
    // Save button
    document.getElementById('save-btn').addEventListener('click', () => {
        vscode.postMessage({ command: 'requestSave' });
    });
    
    // Validate button
    document.getElementById('validate-btn').addEventListener('click', () => {
        vscode.postMessage({ command: 'requestValidate' });
    });
    
    // Toggle view button (latencies vs flat)
    const toggleViewBtn = document.getElementById('toggle-view-btn');
    if (toggleViewBtn) {
        toggleViewBtn.addEventListener('click', () => {
            showLatencies = !showLatencies;
            toggleViewBtn.textContent = showLatencies ? '📊 Latencies' : '📏 Flat';
            toggleViewBtn.title = showLatencies 
                ? 'Currently showing latencies - click for flat view' 
                : 'Currently showing flat view - click for latencies';
            renderHistogram();
        });
    }
    
    // Toggle panel
    document.getElementById('toggle-panel').addEventListener('click', () => {
        document.getElementById('side-panel').classList.toggle('collapsed');
        // Re-render histogram with new width
        setTimeout(renderHistogram, 300);
    });
    
    // Sync panel toggle
    const toggleSyncBtn = document.getElementById('toggle-sync-panel');
    if (toggleSyncBtn) {
        toggleSyncBtn.addEventListener('click', () => {
            const panel = document.getElementById('sync-panel');
            panel.classList.toggle('collapsed');
            toggleSyncBtn.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
            // Re-render histogram to adjust to new available space
            setTimeout(renderHistogram, 300);
        });
    }
    
    // Add sync instruction button
    const addSyncBtn = document.getElementById('add-sync-btn');
    if (addSyncBtn) {
        addSyncBtn.addEventListener('click', showAddSyncDialog);
    }
    
    // MFMA reorder panel toggle
    const toggleMfmaReorderBtn = document.getElementById('toggle-mfma-reorder-panel');
    if (toggleMfmaReorderBtn) {
        toggleMfmaReorderBtn.addEventListener('click', () => {
            const panel = document.getElementById('mfma-reorder-panel');
            panel.classList.toggle('collapsed');
            toggleMfmaReorderBtn.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
            // Re-render histogram to adjust to new available space
            setTimeout(renderHistogram, 300);
        });
    }
    
    // Initialize legend
    renderLegend();
    
    // Window resize handler with debouncing
    let resizeTimeout = null;
    window.addEventListener('resize', () => {
        // Clear previous timeout
        if (resizeTimeout) {
            clearTimeout(resizeTimeout);
        }
        // Debounce: wait 150ms after last resize event before re-rendering
        resizeTimeout = setTimeout(() => {
            if (currentSchedule) {
                renderHistogram();
            }
        }, 150);
    });
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

function updateScheduleInfo() {
    if (!currentSchedule) return;
    
    const info = document.getElementById('schedule-info');
    info.textContent = `${currentSchedule.function_name} | ${currentSchedule.num_mfma} MFMAs | ${currentSchedule.num_codepaths} codepaths`;
}

function updateInstructionList() {
    if (!currentSchedule) return;
    
    const list = document.getElementById('instruction-list');
    list.innerHTML = '';
    
    // First add MFMA as a fixed entry
    const mfmaLi = document.createElement('li');
    mfmaLi.dataset.type = 'MFMA';
    mfmaLi.dataset.index = -1;
    mfmaLi.draggable = false;
    mfmaLi.className = 'inst-fixed';
    mfmaLi.title = 'MFMA instructions are fixed at their positions';
    
    const mfmaColorDiv = document.createElement('div');
    mfmaColorDiv.className = 'inst-color';
    mfmaColorDiv.style.backgroundColor = getColorForType('MFMA');
    mfmaLi.appendChild(mfmaColorDiv);
    
    const mfmaNameSpan = document.createElement('span');
    mfmaNameSpan.className = 'inst-name';
    mfmaNameSpan.textContent = 'MFMA';
    mfmaLi.appendChild(mfmaNameSpan);
    
    const mfmaCountSpan = document.createElement('span');
    mfmaCountSpan.className = 'inst-count';
    mfmaCountSpan.textContent = `(${currentSchedule.num_mfma})`;
    mfmaLi.appendChild(mfmaCountSpan);
    
    const mfmaFixedSpan = document.createElement('span');
    mfmaFixedSpan.className = 'inst-fixed-badge';
    mfmaFixedSpan.textContent = '🔒';
    mfmaFixedSpan.title = 'Fixed position';
    mfmaLi.appendChild(mfmaFixedSpan);
    
    list.appendChild(mfmaLi);
    
    const numCodepaths = currentSchedule.num_codepaths || 1;
    
    // Add other instruction types (directly after MFMA, no separator)
    currentSchedule.instruction_types.forEach((type, index) => {
        // Skip MFMA types (already added above)
        if (type === 'MFMA' || type.startsWith('MFMA')) {
            return;
        }
        
        const li = document.createElement('li');
        li.dataset.type = type;
        li.dataset.index = index;
        li.draggable = true;
        
        // Color indicator
        const colorDiv = document.createElement('div');
        colorDiv.className = 'inst-color';
        colorDiv.style.backgroundColor = getColorForType(type);
        li.appendChild(colorDiv);
        
        // Name
        const nameSpan = document.createElement('span');
        nameSpan.className = 'inst-name';
        nameSpan.textContent = type;
        li.appendChild(nameSpan);
        
        // Count (number of instructions of this type)
        const schedule = currentSchedule.opt_schedule[type];
        let count = 0;
        let numSchedules = 0;
        if (Array.isArray(schedule) && schedule.length > 0) {
            numSchedules = schedule.length;
            if (Array.isArray(schedule[0])) {
                count = schedule[0].length;
            } else if (typeof schedule === 'string') {
                count = '?';
            }
        }
        const countSpan = document.createElement('span');
        countSpan.className = 'inst-count';
        countSpan.textContent = `(${count})`;
        li.appendChild(countSpan);
        
        // Schedule count badge (shared vs per-codepath)
        const schedBadge = document.createElement('span');
        const isShared = numSchedules === 1 && numCodepaths > 1;
        schedBadge.className = `inst-sched-badge ${isShared ? 'sched-shared' : 'sched-split'}`;
        schedBadge.textContent = numSchedules === 1 ? '1' : `${numSchedules}`;
        schedBadge.title = isShared 
            ? `Shared schedule across all ${numCodepaths} codepaths` 
            : `${numSchedules} separate schedule(s) for each codepath`;
        li.appendChild(schedBadge);
        
        // Split button (only show if shared and multiple codepaths exist)
        if (isShared) {
            const splitBtn = document.createElement('button');
            splitBtn.className = 'inst-split-btn';
            splitBtn.textContent = '⑂';
            splitBtn.title = `Split into ${numCodepaths} separate schedules (one per codepath)`;
            splitBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                splitScheduleForType(type);
            });
            li.appendChild(splitBtn);
        }
        
        // Drag events
        li.addEventListener('dragstart', handleDragStart);
        li.addEventListener('dragover', handleDragOver);
        li.addEventListener('dragleave', handleDragLeave);
        li.addEventListener('drop', handleDrop);
        li.addEventListener('dragend', handleDragEnd);
        
        list.appendChild(li);
    });
}

/**
 * Split a shared schedule into separate per-codepath schedules
 */
function splitScheduleForType(type) {
    if (!currentSchedule || !currentSchedule.opt_schedule || !currentSchedule.opt_schedule[type]) {
        updateStatus(`ERROR: Cannot find schedule for ${type}`);
        return;
    }
    
    const schedule = currentSchedule.opt_schedule[type];
    const numCodepaths = currentSchedule.num_codepaths || 1;
    
    if (!Array.isArray(schedule) || schedule.length !== 1) {
        updateStatus(`${type} is already split into multiple schedules`);
        return;
    }
    
    // Duplicate the single schedule for each codepath
    const originalSchedule = schedule[0];
    const newSchedule = [];
    for (let i = 0; i < numCodepaths; i++) {
        // Deep copy the array
        newSchedule.push([...originalSchedule]);
    }
    
    // Update local state
    currentSchedule.opt_schedule[type] = newSchedule;
    
    // Send to extension
    vscode.postMessage({
        command: 'splitSchedule',
        type: type,
        newSchedule: newSchedule
    });
    
    // Re-render
    updateInstructionList();
    renderHistogram();
    
    updateStatus(`Split ${type} into ${numCodepaths} separate schedules`);
}

// Drag and drop handlers for instruction list
let draggedItem = null;

function handleDragStart(e) {
    draggedItem = this;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', this.dataset.type);
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    this.classList.add('drag-over');
}

function handleDragLeave(e) {
    this.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    this.classList.remove('drag-over');
    
    if (draggedItem !== this) {
        const list = document.getElementById('instruction-list');
        const items = Array.from(list.children);
        const draggedIndex = items.indexOf(draggedItem);
        const targetIndex = items.indexOf(this);
        
        if (draggedIndex < targetIndex) {
            this.parentNode.insertBefore(draggedItem, this.nextSibling);
        } else {
            this.parentNode.insertBefore(draggedItem, this);
        }
        
        // Automatically apply the new order
        applyInstructionOrder();
    }
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    document.querySelectorAll('#instruction-list li').forEach(item => {
        item.classList.remove('drag-over');
    });
}

function applyInstructionOrder() {
    const list = document.getElementById('instruction-list');
    
    // Filter out separators and undefined types, and exclude MFMA (it's fixed)
    const newOrder = Array.from(list.children)
        .filter(li => li.dataset.type && !li.classList.contains('inst-separator') && li.dataset.type !== 'MFMA')
        .map(li => li.dataset.type);
    
    // Don't update if empty (something went wrong)
    if (newOrder.length === 0) {
        logWarn('applyInstructionOrder: newOrder is empty, skipping update');
        updateStatus('Error: No instructions to reorder');
        return;
    }
    
    // Update local state
    currentSchedule.instruction_types = newOrder;
    
    // Notify extension
    vscode.postMessage({
        command: 'reorderInstructions',
        newOrder: newOrder
    });
    
    updateStatus('Order applied');
    // Note: histogram will re-render when extension sends updateSchedule message back
}

function renderLegend() {
    const legend = document.getElementById('legend');
    
    // Base colors
    const ldsBase = INSTRUCTION_COLORS.LDS || '#FF7F00';
    const vmemBase = INSTRUCTION_COLORS.VMEM || '#E6D700';
    const valuBase = INSTRUCTION_COLORS.VALU || '#118C13';
    const saluBase = INSTRUCTION_COLORS.SALU || '#A6FFD9';
    
    const categories = [
        { name: 'MFMA', color: INSTRUCTION_COLORS.MFMA },
        { name: 'LDS-A (LRA)', color: adjustBrightness(ldsBase, 25) },
        { name: 'LDS-B (LRB)', color: adjustBrightness(ldsBase, -25) },
        { name: 'VMEM-A (GRA)', color: adjustBrightness(vmemBase, 25) },
        { name: 'VMEM-B (GRB)', color: adjustBrightness(vmemBase, -25) },
        { name: 'Pack/VALU', color: valuBase },
        { name: 'SALU-A (LWSA/GRIncA/LRSA)', color: adjustBrightness(saluBase, 25) },
        { name: 'SALU-B (LWSB/GRIncB/LRSB)', color: adjustBrightness(saluBase, -25) },
        { name: 'SWaitCnt', color: INSTRUCTION_COLORS.WAITCNT },
        { name: 'SBarrier', color: '#303030' },
        { name: 'SNop', color: '#606060' },
    ];
    
    legend.innerHTML = '';
    categories.forEach(cat => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        
        const color = document.createElement('div');
        color.className = 'legend-color';
        color.style.backgroundColor = cat.color;
        item.appendChild(color);
        
        const name = document.createElement('span');
        name.textContent = cat.name;
        item.appendChild(name);
        
        legend.appendChild(item);
    });
}

function renderHistogram() {
    if (!currentSchedule) {
        logInfo('renderHistogram: No currentSchedule, returning');
        return;
    }
    
    logInfo('renderHistogram: Starting', {
        num_mfma: currentSchedule.num_mfma,
        num_codepaths: currentSchedule.num_codepaths,
        instruction_types: currentSchedule.instruction_types,
        opt_schedule_keys: currentSchedule.opt_schedule ? Object.keys(currentSchedule.opt_schedule) : null,
        hasLatencies: !!currentLatencies,
        usingDummy: usingDummyLatencies
    });
    
    const container = document.getElementById('histogram-container');
    const svg = d3.select('#histogram');
    
    // Clear previous content
    svg.selectAll('*').remove();
    
    // Remove any existing tooltips
    d3.selectAll('.tooltip').remove();
    
    const numCodepaths = currentSchedule.num_codepaths || 1;
    const barWidth = 8;
    const barGap = 1;
    
    // Get dimensions
    const containerRect = container.getBoundingClientRect();
    const margin = { top: 20, right: 20, bottom: 45, left: 50 };
    const codepathGap = 55;   // Gap between codepath charts (room for tick marks + labels)
    
    // Calculate chart height dynamically based on available space
    // Account for: toolbar, histogram header, sync panel, mfma reorder panel, and padding
    const toolbar = document.getElementById('toolbar');
    const histogramHeader = document.getElementById('histogram-header');
    const syncPanel = document.getElementById('sync-panel');
    const mfmaReorderPanel = document.getElementById('mfma-reorder-panel');
    
    const toolbarHeight = toolbar ? toolbar.getBoundingClientRect().height : 40;
    const headerHeight = histogramHeader ? histogramHeader.getBoundingClientRect().height : 30;
    const syncPanelHeight = syncPanel ? syncPanel.getBoundingClientRect().height : 150;
    const mfmaReorderPanelHeight = mfmaReorderPanel ? mfmaReorderPanel.getBoundingClientRect().height : 0;
    const padding = 40;  // Extra padding for margins and spacing
    
    // Calculate available height for the histogram SVG
    const windowHeight = window.innerHeight;
    const usedHeight = toolbarHeight + headerHeight + syncPanelHeight + mfmaReorderPanelHeight + padding;
    const availableForHistogram = windowHeight - usedHeight;
    
    // Calculate height per codepath, with min/max constraints
    const totalGapHeight = (numCodepaths - 1) * codepathGap;
    const availableForCharts = availableForHistogram - margin.top - margin.bottom - totalGapHeight;
    // Min 100px, max 250px per codepath
    const heightPerCodepath = Math.max(100, Math.min(250, Math.floor(availableForCharts / numCodepaths)));
    const chartHeight = heightPerCodepath;
    const totalHeight = numCodepaths * chartHeight + (numCodepaths - 1) * codepathGap;
    
    // Build bar data for first codepath to determine width and check for data
    logInfo('renderHistogram: Building bar data for codepath 0');
    const barData0 = buildBarData(0);
    logInfo('renderHistogram: barData0 built', { length: barData0.length });
    
    if (barData0.length === 0) {
        const width = Math.max(800, containerRect.width - 40) - margin.left - margin.right;
        svg.attr('width', width + margin.left + margin.right)
           .attr('height', 200);
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        const hasLatencies = currentLatencies && Object.keys(currentLatencies).length > 0;
        const errorMsg = hasLatencies 
            ? 'Missing latency data for some instructions. Check status bar.'
            : 'No latency data available. Using dummy latencies failed.';
        
        logError('Histogram render failed: No bar data', { 
            hasLatencies,
            currentLatenciesKeys: currentLatencies ? Object.keys(currentLatencies) : null,
            usingDummyLatencies
        });
        
        g.append('text')
            .attr('x', width / 2)
            .attr('y', 80)
            .attr('text-anchor', 'middle')
            .attr('fill', '#f14c4c')
            .attr('font-size', '16px')
            .attr('font-weight', 'bold')
            .text('Error: Latency Data Required');
        
        g.append('text')
            .attr('x', width / 2)
            .attr('y', 110)
            .attr('text-anchor', 'middle')
            .attr('fill', 'white')
            .attr('font-size', '14px')
            .text(errorMsg);
        return;
    }
    
    // Build bar data for all codepaths first
    const allBarData = [];
    let globalMaxHeight = 0;
    for (let cp = 0; cp < numCodepaths; cp++) {
        const barData = cp === 0 ? barData0 : buildBarData(cp);
        allBarData.push(barData);
        const maxH = d3.max(barData, d => d.stats ? d.stats.whisker_high : d.barHeight) || 1;
        if (maxH > globalMaxHeight) globalMaxHeight = maxH;
    }
    
    // In flat view, use a fixed height for all bars
    const flatBarHeight = 1;  // All bars will be this height in flat view
    const effectiveMaxHeight = showLatencies ? globalMaxHeight : flatBarHeight;
    
    // Calculate aligned positions so MFMAs line up across codepaths
    // For each MFMA slot, find the max number of instructions across all codepaths
    const numMfma = currentSchedule.num_mfma || 96;
    const maxInstsPerSlot = new Array(numMfma + 1).fill(0);  // +1 for slot -1 (before MFMA 0)
    
    for (const barData of allBarData) {
        // Count instructions at each MFMA slot
        const instsPerSlot = new Array(numMfma + 1).fill(0);
        for (const bar of barData) {
            if (!bar.isFixed) {
                // Non-MFMA bars: count at their mfmaIndex slot
                // slot 0 = instructions at mfmaIndex -1, slot 1 = at mfmaIndex 0, etc.
                const slotIdx = bar.mfmaIndex + 1;
                if (slotIdx >= 0 && slotIdx <= numMfma) {
                    instsPerSlot[slotIdx]++;
                }
            }
        }
        // Update max
        for (let i = 0; i <= numMfma; i++) {
            maxInstsPerSlot[i] = Math.max(maxInstsPerSlot[i], instsPerSlot[i]);
        }
    }
    
    // Calculate cumulative X positions for each MFMA
    // Structure: [insts at -1] [MFMA 0] [insts at 0] [MFMA 1] [insts at 1] ... [MFMA n-1] [insts at n-1]
    alignedMfmaPositions = [];  // X position for each MFMA (stored globally for drag handler)
    let currentX = maxInstsPerSlot[0];  // Start after instructions at slot -1
    
    for (let mfmaIdx = 0; mfmaIdx < numMfma; mfmaIdx++) {
        alignedMfmaPositions.push(currentX);
        currentX++;  // MFMA bar itself takes 1 slot
        currentX += maxInstsPerSlot[mfmaIdx + 1];  // Instructions after this MFMA
    }
    
    alignedTotalSlots = currentX;  // Store globally for drag handler
    const totalSlots = currentX;
    
    // Assign aligned X positions to each bar in each codepath
    for (let cp = 0; cp < numCodepaths; cp++) {
        const barData = allBarData[cp];
        
        // Group non-MFMA bars by their mfmaIndex
        const instsAtSlot = {};
        for (let i = -1; i < numMfma; i++) {
            instsAtSlot[i] = [];
        }
        
        for (const bar of barData) {
            if (!bar.isFixed) {
                if (instsAtSlot[bar.mfmaIndex]) {
                    instsAtSlot[bar.mfmaIndex].push(bar);
                }
            }
        }
        
        // Assign aligned X positions
        for (const bar of barData) {
            if (bar.isFixed) {
                // MFMA: use the pre-calculated position
                bar.alignedX = alignedMfmaPositions[bar.mfmaIndex];
            } else {
                // Non-MFMA: position within the allocated slot
                const slotBars = instsAtSlot[bar.mfmaIndex];
                const indexInSlot = slotBars.indexOf(bar);
                
                if (bar.mfmaIndex === -1) {
                    // Before MFMA 0
                    bar.alignedX = indexInSlot;
                } else {
                    // After MFMA[mfmaIndex]
                    const mfmaX = alignedMfmaPositions[bar.mfmaIndex];
                    bar.alignedX = mfmaX + 1 + indexInSlot;
                }
            }
        }
    }
    
    // Calculate width based on total slots
    const calculatedWidth = totalSlots * (barWidth + barGap);
    const width = Math.max(800, calculatedWidth, containerRect.width - 40) - margin.left - margin.right;
    
    svg.attr('width', width + margin.left + margin.right)
       .attr('height', totalHeight + margin.top + margin.bottom);
    
    // X scale (shared across all codepaths, using aligned positions)
    const x = d3.scaleLinear()
        .domain([0, totalSlots])
        .range([0, width]);
    
    // Tooltip (shared)
    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0);
    
    // Render each codepath
    for (let cp = 0; cp < numCodepaths; cp++) {
        const barData = allBarData[cp];
        const yOffset = margin.top + cp * (chartHeight + codepathGap);
        
        const g = svg.append('g')
            .attr('class', `codepath-group codepath-${cp}`)
            .attr('transform', `translate(${margin.left},${yOffset})`);
        
        // Y scale for this codepath (using global max for consistency, or flat if in flat view)
        const y = d3.scaleLinear()
            .domain([0, effectiveMaxHeight * 1.1])
            .nice()
            .range([chartHeight, 0]);
        
        // Codepath label
        g.append('text')
            .attr('x', -5)
            .attr('y', -5)
            .attr('text-anchor', 'start')
            .attr('fill', '#4ec9b0')
            .attr('font-size', '11px')
            .attr('font-weight', 'bold')
            .text(`Codepath ${cp}`);
        
        // Y axis
        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0));
        
        // Add X axis tick marks for all codepaths (use aligned positions)
        const mfmaTicks = barData
            .filter(d => d.isFixed)
            .map(d => ({ pos: d.alignedX, label: d.mfmaIndex }));
        
        const xAxis = g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${chartHeight})`);
        
        xAxis.selectAll('.mfma-tick')
            .data(mfmaTicks)
            .join('g')
            .attr('class', 'mfma-tick')
            .attr('transform', d => `translate(${x(d.pos + 0.5)},0)`)
            .call(g => {
                g.append('line')
                    .attr('y2', 6)
                    .attr('stroke', 'white');
                g.append('text')
                    .attr('x', 4)
                    .attr('y', 10)
                    .attr('text-anchor', 'start')
                    .attr('transform', 'rotate(90, 4, 10)')
                    .attr('fill', 'white')
                    .attr('font-size', '8px')
                    .text(d => d.label);
            });
        
        // Create bar groups for this codepath (use aligned X positions)
        const barGroups = g.selectAll('.bar-group')
            .data(barData.map(d => ({ ...d, codepath: cp })))
            .join('g')
            .attr('class', d => d.isFixed ? 'bar-group bar-fixed' : 'bar-group bar-movable')
            .attr('transform', d => `translate(${x(d.alignedX)}, 0)`)
            .style('cursor', d => d.isFixed ? 'not-allowed' : 'grab');
        
        // Draw histogram bar
        barGroups.append('rect')
            .attr('class', 'histogram-bar')
            .attr('x', 0)
            .attr('y', d => showLatencies ? y(d.stats.median) : y(flatBarHeight))
            .attr('width', barWidth)
            .attr('height', d => showLatencies ? Math.max(1, chartHeight - y(d.stats.median)) : Math.max(1, chartHeight - y(flatBarHeight)))
            .attr('fill', d => getColorForType(d.type))
            .attr('opacity', d => d.isFixed ? 0.8 : 0.6);
        
        // Draw whisker line (only in latency view)
        if (showLatencies) {
            barGroups.append('line')
                .attr('class', 'whisker-line')
                .attr('x1', barWidth / 2)
                .attr('x2', barWidth / 2)
                .attr('y1', d => y(d.stats.whisker_high))
                .attr('y2', d => y(d.stats.whisker_low))
                .attr('stroke', '#FFFFFF')
                .attr('stroke-width', 1);
            
            // Draw whisker caps
            const whiskerCapWidth = barWidth * 0.6;
            barGroups.append('line')
                .attr('class', 'whisker-cap-high')
                .attr('x1', (barWidth - whiskerCapWidth) / 2)
                .attr('x2', (barWidth + whiskerCapWidth) / 2)
                .attr('y1', d => y(d.stats.whisker_high))
                .attr('y2', d => y(d.stats.whisker_high))
                .attr('stroke', '#FFFFFF')
                .attr('stroke-width', 1);
            
            barGroups.append('line')
                .attr('class', 'whisker-cap-low')
                .attr('x1', (barWidth - whiskerCapWidth) / 2)
                .attr('x2', (barWidth + whiskerCapWidth) / 2)
                .attr('y1', d => y(d.stats.whisker_low))
                .attr('y2', d => y(d.stats.whisker_low))
                .attr('stroke', '#FFFFFF')
                .attr('stroke-width', 1);
            
            // Draw box
            barGroups.append('rect')
                .attr('class', 'box-rect')
                .attr('x', 0)
                .attr('y', d => y(d.stats.q3))
                .attr('width', barWidth)
                .attr('height', d => Math.max(1, y(d.stats.q1) - y(d.stats.q3)))
                .attr('fill', 'none')
                .attr('stroke', '#FFFFFF')
                .attr('stroke-width', 1.5);
            
            // Draw median line
            barGroups.append('line')
                .attr('class', 'median-line')
                .attr('x1', 0)
                .attr('x2', barWidth)
                .attr('y1', d => y(d.stats.median))
                .attr('y2', d => y(d.stats.median))
                .attr('stroke', d => getColorForType(d.type))
                .attr('stroke-width', 2);
        }
        
        // Interaction rect
        barGroups.append('rect')
            .attr('class', 'interaction-rect')
            .attr('x', 0)
            .attr('y', d => showLatencies ? y(Math.max(d.stats.whisker_high, d.stats.median * 1.1)) : y(flatBarHeight))
            .attr('width', barWidth)
            .attr('height', d => showLatencies ? Math.max(1, chartHeight - y(Math.max(d.stats.whisker_high, d.stats.median * 1.1))) : Math.max(1, chartHeight - y(flatBarHeight)))
            .attr('fill', 'transparent')
            .on('mouseover', function(event, d) {
                d3.select(this.parentNode).select('.histogram-bar').attr('opacity', d.isFixed ? 1 : 0.85);
                tooltip.transition().duration(200).style('opacity', 0.9);
                tooltip.html(`
                    <div class="tooltip-label">${d.label}</div>
                    <div>Type: ${d.type}</div>
                    <div>Codepath: ${d.codepath}</div>
                    <div>Issued at MFMA: ${d.mfmaIndex}</div>
                    ${d.isFixed ? '<div style="color: #888; font-style: italic;">Fixed position</div>' : ''}
                    ${isBarSelected(d) ? '<div style="color: #4ec9b0; font-weight: bold;">✓ Selected</div>' : ''}
                    <hr style="margin: 4px 0; border-color: #555;">
                    <div><b>Box Plot Statistics (cycles):</b></div>
                    <div>Whisker High: ${d.stats.whisker_high.toFixed(2)}</div>
                    <div>Q3 (75%): ${d.stats.q3.toFixed(2)}</div>
                    <div><b>Median: ${d.stats.median.toFixed(2)}</b></div>
                    <div>Q1 (25%): ${d.stats.q1.toFixed(2)}</div>
                    <div>Whisker Low: ${d.stats.whisker_low.toFixed(2)}</div>
                    <div style="margin-top: 4px; color: #aaa;">IQR: ${(d.stats.q3 - d.stats.q1).toFixed(2)}</div>
                `)
                    .style('left', (event.pageX) + 'px')
                    .style('top', (event.pageY + 15) + 'px');
            })
            .on('mouseout', function(event, d) {
                d3.select(this.parentNode).select('.histogram-bar').attr('opacity', d.isFixed ? 0.8 : 0.6);
                tooltip.transition().duration(500).style('opacity', 0);
            })
            .on('click', handleBarClick);
        
        // Drag behavior
        barGroups.filter(d => !d.isFixed)
            .call(d3.drag()
                .on('start', dragBarStart)
                .on('drag', dragBarMove)
                .on('end', dragBarEnd));
    }
    
    // Click on SVG background clears selection
    svg.on('click', function(event) {
        // Only clear if clicking directly on SVG (not on a bar)
        if (event.target === this) {
            clearBarSelection(true);  // Also clear anchor
            const dummyNote = usingDummyLatencies ? ' (using dummy latencies)' : '';
            updateStatus(`Rendered ${numCodepaths} codepath(s), ${barData0.length} instructions, MFMAs aligned${dummyNote}`);
        }
    });
    
    // Update status
    const dummyNote = usingDummyLatencies ? ' ⚠️ Using dummy latencies - run cms-fast for real data' : '';
    updateStatus(`Rendered ${numCodepaths} codepath(s), ${barData0.length} instructions, MFMAs aligned${dummyNote}`);
}

function buildBarData(codepath = 0) {
    if (!currentSchedule) {
        console.log('buildBarData: No currentSchedule');
        return [];
    }
    
    console.log('buildBarData: Starting with', {
        numMfma: currentSchedule.num_mfma,
        numCodepaths: currentSchedule.num_codepaths,
        instructionTypes: currentSchedule.instruction_types,
        codepath: codepath
    });
    
    const numMfma = currentSchedule.num_mfma || 96;
    const codepathKey = `codepath_${codepath}`;
    
    // Check if latency data is available, generate dummy data if not
    // This MUST happen BEFORE processing instructions so stats can be looked up
    let codepathLatencies;
    logInfo('buildBarData: Checking latencies', { 
        hasCurrentLatencies: !!currentLatencies,
        hasCodepathKey: currentLatencies ? !!currentLatencies[codepathKey] : false,
        codepathKey: codepathKey
    });
    
    if (!currentLatencies || !currentLatencies[codepathKey]) {
        // Generate dummy latencies for UI usage
        if (!usingDummyLatencies) {
            logInfo('No latency data available, generating dummy values for UI', { codepath: codepath });
            currentLatencies = generateDummyLatencies();
            usingDummyLatencies = true;
            logInfo('Dummy latencies generated', { 
                success: !!currentLatencies,
                hasCodepathKey: currentLatencies ? !!currentLatencies[codepathKey] : false
            });
        }
        codepathLatencies = currentLatencies ? currentLatencies[codepathKey] : null;
        
        if (!codepathLatencies) {
            logError('Failed to generate dummy latencies', { 
                codepath: codepath,
                currentLatencies: currentLatencies ? Object.keys(currentLatencies) : null
            });
            return [];
        }
    } else {
        codepathLatencies = currentLatencies[codepathKey];
        usingDummyLatencies = false;
    }
    
    logInfo('buildBarData: Using latencies', { 
        numLatencyKeys: Object.keys(codepathLatencies).length,
        sampleKeys: Object.keys(codepathLatencies).slice(0, 5),
        usingDummy: usingDummyLatencies
    });
    
    // Counter for instruction indices per type
    const typeCounters = {};
    
    // Track skipped and processed types for debugging
    const skippedTypes = [];
    const processedTypes = [];
    
    // First, collect all non-MFMA instructions grouped by their target MFMA index
    // instructionsAtMfma[i] = list of instructions scheduled at MFMA index i
    // instructionsAtMfma[-1] = instructions before MFMA 0
    const instructionsAtMfma = {};
    for (let i = -1; i < numMfma; i++) {
        instructionsAtMfma[i] = [];
    }
    
    // Process each instruction type
    for (const instType of currentSchedule.instruction_types) {
        // Skip MFMA - handled separately
        if (instType === 'MFMA' || instType.startsWith('MFMA')) {
            continue;
        }
        
        const schedule = currentSchedule.opt_schedule[instType];
        
        if (!schedule || typeof schedule === 'string') {
            skippedTypes.push({type: instType, reason: schedule ? 'string' : 'null'});
            continue;
        }
        
        // Get the schedule for specified codepath
        let indices = [];
        if (Array.isArray(schedule)) {
            if (codepath < schedule.length) {
                indices = schedule[codepath];
            } else if (schedule.length > 0) {
                indices = schedule[0];
            }
        }
        
        if (!Array.isArray(indices)) {
            skippedTypes.push({type: instType, reason: 'indices not array', schedule: schedule});
            continue;
        }
        
        processedTypes.push({type: instType, numIndices: indices.length, sampleIndices: indices.slice(0, 5)});
        
        // Initialize counter for this type
        if (!(instType in typeCounters)) {
            typeCounters[instType] = 0;
        }
        
        // Group instructions by their MFMA index
        for (const mfmaIndex of indices) {
            const idx = typeCounters[instType]++;
            const label = `${instType}[${idx}]`;
            
            // Get box plot stats from latencies (now guaranteed to exist or be dummy)
            let stats = codepathLatencies[label] || generateDummyStats(4);
            
            // For SYNC instructions, get the actual type (SWaitCnt, SBarrier, SNop) from sync_code
            let displayType = instType;
            if (instType === 'SYNC' && currentSchedule.sync_code && currentSchedule.sync_code[idx]) {
                const syncInst = currentSchedule.sync_code[idx];
                if (syncInst.instruction && syncInst.instruction.type) {
                    displayType = syncInst.instruction.type;
                }
            }
            
            if (instructionsAtMfma[mfmaIndex]) {
                instructionsAtMfma[mfmaIndex].push({
                    type: displayType,
                    mfmaIndex: mfmaIndex,
                    label: label,
                    stats: stats,
                    instIndex: idx,
                    isFixed: false
                });
            }
        }
    }
    
    // Now build the sequential data:
    // [instructions at -1] [MFMA 0] [instructions at 0] [MFMA 1] [instructions at 1] ...
    const data = [];
    let seqIndex = 0;
    
    // Instructions at -1 (before MFMA 0)
    for (const inst of instructionsAtMfma[-1]) {
        data.push({
            ...inst,
            seqIndex: seqIndex++,
            barHeight: inst.stats.whisker_high  // Use whisker_high for Y scale calculation
        });
    }
    
    // For each MFMA, add the MFMA then the instructions scheduled at that index
    for (let mfmaIdx = 0; mfmaIdx < numMfma; mfmaIdx++) {
        const mfmaLabel = `MFMA[${mfmaIdx}]`;
        
        // Get MFMA latency stats (use dummy if not available)
        let mfmaStats = codepathLatencies[mfmaLabel] || generateDummyStats(4);
        
        // Add MFMA box plot (fixed)
        data.push({
            type: 'MFMA',
            mfmaIndex: mfmaIdx,
            label: mfmaLabel,
            stats: mfmaStats,
            instIndex: mfmaIdx,
            isFixed: true,
            seqIndex: seqIndex++,
            barHeight: mfmaStats.whisker_high  // Use whisker_high for Y scale calculation
        });
        
        // Add instructions scheduled at this MFMA index (they execute after this MFMA)
        for (const inst of instructionsAtMfma[mfmaIdx]) {
            data.push({
                ...inst,
                seqIndex: seqIndex++,
                barHeight: inst.stats.whisker_high  // Use whisker_high for Y scale calculation
            });
        }
    }
    
    processedTypes.push({type: 'MFMA', numIndices: numMfma, sampleIndices: [0, 1, 2, 3, 4]});
    
    console.log('buildBarData: Complete', {
        codepath: codepath,
        totalBars: data.length,
        processedTypes: processedTypes,
        skippedTypes: skippedTypes
    });
    
    return data;
}

// Drag handlers for histogram bars
let draggedBar = null;
let dragStartX = 0;
let originalTransform = null;  // Store original transform to restore on drag end
let originalTransforms = {};   // Store transforms for all selected bars during multi-drag

/**
 * Handle click on a bar for selection (with Ctrl for multi-select, Shift for range select)
 */
function handleBarClick(event, d) {
    // Only handle movable bars
    if (d.isFixed) return;
    
    const barGroup = d3.select(this.parentNode);
    const isCtrlClick = event.ctrlKey || event.metaKey;  // metaKey for Mac
    const isShiftClick = event.shiftKey;
    
    if (isShiftClick && selectionAnchor) {
        // Shift+click: select range from anchor to clicked bar
        // Check if same codepath as anchor
        if (selectionAnchor.codepath !== d.codepath) {
            updateStatus('Cannot select range across different codepaths');
            return;
        }
        
        // Get the sequential index range
        const startSeq = Math.min(selectionAnchor.seqIndex, d.seqIndex);
        const endSeq = Math.max(selectionAnchor.seqIndex, d.seqIndex);
        
        // Clear current selection (but keep the anchor)
        clearBarSelection();
        
        // Select all non-MFMA bars in the range
        d3.selectAll('.bar-group').each(function(barData) {
            if (!barData.isFixed && 
                barData.codepath === d.codepath &&
                barData.seqIndex >= startSeq && 
                barData.seqIndex <= endSeq) {
                selectedBars.push(barData);
                d3.select(this).classed('bar-selected', true);
            }
        });
        
    } else if (isCtrlClick) {
        // Ctrl+click: toggle selection
        const existingIndex = selectedBars.findIndex(b => b.label === d.label && b.codepath === d.codepath);
        
        if (existingIndex >= 0) {
            // Already selected - deselect it
            selectedBars.splice(existingIndex, 1);
            barGroup.classed('bar-selected', false);
        } else {
            // Not selected - check if same codepath as existing selections
            if (selectedBars.length > 0 && selectedBars[0].codepath !== d.codepath) {
                updateStatus('Cannot select bars from different codepaths');
                return;
            }
            // Add to selection
            selectedBars.push(d);
            barGroup.classed('bar-selected', true);
        }
        // Update anchor to this bar for subsequent shift-clicks
        selectionAnchor = d;
        
    } else {
        // Regular click: clear selection and select only this bar
        clearBarSelection();
        selectedBars.push(d);
        barGroup.classed('bar-selected', true);
        // Set anchor for shift-click range selection
        selectionAnchor = d;
    }
    
    // Update status
    if (selectedBars.length > 1) {
        updateStatus(`Selected ${selectedBars.length} instructions (Ctrl+click to add, Shift+click for range, drag to move all)`);
    } else if (selectedBars.length === 1) {
        updateStatus(`Selected ${selectedBars[0].label} (Ctrl+click to add, Shift+click for range)`);
    }
    
    // Prevent event from bubbling to drag handlers
    event.stopPropagation();
}

/**
 * Clear all bar selections
 * @param {boolean} clearAnchor - If true, also clear the selection anchor
 */
function clearBarSelection(clearAnchor = false) {
    selectedBars = [];
    d3.selectAll('.bar-group').classed('bar-selected', false);
    if (clearAnchor) {
        selectionAnchor = null;
    }
}

/**
 * Check if a bar is in the current selection
 */
function isBarSelected(d) {
    return selectedBars.some(b => b.label === d.label && b.codepath === d.codepath);
}

function dragBarStart(event, d) {
    // If the dragged bar is not in selection, clear selection and select only this bar
    if (!isBarSelected(d)) {
        clearBarSelection();
        selectedBars.push(d);
        d3.select(this).classed('bar-selected', true);
    }
    
    draggedBar = d;
    dragStartX = event.x;
    
    // Store original transforms for all selected bars
    originalTransforms = {};
    selectedBars.forEach(bar => {
        const barGroup = d3.selectAll('.bar-group').filter(b => b.label === bar.label && b.codepath === bar.codepath);
        originalTransforms[bar.label] = barGroup.attr('transform');
        // Reduce opacity to indicate dragging
        barGroup.select('.histogram-bar').attr('opacity', 0.3);
    });
    
    // Store the original transform of the dragged bar
    originalTransform = d3.select(this).attr('transform');
}

function dragBarMove(event, d) {
    // Visual feedback during drag - apply delta to all selected bars
    const deltaX = event.x - dragStartX;
    
    selectedBars.forEach(bar => {
        const barGroup = d3.selectAll('.bar-group').filter(b => b.label === bar.label && b.codepath === bar.codepath);
        const origTransform = originalTransforms[bar.label];
        const match = origTransform ? origTransform.match(/translate\(([^,]+),\s*([^)]+)\)/) : null;
        const baseX = match ? parseFloat(match[1]) : 0;
        const baseY = match ? parseFloat(match[2]) : 0;
        
        barGroup.attr('transform', `translate(${baseX + deltaX}, ${baseY})`);
    });
}

function dragBarEnd(event, d) {
    // Restore original transforms for all selected bars
    selectedBars.forEach(bar => {
        const barGroup = d3.selectAll('.bar-group').filter(b => b.label === bar.label && b.codepath === bar.codepath);
        barGroup.attr('transform', originalTransforms[bar.label]);
        // Restore opacity
        barGroup.select('.histogram-bar').attr('opacity', bar.isFixed ? 0.8 : 0.6);
    });
    
    if (!draggedBar) return;
    
    // Calculate target MFMA index based on drop position
    const svg = document.getElementById('histogram');
    const rect = svg.getBoundingClientRect();
    const margin = { left: 50, right: 20 };
    const width = rect.width - margin.left - margin.right;
    
    const relativeX = event.sourceEvent.clientX - rect.left - margin.left;
    const codepath = draggedBar.codepath !== undefined ? draggedBar.codepath : 0;
    
    // Convert X position to slot index using aligned layout
    const slotIndex = (relativeX / width) * alignedTotalSlots;
    
    // Find which MFMA region this falls into using aligned positions
    let targetMfma = -1;
    for (let mfmaIdx = 0; mfmaIdx < alignedMfmaPositions.length; mfmaIdx++) {
        if (slotIndex >= alignedMfmaPositions[mfmaIdx]) {
            targetMfma = mfmaIdx;
        } else {
            break;
        }
    }
    
    // Clamp to valid range
    targetMfma = Math.max(-1, Math.min(targetMfma, currentSchedule.num_mfma - 1));
    
    // Calculate the relative movement based on the dragged bar
    const deltaMove = targetMfma - draggedBar.mfmaIndex;
    
    if (deltaMove !== 0) {
        if (selectedBars.length > 1) {
            // Multi-bar move: move all selected bars by the same relative amount
            const moves = selectedBars.map(bar => ({
                instruction: bar.label,
                fromMfma: bar.mfmaIndex,
                toMfma: Math.max(-1, Math.min(bar.mfmaIndex + deltaMove, currentSchedule.num_mfma - 1))
            }));
            
            vscode.postMessage({
                command: 'moveInstructions',
                moves: moves,
                codepath: codepath
            });
            
            updateStatus(`Moved ${selectedBars.length} instructions by ${deltaMove > 0 ? '+' : ''}${deltaMove}`);
        } else {
            // Single bar move
            vscode.postMessage({
                command: 'moveInstruction',
                instruction: draggedBar.label,
                fromMfma: draggedBar.mfmaIndex,
                toMfma: targetMfma,
                codepath: codepath
            });
        }
    }
    
    draggedBar = null;
    originalTransform = null;
    originalTransforms = {};
}

// ============================================================================
// Sync Panel Functions
// ============================================================================

/**
 * Initialize sync panel event delegation (call once on startup)
 */
function initSyncPanelEvents() {
    const tbody = document.getElementById('sync-table-body');
    if (!tbody || tbody.dataset.eventsInitialized) return;
    
    tbody.dataset.eventsInitialized = 'true';
    
    // Use event delegation for all button clicks in the table
    tbody.addEventListener('click', function(e) {
        const target = e.target;
        
        // Find the button that was clicked
        const button = target.closest('button');
        if (!button) return;
        
        // Find the row and get the row index
        const row = button.closest('tr');
        if (!row) return;
        
        const rowIndex = Array.from(tbody.children).indexOf(row);
        if (rowIndex < 0 || !currentSchedule || !currentSchedule.sync_code) return;
        
        const item = currentSchedule.sync_code[rowIndex];
        if (!item) return;
        
        e.preventDefault();
        e.stopPropagation();
        
        // Check which action
        if (button.classList.contains('edit-sync-btn')) {
            showEditSyncDialog(item, rowIndex);
        } else if (button.classList.contains('delete-sync-btn')) {
            deleteSyncInstruction(item, rowIndex);
        }
    });
}

/**
 * Render the sync instructions panel
 */
function renderSyncPanel() {
    const tbody = document.getElementById('sync-table-body');
    if (!tbody) return;
    
    // Initialize event delegation if not done
    initSyncPanelEvents();
    
    tbody.innerHTML = '';
    
    if (!currentSchedule || !currentSchedule.sync_code) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="7" class="no-data">No sync instructions loaded</td>';
        tbody.appendChild(tr);
        return;
    }
    
    currentSchedule.sync_code.forEach((item, idx) => {
        const row = createSyncRow(item, idx);
        tbody.appendChild(row);
    });
}

/**
 * Create a table row for a sync instruction
 */
function createSyncRow(item, rowIndex) {
    const tr = document.createElement('tr');
    const inst = item.instruction;
    const index = item.index;
    
    // Name column (SYNC[0], SYNC[1], etc.)
    const nameTd = document.createElement('td');
    nameTd.textContent = `SYNC[${rowIndex}]`;
    nameTd.className = 'sync-name';
    tr.appendChild(nameTd);
    
    // Index column (MFMA index)
    const indexTd = document.createElement('td');
    indexTd.textContent = index;
    indexTd.className = 'sync-index';
    tr.appendChild(indexTd);
    
    // Type column
    const typeTd = document.createElement('td');
    typeTd.textContent = inst.type;
    typeTd.className = `sync-type sync-type-${inst.type.toLowerCase()}`;
    tr.appendChild(typeTd);
    
    // dscnt column
    const dscntTd = document.createElement('td');
    if (inst.type === 'SWaitCnt') {
        dscntTd.innerHTML = formatWaitCountCell(inst.dscnt, 'dscnt', rowIndex);
    } else {
        dscntTd.textContent = '-';
        dscntTd.className = 'sync-na';
    }
    tr.appendChild(dscntTd);
    
    // vlcnt column
    const vlcntTd = document.createElement('td');
    if (inst.type === 'SWaitCnt') {
        vlcntTd.innerHTML = formatWaitCountCell(inst.vlcnt, 'vlcnt', rowIndex);
    } else {
        vlcntTd.textContent = '-';
        vlcntTd.className = 'sync-na';
    }
    tr.appendChild(vlcntTd);
    
    // Comment column
    const commentTd = document.createElement('td');
    commentTd.textContent = inst.comment || '';
    commentTd.className = 'sync-comment';
    tr.appendChild(commentTd);
    
    // Actions column
    const actionsTd = document.createElement('td');
    actionsTd.className = 'sync-actions';
    
    const editBtn = document.createElement('button');
    editBtn.textContent = '✏️';
    editBtn.title = 'Edit';
    editBtn.className = 'icon-btn-small edit-sync-btn';
    editBtn.type = 'button';
    actionsTd.appendChild(editBtn);
    
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '🗑️';
    deleteBtn.title = 'Delete';
    deleteBtn.className = 'icon-btn-small delete-sync-btn';
    deleteBtn.type = 'button';
    actionsTd.appendChild(deleteBtn);
    
    tr.appendChild(actionsTd);
    
    return tr;
}

/**
 * Format a wait count cell (dscnt, vlcnt, vscnt)
 */
function formatWaitCountCell(value, fieldName, rowIndex) {
    if (value === -1 || value === undefined || value === null) {
        return '<span class="sync-na">-1</span>';
    }
    
    if (typeof value === 'object' && value.expr) {
        // Dynamic value
        const displayValue = value.values ? value.values[currentCodepath] : '?';
        return `<span class="sync-dynamic" title="${value.expr}">
            <span class="dynamic-badge">D</span>
            ${displayValue}
        </span>`;
    }
    
    // Hardcoded value
    return `<span class="sync-value">${value}</span>`;
}

/**
 * Show dialog to add a new sync instruction
 */
function showAddSyncDialog() {
    const dialog = createSyncDialog('Add Sync Instruction', null, (data) => {
        const instruction = {
            index: data.index,
            instruction: buildSyncInstruction(data)
        };
        
        // Add to schedule
        if (!currentSchedule.sync_code) {
            currentSchedule.sync_code = [];
        }
        currentSchedule.sync_code.push(instruction);
        currentSchedule.sync_code.sort((a, b) => a.index - b.index);
        
        // Update sync indices in optSchedule
        updateSyncIndices();
        
        // Send to extension
        vscode.postMessage({
            command: 'addSyncInstruction',
            instruction: instruction
        });
        
        renderSyncPanel();
    });
    
    document.body.appendChild(dialog);
}

/**
 * Show dialog to edit an existing sync instruction
 */
function showEditSyncDialog(item, rowIndex) {
    const dialog = createSyncDialog('Edit Sync Instruction', item, (data) => {
        const instruction = {
            index: data.index,
            instruction: buildSyncInstruction(data)
        };
        
        // Update in schedule
        currentSchedule.sync_code[rowIndex] = instruction;
        currentSchedule.sync_code.sort((a, b) => a.index - b.index);
        
        // Update sync indices in optSchedule
        updateSyncIndices();
        
        // Send to extension
        vscode.postMessage({
            command: 'updateSyncCode',
            syncCode: currentSchedule.sync_code
        });
        
        renderSyncPanel();
    });
    
    document.body.appendChild(dialog);
}

/**
 * Delete a sync instruction
 */
function deleteSyncInstruction(item, rowIndex) {
    try {
        // Remove from local array
        if (!currentSchedule || !currentSchedule.sync_code) {
            updateStatus('ERROR: No schedule loaded');
            return;
        }
        
        if (rowIndex < 0 || rowIndex >= currentSchedule.sync_code.length) {
            updateStatus('ERROR: Invalid row index');
            return;
        }
        
        const deletedItem = currentSchedule.sync_code[rowIndex];
        currentSchedule.sync_code.splice(rowIndex, 1);
        
        // Update sync indices in optSchedule
        updateSyncIndices();
        
        // If no syncs left, remove SYNC from instruction_types and opt_schedule
        if (currentSchedule.sync_code.length === 0) {
            if (currentSchedule.instruction_types) {
                currentSchedule.instruction_types = currentSchedule.instruction_types.filter(t => t !== 'SYNC');
            }
            if (currentSchedule.opt_schedule && currentSchedule.opt_schedule['SYNC']) {
                delete currentSchedule.opt_schedule['SYNC'];
            }
        }
        
        // Send to extension
        vscode.postMessage({
            command: 'removeSyncInstruction',
            index: item.index,
            instructionIndex: rowIndex
        });
        
        // Re-render the panel
        renderSyncPanel();
        
        // Also update the histogram to reflect the change
        renderHistogram();
        
        updateStatus(`Deleted sync instruction at MFMA ${item.index}`);
    } catch (err) {
        updateStatus('ERROR: ' + (err.message || 'Failed to delete'));
    }
}

/**
 * Create the sync instruction dialog
 */
function createSyncDialog(title, existingItem, onSave) {
    const overlay = document.createElement('div');
    overlay.className = 'dialog-overlay';
    
    const dialog = document.createElement('div');
    dialog.className = 'dialog';
    
    const inst = existingItem ? existingItem.instruction : null;
    const index = existingItem ? existingItem.index : 0;
    
    dialog.innerHTML = `
        <div class="dialog-header">
            <h3>${title}</h3>
            <button class="dialog-close">&times;</button>
        </div>
        <div class="dialog-content">
            <div class="form-group">
                <label>Type:</label>
                <select id="sync-type">
                    <option value="SWaitCnt" ${inst?.type === 'SWaitCnt' ? 'selected' : ''}>SWaitCnt</option>
                    <option value="SBarrier" ${inst?.type === 'SBarrier' ? 'selected' : ''}>SBarrier</option>
                    <option value="SNop" ${inst?.type === 'SNop' ? 'selected' : ''}>SNop</option>
                </select>
            </div>
            <div class="form-group">
                <label>MFMA Index:</label>
                <input type="number" id="sync-index" value="${index}" min="-1" max="${currentSchedule?.num_mfma || 96}">
            </div>
            <div id="swaitcnt-fields" class="conditional-fields">
                <div class="form-group">
                    <label>dscnt:</label>
                    ${createWaitCountEditor('dscnt', inst?.dscnt)}
                </div>
                <div class="form-group">
                    <label>vlcnt:</label>
                    ${createWaitCountEditor('vlcnt', inst?.vlcnt)}
                </div>
                <div class="form-group">
                    <label>vscnt:</label>
                    ${createWaitCountEditor('vscnt', inst?.vscnt)}
                </div>
            </div>
            <div id="snop-fields" class="conditional-fields" style="display: none;">
                <div class="form-group">
                    <label>Count:</label>
                    <input type="number" id="snop-count" value="${inst?.count || 1}" min="1">
                </div>
            </div>
            <div class="form-group">
                <label>Comment:</label>
                <input type="text" id="sync-comment" value="${inst?.comment || ''}" placeholder="Description...">
            </div>
        </div>
        <div class="dialog-footer">
            <button class="btn-secondary" id="dialog-cancel">Cancel</button>
            <button class="btn-primary" id="dialog-save">Save</button>
        </div>
    `;
    
    overlay.appendChild(dialog);
    
    // Event handlers
    const closeDialog = () => overlay.remove();
    
    overlay.querySelector('.dialog-close').onclick = closeDialog;
    overlay.querySelector('#dialog-cancel').onclick = closeDialog;
    overlay.onclick = (e) => { if (e.target === overlay) closeDialog(); };
    
    // Type change handler
    const typeSelect = dialog.querySelector('#sync-type');
    const swaitcntFields = dialog.querySelector('#swaitcnt-fields');
    const snopFields = dialog.querySelector('#snop-fields');
    
    typeSelect.onchange = () => {
        const type = typeSelect.value;
        swaitcntFields.style.display = type === 'SWaitCnt' ? 'block' : 'none';
        snopFields.style.display = type === 'SNop' ? 'block' : 'none';
    };
    typeSelect.onchange(); // Initialize
    
    // Mode select handlers (hardcoded vs dynamic)
    dialog.querySelectorAll('.mode-select').forEach(select => {
        select.onchange = function() {
            const editor = this.closest('.wait-count-editor');
            const hardcodedInput = editor.querySelector('.hardcoded-input');
            const dynamicInput = editor.querySelector('.dynamic-input');
            
            if (this.value === 'hardcoded') {
                hardcodedInput.style.display = 'block';
                dynamicInput.style.display = 'none';
            } else {
                hardcodedInput.style.display = 'none';
                dynamicInput.style.display = 'block';
            }
        };
    });
    
    // Rule select handlers (show/hide N input based on rule)
    dialog.querySelectorAll('.rule-select').forEach(select => {
        select.onchange = function() {
            const editor = this.closest('.wait-count-editor');
            const nInput = editor.querySelector('.n-input');
            
            if (this.value === 'finish_all') {
                nInput.style.display = 'none';
            } else {
                nInput.style.display = 'block';
            }
        };
    });
    
    // Save handler
    dialog.querySelector('#dialog-save').onclick = () => {
        const data = collectDialogData(dialog);
        onSave(data);
        closeDialog();
    };
    
    return overlay;
}

/**
 * Create wait count editor HTML (hardcoded/dynamic selector)
 */
function createWaitCountEditor(field, value) {
    const isDynamic = typeof value === 'object' && value?.expr;
    const hardcodedValue = isDynamic ? -1 : (value ?? -1);
    const expr = isDynamic ? value.expr : '';
    
    return `
        <div class="wait-count-editor" data-field="${field}">
            <select class="mode-select" data-field="${field}">
                <option value="hardcoded" ${!isDynamic ? 'selected' : ''}>Hardcoded</option>
                <option value="dynamic" ${isDynamic ? 'selected' : ''}>Dynamic</option>
            </select>
            <div class="hardcoded-input" ${isDynamic ? 'style="display:none"' : ''}>
                <input type="number" class="wait-value" value="${hardcodedValue}" min="-1">
            </div>
            <div class="dynamic-input" ${!isDynamic ? 'style="display:none"' : ''}>
                ${createDynamicRuleEditor(field, expr)}
            </div>
        </div>
    `;
}

/**
 * Create dynamic rule editor (rule selector + type selector)
 */
function createDynamicRuleEditor(field, existingExpr) {
    // Parse existing expression to get rule and types
    const parsed = parseDynamicExpression(existingExpr);
    const isLds = field === 'dscnt';
    const prefix = isLds ? 'dscnt' : 'vlcnt';
    
    // Get available types from optSchedule
    const availableTypes = getAvailableTypes(isLds);
    
    return `
        <select class="rule-select" data-field="${field}">
            <option value="finish_all" ${parsed.rule === 'finish_all' ? 'selected' : ''}>Wait for all to finish</option>
            <option value="finish_n" ${parsed.rule === 'finish_n' ? 'selected' : ''}>Wait for N to finish</option>
            <option value="keep_n" ${parsed.rule === 'keep_n' ? 'selected' : ''}>Keep N in flight</option>
        </select>
        <div class="type-selector">
            <label>Types:</label>
            <select class="type-select" data-field="${field}" multiple>
                ${availableTypes.map(t => 
                    `<option value="${t}" ${parsed.types.includes(t) ? 'selected' : ''}>${t}</option>`
                ).join('')}
            </select>
        </div>
        <div class="n-input" ${parsed.rule === 'finish_all' ? 'style="display:none"' : ''}>
            <label>N:</label>
            <input type="number" class="n-value" value="${parsed.n || 1}" min="1">
        </div>
    `;
}

/**
 * Get available instruction types for the type selector
 */
function getAvailableTypes(isLds) {
    if (!currentSchedule || !currentSchedule.opt_schedule) return [];
    
    const prefix = isLds ? 'LR' : 'GR';
    return Object.keys(currentSchedule.opt_schedule)
        .filter(key => key.startsWith(prefix))
        .sort();
}

/**
 * Parse a dynamic expression to extract rule, types, and n
 */
function parseDynamicExpression(expr) {
    if (!expr) return { rule: 'finish_all', types: [], n: null };
    
    // Match patterns like: dscnt_after_finish(optSchedule, 'LRA0', 22)
    // or: dscnt_after_finish(optSchedule, ['LRA0', 'LRB0'], 22)
    
    let rule = 'finish_all';
    let types = [];
    let n = null;
    
    if (expr.includes('after_finish')) rule = 'finish_all';
    else if (expr.includes('after_n_finish')) rule = 'finish_n';
    else if (expr.includes('with_n_inflight')) rule = 'keep_n';
    
    // Extract types
    const typesMatch = expr.match(/\[\s*'([^']+)'(?:\s*,\s*'([^']+)')*\s*\]/);
    if (typesMatch) {
        types = expr.match(/'([^']+)'/g)?.map(s => s.replace(/'/g, '')) || [];
    } else {
        const singleType = expr.match(/,\s*'([^']+)'/);
        if (singleType) types = [singleType[1]];
    }
    
    // Extract n for finish_n or keep_n
    if (rule !== 'finish_all') {
        const nMatch = expr.match(/,\s*(\d+)\s*,/);
        if (nMatch) n = parseInt(nMatch[1]);
    }
    
    return { rule, types, n };
}

/**
 * Build a dynamic expression from rule editor state
 */
function buildDynamicExpression(field, rule, types, n, index) {
    const prefix = field === 'dscnt' ? 'dscnt' : 'vlcnt';
    const typesStr = types.length === 1 ? `'${types[0]}'` : `[${types.map(t => `'${t}'`).join(', ')}]`;
    
    if (rule === 'finish_all') {
        return `${prefix}_after_finish(optSchedule, ${typesStr}, ${index})`;
    } else if (rule === 'finish_n') {
        return `${prefix}_after_n_finish(optSchedule, ${typesStr}, ${n}, ${index})`;
    } else if (rule === 'keep_n') {
        return `${prefix}_with_n_inflight(optSchedule, ${typesStr}, ${n}, ${index})`;
    }
    
    return '-1';
}

/**
 * Collect data from the dialog form
 */
function collectDialogData(dialog) {
    const type = dialog.querySelector('#sync-type').value;
    const index = parseInt(dialog.querySelector('#sync-index').value);
    const comment = dialog.querySelector('#sync-comment').value;
    
    const data = { type, index, comment };
    
    if (type === 'SWaitCnt') {
        data.dscnt = collectWaitCountValue(dialog, 'dscnt', index);
        data.vlcnt = collectWaitCountValue(dialog, 'vlcnt', index);
        data.vscnt = collectWaitCountValue(dialog, 'vscnt', index);
    } else if (type === 'SNop') {
        data.count = parseInt(dialog.querySelector('#snop-count').value);
    }
    
    return data;
}

/**
 * Collect wait count value from the editor
 */
function collectWaitCountValue(dialog, field, index) {
    const editor = dialog.querySelector(`.wait-count-editor[data-field="${field}"]`);
    const mode = editor.querySelector('.mode-select').value;
    
    if (mode === 'hardcoded') {
        return parseInt(editor.querySelector('.wait-value').value);
    } else {
        const rule = editor.querySelector('.rule-select').value;
        const typeSelect = editor.querySelector('.type-select');
        const types = Array.from(typeSelect.selectedOptions).map(o => o.value);
        const nInput = editor.querySelector('.n-value');
        const n = nInput ? parseInt(nInput.value) : null;
        
        const expr = buildDynamicExpression(field, rule, types, n, index);
        
        // Compute values for each codepath
        const values = [];
        for (let cp = 0; cp < (currentSchedule?.num_codepaths || 1); cp++) {
            // For now, just store placeholder - actual computation happens in Python
            values.push(-1);
        }
        
        return { expr, values };
    }
}

/**
 * Build a sync instruction object from dialog data
 */
function buildSyncInstruction(data) {
    const inst = { type: data.type };
    
    if (data.type === 'SWaitCnt') {
        inst.dscnt = data.dscnt;
        inst.vlcnt = data.vlcnt;
        inst.vscnt = data.vscnt;
    } else if (data.type === 'SNop') {
        inst.count = data.count;
    }
    
    inst.comment = data.comment || '';
    
    return inst;
}

/**
 * Update SYNC indices in optSchedule to match sync_code
 */
function updateSyncIndices() {
    if (!currentSchedule || !currentSchedule.sync_code) return;
    
    const indices = currentSchedule.sync_code.map(item => item.index);
    
    // Update optSchedule SYNC
    if (currentSchedule.opt_schedule) {
        currentSchedule.opt_schedule['SYNC'] = [indices];
    }
}

// ============================================================================
// MFMA Reorder Panel Functions
// ============================================================================

/**
 * Render the MFMA reorder panel
 */
function renderMfmaReorderPanel() {
    const infoEl = document.getElementById('mfma-reorder-info');
    const listEl = document.getElementById('mfma-reorder-list');
    
    if (!infoEl || !listEl) return;
    
    if (!currentSchedule || !currentSchedule.mfma_reorder || currentSchedule.mfma_reorder.length === 0) {
        infoEl.textContent = 'No MFMA reorder defined';
        infoEl.style.display = 'block';
        listEl.innerHTML = '';
        listEl.style.display = 'none';
        return;
    }
    
    const reorder = currentSchedule.mfma_reorder;
    const numMfma = currentSchedule.num_mfma || reorder.length;
    
    infoEl.innerHTML = `<strong>${reorder.length} MFMAs</strong> - Execution order maps logical MFMA index to physical execution position`;
    infoEl.style.display = 'block';
    
    // Build the grid showing reorder mapping
    listEl.innerHTML = '';
    listEl.style.display = 'grid';
    
    // Add header row
    const headerRow = document.createElement('div');
    headerRow.className = 'mfma-reorder-header';
    headerRow.innerHTML = `
        <span class="mfma-col-label">Position</span>
        <span class="mfma-col-label">→</span>
        <span class="mfma-col-label">MFMA#</span>
    `;
    listEl.appendChild(headerRow);
    
    // Show the reorder mapping - position i executes MFMA reorder[i]
    // Group by quadrants for readability
    const quadrantSize = Math.ceil(numMfma / 4);
    let currentQuadrant = -1;
    
    reorder.forEach((mfmaIndex, position) => {
        const quadrant = Math.floor(position / quadrantSize);
        
        // Add quadrant separator
        if (quadrant !== currentQuadrant) {
            currentQuadrant = quadrant;
            const separator = document.createElement('div');
            separator.className = 'mfma-quadrant-separator';
            separator.textContent = `Quadrant ${quadrant + 1} (positions ${quadrant * quadrantSize}-${Math.min((quadrant + 1) * quadrantSize - 1, numMfma - 1)})`;
            listEl.appendChild(separator);
        }
        
        const row = document.createElement('div');
        row.className = 'mfma-reorder-row';
        
        // Highlight if position != mfmaIndex (reordered)
        if (position !== mfmaIndex) {
            row.classList.add('mfma-reordered');
        }
        
        row.innerHTML = `
            <span class="mfma-position">${position}</span>
            <span class="mfma-arrow">→</span>
            <span class="mfma-index">${mfmaIndex}</span>
        `;
        
        listEl.appendChild(row);
    });
    
    // Add summary
    const reorderedCount = reorder.filter((val, idx) => val !== idx).length;
    const summary = document.createElement('div');
    summary.className = 'mfma-reorder-summary';
    summary.textContent = `${reorderedCount} of ${reorder.length} MFMAs are reordered from their natural position`;
    listEl.appendChild(summary);
}
