/**
 * CMS Schedule Editor - Webview Panel Provider
 * 
 * Manages the webview panel that displays the interactive histogram.
 */

import * as vscode from 'vscode';
import { ScheduleData, LatencyData, WebviewMessage, INSTRUCTION_CLASS_COLORS, LogLevel } from './types';
import { PythonBridge } from './pythonBridge';
import { log, logError } from './logger';

export class ScheduleEditorPanel {
    public static currentPanel: ScheduleEditorPanel | undefined;
    public static readonly viewType = 'cmsScheduleEditor';

    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _schedule: ScheduleData | undefined;
    private _latencies: LatencyData | undefined;
    private _pythonBridge: PythonBridge;
    private _disposables: vscode.Disposable[] = [];
    private _disposeCallbacks: (() => void)[] = [];

    public static createOrShow(
        extensionUri: vscode.Uri, 
        schedule: ScheduleData,
        pythonBridge: PythonBridge
    ) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        // If we already have a panel, show it
        if (ScheduleEditorPanel.currentPanel) {
            ScheduleEditorPanel.currentPanel._panel.reveal(column);
            ScheduleEditorPanel.currentPanel.updateSchedule(schedule);
            return;
        }

        // Create a new panel
        const panel = vscode.window.createWebviewPanel(
            ScheduleEditorPanel.viewType,
            'CMS Schedule Editor',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(extensionUri, 'media'),
                    vscode.Uri.joinPath(extensionUri, 'webview')
                ]
            }
        );

        ScheduleEditorPanel.currentPanel = new ScheduleEditorPanel(
            panel, 
            extensionUri, 
            schedule,
            pythonBridge
        );
    }

    private constructor(
        panel: vscode.WebviewPanel, 
        extensionUri: vscode.Uri,
        schedule: ScheduleData,
        pythonBridge: PythonBridge
    ) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._schedule = schedule;
        this._pythonBridge = pythonBridge;

        // Set the webview's initial html content
        this._update();

        // Listen for when the panel is disposed
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        // Handle messages from the webview
        this._panel.webview.onDidReceiveMessage(
            (message: WebviewMessage) => this._handleMessage(message),
            null,
            this._disposables
        );
    }

    public dispose() {
        ScheduleEditorPanel.currentPanel = undefined;

        // Call dispose callbacks
        for (const callback of this._disposeCallbacks) {
            try {
                callback();
            } catch (e) {
                logError('Error in dispose callback', e);
            }
        }

        // Clean up resources
        this._panel.dispose();

        while (this._disposables.length) {
            const d = this._disposables.pop();
            if (d) {
                d.dispose();
            }
        }
    }

    /**
     * Register a callback to be called when the panel is disposed
     */
    public onDispose(callback: () => void) {
        this._disposeCallbacks.push(callback);
    }

    /**
     * Update the schedule data and refresh the view
     */
    public updateSchedule(schedule: ScheduleData) {
        this._schedule = schedule;
        this._panel.webview.postMessage({
            command: 'updateSchedule',
            schedule: schedule
        });
    }

    /**
     * Update the latency data
     */
    public updateLatencies(latencies: LatencyData) {
        this._latencies = latencies;
        this._panel.webview.postMessage({
            command: 'updateLatencies',
            latencies: latencies
        });
    }

    /**
     * Get the current schedule data
     */
    public getCurrentSchedule(): ScheduleData | undefined {
        return this._schedule;
    }

    /**
     * Handle messages from the webview
     */
    private async _handleMessage(message: WebviewMessage) {
        switch (message.command) {
            case 'ready':
                // Webview is ready, send initial data
                if (this._schedule) {
                    this._panel.webview.postMessage({
                        command: 'updateSchedule',
                        schedule: this._schedule
                    });
                }
                if (this._latencies) {
                    this._panel.webview.postMessage({
                        command: 'updateLatencies',
                        latencies: this._latencies
                    });
                }
                break;

            case 'reorderInstructions':
                // Update instruction type order
                if (this._schedule && message.newOrder) {
                    this._schedule.instruction_types = message.newOrder;
                    // Reorder opt_schedule to match
                    const newOptSchedule: Record<string, any> = {};
                    for (const key of message.newOrder) {
                        if (key in this._schedule.opt_schedule) {
                            newOptSchedule[key] = this._schedule.opt_schedule[key];
                        }
                    }
                    // Add any keys not in newOrder
                    for (const key of Object.keys(this._schedule.opt_schedule)) {
                        if (!(key in newOptSchedule)) {
                            newOptSchedule[key] = this._schedule.opt_schedule[key];
                        }
                    }
                    this._schedule.opt_schedule = newOptSchedule;
                    
                    // Send updated schedule back to webview
                    this._panel.webview.postMessage({
                        command: 'updateSchedule',
                        schedule: this._schedule
                    });
                    log('info', `Reordered instructions: ${message.newOrder.join(', ')}`, 'extension');
                }
                break;

            case 'moveInstruction':
                // Move instruction to different MFMA index
                if (this._schedule && message.instruction && message.toMfma !== undefined) {
                    const moved = this._moveInstruction(
                        message.instruction,
                        message.fromMfma,
                        message.toMfma,
                        message.codepath || 0
                    );
                    if (moved) {
                        // Send updated schedule back to webview
                        this._panel.webview.postMessage({
                            command: 'updateSchedule',
                            schedule: this._schedule
                        });
                        log('info', `Moved ${message.instruction} from MFMA ${message.fromMfma} to ${message.toMfma}`, 'extension');
                    }
                }
                break;

            case 'moveInstructions':
                // Move multiple instructions at once (multi-select)
                if (this._schedule && message.moves && Array.isArray(message.moves)) {
                    const codepath = message.codepath || 0;
                    let allMoved = true;
                    const movedLabels: string[] = [];
                    
                    for (const move of message.moves) {
                        const moved = this._moveInstruction(
                            move.instruction,
                            move.fromMfma,
                            move.toMfma,
                            codepath
                        );
                        if (moved) {
                            movedLabels.push(move.instruction);
                        } else {
                            allMoved = false;
                        }
                    }
                    
                    if (movedLabels.length > 0) {
                        // Send updated schedule back to webview
                        this._panel.webview.postMessage({
                            command: 'updateSchedule',
                            schedule: this._schedule
                        });
                        log('info', `Moved ${movedLabels.length} instructions: ${movedLabels.join(', ')}`, 'extension');
                    }
                }
                break;

            case 'splitSchedule':
                // Split a shared schedule into per-codepath schedules
                if (this._schedule && message.type && message.newSchedule) {
                    if (this._schedule.opt_schedule) {
                        this._schedule.opt_schedule[message.type] = message.newSchedule;
                        log('info', `Split schedule for ${message.type} into ${message.newSchedule.length} codepaths`, 'extension');
                    }
                }
                break;

            case 'updateSyncCode':
                // Update the entire sync code array
                if (this._schedule && message.syncCode) {
                    this._schedule.sync_code = message.syncCode;
                }
                break;

            case 'addSyncInstruction':
                // Add a new sync instruction
                if (this._schedule && message.instruction) {
                    if (!this._schedule.sync_code) {
                        this._schedule.sync_code = [];
                    }
                    this._schedule.sync_code.push(message.instruction);
                    // Sort by index
                    this._schedule.sync_code.sort((a, b) => a.index - b.index);
                }
                break;

            case 'removeSyncInstruction':
                // Remove a sync instruction
                if (this._schedule && this._schedule.sync_code) {
                    this._schedule.sync_code = this._schedule.sync_code.filter(
                        (item, idx) => !(item.index === message.index && idx === message.instructionIndex)
                    );
                    
                    // Update opt_schedule SYNC indices
                    if (this._schedule.opt_schedule) {
                        const syncIndices = this._schedule.sync_code.map(item => item.index);
                        if (syncIndices.length > 0) {
                            this._schedule.opt_schedule['SYNC'] = [syncIndices];
                        } else {
                            // No syncs left - remove SYNC from opt_schedule and instruction_types
                            delete this._schedule.opt_schedule['SYNC'];
                            if (this._schedule.instruction_types) {
                                this._schedule.instruction_types = this._schedule.instruction_types.filter(
                                    t => t !== 'SYNC'
                                );
                            }
                        }
                    }
                    
                    // Send updated schedule back to webview
                    this._panel.webview.postMessage({ 
                        command: 'updateSchedule', 
                        schedule: this._schedule 
                    });
                }
                break;

            case 'requestSave':
                vscode.commands.executeCommand('cms.saveSchedule');
                break;

            case 'requestRun':
                vscode.commands.executeCommand('cms.runSchedule');
                break;

            case 'requestValidate':
                vscode.commands.executeCommand('cms.validateSchedule');
                break;

            case 'error':
                logError(message.message, undefined, 'webview');
                vscode.window.showErrorMessage(message.message);
                break;

            case 'log':
                // Handle log messages from webview
                const logLevel: LogLevel = message.level || 'info';
                const logMessage = message.details 
                    ? `${message.message}: ${JSON.stringify(message.details)}`
                    : message.message;
                log(logLevel, logMessage, 'webview');
                break;
        }
    }

    /**
     * Move an instruction to a different MFMA index
     * 
     * @param instructionLabel - Label like "LRA0[2]" or "SYNC[0]" (type + index)
     * @param fromMfma - Original MFMA index
     * @param toMfma - Target MFMA index
     * @param codepath - Which codepath to modify (0-indexed)
     * @returns true if the move was successful
     */
    private _moveInstruction(
        instructionLabel: string,
        fromMfma: number,
        toMfma: number,
        codepath: number
    ): boolean {
        if (!this._schedule) {
            return false;
        }

        // Parse the instruction label: "LRA0[2]" -> type="LRA0", index=2
        const match = instructionLabel.match(/^([A-Za-z0-9]+)\[(\d+)\]$/);
        if (!match) {
            logError(`Invalid instruction label format: ${instructionLabel}`, undefined, 'extension');
            return false;
        }

        const instType = match[1];
        const instIndex = parseInt(match[2], 10);

        // Skip MFMA instructions - they are fixed
        if (instType === 'MFMA') {
            log('warn', 'Cannot move MFMA instructions', 'extension');
            return false;
        }

        // Handle SYNC instructions specially - they're stored in sync_code array
        if (instType === 'SYNC') {
            return this._moveSyncInstruction(instIndex, fromMfma, toMfma);
        }

        // Find the instruction type in opt_schedule
        const schedule = this._schedule.opt_schedule[instType];
        if (!schedule || !Array.isArray(schedule)) {
            logError(`Instruction type ${instType} not found in opt_schedule`, undefined, 'extension');
            return false;
        }

        // Get the codepath array
        let codepathArray: number[];
        if (codepath < schedule.length) {
            codepathArray = schedule[codepath];
        } else if (schedule.length > 0) {
            // Use first codepath if specified one doesn't exist
            codepathArray = schedule[0];
        } else {
            logError(`No codepath data for ${instType}`, undefined, 'extension');
            return false;
        }

        // Check if the index is valid
        if (instIndex < 0 || instIndex >= codepathArray.length) {
            logError(`Index ${instIndex} out of range for ${instType} (length: ${codepathArray.length})`, undefined, 'extension');
            return false;
        }

        // Verify the current MFMA index matches
        if (codepathArray[instIndex] !== fromMfma) {
            log('warn', `Expected ${instType}[${instIndex}] at MFMA ${fromMfma}, but found ${codepathArray[instIndex]}`, 'extension');
        }

        // Update the MFMA index
        codepathArray[instIndex] = toMfma;

        // If there's only one codepath array and multiple codepaths, update all
        // (some instructions share the same schedule across all codepaths)
        if (schedule.length === 1 && this._schedule.num_codepaths > 1) {
            // Already updated via reference
        }

        return true;
    }

    /**
     * Move a SYNC instruction to a different MFMA index
     * 
     * SYNC instructions are stored in sync_code array with an 'index' property.
     * We also need to update opt_schedule['SYNC'] to keep them in sync.
     * 
     * @param syncIndex - Index into the sync_code array (0-indexed)
     * @param fromMfma - Original MFMA index
     * @param toMfma - Target MFMA index
     * @returns true if the move was successful
     */
    private _moveSyncInstruction(
        syncIndex: number,
        fromMfma: number,
        toMfma: number
    ): boolean {
        if (!this._schedule || !this._schedule.sync_code) {
            logError('No sync_code in schedule', undefined, 'extension');
            return false;
        }

        // Check if the sync index is valid
        if (syncIndex < 0 || syncIndex >= this._schedule.sync_code.length) {
            logError(`SYNC index ${syncIndex} out of range (length: ${this._schedule.sync_code.length})`, undefined, 'extension');
            return false;
        }

        const syncEntry = this._schedule.sync_code[syncIndex];

        // Verify the current MFMA index matches
        if (syncEntry.index !== fromMfma) {
            log('warn', `Expected SYNC[${syncIndex}] at MFMA ${fromMfma}, but found ${syncEntry.index}`, 'extension');
        }

        // Update the MFMA index in sync_code
        syncEntry.index = toMfma;

        // Also update opt_schedule['SYNC'] to keep them in sync
        if (this._schedule.opt_schedule['SYNC']) {
            // Rebuild the SYNC indices from sync_code
            const syncIndices = this._schedule.sync_code.map(s => s.index);
            this._schedule.opt_schedule['SYNC'] = [syncIndices];
        }

        // Re-sort sync_code by index to maintain order
        this._schedule.sync_code.sort((a, b) => a.index - b.index);

        log('info', `Moved SYNC[${syncIndex}] from MFMA ${fromMfma} to ${toMfma}`, 'extension');
        return true;
    }

    /**
     * Update the webview content
     */
    private _update() {
        this._panel.title = 'CMS Schedule Editor';
        this._panel.webview.html = this._getHtmlContent();
    }

    /**
     * Generate the HTML content for the webview
     */
    private _getHtmlContent(): string {
        const webview = this._panel.webview;

        // Get URIs for local resources
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'histogram.js')
        );
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'histogram.css')
        );

        // Use a nonce to only allow specific scripts
        const nonce = getNonce();

        // Serialize colors for the webview
        const colorsJson = JSON.stringify(INSTRUCTION_CLASS_COLORS);

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}' https://d3js.org; img-src ${webview.cspSource};">
    <link href="${styleUri}" rel="stylesheet">
    <title>CMS Schedule Editor</title>
</head>
<body>
    <div id="app">
        <div id="toolbar">
            <button id="run-btn" title="Run Schedule (Ctrl+Shift+R)">▶ Run</button>
            <button id="save-btn" title="Save Schedule (Ctrl+S)">💾 Save</button>
            <button id="validate-btn" title="Validate Schedule">✓ Validate</button>
            <button id="toggle-view-btn" title="Toggle between latency view and flat view">📊 Latencies</button>
            <span id="status">Ready</span>
            <span id="schedule-info"></span>
        </div>
        <div id="main-content">
            <div id="left-content">
                <div id="histogram-container">
                    <div id="histogram-header">
                        <h3>Instruction Histogram</h3>
                        <div id="legend"></div>
                    </div>
                    <svg id="histogram"></svg>
                </div>
                <div id="sync-panel" class="collapsible-panel">
                    <div class="panel-header">
                        <h3>Sync Instructions</h3>
                        <div class="panel-actions">
                            <button id="add-sync-btn" class="icon-btn" title="Add Sync Instruction">+ Add</button>
                            <button id="toggle-sync-panel" class="icon-btn" title="Toggle Panel">▼</button>
                        </div>
                    </div>
                    <div id="sync-panel-content">
                        <table id="sync-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Index</th>
                                    <th>Type</th>
                                    <th>dscnt</th>
                                    <th>vlcnt</th>
                                    <th>Comment</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="sync-table-body">
                            </tbody>
                        </table>
                    </div>
                </div>
                <div id="mfma-reorder-panel" class="collapsible-panel collapsed">
                    <div class="panel-header">
                        <h3>MFMA Reorder</h3>
                        <div class="panel-actions">
                            <button id="toggle-mfma-reorder-panel" class="icon-btn" title="Toggle Panel">▶</button>
                        </div>
                    </div>
                    <div id="mfma-reorder-content">
                        <p id="mfma-reorder-info" class="panel-hint">No MFMA reorder defined</p>
                        <div id="mfma-reorder-list" class="mfma-reorder-grid"></div>
                    </div>
                </div>
            </div>
            <div id="side-panel" class="collapsible">
                <div id="panel-header">
                    <h3>Instruction Order</h3>
                    <button id="toggle-panel" title="Toggle Panel">◀</button>
                </div>
                <p class="panel-hint">Drag to reorder (auto-applies)</p>
                <ul id="instruction-list"></ul>
            </div>
        </div>
    </div>
    
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script nonce="${nonce}">
        const INSTRUCTION_COLORS = ${colorsJson};
    </script>
    <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
    }
}

/**
 * Generate a nonce for Content Security Policy
 */
function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
