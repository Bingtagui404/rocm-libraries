/**
 * CMS Schedule Editor - VS Code Extension Entry Point
 */

import * as vscode from 'vscode';
import { ScheduleEditorPanel } from './scheduleEditorPanel';
import { PythonBridge } from './pythonBridge';
import { CMSConfig } from './types';
import { initLogger, logInfo, logError, logWarn, showOutputChannel } from './logger';

let pythonBridge: PythonBridge | undefined;
let extensionPath: string;

export function activate(context: vscode.ExtensionContext) {
    // Store extension path for finding bundled scripts
    extensionPath = context.extensionUri.fsPath;

    // Initialize logger first
    const config = getConfig();
    initLogger(context, { outputDir: config.outputDir || undefined });
    logInfo('CMS Schedule Editor is now active');

    // Initialize Python bridge with extension path
    pythonBridge = new PythonBridge(config, extensionPath);

    // Register command: Open Schedule Editor
    const openEditorCommand = vscode.commands.registerCommand('cms.openEditor', async () => {
        try {
            await openScheduleEditor(context);
        } catch (error) {
            logError('Failed to open CMS editor', error);
            vscode.window.showErrorMessage(`Failed to open CMS editor: ${error}`);
        }
    });

    // Register command: Run Schedule
    const runScheduleCommand = vscode.commands.registerCommand('cms.runSchedule', async () => {
        try {
            await runSchedule(context);
        } catch (error) {
            logError('Failed to run schedule', error);
            vscode.window.showErrorMessage(`Failed to run schedule: ${error}`);
        }
    });

    // Register command: Save Schedule
    const saveScheduleCommand = vscode.commands.registerCommand('cms.saveSchedule', async () => {
        try {
            await saveSchedule(context);
        } catch (error) {
            logError('Failed to save schedule', error);
            vscode.window.showErrorMessage(`Failed to save schedule: ${error}`);
        }
    });

    // Register command: Reload Latencies
    const reloadLatenciesCommand = vscode.commands.registerCommand('cms.reloadLatencies', async () => {
        try {
            await reloadLatencies(context);
        } catch (error) {
            logError('Failed to reload latencies', error);
            vscode.window.showErrorMessage(`Failed to reload latencies: ${error}`);
        }
    });

    // Register command: Validate Schedule
    const validateScheduleCommand = vscode.commands.registerCommand('cms.validateSchedule', async () => {
        try {
            await validateSchedule(context);
        } catch (error) {
            logError('Failed to validate schedule', error);
            vscode.window.showErrorMessage(`Failed to validate schedule: ${error}`);
        }
    });

    // Register command: Show Log
    const showLogCommand = vscode.commands.registerCommand('cms.showLog', () => {
        showOutputChannel();
    });

    // Watch for configuration changes
    const configWatcher = vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('cms')) {
            pythonBridge = new PythonBridge(getConfig(), extensionPath);
        }
    });

    context.subscriptions.push(
        openEditorCommand,
        runScheduleCommand,
        saveScheduleCommand,
        reloadLatenciesCommand,
        validateScheduleCommand,
        showLogCommand,
        configWatcher
    );
}

export function deactivate() {
    // Clean up
}

/**
 * Get current configuration
 */
function getConfig(): CMSConfig {
    const config = vscode.workspace.getConfiguration('cms');
    return {
        customSchedulePath: config.get<string>('customSchedulePath') || 'Tensile/Components/CustomSchedule.py',
        yamlPath: config.get<string>('yamlPath') || '',
        outputDir: config.get<string>('outputDir') || '',
        pythonPath: config.get<string>('pythonPath') || 'python',
    };
}

/**
 * Get the full path to CustomSchedule.py
 */
function getCustomScheduleFullPath(): string {
    const config = getConfig();
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        throw new Error('No workspace folder open');
    }
    return vscode.Uri.joinPath(workspaceFolder.uri, config.customSchedulePath).fsPath;
}

/**
 * Open the Schedule Editor
 */
async function openScheduleEditor(context: vscode.ExtensionContext) {
    const config = getConfig();
    
    if (!pythonBridge) {
        pythonBridge = new PythonBridge(config, extensionPath);
    }

    const customSchedulePath = getCustomScheduleFullPath();

    // Get full yaml path if configured
    const yamlPath = config.yamlPath ? 
        (config.yamlPath.startsWith('/') ? config.yamlPath : 
         vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0].uri, config.yamlPath).fsPath) : 
        undefined;

    if (!yamlPath) {
        logError('YAML path must be configured to load schedules. Please set cms.yamlPath in settings.');
        vscode.window.showErrorMessage('YAML path must be configured to load schedules. Please set cms.yamlPath in settings.');
        return;
    }

    // Show progress while loading
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Loading CMS Schedule...',
        cancellable: false
    }, async (progress) => {
        // Load schedule using actual indices from Tensile pipeline
        progress.report({ message: 'Loading schedule from YAML via Tensile...' });
        const schedule = await pythonBridge!.loadSchedule(customSchedulePath, yamlPath);
        
        const transposeInfo = (schedule.markers as any).transpose ? ` [${(schedule.markers as any).transpose}]` : '';
        vscode.window.showInformationMessage(
            `Loaded schedule: ${schedule.function_name || 'unnamed'}${transposeInfo} ` +
            `(${schedule.num_mfma} MFMAs, ${schedule.num_codepaths} codepaths)`
        );

        // Create or show the panel
        ScheduleEditorPanel.createOrShow(context.extensionUri, schedule, pythonBridge!);

        // Try to load latencies if output dir is configured
        if (config.outputDir) {
            progress.report({ message: 'Loading latencies...' });
            try {
                const latencyPath = vscode.Uri.joinPath(
                    vscode.Uri.file(config.outputDir), 
                    'figures', 
                    'latency_dict.json'
                ).fsPath;
                const latencies = await pythonBridge!.readLatencyJson(latencyPath);
                ScheduleEditorPanel.currentPanel?.updateLatencies(latencies);
            } catch {
                // Latencies not available yet - that's OK
                logInfo('No latency data available yet - run cms-fast to generate');
            }
        }
    });
}

/**
 * Run the schedule (cms-fast)
 */
async function runSchedule(context: vscode.ExtensionContext) {
    const config = getConfig();
    
    if (!config.yamlPath || !config.outputDir) {
        const result = await vscode.window.showWarningMessage(
            'CMS configuration incomplete. Please set yamlPath and outputDir in settings.',
            'Open Settings'
        );
        if (result === 'Open Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'cms');
        }
        return;
    }

    // Use bundled run_all.sh from the extension
    const runAllScript = pythonBridge!.getRunAllScriptPath();

    // Save schedule first
    await saveSchedule(context);

    // Run cms-fast in terminal
    const terminal = vscode.window.createTerminal({
        name: 'CMS Run',
        cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
    });
    terminal.show();
    terminal.sendText(`${runAllScript} -m cms-fast -y ${config.yamlPath} -o ${config.outputDir}`);

    // Watch for latency file changes
    const latencyPath = vscode.Uri.joinPath(
        vscode.Uri.file(config.outputDir),
        'figures',
        'latency_dict.json'
    );
    
    // Use glob pattern for watcher (works better with remote files)
    const watchPattern = new vscode.RelativePattern(
        vscode.Uri.file(config.outputDir),
        'figures/latency_dict.json'
    );
    const watcher = vscode.workspace.createFileSystemWatcher(watchPattern);
    
    const onLatencyChange = async () => {
        try {
            // Small delay to ensure file is fully written
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            logInfo('Latency file changed, reloading...');
            const latencies = await pythonBridge!.readLatencyJson(latencyPath.fsPath);
            ScheduleEditorPanel.currentPanel?.updateLatencies(latencies);
            logInfo('Latencies updated from file watcher');
            vscode.window.showInformationMessage('Latencies updated from file watcher');
        } catch (error) {
            logError('Failed to read latencies', error);
            vscode.window.showErrorMessage(`Failed to read latencies: ${error}`);
        }
    };

    watcher.onDidCreate(onLatencyChange);
    watcher.onDidChange(onLatencyChange);

    // Also poll for file changes as a fallback (every 5 seconds for 10 minutes)
    let pollCount = 0;
    const maxPolls = 120; // 10 minutes / 5 seconds
    let lastModified = 0;
    
    const pollInterval = setInterval(async () => {
        pollCount++;
        if (pollCount > maxPolls) {
            clearInterval(pollInterval);
            watcher.dispose();
            return;
        }
        
        try {
            const stat = await vscode.workspace.fs.stat(latencyPath);
            if (stat.mtime > lastModified) {
                lastModified = stat.mtime;
                if (pollCount > 1) { // Skip first check (initial state)
                    logInfo('Latency file modified (detected by polling)');
                    const latencies = await pythonBridge!.readLatencyJson(latencyPath.fsPath);
                    ScheduleEditorPanel.currentPanel?.updateLatencies(latencies);
                    logInfo('Latencies updated');
                    vscode.window.showInformationMessage('Latencies updated');
                }
            }
        } catch {
            // File doesn't exist yet - that's OK
        }
    }, 5000);

    // Clean up on panel dispose
    ScheduleEditorPanel.currentPanel?.onDispose(() => {
        clearInterval(pollInterval);
        watcher.dispose();
    });
}

/**
 * Save the current schedule
 */
async function saveSchedule(context: vscode.ExtensionContext) {
    const panel = ScheduleEditorPanel.currentPanel;
    if (!panel) {
        vscode.window.showWarningMessage('No schedule editor is open');
        return;
    }

    const schedule = panel.getCurrentSchedule();
    if (!schedule) {
        vscode.window.showWarningMessage('No schedule data to save');
        return;
    }

    const config = getConfig();
    const customSchedulePath = getCustomScheduleFullPath();
    
    // Get full yaml path if configured
    const yamlPath = config.yamlPath ? 
        (config.yamlPath.startsWith('/') ? config.yamlPath : 
         vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0].uri, config.yamlPath).fsPath) : 
        undefined;

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Saving schedule...',
        cancellable: false
    }, async () => {
        await pythonBridge!.writeSchedule(customSchedulePath, schedule, yamlPath);
    });

    vscode.window.showInformationMessage('Schedule saved');
}

/**
 * Reload latencies from file
 */
async function reloadLatencies(context: vscode.ExtensionContext) {
    const config = getConfig();
    
    if (!config.outputDir) {
        vscode.window.showWarningMessage('Output directory not configured');
        return;
    }

    const latencyPath = vscode.Uri.joinPath(
        vscode.Uri.file(config.outputDir),
        'figures',
        'latency_dict.json'
    ).fsPath;

    try {
        const latencies = await pythonBridge!.readLatencyJson(latencyPath);
        ScheduleEditorPanel.currentPanel?.updateLatencies(latencies);
        logInfo('Latencies reloaded');
        vscode.window.showInformationMessage('Latencies reloaded');
    } catch (error) {
        logError('Failed to load latencies', error);
        vscode.window.showErrorMessage(`Failed to load latencies: ${error}`);
    }
}

/**
 * Validate the current schedule using CMSValidator
 */
async function validateSchedule(context: vscode.ExtensionContext) {
    const config = getConfig();
    
    if (!config.yamlPath) {
        vscode.window.showWarningMessage('YAML path not configured. Please set cms.yamlPath in settings.');
        return;
    }

    // Get full yaml path
    const yamlPath = config.yamlPath.startsWith('/') ? 
        config.yamlPath : 
        vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0].uri, config.yamlPath).fsPath;

    // First save the schedule to ensure we validate the latest changes
    await saveSchedule(context);

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Validating schedule...',
        cancellable: false
    }, async () => {
        try {
            const result = await pythonBridge!.validateSchedule(yamlPath);
            
            if (result.valid) {
                logInfo('Schedule validation passed', result.message);
                vscode.window.showInformationMessage(`✓ Schedule is valid${result.message ? ': ' + result.message : ''}`);
            } else {
                logWarn('Schedule validation failed', result.message);
                vscode.window.showWarningMessage(`✗ Validation failed: ${result.message}`);
            }
        } catch (error) {
            logError('Validation error', error);
            vscode.window.showErrorMessage(`Validation error: ${error}`);
        }
    });
}
