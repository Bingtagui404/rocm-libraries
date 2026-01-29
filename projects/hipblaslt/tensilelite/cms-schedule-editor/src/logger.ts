/**
 * CMS Schedule Editor - Centralized Logger
 * 
 * Provides logging to both VSCode Output Channel and a log file.
 * Errors are accessible to AI models via the log file.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export type LogLevel = 'info' | 'warn' | 'error';

interface LoggerConfig {
    outputDir?: string;
    logFileName?: string;
    maxLogSize?: number;  // Max log file size in bytes before truncation
}

let outputChannel: vscode.OutputChannel | undefined;
let logFilePath: string | undefined;
let maxLogSize: number = 1024 * 1024;  // Default 1MB

/**
 * Initialize the logger with configuration
 */
export function initLogger(context: vscode.ExtensionContext, config: LoggerConfig = {}): void {
    // Create Output Channel
    outputChannel = vscode.window.createOutputChannel('CMS Schedule Editor');
    context.subscriptions.push(outputChannel);

    // Determine log file path
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const baseDir = config.outputDir || workspaceFolder || context.extensionUri.fsPath;
    const fileName = config.logFileName || 'cms-editor.log';
    logFilePath = path.join(baseDir, fileName);

    if (config.maxLogSize) {
        maxLogSize = config.maxLogSize;
    }

    // Log initialization
    log('info', `CMS Schedule Editor logger initialized`);
    log('info', `Log file: ${logFilePath}`);
}

/**
 * Get the current timestamp formatted for logging
 */
function getTimestamp(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

/**
 * Format a log message with timestamp and level
 */
function formatMessage(level: LogLevel, message: string, source?: string): string {
    const timestamp = getTimestamp();
    const levelStr = level.toUpperCase().padEnd(5);
    const sourceStr = source ? `[${source}] ` : '';
    return `[${timestamp}] [${levelStr}] ${sourceStr}${message}`;
}

/**
 * Write to the log file, handling rotation if needed
 */
function writeToFile(formattedMessage: string): void {
    if (!logFilePath) return;

    try {
        // Check file size and truncate if needed
        if (fs.existsSync(logFilePath)) {
            const stats = fs.statSync(logFilePath);
            if (stats.size > maxLogSize) {
                // Keep the last half of the file
                const content = fs.readFileSync(logFilePath, 'utf8');
                const halfPoint = Math.floor(content.length / 2);
                const newContent = '... [log truncated] ...\n' + content.slice(halfPoint);
                fs.writeFileSync(logFilePath, newContent, 'utf8');
            }
        }

        // Append the new message
        fs.appendFileSync(logFilePath, formattedMessage + '\n', 'utf8');
    } catch (err) {
        // Fallback to console if file writing fails
        console.error('Failed to write to log file:', err);
    }
}

/**
 * Log a message at the specified level
 */
export function log(level: LogLevel, message: string, source?: string): void {
    const formattedMessage = formatMessage(level, message, source);

    // Write to Output Channel
    if (outputChannel) {
        outputChannel.appendLine(formattedMessage);
    }

    // Write to log file
    writeToFile(formattedMessage);

    // Also log to console for development
    switch (level) {
        case 'error':
            console.error(formattedMessage);
            break;
        case 'warn':
            console.warn(formattedMessage);
            break;
        default:
            console.log(formattedMessage);
    }
}

/**
 * Log an info message
 */
export function logInfo(message: string, source?: string): void {
    log('info', message, source);
}

/**
 * Log a warning message
 */
export function logWarn(message: string, source?: string): void {
    log('warn', message, source);
}

/**
 * Log an error message, optionally with an Error object
 */
export function logError(message: string, error?: Error | unknown, source?: string): void {
    let fullMessage = message;
    if (error) {
        if (error instanceof Error) {
            fullMessage += `: ${error.message}`;
            if (error.stack) {
                fullMessage += `\n${error.stack}`;
            }
        } else {
            fullMessage += `: ${String(error)}`;
        }
    }
    log('error', fullMessage, source);
}

/**
 * Show the Output Channel to the user
 */
export function showOutputChannel(): void {
    outputChannel?.show();
}

/**
 * Get the log file path (useful for displaying to users)
 */
export function getLogFilePath(): string | undefined {
    return logFilePath;
}

/**
 * Clear the log (both Output Channel and file)
 */
export function clearLog(): void {
    outputChannel?.clear();
    if (logFilePath && fs.existsSync(logFilePath)) {
        fs.writeFileSync(logFilePath, '', 'utf8');
    }
    log('info', 'Log cleared');
}
