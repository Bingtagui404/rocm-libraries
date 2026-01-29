/**
 * CMS Schedule Editor - Python Bridge
 * 
 * Communicates with CMSScheduleIO.py for schedule parsing and serialization.
 * The CMSScheduleIO.py script is bundled with the extension in the src/ folder.
 */

import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { CMSConfig, MarkerInfo, ScheduleData, LatencyData, ScheduleInfoResult } from './types';

/**
 * Result from parse_schedule - only kernel_settings_raw and metadata
 */
interface ParseScheduleResult {
    function_name: string;
    kernel_settings_raw: string[];
    transpose: string | null;
    branch_lines: [number, number];
    markers: MarkerInfo;
    dynamic_sync_expressions?: Record<number, Record<string, string>>;
}

export class PythonBridge {
    private config: CMSConfig;
    private workspaceRoot: string;
    private extensionPath: string;

    constructor(config: CMSConfig, extensionPath: string) {
        this.config = config;
        this.workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        this.extensionPath = extensionPath;
    }

    /**
     * Get the path to CMSScheduleIO.py script bundled with the extension
     */
    private getCMSIOScriptPath(): string {
        // The script is bundled in the extension's src/ folder
        return path.join(this.extensionPath, 'src', 'CMSScheduleIO.py');
    }

    /**
     * Get the path to the bundled run_all.sh script
     */
    public getRunAllScriptPath(): string {
        return path.join(this.extensionPath, 'src', 'run_all.sh');
    }

    /**
     * Run a Python command and return the output
     */
    private async runPython(args: string[]): Promise<string> {
        return new Promise((resolve, reject) => {
            const scriptPath = this.getCMSIOScriptPath();
            const pythonArgs = [scriptPath, ...args];
            
            console.log(`Running: ${this.config.pythonPath} ${pythonArgs.join(' ')}`);

            const proc = cp.spawn(this.config.pythonPath, pythonArgs, {
                cwd: this.workspaceRoot,
                env: { ...process.env }
            });

            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            proc.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    resolve(stdout.trim());
                } else {
                    // Try to parse error from stderr
                    try {
                        const errorObj = JSON.parse(stderr);
                        reject(new Error(errorObj.error || stderr));
                    } catch {
                        reject(new Error(stderr || `Process exited with code ${code}`));
                    }
                }
            });

            proc.on('error', (err) => {
                reject(err);
            });
        });
    }

    /**
     * Find GUI schedule markers in CustomSchedule.py
     */
    async findMarkers(customSchedulePath: string, yamlPath?: string): Promise<MarkerInfo> {
        const args = ['find-markers', customSchedulePath];
        if (yamlPath) {
            args.push('--yaml', yamlPath);
        }
        const output = await this.runPython(args);
        return JSON.parse(output) as MarkerInfo;
    }

    /**
     * Get the actual ScheduleInfo by loading the YAML through Tensile pipeline.
     * This returns the real computed indices, not parsed from Python source.
     */
    async getScheduleInfoFromYaml(yamlPath: string, arch: string = 'gfx950'): Promise<ScheduleInfoResult> {
        const args = ['get-schedule-info', yamlPath, '--arch', arch];
        const output = await this.runPython(args);
        return JSON.parse(output) as ScheduleInfoResult;
    }

    /**
     * Parse the schedule between markers (from Python source code).
     * NOTE: This now only extracts kernel_settings_raw and checks for dynamic values.
     * For optSchedule, syncCode, mfmaReorder, use getScheduleInfoFromYaml() instead.
     */
    async parseScheduleFromSource(customSchedulePath: string, yamlPath?: string): Promise<ParseScheduleResult> {
        const args = ['parse', customSchedulePath];
        if (yamlPath) {
            args.push('--yaml', yamlPath);
        }
        const output = await this.runPython(args);
        return JSON.parse(output) as ParseScheduleResult;
    }

    /**
     * Load schedule by combining:
     * 1. Actual computed values from Tensile pipeline (optSchedule, syncCode, mfmaReorder)
     * 2. Parsed source code for kernel_settings_raw (written back as-is)
     * 3. Marker info for writing back to file
     * 
     * This is the preferred method for loading schedules.
     */
    async loadSchedule(customSchedulePath: string, yamlPath: string): Promise<ScheduleData> {
        // Get the actual schedule data from YAML via Tensile pipeline
        const scheduleInfo = await this.getScheduleInfoFromYaml(yamlPath);
        
        if (!scheduleInfo.has_schedule || !scheduleInfo.schedule_info) {
            throw new Error('No custom schedule found for this YAML configuration');
        }
        
        // Get marker info for the file location
        const markers = await this.findMarkers(customSchedulePath, yamlPath);
        
        // Parse source code for kernel_settings_raw and dynamic expressions
        const sourceParsed = await this.parseScheduleFromSource(customSchedulePath, yamlPath);
        
        // Merge dynamic expressions into sync_code
        let syncCode = scheduleInfo.schedule_info.syncCode;
        if (sourceParsed.dynamic_sync_expressions) {
            syncCode = this.mergeDynamicExpressions(
                syncCode, 
                sourceParsed.dynamic_sync_expressions
            );
        }
        
        // Build ScheduleData from combined sources
        const scheduleData: ScheduleData = {
            function_name: markers.function_name,
            instruction_types: Object.keys(scheduleInfo.schedule_info.optSchedule),
            opt_schedule: scheduleInfo.schedule_info.optSchedule,
            num_codepaths: scheduleInfo.schedule_info.numCodePaths,
            num_mfma: scheduleInfo.schedule_info.numMfma,
            sync_indices: this.extractSyncIndices(scheduleInfo.schedule_info.optSchedule),
            nglshift: scheduleInfo.schedule_info.nglshift,
            nllshift: scheduleInfo.schedule_info.nllshift,
            markers: markers,
            // FROM Tensile pipeline (computed values) + dynamic expressions from source:
            sync_code: syncCode,
            mfma_reorder: scheduleInfo.schedule_info.mfmaReorder,
            // FROM source parsing (raw strings):
            kernel_settings_raw: sourceParsed.kernel_settings_raw,
        };
        
        return scheduleData;
    }

    /**
     * Merge dynamic expressions from source parsing into sync_code from Tensile pipeline.
     * Converts hardcoded values to { expr: "...", values: [computed_value] } format.
     */
    private mergeDynamicExpressions(
        syncCode: any[], 
        dynamicExprs: Record<number, Record<string, string>>
    ): any[] {
        return syncCode.map(item => {
            const index = item.index;
            const exprs = dynamicExprs[index];
            
            if (!exprs || item.instruction.type !== 'SWaitCnt') {
                return item;
            }
            
            // Clone the instruction to avoid mutating original
            const newInst = { ...item.instruction };
            
            // For each field with a dynamic expression, convert to { expr, values } format
            for (const field of ['dscnt', 'vlcnt', 'vscnt']) {
                if (exprs[field]) {
                    const computedValue = newInst[field];
                    newInst[field] = {
                        expr: exprs[field],
                        values: [computedValue]  // Single codepath value for now
                    };
                }
            }
            
            return { ...item, instruction: newInst };
        });
    }

    /**
     * Extract SYNC indices from optSchedule
     */
    private extractSyncIndices(optSchedule: Record<string, number[][]>): number[] {
        const syncData = optSchedule['SYNC'];
        if (syncData && syncData.length > 0) {
            // Return the first codepath's SYNC indices, filtering out -1
            return syncData[0].filter(idx => idx >= 0);
        }
        return [];
    }

    /**
     * Write schedule back to file
     */
    async writeSchedule(customSchedulePath: string, schedule: ScheduleData, yamlPath?: string): Promise<boolean> {
        // Write schedule to a temp file first
        const tempFile = path.join(this.workspaceRoot, '.cms_schedule_temp.json');
        
        try {
            fs.writeFileSync(tempFile, JSON.stringify(schedule, null, 2));
            
            const args = ['write', customSchedulePath, '--schedule', tempFile];
            if (yamlPath) {
                args.push('--yaml', yamlPath);
            }
            const output = await this.runPython(args);
            const result = JSON.parse(output);
            return result.success === true;
        } finally {
            // Clean up temp file
            try {
                fs.unlinkSync(tempFile);
            } catch {
                // Ignore cleanup errors
            }
        }
    }

    /**
     * Read latency JSON file (works over Remote-SSH)
     */
    async readLatencyJson(latencyPath: string): Promise<LatencyData> {
        const uri = vscode.Uri.file(latencyPath);
        const data = await vscode.workspace.fs.readFile(uri);
        const content = Buffer.from(data).toString('utf8');
        return JSON.parse(content) as LatencyData;
    }

    /**
     * Validate a schedule using CMSValidator
     */
    async validateSchedule(yamlPath: string): Promise<{ valid: boolean; message: string }> {
        const args = ['validate', yamlPath];
        const output = await this.runPython(args);
        return JSON.parse(output) as { valid: boolean; message: string };
    }
}
