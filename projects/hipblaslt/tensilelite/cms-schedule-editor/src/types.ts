/**
 * CMS Schedule Editor - Type Definitions
 */

/**
 * Information about GUI schedule markers found in CustomSchedule.py
 */
export interface MarkerInfo {
    function_name: string;
    start_line: number;
    end_line: number;
    decorator_start: number | null;
}

/**
 * Parsed schedule data from CustomSchedule.py
 */
export interface ScheduleData {
    function_name: string;
    instruction_types: string[];
    opt_schedule: Record<string, number[][] | string>;
    num_codepaths: number;
    num_mfma: number;
    sync_indices: number[];
    nglshift: number;
    nllshift: number;
    markers: MarkerInfo;
    /** Kernel settings from the schedule (e.g., MfmaInitCVgprs, UsePLRPack) - DEPRECATED, use kernel_settings_raw */
    kernel_settings?: KernelSettings;
    /** Kernel settings as raw strings (written back as-is) */
    kernel_settings_raw?: string[];
    /** Sync code instructions with their indices */
    sync_code?: IndexedSyncInstruction[];
    /** MFMA reordering array (optional) */
    mfma_reorder?: number[];
}

/**
 * Result from get-schedule-info command (actual computed indices from Tensile pipeline)
 */
export interface ScheduleInfoResult {
    has_schedule: boolean;
    schedule_info: {
        numCodePaths: number;
        numMfma: number;
        optSchedule: Record<string, number[][]>;
        nglshift: number;
        nllshift: number;
        /** Sync code instructions with their MFMA indices (from ScheduleInfo.syncCode) */
        syncCode: IndexedSyncInstruction[];
        /** MFMA reordering array (from ScheduleInfo.mfmaReorder) */
        mfmaReorder: number[];
    } | null;
    kernel_config: {
        MacroTile0: number | null;
        MacroTile1: number | null;
        DepthU: number | null;
        MatrixInstruction: number[] | null;
    };
}

/**
 * Box plot statistics for a single instruction
 */
export interface BoxPlotStats {
    whisker_low: number;
    q1: number;
    median: number;
    q3: number;
    whisker_high: number;
}

/**
 * Latency data per codepath from analyze.py
 * Each instruction has full box plot statistics
 */
export interface LatencyData {
    [codepath: string]: {
        [instructionKey: string]: BoxPlotStats;
    };
}

/**
 * Sync instruction type - the three types that can be added/edited by the user
 */
export type SyncInstructionType = 'SWaitCnt' | 'SBarrier' | 'SNop';

/**
 * Dynamic value - a value computed from a Python expression
 * Used for dscnt/vlcnt/vscnt in SWaitCnt instructions
 */
export interface DynamicValue {
    /** The Python expression that computes this value */
    expr: string;
    /** Computed values, one per codepath */
    values: number[];
}

/**
 * A wait count value can be either hardcoded (number) or dynamic (computed from expression)
 * -1 means "not set" for hardcoded values
 */
export type WaitCountValue = number | DynamicValue;

/**
 * Helper to check if a wait count value is dynamic
 */
export function isDynamicValue(value: WaitCountValue): value is DynamicValue {
    return typeof value === 'object' && 'expr' in value && 'values' in value;
}

/**
 * SWaitCnt instruction - wait for memory operations to complete
 */
export interface SWaitCntInstruction {
    type: 'SWaitCnt';
    /** Data share count - wait for LDS operations (-1 = not set) */
    dscnt: WaitCountValue;
    /** Vector memory count - wait for VMEM operations (-1 = not set) */
    vlcnt: WaitCountValue;
    /** Vector store count - wait for vector stores (-1 = not set) */
    vscnt: WaitCountValue;
    /** Optional comment describing this wait */
    comment?: string;
}

/**
 * SBarrier instruction - synchronization barrier between waves
 */
export interface SBarrierInstruction {
    type: 'SBarrier';
    /** Optional comment describing this barrier */
    comment?: string;
}

/**
 * SNop instruction - insert delay/no-operation cycles
 */
export interface SNopInstruction {
    type: 'SNop';
    /** Number of nop cycles to insert */
    count: number;
    /** Optional comment describing this nop */
    comment?: string;
}

/**
 * Union type for all sync instructions
 */
export type SyncInstruction = SWaitCntInstruction | SBarrierInstruction | SNopInstruction;

/**
 * A sync instruction with its MFMA index position
 */
export interface IndexedSyncInstruction {
    /** MFMA index where this sync instruction occurs (-1 = before first MFMA) */
    index: number;
    /** The sync instruction */
    instruction: SyncInstruction;
}

/**
 * Available dynamic rules for wait count computation
 */
export type DynamicRuleId = 'finish_all' | 'finish_n' | 'keep_n';

/**
 * Dynamic rule configuration for the GUI rule builder
 */
export interface DynamicRule {
    /** Which rule to use */
    ruleId: DynamicRuleId;
    /** Instruction types to apply the rule to */
    types: string[];
    /** N value for finish_n and keep_n rules */
    n?: number;
}

/**
 * Kernel settings that can be modified by the schedule
 */
export interface KernelSettings {
    [key: string]: boolean | number | string;
}

/**
 * Instruction class colors (matching analyze.py INSTRUCTION_CLASS_COLORS)
 */
export const INSTRUCTION_CLASS_COLORS: Record<string, string> = {
    MFMA: "#0A5B0C",      // Dark Green
    LDS: "#FF7F00",       // Orange (LRA, LRB)
    VMEM: "#E6D700",      // Yellow (GRA, GRB)
    WAITCNT: "#000000",   // Black
    BARRIER: "#999999",   // Grey
    NOP: "#999999",       // Grey
    SALU: "#A6FFD9",      // Light Mint
    VALU: "#118C13",      // Green
    BRANCH: "#E78AC3",    // Magenta
    OTHER: "#A6D854",     // Lime
};

/**
 * Get color for an instruction type
 */
export function getColorForInstructionType(instType: string): string {
    // Map instruction type to class
    if (instType.startsWith('MFMA')) {
        return INSTRUCTION_CLASS_COLORS.MFMA;
    }
    if (instType.startsWith('LRA') || instType.startsWith('LRB') || 
        instType.startsWith('LRS') || instType === 'LDS') {
        return INSTRUCTION_CLASS_COLORS.LDS;
    }
    if (instType.startsWith('GRA') || instType.startsWith('GRB') ||
        instType.startsWith('GRInc')) {
        return INSTRUCTION_CLASS_COLORS.VMEM;
    }
    if (instType === 'SYNC' || instType.startsWith('WAIT')) {
        return INSTRUCTION_CLASS_COLORS.WAITCNT;
    }
    if (instType.startsWith('Pack')) {
        return INSTRUCTION_CLASS_COLORS.VALU;
    }
    if (instType.startsWith('LW') || instType === 'LCC') {
        return INSTRUCTION_CLASS_COLORS.SALU;
    }
    return INSTRUCTION_CLASS_COLORS.OTHER;
}

/**
 * Log level for messages from webview
 */
export type LogLevel = 'info' | 'warn' | 'error';

/**
 * Move instruction request for multi-select
 */
export interface MoveInstructionRequest {
    instruction: string;
    fromMfma: number;
    toMfma: number;
}

/**
 * Message types for webview communication
 */
export type WebviewMessage = 
    | { command: 'updateSchedule'; schedule: ScheduleData }
    | { command: 'updateLatencies'; latencies: LatencyData }
    | { command: 'reorderInstructions'; newOrder: string[] }
    | { command: 'moveInstruction'; instruction: string; fromMfma: number; toMfma: number; codepath: number }
    | { command: 'moveInstructions'; moves: MoveInstructionRequest[]; codepath: number }
    | { command: 'splitSchedule'; type: string; newSchedule: number[][] }
    | { command: 'updateSyncCode'; syncCode: IndexedSyncInstruction[] }
    | { command: 'addSyncInstruction'; instruction: IndexedSyncInstruction }
    | { command: 'removeSyncInstruction'; index: number; instructionIndex: number }
    | { command: 'requestSave' }
    | { command: 'requestRun' }
    | { command: 'requestValidate' }
    | { command: 'ready' }
    | { command: 'error'; message: string }
    | { command: 'log'; level: LogLevel; message: string; details?: unknown };

/**
 * Configuration for the CMS Schedule Editor
 */
export interface CMSConfig {
    customSchedulePath: string;
    yamlPath: string;
    outputDir: string;
    pythonPath: string;
}
