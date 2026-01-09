################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from .CustomKernels import getCustomKernelConfig
from . import SolutionLibrary
from .CustomYamlLoader import load_yaml_stream
from Tensile import __version__
from Tensile.Common import printExit, printWarning, print2, \
                           versionIsCompatible, IsaInfo
from Tensile.Common.Architectures import gfxToIsa
from Tensile.SolutionStructs import Solution, ProblemSizes
from Tensile.SolutionStructs.Problem import ProblemType, problemTypeToEnum

from typing import NamedTuple, List, Dict
import os
import sys
import subprocess
import re

try:
    import orjson as json
except ImportError:
    try:
        import ujson as json
        print2("orjson not installed. Fallback to ujson.")
    except ImportError:
        try:
            import simplejson as json
            print2("orjson, ujson not installed. Fallback to simplejson.")
        except ImportError:
            import json
            print2("orjson, ujson, simplejson not installed. Fallback to json.")

try:
    import yaml
except ImportError:
    printExit(
        "You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions."
    )

try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader
    printWarning("CSafeLoader not installed. Fallback to SafeLoader.")

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    msgpack = None
    HAS_MSGPACK = False
    print("Message pack python library not detected. Must use YAML backend instead.")


###################
# Writing functions
###################
def write(filename_noExt, data, format="yaml"):
    """Writes data to file with specified format; extension is appended based on format."""
    if format == "yaml":
        filename = filename_noExt + ".yaml"
        writeYAML(filename, data)
    elif format == "json":
        filename = filename_noExt + ".json"
        writeJson(filename, data)
    elif format == "msgpack":
        filename = filename_noExt + ".dat"
        writeMsgPack(filename, data)
    else:
        printExit("Unrecognized write format {}".format(format))


def writeYAML(filename, data, **kwargs):
    """Writes data to file in YAML format."""
    # set default kwags for yaml dump
    if "explicit_start" not in kwargs:
        kwargs["explicit_start"] = True
    if "explicit_end" not in kwargs:
        kwargs["explicit_end"] = True
    if "default_flow_style" not in kwargs:
        kwargs["default_flow_style"] = None

    with open(filename, "w") as f:
        yaml.dump(data, f, **kwargs)

def writeJson(filename, data):
    """Writes data to file in json format."""
    with open(filename, "w") as f:
        json_object = json.dumps(data, option=json.OPT_INDENT_2).decode("utf-8") if 'orjson' in sys.modules else json.dumps(data, indent=2)
        f.write(json_object)

def writeMsgPack(filename, data):
    """Writes data to file in Message Pack format."""
    if not HAS_MSGPACK:
        printExit("Message pack python library not detected. Install msgpack to use msgpack format.")
    with open(filename, "wb") as f:
        msgpack.pack(data, f)

def writeSolutions(filename, problemSizes, biasTypeArgs, activationArgs, solutions, cache=False, format="yaml"):
    """Writes solution file in YAML, JSON, or msgpack format."""

    # Adjust filename extension based on format
    base, _ext = os.path.splitext(filename)
    if format == "msgpack":
        filename = base + ".dat"
    elif format == "json":
        filename = base + ".json"
    else:  # yaml
        filename = base + ".yaml"

    # convert objects to nested dictionaries
    solutionStates = []

    if cache and os.path.exists(filename):
        solData = read(filename)  # Auto-detects format based on extension
        if solData is not None:
            if biasTypeArgs and activationArgs:
                solutionStates = solData[4:]
            elif biasTypeArgs or activationArgs:
                solutionStates = solData[3:]
            else:
                solutionStates = solData[2:]
    elif solutions is not None:
        for solution in solutions:
            solutionState = solution.getAttributes()
            solutionState["ProblemType"] = solutionState["ProblemType"].state
            problemTypeToEnum(solutionState["ProblemType"])
            isa = solutionState["ISA"]
            solutionState["ISA"] = [isa[0], isa[1], isa[2]]
            solutionStates.append(solutionState)
    
    # write dictionaries based on format
    if format == "yaml":
        # Original YAML format with special header
        with open(filename, "w") as f:
            f.write("- MinimumRequiredVersion: {}\n".format(__version__))
            f.write("- ProblemSizes:\n")
            if problemSizes:
                for sizeRange in problemSizes.ranges:
                    f.write("  - Range: {}\n".format(sizeRange))
                for problemExact in problemSizes.exacts:
                    #FIXME-problem, this ignores strides:
                    f.write("  - Exact: {}\n".format(problemExact))
            # Write BiasTypeArgs
            # BUG FIX: Don't use format([]) as it produces "[[]]" instead of "[]"
            # When biasTypes is empty, we should write "[]" not "[[]]"
            if biasTypeArgs:
                bias_type_values = [btype.toChar().lower() for btype in biasTypeArgs.biasTypes]
                if bias_type_values:
                    # Non-empty list: write as "[value1, value2, ...]"
                    f.write("- BiasTypeArgs: [{}]\n".format(', '.join(bias_type_values)))
                else:
                    # Empty list: write as "[]" not "[[]]"
                    f.write("- BiasTypeArgs: []\n")
            
            # Write ActivationArgs
            # BUG FIX: When settingList is empty, write "[]" not just the key with no value
            # Previously, writing only "- ActivationArgs:\n" results in YAML parsing it as None
            if activationArgs:
                if activationArgs.settingList:
                    # Non-empty list: write with nested structure
                    f.write("- ActivationArgs:\n")
                    for setting in activationArgs.settingList:
                        f.write("  - [Enum: %s]\n"%(setting.activationEnum))
                else:
                    # Empty list: write as "[]" not None
                    f.write("- ActivationArgs: []\n")
            if solutionStates:  # Only dump if we have solution states
                yaml.dump(solutionStates, f, default_flow_style=None)
    
    elif format in ["json", "msgpack"]:
        # Structured format for JSON/msgpack
        data = [
            {"MinimumRequiredVersion": __version__}
        ]
        
        # Add ProblemSizes (convert custom objects to lists for serialization)
        # Must match the format expected by Problem.py parser:
        # ProblemSizes is a list of dictionaries, each with "Range" or "Exact" key
        problemSizesData = []
        if problemSizes:
            if problemSizes.ranges:
                for r in problemSizes.ranges:
                    # Reconstruct the Range config from ProblemSizeRange attributes
                    # This mirrors the logic in ProblemSizeRange.__str__()
                    range_config = []
                    sizedIdx = 0
                    mappedIdx = 0
                    for i in range(len(r.indexIsSized)):
                        if r.indexIsSized[i]:
                            # This index is sized, add the 4-element list [start, step, increment, end]
                            range_config.append(list(r.indicesSized[sizedIdx]))
                            sizedIdx += 1
                        else:
                            # This index is mapped, add the mapping index
                            range_config.append(r.indicesMapped[mappedIdx])
                            mappedIdx += 1
                    problemSizesData.append({"Range": range_config})
            if problemSizes.exacts:
                for e in problemSizes.exacts:
                    # Handle both ExactList and ExactDict
                    # ExactList: simple list of sizes
                    # ExactDict: dict with sizes and optional strides
                    if hasattr(e, 'stridesA') and (e.stridesA or e.stridesB or e.stridesC or e.stridesD):
                        # This is an ExactDict with strides, serialize as dict
                        exact_data = {"sizes": list(e.sizes)}
                        if e.stridesA:
                            exact_data["stridesA"] = list(e.stridesA)
                        if e.stridesB:
                            exact_data["stridesB"] = list(e.stridesB)
                        if e.stridesC:
                            exact_data["stridesC"] = list(e.stridesC)
                        if e.stridesD:
                            exact_data["stridesD"] = list(e.stridesD)
                        if hasattr(e, 'count') and e.count:
                            exact_data["count"] = e.count
                        problemSizesData.append({"Exact": exact_data})
                    else:
                        # This is an ExactList or ExactDict without strides, serialize as list
                        problemSizesData.append({"Exact": list(e.sizes)})
        # Wrap in ProblemSizes key to match expected format
        data.append({"ProblemSizes": problemSizesData})
        
        # Add optional fields (only if they exist)
        if biasTypeArgs:
            data.append({"BiasTypeArgs": [btype.toChar().lower() for btype in biasTypeArgs.biasTypes]})
            
        if activationArgs:
            # Convert ActivationType objects to strings for serialization
            data.append({"ActivationArgs": [{"Enum": str(setting.activationEnum)} for setting in activationArgs.settingList]})
        
        # Add solutions
        data.extend(solutionStates)
        
        # Write to file
        if format == "msgpack":
            if not HAS_MSGPACK:
                printExit("Message pack python library not detected. Install msgpack to use msgpack format.")
            with open(filename, "wb") as f:
                msgpack.pack(data, f, use_bin_type=True)
        else:  # json
            writeJson(filename, data)


###############################
# Reading and parsing functions
###############################
def read(filename, customizedLoader=False):
    name, extension = os.path.splitext(filename)
    
    if extension == ".yaml":
        result = load_yaml_stream(filename, yamlLoader) if customizedLoader else readYAML(filename)
    elif extension == ".json":
        result = readJson(filename)
    elif extension == ".dat":
        result = readMsgPack(filename)
    else:
        printExit("Unrecognized read format {}".format(extension))
    
    return result


def readYAML(filename):
    """Reads and returns YAML data from file."""
    with open(filename, "r") as f:
        data = yaml.load(f, yamlLoader)
    
    return data


def readJson(filename):
    """Reads and returns JSON data from file."""
    with open(filename, "r") as f:
        data = json.loads(f.read())
    
    return data


def readMsgPack(filename):
    """Reads and returns msgpack data from file."""
    if not HAS_MSGPACK:
        printExit("Message pack python library not detected. Install msgpack to use msgpack format.")
    with open(filename, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    
    return data


def parseSolutionsFile(
        filename,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap
    ):
    """Wrapper function to read and parse a solutions file."""
    return parseSolutionsData(
               read(filename),
               filename,
               assembler,
               splitGSU,
               printSolutionRejectionReason,
               printIndexAssignmentInfo,
               isaInfoMap
            )


def parseSolutionsData(
        data,
        srcFile,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap
    ):
    """Parses problem sizes and solutions from the data of a solutions file."""
    if len(data) < 3:
        printExit("Solution file {} is missing required fields (len = {} < 3" \
                .format(srcFile, len(data)))

    versionString = data[0]["MinimumRequiredVersion"]
    if not versionIsCompatible(versionString):
        printWarning("Version = {} in solution file {} does not match Tensile version = {}" \
                .format(srcFile, versionString, __version__) )

    if "ProblemSizes" not in data[1]:
        printExit("Solution file {} doesn't begin with ProblemSizes".format(srcFile))

    problemSizesConfig = data[1]["ProblemSizes"]
    solutionStartIdxInData = 2
    if (len(data) > solutionStartIdxInData) and "BiasTypeArgs" in data[solutionStartIdxInData]:
        solutionStartIdxInData += 1
    if (len(data) > solutionStartIdxInData) and "ActivationArgs" in data[solutionStartIdxInData]:
        solutionStartIdxInData += 1

    solutions = []
    for i in range(solutionStartIdxInData, len(data)):
        solutionState = data[i]
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        solutionObject = Solution(
                             solutionState,
                             splitGSU,
                             printSolutionRejectionReason,
                             printIndexAssignmentInfo,
                             assembler,
                             isaInfoMap,
                             srcFile
                         )
        solutions.append(solutionObject)
    problemType = solutions[0]["ProblemType"]
    problemSizes = ProblemSizes(problemType, problemSizesConfig)
    return (problemSizes, solutions)


class LibraryLogic(NamedTuple):
    """Return tuple for parseLibraryLogicData()"""
    schedule: str
    architecture: str
    problemType: ProblemType
    solutions: list
    exactLogic: list
    library: SolutionLibrary.MasterSolutionLibrary

def parseLibraryLogicFile(
        filename,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap: Dict[str, IsaInfo],
        lazyLibraryLoading: bool
    ):
    """Wrapper function to read and parse a library logic file."""
    return parseLibraryLogicData(
               read(filename, True),
               filename,
               assembler,
               splitGSU,
               printSolutionRejectionReason,
               printIndexAssignmentInfo,
               isaInfoMap,
               lazyLibraryLoading
           )


def parseLibraryLogicData(
        data,
        srcFile,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap: Dict[str, IsaInfo],
        lazyLibraryLoading: bool
    ):
    """Parses the data of a library logic file."""
    if isinstance(data, List):
        data = parseLibraryLogicList(data, srcFile)

    if "CUCount" not in data:
        data["CUCount"] = None

    if not versionIsCompatible(data["MinimumRequiredVersion"]):
        printWarning("Version = {} in library logic file {} does not match Tensile version = {}" \
                .format(srcFile, data["MinimumRequiredVersion"], __version__) )

    # unpack problemType
    problemType = ProblemType(data["ProblemType"], printIndexAssignmentInfo)

    # unpack solution
    def solutionStateToSolution(solutionState, assembler, isaInfoMap) -> Solution:
        if solutionState["KernelLanguage"] == "Assembly":
            solutionState["ISA"] = gfxToIsa(data["ArchitectureName"])
        solutionState["CUCount"] = data["CUCount"]
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        if solutionState["CustomKernelName"]:
            isp = {}
            if "InternalSupportParams" in solutionState:
                isp = solutionState["InternalSupportParams"]
            customConfig = getCustomKernelConfig(solutionState["CustomKernelName"], isp)
            for key, value in customConfig.items():
                solutionState[key] = value

            if "MatrixInstruction" in customConfig and len(customConfig["MatrixInstruction"]) != 4:
                raise ValueError(f"Custom kernel MatrixInstruction can only be of length 4, found {customConfig['MatrixInstruction']}")
        # overwrite problemType if any
        solutionState["ProblemType"] = problemType
        solutionObject = Solution(
                             solutionState,
                             splitGSU,
                             printSolutionRejectionReason,
                             printIndexAssignmentInfo,
                             assembler,
                             isaInfoMap,
                             srcFile
                         )
        return solutionObject

    solutions = [solutionStateToSolution(solutionState, assembler, isaInfoMap) for solutionState in data["Solutions"]]

    newLibrary, _ = SolutionLibrary.MasterSolutionLibrary.FromOriginalState(
        data,
        solutions,
        splitGSU,
        printSolutionRejectionReason,
        printIndexAssignmentInfo,
        assembler,
        isaInfoMap,
        lazyLibraryLoading
    )

    return LibraryLogic(data["ScheduleName"], data["ArchitectureName"], problemType, solutions, \
            data.get("ExactLogic"), newLibrary)


def parseLibraryLogicList(data, srcFile="?"):
    """Parses the data of a matching table style library logic file."""
    if len(data) < 9:
        printExit("Library logic file {} is missing required fields (len = {} < 9)" \
                .format(srcFile, len(data)))

    rv = {}
    rv["MinimumRequiredVersion"] = data[0]["MinimumRequiredVersion"]
    rv["ScheduleName"] = data[1]
    rv["DeviceNames"] = data[3]
    rv["ProblemType"] = data[4]
    rv["Solutions"] = data[5]

    if type(data[2]) is dict:
        rv["ArchitectureName"] = data[2]["Architecture"]
        rv["CUCount"] = data[2]["CUCount"]
    else:
        rv["ArchitectureName"] = data[2]
        rv["CUCount"] = None

    # TODOBEN: figure out what to do with these...
    rv["ExactLogic"] = data[7]
    rv["RangeLogic"] = data[8]

    # optional fields
    if len(data) > 10 and data[10]:
        rv["PerfMetric"] = data[10]

    # library logic fields
    libraryType = None
    if len(data) > 11 and data[11]:
        libraryType = data[11]
    else:
        printExit("Library logic file {} is missing required field matching property." \
                .format(srcFile))
    if libraryType == "FreeSize":
        rv["LibraryType"] = "FreeSize"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = None
        rv["Library"]["table"] = [0, len(data[5])]
        rv["Library"]["distance"] = None
        rv["Library"]["type"] = "FreeSize"
    elif libraryType == "Prediction":
        rv["LibraryType"] = "Prediction"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = None
        rv["Library"]["table"] = [0, len(data[5])]
        rv["Library"]["distance"] = None
        rv["Library"]["type"] = "Prediction"
    else:
        rv["LibraryType"] = "Matching"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = data[6]
        rv["Library"]["table"] = data[7]
        rv["Library"]["distance"] = libraryType
        rv["Library"]["type"] = "Matching"

    return rv


def rawLibraryLogic(data):
    """Returns a tuple of the data in a library logic file."""
    versionString = data[0]
    scheduleName = data[1]
    architectureName = data[2]
    deviceNames = data[3]
    problemTypeState = data[4]
    solutionStates = data[5]
    indexOrder = data[6]
    exactLogic = data[7]
    rangeLogic = data[8]
    otherFields = []

    dataLength = len(data)
    if dataLength > 9:
        for idx in range(9, dataLength):
            otherFields.append(data[idx])

    return (versionString, scheduleName, architectureName, deviceNames,\
            problemTypeState, solutionStates, indexOrder, exactLogic, rangeLogic, otherFields)


#################
# Other functions
#################
def getCUCount() -> int:
    """Return the number of CU Count in current Hardware."""
    CU = os.environ.get("CU", None)
    if CU is None:
        try:
            res = subprocess.run("rocminfo | grep Compute", stdout=subprocess.PIPE, shell=True, env={**os.environ, "ROCR_VISIBLE_DEVICES": "0"})
            CU_RE = r"Compute Unit:(?P<COMPUTE_UNIT>[\w ]+)"
            lines = res.stdout.decode("utf-8").strip().split('\n')
            if lines:
                match = re.search(CU_RE, lines[-1])
                if match:
                    CU = int(match.group('COMPUTE_UNIT').strip())
        except Exception:
            pass

    if CU is None:
        printExit("Failed to get Compute Unit count from rocminfo or env variable 'CU'")

    return int(CU)

def createLibraryLogic(schedulePrefix, architectureName, deviceNames, libraryType, logicTuple):
    """Creates the data for a library logic file suitable for writing to YAML."""
    problemType = logicTuple[0]
    solutions = logicTuple[1]
    indexOrder = logicTuple[2]
    exactLogic = logicTuple[3]
    rangeLogic = logicTuple[4]

    tileSelection = False
    if len(logicTuple) > 5 and logicTuple[5]:
        tileSelection = True

    data = []
    # Tensile version
    data.append({"MinimumRequiredVersion": __version__})
    # schedule name
    data.append(schedulePrefix)  # change from Tensile to vega10
    # schedule architecture name and get CU count
    CUCount=getCUCount()
    data.append({"Architecture": architectureName, "CUCount": CUCount} if architectureName=="gfx942" and CUCount and CUCount!=304 else architectureName)
    # schedule device names
    data.append(deviceNames)
    # problem type
    problemTypeState = problemType.state
    problemTypeState["DataType"] = \
            problemTypeState["DataType"].value
    problemTypeState["DataTypeA"] = \
            problemTypeState["DataTypeA"].value
    problemTypeState["DataTypeB"] = \
            problemTypeState["DataTypeB"].value
    problemTypeState["DataTypeE"] = \
            problemTypeState["DataTypeE"].value
    problemTypeState["DataTypeAmaxD"] = \
            problemTypeState["DataTypeAmaxD"].value
    problemTypeState["DestDataType"] = \
            problemTypeState["DestDataType"].value
    problemTypeState["ComputeDataType"] = \
            problemTypeState["ComputeDataType"].value
    problemTypeState["BiasDataTypeList"] = \
            [btype.value for btype in problemTypeState["BiasDataTypeList"]]
    problemTypeState["ActivationComputeDataType"] = \
            problemTypeState["ActivationComputeDataType"].value
    problemTypeState["ActivationType"] = \
            problemTypeState["ActivationType"].value
    problemTypeState["F32XdlMathOp"] = \
            problemTypeState["F32XdlMathOp"].value
    if "DataTypeMetadata" in problemTypeState:
        problemTypeState["DataTypeMetadata"] = \
                problemTypeState["DataTypeMetadata"].value
    data.append(problemTypeState)
    # solutions
    solutionList = []
    for solution in solutions:
        solutionState = solution.getAttributes()
        del solutionState["ProblemType"]
        isa = solutionState["ISA"]
        solutionState["ISA"] = [isa[0], isa[1], isa[2]]
        solutionList.append(solutionState)

    if tileSelection:
        tileSolutions = logicTuple[5]
        for solution in tileSolutions:
            solutionState = solution.getAttributes()
            del solutionState["ProblemType"]
            solutionList.append(solutionState)

    data.append(solutionList)
    # index order
    data.append(indexOrder)

    # exactLogic
    exactLogicList = []
    if exactLogic:
        for key in exactLogic:
            exactLogicList.append([list(key), exactLogic[key]])
        data.append(exactLogicList)
    else:
        data.append(None)

    # rangeLogic
    data.append(rangeLogic)

    if tileSelection:
        tileSelectionLogic = {}
        tileSelectionIndices = logicTuple[6]
        tileSelectionLogic["TileSelectionIndices"] = tileSelectionIndices
        data.append(tileSelectionLogic)
    else:
        data.append(None)

    data.append(logicTuple[7]) # PerfMetric
    data.append(libraryType) # LibraryType
    return data
