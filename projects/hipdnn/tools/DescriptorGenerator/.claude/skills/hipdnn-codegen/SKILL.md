---
name: hipdnn-codegen
description: Generate hipDNN operation boilerplate from a FlatBuffer schema. Use when the user wants to add a new operation type to hipDNN, or generate descriptor/packer/unpacker code.
argument-hint: "<schema-path-or-op-name> [mode: backend|frontend|full|lift-only]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, AskUserQuestion
---

# hipDNN Code Generator Skill

Generate all boilerplate code needed to add a new operation to hipDNN from a FlatBuffer schema.

## Arguments

- `$ARGUMENTS` can contain:
  - **Schema or operation**: Path to `.fbs` schema file, path to existing YAML config, or operation name (e.g., `convolution_fwd`)
  - **Mode** (one of):
    - `backend` (default) - Descriptor, Packer, Unpacker + backend tests
    - `frontend` - Node, Attributes, Graph method + frontend tests
    - `full` - Everything (backend + frontend)
    - `lift-only` - Unpacker + lifting fragments for existing operations

## Directory Locations

Determine the hipDNN project root by finding the nearest parent directory containing `tools/DescriptorGenerator/`. Set paths relative to that:

```
HIPDNN_SRC=<path to projects/hipdnn>
CODEGEN=$HIPDNN_SRC/tools/DescriptorGenerator
VENV=$CODEGEN/.venv
```

Hint: if the current working directory is inside a rocm-libraries worktree, the hipDNN root is at `<worktree>/projects/hipdnn/`. If invoked from a standalone hipDNN checkout, it is the repo root.

## Execution Steps

### 1. Parse Arguments and Set Up

Parse `$ARGUMENTS` for:
- Schema path, config path, or operation name
- Mode (default: `backend`)

Locate the hipDNN project root. Verify the codegen venv exists:
```bash
cd $CODEGEN
if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi
```

### 2. Determine Input Type

- If argument is a `.fbs` file path: proceed to Step 3 (create YAML config from schema)
- If argument is a `.yaml` file path: skip to Step 5 (run generator)
- If argument is an operation name and `configs/<name>.yaml` exists: skip to Step 5
- If none of the above: ask the user to provide a schema or config path

### 3. Create YAML Config from FBS Schema

Read the FBS schema file. Map fields to YAML config following these rules:

| FBS Field Pattern | YAML Section | YAML `type` |
|---|---|---|
| `*_tensor_uid: long` | `tensor_fields` | (tensors are UIDs) |
| `field: [long]` | `data_fields` | `vector_int64` |
| `field: SomeEnum` | `data_fields` | `mode` |
| `field: float` | `data_fields` | `scalar_float` |
| `field: long` (non-UID) | `data_fields` | `scalar_int64` |
| `field: bool` | `data_fields` | `bool` |
| `field: [long]` (array of UIDs) | `tensor_array_fields` | (tensor arrays) |

Ask the user for information the schema alone cannot provide:
- **Operation name** (PascalCase, e.g., `ConvolutionFwd`)
- **Descriptor type enum name** (e.g., `HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`)
- **Operation attribute prefix** (e.g., `HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD`)
- **Whether any attributes are shared** with other existing operations
- **Whether this operation has a compute data type**
- **Test tensor UIDs** (suggest non-conflicting range: existing ranges documented in the reference doc)

For frontend mode or full mode, also ask:
- **Graph method name** (e.g., `conv_fprop`)
- **NodeType enum value** (e.g., `CONVOLUTION_FPROP`)
- **FlatBuffer NodeAttributes union type** (e.g., `ConvolutionFwdAttributes`)

For any question the user skips, use sensible defaults derived from the operation name.

Write the config to `$CODEGEN/configs/<operation>.yaml`.

Use `$CODEGEN/configs/convolution_fwd.yaml` as the reference template for all config fields.

### 4. Handle Mode Enum Fields

If the FBS schema references enum types, check whether each enum already exists in the backend:

```bash
grep -r "HIPDNN_TYPE_" $HIPDNN_SRC/backend/include/HipdnnBackendAttributeType.h
```

For each new enum type not already in the backend, inform the user that additional files must be created:
1. Backend C-API enum header (`backend/include/Hipdnn<Foo>Mode.h`)
2. Type tag entry in `HipdnnBackendAttributeType.h`
3. SDK conversion functions in `DataTypeConversion.hpp/.cpp`
4. Shared helpers in `DescriptorAttributeUtils.hpp/.cpp`
5. String utility case in `BackendEnumStringUtils.hpp`
6. Frontend converter in `Types.hpp`

Ask the user if they want to create these now or handle them separately.

### 5. Run the Generator

```bash
cd $CODEGEN
OUTPUT_DIR=/tmp/hipdnn-codegen-output
rm -rf $OUTPUT_DIR
$VENV/bin/python generate.py \
    --config configs/<operation>.yaml \
    --output-dir $OUTPUT_DIR \
    --mode $MODE
```

If the generator fails, show the error and stop.

List all generated files:
```bash
find $OUTPUT_DIR -type f | sort
```

### 6. Place Generated Files

Copy each generated file to its target location in the project tree.

**Backend files** (for `backend` or `full` mode):

| Generated Path | Target |
|----------------|--------|
| `backend/src/descriptors/<Op>OperationDescriptor.hpp` | `$HIPDNN_SRC/backend/src/descriptors/` |
| `backend/src/descriptors/<Op>OperationDescriptor.cpp` | `$HIPDNN_SRC/backend/src/descriptors/` |
| `frontend/include/hipdnn_frontend/detail/<Op>Packer.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/detail/` |
| `frontend/include/hipdnn_frontend/detail/<Op>Unpacker.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/detail/` |
| `backend/tests/descriptors/Test<Op>OperationDescriptor.cpp` | `$HIPDNN_SRC/backend/tests/descriptors/` |
| `backend/tests/descriptors/TestGraphDescriptor<Op>.cpp` | `$HIPDNN_SRC/backend/tests/descriptors/` |
| `backend/tests/descriptors/Test<Op>OperationFromNode.cpp` | `$HIPDNN_SRC/backend/tests/descriptors/` |
| `tests/frontend/Integration<Op>DescriptorLowering.cpp` | `$HIPDNN_SRC/tests/frontend/` |

**Frontend files** (for `frontend` or `full` mode):

| Generated Path | Target |
|----------------|--------|
| `frontend/include/hipdnn_frontend/attributes/<Op>Attributes.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/attributes/` |
| `frontend/include/hipdnn_frontend/node/<Op>Node.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/node/` |
| `frontend/tests/Test<Op>Attributes.cpp` | `$HIPDNN_SRC/frontend/tests/` |
| `frontend/tests/Test<Op>Node.cpp` | `$HIPDNN_SRC/frontend/tests/` |
| `frontend/tests/TestGraph<Op>.cpp` | `$HIPDNN_SRC/frontend/tests/` |

### 7. Insert Fragment Snippets

Read each fragment file from the output and insert it into the correct shared file.

**CRITICAL**: Read each target file FIRST to find the correct insertion point. Never blindly append.

**Backend fragments** (for `backend` or `full` mode):

| Fragment | Target File | Insertion Point |
|----------|-------------|----------------|
| `fragments/attribute_enum_block.txt` | `$HIPDNN_SRC/backend/include/HipdnnBackendAttributeName.h` | After the last operation attribute enum range. Assign next available range (read existing ranges to find the next one). Replace `PLACEHOLDER_VALUE` with actual values. |
| `fragments/descriptor_type_enum.txt` | `$HIPDNN_SRC/backend/include/HipdnnBackendDescriptorType.h` | Before the closing brace of `hipdnnBackendDescriptorType_t` enum. |
| `fragments/string_utils_block.txt` | `$HIPDNN_SRC/backend/src/BackendEnumStringUtils.hpp` | In the appropriate switch statements (descriptor type name + attribute name). |
| `fragments/factory_case.txt` | `$HIPDNN_SRC/backend/src/descriptors/DescriptorFactory.cpp` | Add `#include` at top, add `case` in `create()` switch. |
| `fragments/cmake_entries.txt` | Multiple CMakeLists.txt files | Add source to `backend/src/CMakeLists.txt`, tests to `backend/tests/CMakeLists.txt` and `tests/frontend/CMakeLists.txt`. |

**Lifting fragments** (for `backend`, `full`, or `lift-only` mode):

| Fragment | Target File | Insertion Point |
|----------|-------------|----------------|
| `fragments/node_factory_case.txt` | `$HIPDNN_SRC/backend/src/descriptors/NodeFactory.cpp` | In the `createFromNode()` switch. |
| `fragments/operation_unpacker_case.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/detail/OperationUnpacker.hpp` | In the `createNodeForType()` switch. |
| `fragments/operation_type_enum.txt` | `$HIPDNN_SRC/backend/include/HipdnnOperationType.h` | Before the closing brace of the enum. |
| `fragments/node_unpack_override.txt` | Frontend node header | Add method to the node class. |

**Frontend fragments** (for `frontend` or `full` mode):

| Fragment | Target File | Insertion Point |
|----------|-------------|----------------|
| `fragments/graph_method.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/Graph.hpp` | After the last operation method (find the pattern `_sub_nodes.emplace_back`). |
| `fragments/graph_includes.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/Graph.hpp` | In the includes section at the top, grouped with other operation includes. |
| `fragments/deserialize_case.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/Graph.hpp` | In the `deserializeFromFlatBuffer()` switch. |
| `fragments/frontend_cmake_entries.txt` | `$HIPDNN_SRC/frontend/tests/CMakeLists.txt` | Add test files to the test target source list. |

### 8. Add Enum Test Coverage

After inserting enum fragments, add test entries to `$HIPDNN_SRC/backend/tests/TestBackendEnumStringUtils.cpp`:

- One `EXPECT_STREQ` for the descriptor type name
- One `EXPECT_STREQ` per new attribute enum value
- Place entries near existing similar tests

### 9. Apply Descriptor Lifting Additions (lift-only and full modes)

If `fragments/descriptor_lifting_additions.txt` exists:
- Read it for the exact changes needed to the existing descriptor `.hpp` and `.cpp`
- Apply each change (add `#include <unordered_map>`, `fromNode()` declaration, `_name` member to `.hpp`; add operation name/type handling and `fromNode()` impl to `.cpp`)

### 10. Wire Frontend Node (if node class exists or was generated)

For `backend` mode when a frontend node already exists, or for `full` mode:
- Add `#include` for packer header to the node class
- Add `create_operation()` override calling the packer
- Add `#include` for unpacker header
- Add `unpack_from_descriptor()` override calling the unpacker

### 11. Ask About Operation-Specific Logic

For `frontend` or `full` mode, ask the user about:

**infer_properties_node()**:
- "How should output dimensions be inferred?"
  - Option 1: Output matches input dimensions (e.g., batchnorm, pointwise unary)
  - Option 2: Output is broadcast of inputs (e.g., pointwise binary)
  - Option 3: Custom formula (ask user to describe, generate TODO stub with the description)
  - Option 4: Leave as stub (default)

**pre_validate_node()**:
- The generated code already checks for null required tensors and non-empty dimensions
- "Are there additional validation rules?"
  - e.g., "input channels must match weight channels"
  - e.g., "stride and dilation must be > 0"
  - Default: leave with just the standard null/dim checks

If the user chooses stubs, ensure the TODO comments are descriptive enough for later implementation.

### 12. Build and Test

Build to verify everything compiles. Use the ROCm Clang toolchain:
```bash
cd $HIPDNN_SRC
mkdir -p build && cd build
cmake .. -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=<repo-root>/cmake/toolchains/rocm-clang.cmake
ninja 2>&1 | tail -100
```

If build succeeds, run unit tests:
```bash
ninja unit-check 2>&1 | tail -50
```

### 13. Report Results

Summarize what was generated and placed:
- List all files created/modified
- Note any stubs that need manual implementation (infer_properties, validation, integration test)
- Note any fragment insertions that need manual verification (enum value ranges, CMake)
- If build/tests passed, confirm
- If anything failed, show the error

## Error Handling

- If the FBS schema has field types not supported by the generator, report them and ask the user how to handle
- If enum value ranges conflict with existing values, report and ask the user for the correct range
- If the generator script fails, show the full error output
- If build fails after placement, show the last 100 lines and help diagnose
- If a target file for fragment insertion cannot be found, report and skip that insertion

## Notes

- The integration test (`Integration<Op>DescriptorLowering.cpp`) is always a STUB. Remind the user it needs manual implementation.
- For `lift-only` mode, existing descriptor files are modified in-place per `descriptor_lifting_additions.txt`.
- `mode` type is REQUIRED for all enum fields in new operations. Never use the legacy `enum` type.
- Read `$CODEGEN/CLAUDE.md` for the full detailed post-generation workflow if you need additional context on any step.
- Always use `convolution_fwd.yaml` as the reference config when creating new configs.
