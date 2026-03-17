# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Template rendering orchestrator."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .models import OperationConfig


class DescriptorGenerator:
    """Renders all templates for a given OperationConfig."""

    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_lift_only(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render only lifting-related templates. Returns list of written files."""
        written = []

        # Lifting file templates
        lift_templates = {
            "unpacker.hpp.j2": Path("frontend/include/hipdnn_frontend/detail")
            / config.unpacker_filename,
            "test_from_node.cpp.j2": Path("backend/tests/descriptors")
            / config.test_from_node_filename,
        }

        for template_name, rel_path in lift_templates.items():
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(str(rel_path))

        # Lifting fragment templates
        lift_fragments = {
            "fragments/node_factory_case.j2": "node_factory_case.txt",
            "fragments/operation_unpacker_case.j2": "operation_unpacker_case.txt",
            "fragments/operation_type_enum.j2": "operation_type_enum.txt",
            "fragments/node_unpack_override.j2": "node_unpack_override.txt",
        }

        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)

        for template_name, filename in lift_fragments.items():
            out_path = fragments_dir / filename
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(f"fragments/{filename}")

        # Generate descriptor lifting additions (manual insertion guide)
        additions = self._render_descriptor_lifting_additions(config)
        additions_path = fragments_dir / "descriptor_lifting_additions.txt"
        additions_path.write_text(additions)
        written.append("fragments/descriptor_lifting_additions.txt")

        return written

    def _render_descriptor_lifting_additions(self, config: OperationConfig) -> str:
        """Generate a text file showing what to add to existing descriptor files."""
        lines = []
        cn = config.class_name

        lines.append(f"# Descriptor Lifting Additions for {cn}")
        lines.append(f"# Add these changes to the existing {cn}.hpp/.cpp files.")
        lines.append("")

        # --- HPP additions ---
        lines.append("=" * 72)
        lines.append(f"# {cn}.hpp — Add these to the class declaration")
        lines.append("=" * 72)
        lines.append("")
        lines.append("# 1. Add include (at top of file):")
        lines.append("#include <unordered_map>")
        lines.append("")
        lines.append("# 2. Add public static method (after buildNode declaration):")
        lines.append(f"    static std::shared_ptr<{cn}>")
        lines.append(f"        fromNode(const {config.fbs_namespace}::NodeT& nodeT,")
        lines.append(
            "                 const std::unordered_map<int64_t, "
            "std::shared_ptr<TensorDescriptor>>& tensorMap);"
        )
        lines.append("")
        lines.append("# 3. Add private member (after _data):")
        lines.append("    std::string _name;")
        lines.append("")

        # --- CPP additions ---
        lines.append("=" * 72)
        lines.append(f"# {cn}.cpp — Add these changes")
        lines.append("=" * 72)
        lines.append("")
        lines.append("# 1. Add include:")
        lines.append('#include "HipdnnOperationType.h"')
        lines.append("")

        lines.append("# 2. In setAttribute switch, add before default:")
        lines.append("    case HIPDNN_ATTR_OPERATION_NAME_EXT:")
        lines.append("        setString(_name,")
        lines.append("                  attributeType,")
        lines.append("                  elementCount,")
        lines.append("                  arrayOfElements,")
        lines.append(f'                  "{cn}::setAttribute()");')
        lines.append("        break;")
        lines.append("")

        lines.append("# 3. In getAttribute switch, add before default:")
        lines.append("    case HIPDNN_ATTR_OPERATION_NAME_EXT:")
        lines.append("        getString(_name,")
        lines.append("                  attributeType,")
        lines.append("                  requestedElementCount,")
        lines.append("                  elementCount,")
        lines.append("                  arrayOfElements,")
        lines.append(f'                  "{cn}::getAttribute()");')
        lines.append("        break;")

        if config.operation_type_enum:
            lines.append("    case HIPDNN_ATTR_OPERATION_TYPE_EXT:")
            lines.append(f"        getOperationType({config.operation_type_enum},")
            lines.append("                         attributeType,")
            lines.append("                         requestedElementCount,")
            lines.append("                         elementCount,")
            lines.append("                         arrayOfElements,")
            lines.append(f'                         "{cn}::getAttribute()");')
            lines.append("        break;")
        lines.append("")

        lines.append("# 4. In buildNode(), add before compute_data_type:")
        lines.append("    node->name = _name;")
        lines.append("")

        lines.append("# 5. In toString(), prepend name to output:")
        lines.append('    str += "name=" + _name;')
        lines.append("")

        lines.append("# 6. Add fromNode() implementation (at end of file):")
        content = self._render_template("descriptor.cpp.j2", config)
        # Extract fromNode from the rendered output
        from_node_start = content.find(f"std::shared_ptr<{cn}> {cn}::fromNode(")
        if from_node_start >= 0:
            # Find the closing brace
            brace_depth = 0
            i = content.index("{", from_node_start)
            for j in range(i, len(content)):
                if content[j] == "{":
                    brace_depth += 1
                elif content[j] == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        from_node_end = j + 1
                        break
            lines.append(content[from_node_start:from_node_end])
        lines.append("")

        return "\n".join(lines) + "\n"

    def render(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render all templates and write to output_dir. Returns list of written files."""
        written = []

        # Template -> output path mapping
        file_templates = {
            "descriptor.hpp.j2": Path("backend/src/descriptors")
            / config.header_filename,
            "descriptor.cpp.j2": Path("backend/src/descriptors")
            / config.source_filename,
            "packer.hpp.j2": Path("frontend/include/hipdnn_frontend/detail")
            / config.packer_filename,
            "test_descriptor.cpp.j2": Path("backend/tests/descriptors")
            / config.test_descriptor_filename,
            "test_graph_ops.cpp.j2": Path("backend/tests/descriptors")
            / config.test_graph_filename,
            "test_from_node.cpp.j2": Path("backend/tests/descriptors")
            / config.test_from_node_filename,
            "test_integration.cpp.j2": Path("tests/frontend")
            / config.test_integration_filename,
            "unpacker.hpp.j2": Path("frontend/include/hipdnn_frontend/detail")
            / config.unpacker_filename,
        }

        for template_name, rel_path in file_templates.items():
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(str(rel_path))

        # Fragment templates
        fragment_templates = {
            "fragments/attribute_enum_block.j2": "attribute_enum_block.txt",
            "fragments/descriptor_type_enum.j2": "descriptor_type_enum.txt",
            "fragments/string_utils_block.j2": "string_utils_block.txt",
            "fragments/factory_case.j2": "factory_case.txt",
            "fragments/cmake_entries.j2": "cmake_entries.txt",
            "fragments/node_factory_case.j2": "node_factory_case.txt",
            "fragments/operation_unpacker_case.j2": "operation_unpacker_case.txt",
            "fragments/operation_type_enum.j2": "operation_type_enum.txt",
            "fragments/node_unpack_override.j2": "node_unpack_override.txt",
        }

        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)

        for template_name, filename in fragment_templates.items():
            out_path = fragments_dir / filename
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(f"fragments/{filename}")

        return written

    def _render_template(self, template_name: str, config: OperationConfig) -> str:
        try:
            template = self.env.get_template(template_name)
            return template.render(op=config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to render template '{template_name}' for "
                f"operation '{config.name}': {e}"
            ) from e
