"""Script to visualize LangGraph architectures from specified Python files.

Usage:
    python visualize_graphs.py file1.py:function_name file2.py:function_name [...]

Examples:
    python visualize_graphs.py 01_router_handoff_command.py:create_handoff_graph
    python visualize_graphs.py 01_router_handoff_command.py:create_handoff_graph 02_router_handoff_tools.py:create_supervisor_agent
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .graph_utils import print_graph_info, save_graph_both


def parse_file_function(spec: str) -> tuple[str, str]:
    """Parse file:function specification.

    Args:
        spec: String in format "filename.py:function_name"

    Returns:
        Tuple of (filename, function_name)

    Raises:
        ValueError: If spec format is invalid
    """
    if ":" not in spec:
        msg = f"Invalid spec '{spec}'. Expected format: 'filename.py:function_name'"
        raise ValueError(msg)

    filename, function_name = spec.rsplit(":", 1)

    # Validate file exists
    if not Path(filename).exists():
        msg = f"File '{filename}' not found"
        raise FileNotFoundError(msg)

    # Remove .py extension for import
    module_name = filename.replace(".py", "")

    return module_name, function_name


def import_and_call_function(module_name: str, function_name: str) -> Any:
    """Import module and call specified function."""
    try:
        # Import module (handles numeric prefixes automatically)
        module = importlib.import_module(module_name)

        # Get function from module
        if not hasattr(module, function_name):
            available_functions = [
                name for name in dir(module) if callable(getattr(module, name)) and not name.startswith("_")
            ]
            msg = f"Function '{function_name}' not found in '{module_name}'. Available functions: {available_functions}"
            raise AttributeError(msg)

        func = getattr(module, function_name)

        # Call function and return result
        logger.info(f"Calling {module_name}.{function_name}()")
        return func()

    except Exception as e:
        logger.error(f"Error importing/calling {module_name}.{function_name}: {e}")
        raise


def create_safe_filename(module_name: str, function_name: str) -> str:
    """Create safe filename from module and function names."""
    # Replace problematic characters
    return f"{module_name}_{function_name}".replace("-", "_").replace(".", "_")


def main() -> None:
    """Main function to visualize graphs from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize LangGraph architectures from Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 01_router_handoff_command.py:create_handoff_graph
  %(prog)s 01_router_handoff_command.py:create_handoff_graph 02_router_handoff_tools.py:create_workflow_graph
  %(prog)s 03_agent_handoff_tools.py:create_tool_handoff_graph --output-dir ./my_graphs
        """,
    )

    parser.add_argument(
        "specs", nargs="+", help="File and function specifications in format 'filename.py:function_name'"
    )

    parser.add_argument(
        "--output-dir", default="graphs", help="Output directory for graph files (default: graphs)"
    )

    parser.add_argument("--list-functions", metavar="FILE", help="List available functions in a Python file")

    args = parser.parse_args()

    # Handle list-functions option
    if args.list_functions:
        try:
            module_name = args.list_functions.replace(".py", "")
            module = importlib.import_module(module_name)
            functions = [
                name for name in dir(module) if callable(getattr(module, name)) and not name.startswith("_")
            ]
            logger.info(f"Available functions in {args.list_functions}:")
            for func in functions:
                logger.info(f"  - {func}")
        except Exception as e:
            logger.error(f"Error listing functions: {e}")
        return

    logger.info("🎨 Visualizing LangGraph Architectures")

    # Create output directory (including parent directories)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs_created = []

    # Process each specification
    for spec in args.specs:
        try:
            # Parse specification
            module_name, function_name = parse_file_function(spec)

            # Import and call function
            graph = import_and_call_function(module_name, function_name)

            # Print graph information
            logger.info("\n" + "=" * 60)
            logger.info(f"GRAPH: {module_name}.{function_name}")
            logger.info("=" * 60)
            print_graph_info(graph)

            # Create safe filename
            base_name = create_safe_filename(module_name, function_name)

            # Save graph visualizations
            logger.info(f"\n📊 Saving visualizations for {base_name}...")

            # Save both basic and detailed versions
            basic_file = output_dir / base_name
            detailed_file = output_dir / f"{base_name}_detailed"

            save_graph_both(graph, str(basic_file), xray=0)
            save_graph_both(graph, str(detailed_file), xray=1)

            graphs_created.extend(
                [f"{base_name}.png/.mmd (basic view)", f"{base_name}_detailed.png/.mmd (detailed view)"]
            )

        except Exception as e:
            logger.error(f"❌ Failed to process '{spec}': {e}")
            continue

    # Summary
    if graphs_created:
        logger.info(f"\n✅ Graph visualizations saved to '{output_dir}/' directory")
        logger.info("📁 Files created:")
        for file in graphs_created:
            logger.info(f"  - {file}")

        logger.info("\n💡 You can:")
        logger.info("  1. Open PNG files to view the graph diagrams")
        logger.info("  2. Use .mmd files with Mermaid tools/editors")
        logger.info("  3. View the graphs online at https://mermaid.live/")
    else:
        logger.warning("No graphs were successfully created")
        sys.exit(1)


if __name__ == "__main__":
    main()
