"""Graph visualization utilities for LangGraph workflows."""

import os
from pathlib import Path
from typing import Any

from loguru import logger


def save_graph_png(
    graph: Any, filename: str = "graph", output_dir: str = "../graphs", *, xray: int = 0
) -> None:
    """Save the graph as a PNG image file.

    Args:
        graph: The LangGraph workflow graph
        filename: Name of the output file (without extension)
        output_dir: Directory to save the graph files
        xray: X-ray level for graph visualization (0=basic, 1=detailed)
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)

        # Generate PNG and save to file
        png_data = graph.get_graph(xray=xray).draw_mermaid_png()
        png_path = Path(output_dir) / f"{filename}.png"

        with png_path.open("wb") as f:
            f.write(png_data)

        logger.info(f"Graph PNG saved to: {png_path.absolute()}")

    except Exception as e:
        logger.error(f"Error saving graph PNG: {e}")


def save_graph_mermaid(
    graph: Any, filename: str = "graph", output_dir: str = "graphs", *, xray: int = 0
) -> None:
    """Save the graph as a Mermaid diagram file.

    Args:
        graph: The LangGraph workflow graph
        filename: Name of the output file (without extension)
        output_dir: Directory to save the graph files
        xray: X-ray level for graph visualization (0=basic, 1=detailed)
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)

        # Generate Mermaid diagram and save to file
        mermaid_code = graph.get_graph(xray=xray).draw_mermaid()
        mermaid_path = Path(output_dir) / f"{filename}.mmd"

        with open(mermaid_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)

        logger.info(f"Graph Mermaid diagram saved to: {mermaid_path.absolute()}")

    except Exception as e:
        logger.error(f"Error saving graph Mermaid: {e}")


def save_graph_both(
    graph: Any, filename: str = "graph", output_dir: str = "graphs", *, xray: int = 0
) -> None:
    """Save the graph as both PNG and Mermaid files.

    Args:
        graph: The LangGraph workflow graph
        filename: Name of the output file (without extension)
        output_dir: Directory to save the graph files
        xray: X-ray level for graph visualization (0=basic, 1=detailed)
    """
    save_graph_png(graph, filename, output_dir, xray=xray)
    save_graph_mermaid(graph, filename, output_dir, xray=xray)


def print_graph_info(graph: Any, *, xray: int = 0) -> None:
    """Print basic information about the graph structure.

    Args:
        graph: The LangGraph workflow graph
        xray: X-ray level for graph visualization (0=basic, 1=detailed)
    """
    try:
        graph_repr = graph.get_graph(xray=xray)

        logger.info("Graph Structure:")
        logger.info(f"  Nodes: {len(graph_repr.nodes)}")
        logger.info(f"  Edges: {len(graph_repr.edges)}")

        logger.info("  Node names:")
        for node in graph_repr.nodes:
            logger.info(f"    - {node}")

        logger.info("  Edges:")
        for edge in graph_repr.edges:
            logger.info(f"    - {edge}")

    except Exception as e:
        logger.error(f"Error printing graph info: {e}")


def open_graph_file(filename: str = "graph", output_dir: str = "graphs", file_type: str = "png") -> None:
    """Attempt to open the saved graph file with the default system application.

    Args:
        filename: Name of the file (without extension)
        output_dir: Directory containing the graph files
        file_type: Type of file to open ('png' or 'mmd')
    """
    try:
        file_path = Path(output_dir) / f"{filename}.{file_type}"

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return

        # Try to open with default system application
        if os.name == "nt":  # Windows
            os.startfile(file_path)
        elif os.name == "posix":  # macOS and Linux
            os.system(f'open "{file_path}"' if os.uname().sysname == "Darwin" else f'xdg-open "{file_path}"')

        logger.info(f"Opened graph file: {file_path}")

    except Exception as e:
        logger.error(f"Error opening graph file: {e}")
