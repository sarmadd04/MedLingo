"""
medlingo/inference/cli.py

MedLingo Interactive Diagnostic CLI.

Provides a command-line interface for on-demand inference against trained
MedLingo weights. Supports single-query mode, interactive session mode,
and batch evaluation from a JSON file.

Usage:
  # Single query with image
  python -m medlingo.inference.cli --image chest_xray.jpg \\
      --query "Is there evidence of pneumothorax?"

  # Interactive session (prompts for input)
  python -m medlingo.inference.cli --interactive

  # Batch evaluation
  python -m medlingo.inference.cli --batch queries.json --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from medlingo.inference.engine import MedLingoEngine, DiagnosticRequest
from medlingo.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          MedLingo — Hierarchical Clinical Diagnostics        ║
║  Resource-Efficient Multi-Agent Vision-Language Intelligence ║
╚══════════════════════════════════════════════════════════════╝
Type 'help' for commands, 'quit' to exit.
"""

_HELP_TEXT = """
Commands:
  image <path>        Set the current image file for analysis
  clear image         Remove the current image
  query <text>        Run a diagnostic query
  stats               Show engine performance statistics
  help                Show this help message
  quit / exit         Terminate the session
"""


def run_single(
    query: str,
    image: Optional[str] = None,
    config: Optional[str] = None,
    device: str = "auto",
    verbose: bool = False,
) -> None:
    """Execute a single diagnostic query and print the result."""
    setup_logging("INFO")

    with MedLingoEngine(config_path=config, device=device, verbose=verbose) as engine:
        request = DiagnosticRequest(
            query=query,
            image_path=image,
            request_id="cli-single",
        )
        response = engine.diagnose(request)
        print("\n" + str(response) + "\n")
        print(f"Latency: {response.latency_ms:.0f} ms | "
              f"Confidence: {response.overall_confidence}")


def run_interactive(
    config: Optional[str] = None,
    device: str = "auto",
) -> None:
    """Launch an interactive diagnostic session."""
    setup_logging("WARNING")    # Suppress model loading noise in interactive mode
    print(_BANNER)

    current_image: Optional[str] = None
    request_counter = 0

    with MedLingoEngine(config_path=config, device=device) as engine:
        print("Loading MedLingo pipeline... (this may take a moment)\n")

        while True:
            try:
                raw = input("medlingo> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession terminated.")
                break

            if not raw:
                continue

            if raw.lower() in ("quit", "exit"):
                print("Goodbye.")
                break

            if raw.lower() == "help":
                print(_HELP_TEXT)
                continue

            if raw.lower() == "stats":
                print(json.dumps(engine.stats, indent=2))
                continue

            if raw.lower().startswith("image "):
                path_str = raw[6:].strip()
                if not Path(path_str).exists():
                    print(f"File not found: {path_str}")
                else:
                    current_image = path_str
                    print(f"Image set: {path_str}")
                continue

            if raw.lower() == "clear image":
                current_image = None
                print("Image cleared.")
                continue

            # Everything else is treated as a diagnostic query
            query = raw
            if raw.lower().startswith("query "):
                query = raw[6:].strip()

            request_counter += 1
            request = DiagnosticRequest(
                query=query,
                image_path=current_image,
                request_id=f"interactive-{request_counter}",
            )

            print("\nAnalyzing...\n")
            try:
                response = engine.diagnose(request)
                print(str(response))
                print(f"\n[{response.latency_ms:.0f} ms | {response.overall_confidence}]\n")
            except Exception as exc:
                print(f"Error during inference: {exc}")
                logger.exception("Inference error")


def run_batch(
    batch_file: str,
    output_file: str,
    config: Optional[str] = None,
    device: str = "auto",
) -> None:
    """Run inference on a batch of queries from a JSON file."""
    setup_logging("INFO")

    with open(batch_file) as f:
        queries = json.load(f)

    requests = [
        DiagnosticRequest(
            query=q["query"],
            image_path=q.get("image_path"),
            image_description=q.get("image_description"),
            request_id=q.get("id", str(i)),
        )
        for i, q in enumerate(queries)
    ]

    results = []
    with MedLingoEngine(config_path=config, device=device) as engine:
        for req in requests:
            logger.info("Processing request: %s", req.request_id)
            resp = engine.diagnose(req)
            results.append(resp.to_dict())

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Batch complete — %d results saved to %s", len(results), output_file)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedLingo Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m medlingo.inference.cli --image scan.jpg --query "Any abnormality?"
  python -m medlingo.inference.cli --interactive
  python -m medlingo.inference.cli --batch queries.json --output results.json
        """,
    )
    parser.add_argument("--query",       type=str, help="Diagnostic query string.")
    parser.add_argument("--image",       type=str, help="Path to medical image file.")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive session.")
    parser.add_argument("--batch",       type=str,
                        help="Path to JSON file with batch queries.")
    parser.add_argument("--output",      type=str, default="results.json",
                        help="Output path for batch results (default: results.json).")
    parser.add_argument("--config",      type=str, default=None,
                        help="Path to inference_config.yaml.")
    parser.add_argument("--device",      type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--verbose",     action="store_true",
                        help="Log routing and SPR details.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.interactive:
        run_interactive(config=args.config, device=args.device)
    elif args.batch:
        run_batch(args.batch, args.output, config=args.config, device=args.device)
    elif args.query:
        run_single(
            query=args.query,
            image=args.image,
            config=args.config,
            device=args.device,
            verbose=args.verbose,
        )
    else:
        print("No mode specified. Use --query, --interactive, or --batch.")
        print("Run with --help for usage information.")
        sys.exit(1)


def main():
    """setuptools entry point — called by the `medlingo-infer` console script."""
    args = _parse_args()
    if args.interactive:
        run_interactive(config=args.config, device=args.device)
    elif args.batch:
        run_batch(args.batch, args.output, config=args.config, device=args.device)
    elif args.query:
        run_single(
            query=args.query,
            image=args.image,
            config=args.config,
            device=args.device,
            verbose=args.verbose,
        )
    else:
        import sys
        print("No mode specified. Use --query, --interactive, or --batch.")
        sys.exit(1)
