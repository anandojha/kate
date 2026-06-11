"""Command-line entry point invoked by ``python -m glide``; delegates to the CLI."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
