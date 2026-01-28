"""
Entry point para ejecutar como modulo.

Uso:
    python -m src_v2 --help
    python -m src_v2 train --help
    python -m src_v2 evaluate checkpoint.pt
    python -m src_v2 predict image.png --checkpoint model.pt
    python -m src_v2 warp input/ output/ --checkpoint model.pt
"""

from src_v2.cli import main

if __name__ == "__main__":
    main()
