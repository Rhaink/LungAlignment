#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Evaluate ensemble using a JSON config file.'
    )
    parser.add_argument('--config', required=True, help='Path to ensemble config JSON')
    parser.add_argument('--out', help='Optional output log file')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    config = json.loads(config_path.read_text())
    models = config.get('models')
    if not isinstance(models, list) or not models:
        raise SystemExit('Config must include a non-empty "models" list.')

    for model in models:
        if not Path(model).is_file():
            raise SystemExit(f"Missing checkpoint: {model}")

    cmd = [sys.executable, '-m', 'src_v2', 'evaluate-ensemble', *models]
    if config.get('tta', False):
        cmd.append('--tta')
    if config.get('clahe', False):
        cmd.append('--clahe')

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd)

    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
