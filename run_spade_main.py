"""Single entry point for the clean-field SPADE main method.

Batch IDs identify result folders. Algorithm settings come from
spade_main_config.json and stay independent of the batch ID.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    configuration = json.loads((root / 'spade_main_config.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=list(configuration['case_wall_seconds']))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-id', default='example')
    parser.add_argument('--print-command', action='store_true')
    arguments = parser.parse_args()
    command = [sys.executable, str(root / 'run_search.py'), '--root', str(root),
               '--case', arguments.case, '--seed', str(arguments.seed),
               '--device', arguments.device, '--run-id', arguments.batch_id,
               '--seconds', str(configuration['case_wall_seconds'][arguments.case])]
    for key, value in configuration['search'].items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    if arguments.print_command:
        print(json.dumps(command, ensure_ascii=False))
    else:
        raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == '__main__':
    main()
