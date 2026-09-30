"""Command-line entrypoints for the same analysis used by the upload service."""
import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='music-detector')
    commands = parser.add_subparsers(dest='command', required=True)
    server = commands.add_parser('serve', help='Start the private local upload UI')
    server.add_argument('--port', type=int, default=8765)
    server.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    analysis = commands.add_parser('analyze', help='Analyze a native stereo recording')
    analysis.add_argument('audio', type=Path)
    analysis.add_argument('--artifacts', default='F,H,SC')
    analysis.add_argument('--research', action='store_true')
    analysis.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.command == 'serve':
        import uvicorn
        os.environ['MUSIC_DETECTOR_DEVICE'] = args.device
        uvicorn.run('music_detector.api:app', host='127.0.0.1', port=args.port)
    else:
        from .analysis import analyze
        # Third-party frontends print progress; keep stdout valid JSON.
        with redirect_stdout(sys.stderr):
            result = analyze(args.audio, args.artifacts.split(','), research=args.research, device=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
