"""Explicit local OAuth reuse; never log, serialize, or accept URL credentials."""
import configparser
import json
import os
import re
import shutil
import subprocess


def drive_token(rclone_remote=None):
    if rclone_remote is None:
        token = os.environ.get('MUSIC_DETECTOR_DRIVE_TOKEN', '')
        if not token:
            raise ValueError('Set MUSIC_DETECTOR_DRIVE_TOKEN or select an already configured --rclone-remote.')
        return token
    if not re.fullmatch(r'[A-Za-z0-9_-]+', rclone_remote):
        raise ValueError('Use a simple configured rclone remote name, without a colon or path.')
    executable = shutil.which('rclone')
    if not executable:
        raise ValueError('rclone is not installed; use explicit OAuth environment credentials instead.')
    # Read-only call refreshes the configured OAuth token when needed.
    refreshed = subprocess.run([executable, 'about', rclone_remote + ':', '--json'],
                               capture_output=True, timeout=60)
    if refreshed.returncode:
        raise ValueError('Configured rclone authorization could not refresh. Reconnect this remote locally.')
    result = subprocess.run([executable, 'config', 'show', rclone_remote],
                            capture_output=True, text=True, timeout=10)
    try:
        if result.returncode:
            raise ValueError()
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(result.stdout)
        token = json.loads(parser[rclone_remote]['token'])['access_token']
        if not isinstance(token, str) or not token:
            raise ValueError()
    except Exception:
        raise ValueError('Cannot read the selected rclone OAuth token; configure it locally.') from None
    return token
