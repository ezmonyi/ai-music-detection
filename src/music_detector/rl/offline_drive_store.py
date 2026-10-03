"""Task-rooted rclone access, including honest Drive MD5 vs SHA-256 receipts."""
from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from .offline_archive import hashes


class RcloneStore:
    def __init__(self, config, folder_id, *, executable="rclone", proxy=None, private_log=None):
        self.config, self.folder_id = Path(config), folder_id
        profiles = configparser.ConfigParser(interpolation=None)
        profiles.read(self.config)
        selected = profiles["offline_drive"]
        if selected.get("type") != "drive" or selected.get("root_folder_id") != folder_id:
            raise ValueError("Upload profile is not rooted at the approved dataset folder")
        if self.config.stat().st_mode & 0o077:
            raise ValueError("Drive credentials require mode 0600")
        self.env = dict(os.environ)
        for key in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
            self.env.pop(key,None)
        if proxy:
            if proxy != "http://127.0.0.1:18991":
                raise ValueError("Only the explicitly authorized loopback tunnel is supported")
            self.env.update(HTTP_PROXY=proxy,HTTPS_PROXY=proxy)
        self.command = [executable,"--config",str(self.config),"--contimeout","20s",
                        "--timeout","5m","--retries","3","--low-level-retries","3"]
        self.private_log = Path(private_log or self.config.parent/"upload_runtime.log")
        fd = os.open(self.private_log,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
        self.errors = os.fdopen(fd,"ab",buffering=0)

    def close(self):
        self.errors.close()

    @staticmethod
    def remote(name):
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
            raise ValueError("Unsafe remote object name")
        return "offline_drive:"+name

    def run(self,*args):
        result = subprocess.run([*self.command,*args],stdout=subprocess.PIPE,
                                stderr=self.errors,env=self.env,check=False)
        if result.returncode:
            raise RuntimeError("rclone operation failed; inspect the private runtime log")
        return result.stdout

    def upload(self,path,name):
        expected = hashes(path)
        self.run("copyto",str(path),self.remote(name),"--immutable","--checksum","--transfers","1")
        rows = json.loads(self.run("lsjson",self.remote(name),"--hash","--hash-type","MD5"))
        if not isinstance(rows,list) or len(rows) != 1:
            raise ValueError("Drive object listing is not unique")
        record = rows[0]
        if record["Size"] != expected["bytes"] or record.get("Hashes",{}).get("MD5","").lower() != expected["md5"]:
            raise ValueError("Uploaded Drive object size/MD5 mismatch")
        return {"path":name,**expected,"drive_id":record.get("ID"),
                "drive_size_md5_verified":True,"remote_sha256_readback_verified":False}

    def download_sha256(self,name):
        process = subprocess.Popen([*self.command,"cat",self.remote(name)],
            stdout=subprocess.PIPE,stderr=self.errors,env=self.env)
        digest = hashlib.sha256()
        try:
            for block in iter(lambda: process.stdout.read(4*1024*1024),b""):
                digest.update(block)
            if process.wait():
                raise RuntimeError("rclone readback failed; inspect private runtime log")
        finally:
            process.stdout.close()
        return digest.hexdigest()
