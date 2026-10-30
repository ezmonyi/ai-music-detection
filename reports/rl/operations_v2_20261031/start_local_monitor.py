"""Project-only private TensorBoard; provider operations stay in their skill env."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

OPS=Path(__file__).resolve().parent
MONITOR=OPS.parent/'rl_live_monitor_20261031'
RUNTIME=MONITOR/'runtime'
CONDA='/Users/yi/miniforge3/bin/conda'


def main():
    RUNTIME.mkdir(parents=True,exist_ok=True)
    receipt=RUNTIME/'LOCAL_MONITORS.json'
    if receipt.exists():
        print(receipt.read_text());return
    with socket.socket() as check:
        check.bind(('127.0.0.1',16006))
    env={k:v for k,v in os.environ.items() if not k.startswith('MATPOOL_') and k!='MATPILOT_CONFIRMED_MATPOOL_ACTION'}
    commands={
        'tensorboard':[CONDA,'run','--no-capture-output','-n','music-rl-monitor','tensorboard',
            '--logdir',str(RUNTIME/'tb_mirror/runs'),'--host','127.0.0.1','--port','16006','--reload_interval','5'],
        'event_mirror':[CONDA,'run','--no-capture-output','-n','matpool-machine-provider','python',str(OPS/'live_event_mirror.py')],
    }
    processes={}
    for name,command in commands.items():
        with (RUNTIME/(name+'.log')).open('ab') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        processes[name]={'pid':child.pid,'command':command,'log':str(RUNTIME/(name+'.log'))}
    value={'utc':datetime.now(timezone.utc).isoformat(),'node_id':2128889,'processes':processes,
        'tensorboard_url':'http://127.0.0.1:16006','gpu_monitor_url':'https://matpool.com/node/ZW703Y/monitor',
        'remote_public_ports_opened':False,'private_monitor_only':True,'training_started':False}
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    for _ in range(40):
        try:
            with urllib.request.urlopen('http://127.0.0.1:16006/data/environment',timeout=2) as response:
                result=json.load(response)
            value['tensorboard_health_http']=200;value['tensorboard_environment']=result
            receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2));return
        except Exception:time.sleep(.5)
    raise RuntimeError('Private TensorBoard not healthy; preserve launch logs and diagnose')


if __name__=='__main__':main()
