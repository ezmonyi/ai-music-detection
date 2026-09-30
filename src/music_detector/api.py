"""Loopback-only upload service; uploaded audio is temporary and never sent out."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import json
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .analysis import DEPLOYMENT_BUNDLE, analyze
from .artifacts import catalogue
from .audio import MAX_UPLOAD_BYTES

STATIC = Path(__file__).parent / 'static'
MAX_JOBS = 8
RESULT_TTL = 3600


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD_BYTES + 65536
        size = 0

        class TooLarge(Exception):
            pass

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get('body', b''))
            if size > limit:
                raise TooLarge()
            return message

        try:
            length = dict(scope.get('headers', [])).get(b'content-length')
            if length:
                try:
                    declared = int(length)
                    if declared < 0:
                        raise ValueError()
                except ValueError:
                    return await JSONResponse({'detail': 'Invalid Content-Length.'}, status_code=400)(scope, receive, send)
                if declared > limit:
                    raise TooLarge()
            await self.app(scope, bounded_receive, send)
        except TooLarge:
            await JSONResponse({'detail': 'Request exceeds 100 MiB audio upload limit.'}, status_code=413)(scope, receive, send)


def create_app(analyzer=analyze) -> FastAPI:
    jobs = {}
    lock = threading.Lock()
    slots = threading.BoundedSemaphore(MAX_JOBS)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='audio-analysis')

    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title='Music Artifact Evidence', version='0.1.0', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver', '[::1]'])
    app.add_middleware(BodyLimitMiddleware)

    @app.middleware('http')
    async def same_origin(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin:
            if urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'Cross-origin uploads are not permitted.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'model_installed': DEPLOYMENT_BUNDLE.is_file()}

    @app.get('/api/artifacts')
    def artifacts():
        return {'artifacts': catalogue(), 'model_status': {
            'installed': DEPLOYMENT_BUNDLE.is_file(), 'upload_max_bytes': MAX_UPLOAD_BYTES,
            'minimum_duration_seconds': 30, 'maximum_duration_seconds': 3600,
            'native_channels_required': 2, 'audio_retention': 'deleted immediately after each job',
            'probability_scope': 'separately calibrated retrospective reference-corpus model; not universal authorship'}}

    def update(uid, **values):
        with lock:
            jobs[uid].update(values)

    def work(uid, directory, path, name, selected, research):
        try:
            update(uid, status='running', stage='校验音频')
            result = analyzer(path, selected, display_name=name, research=research,
                              progress=lambda stage: update(uid, stage=stage))
            update(uid, status='complete', stage='分析完成', result=result, finished_at=time.time())
        except (ValueError, RuntimeError) as error:
            update(uid, status='failed', stage='分析未完成', error=str(error), finished_at=time.time())
        except Exception:
            # Do not publish stack traces, temporary paths or credentials to UI.
            update(uid, status='failed', stage='分析未完成', error='Unexpected analysis error; no probability was produced.',
                   finished_at=time.time())
        finally:
            directory.cleanup()
            slots.release()

    @app.post('/api/analyses', status_code=202)
    async def submit(audio: UploadFile = File(...), artifacts: str = Form(...), research: bool = Form(False)):
        try:
            selected = json.loads(artifacts)
        except json.JSONDecodeError as error:
            raise HTTPException(400, 'artifacts must be a JSON array.') from error
        registry = {r['id']: r for r in catalogue()}
        if (not isinstance(selected, list) or not selected or
            not all(isinstance(x, str) for x in selected) or len(selected) != len(set(selected)) or
            any(x not in registry for x in selected)):
            raise HTTPException(400, 'Choose a nonempty set of known artifact IDs without duplicates.')
        if any(not registry[x]['available'] for x in selected):
            raise HTTPException(409, 'A selected artifact backend is unavailable; no substitute will be used.')
        if any(registry[x]['research_only'] for x in selected) and not research:
            raise HTTPException(400, 'Research-only artifacts require explicit opt-in.')
        if not slots.acquire(blocking=False):
            raise HTTPException(429, 'Analysis queue is full; retry after an active job finishes.')
        directory = tempfile.TemporaryDirectory(prefix='music-artifact-')
        path = Path(directory.name) / 'upload.audio'
        try:
            size = 0
            with path.open('xb') as stream:
                while block := await audio.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, 'Audio exceeds 100 MiB.')
                    stream.write(block)
            if not size:
                raise HTTPException(400, 'Audio file is empty.')
            name = (audio.filename or 'audio').replace('\\', '/').rsplit('/', 1)[-1][:160]
            uid = uuid.uuid4().hex
            with lock:
                for old in list(jobs):
                    if time.time() - jobs[old].get('finished_at', time.time()) > RESULT_TTL:
                        del jobs[old]
                jobs[uid] = {'id': uid, 'status': 'queued', 'stage': '等待分析', 'created_at': time.time()}
            pool.submit(work, uid, directory, path, name, selected, research)
            return {'id': uid, 'status': 'queued'}
        except Exception:
            directory.cleanup()
            slots.release()
            raise
        finally:
            await audio.close()

    @app.get('/api/analyses/{uid}')
    def get_result(uid: str):
        with lock:
            value = jobs.get(uid)
            if value is not None and time.time() - value.get('finished_at', time.time()) > RESULT_TTL:
                del jobs[uid]
                value = None
            if value is None:
                raise HTTPException(404, 'Unknown or expired analysis.')
            return dict(value)

    @app.get('/api/analyses/{uid}/download')
    def download_result(uid: str):
        job = get_result(uid)
        if job['status'] != 'complete':
            raise HTTPException(409, 'Analysis is not complete; no result is available to download.')
        return JSONResponse({'analysis_id': uid, **job['result']}, headers={
            'Content-Disposition': f'attachment; filename="music-evidence-{uid}.json"'})

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    if STATIC.is_dir():
        app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app


app = create_app()
