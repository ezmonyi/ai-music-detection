import time
from pathlib import Path

from fastapi.testclient import TestClient

from music_detector.api import create_app


def test_actual_upload_lifecycle_and_audio_cleanup():
    observed = {}
    def analyze(path, families, **kwargs):
        observed['path'] = Path(path)
        assert path.read_bytes() == b'test-payload'
        kwargs['progress']('measuring')
        return {'prediction': {'probability': None}, 'selected_artifacts': families}
    with TestClient(create_app(analyze)) as client:
        response = client.post('/api/analyses', files={'audio': ('../song.wav', b'test-payload')},
                               data={'artifacts': '["F"]'})
        assert response.status_code == 202
        uid = response.json()['id']
        for _ in range(50):
            result = client.get('/api/analyses/' + uid).json()
            if result['status'] in ('complete', 'failed'):
                break
            time.sleep(.01)
        assert result['status'] == 'complete'
        assert result['result']['prediction']['probability'] is None
        download = client.get('/api/analyses/' + uid + '/download')
        assert download.status_code == 200
        assert download.json()['analysis_id'] == uid
        assert download.headers['content-disposition'].startswith('attachment;')
    assert not observed['path'].exists()


def test_artifact_errors_and_same_origin(monkeypatch):
    import music_detector.api as api
    original = api.catalogue
    monkeypatch.setattr(api, 'catalogue', lambda: [dict(row, available=False) if row['id'] == 'S' else row
                                                 for row in original()])
    with TestClient(create_app()) as client:
        for values, status in [('[]', 400), ('["nope"]', 400), ('["F","F"]', 400), ('["S"]', 409), ('["BC"]', 400)]:
            response = client.post('/api/analyses', files={'audio': ('song.wav', b'x')}, data={'artifacts': values})
            assert response.status_code == status
        response = client.post('/api/analyses', headers={'origin': 'https://unrelated.example'},
                               files={'audio': ('song.wav', b'x')}, data={'artifacts': '["F"]'})
        assert response.status_code == 403
        assert client.get('/api/analyses/missing').status_code == 404
        assert client.get('/api/analyses/missing/download').status_code == 404


def test_invalid_audio_failure_is_explicit_and_temporary_file_removed():
    observed = {}
    def fail(path, *args, **kwargs):
        observed['path'] = path
        raise ValueError('Invalid audio; no probability computed.')
    with TestClient(create_app(fail)) as client:
        response = client.post('/api/analyses', files={'audio': ('broken.wav', b'broken')},
                               data={'artifacts': '["F"]'})
        uid = response.json()['id']
        for _ in range(50):
            result = client.get('/api/analyses/' + uid).json()
            if result['status'] == 'failed':
                break
            time.sleep(.01)
        assert result['status'] == 'failed'
        assert 'result' not in result
        assert client.get('/api/analyses/' + uid + '/download').status_code == 409
    assert not observed['path'].exists()
