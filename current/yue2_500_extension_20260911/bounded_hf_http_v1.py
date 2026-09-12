"""Per-process HF HTTP timeouts without changing TLS or authentication hooks."""
import httpx
from huggingface_hub.utils._http import default_client_factory,set_client_factory


def factory():
    client=default_client_factory()
    client.timeout=httpx.Timeout(connect=20,read=90,write=90,pool=60)
    return client


def install():
    set_client_factory(factory)
