"""Verify local byte-for-byte mirrors without rewriting historical NFS paths."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def main():
    checked=[]
    for name,key,pin in [
        ('results/sc_native2174_v6_v1','products','9b827f25c33794efed0938682bbf9a67d579a03c2d9f97409078ba6949d31bd5'),
        ('results/equal60_exploratory_package_v6_v1','files','873a49081e07ce5d1098fe71d933a055410b010d04c6fd2b9e6db66a5dec781f')]:
        folder=ROOT/name; marker=folder/'COMMIT.json'
        if sha(marker)!=pin: raise ValueError('Remote publication identity differs')
        products=json.loads(marker.read_text())[key]
        actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=marker}
        if actual!=set(products): raise ValueError('Mirror inventory differs')
        for relative,record in products.items():
            p=folder/relative
            if p.is_symlink() or p.resolve()!=p or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
                raise ValueError('Mirror differs: '+str(p))
        checked.append(dict(path=str(folder),commit_sha256=pin,products=len(products),
            product_bytes=sum(r['bytes'] for r in products.values())))
    receipt=dict(passed=True,publications=checked,code_sha256=sha(Path(__file__)),
        scope='local mirror bytes and inventory; embedded NFS provenance paths intentionally unchanged',
        classifier_results_included=False)
    output=ROOT/'audit/v6_local_input_mirror_v1.json'
    with output.open('x') as f: json.dump(receipt,f,sort_keys=True,indent=2); f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__': main()
