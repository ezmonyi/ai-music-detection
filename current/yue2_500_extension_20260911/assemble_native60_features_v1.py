"""Assemble committed YuE2 exact60 descriptors; no prediction or fitting."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0, str(RC/'code'))
import prepare_evaluation_inputs_v4 as reference


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    bound = {}
    def check(path, evidence=None):
        path = path.resolve(strict=True)
        value = dict(bytes=path.stat().st_size, sha256=sha(path))
        if evidence is not None:
            assert value['sha256'] == evidence['sha256'] and value['bytes'] == evidence['bytes'], str(path)
        if str(path) in bound:
            assert bound[str(path)] == value
        bound[str(path)] = value
        return value
    source = ROOT/'native60_inputs_v1'
    assert check(source/'COMMIT.json')['sha256'] == 'cde960d1f4ddb29d9751ac8d3bbecf28b6806a5fc9dc05cf299618ec2d8b02ea'
    source_commit = read(source/'COMMIT.json')
    check(source/'metadata.json', source_commit['products']['metadata.json'])
    rows = read(source/'metadata.json')
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)) == 276
    assert sum(r['role']=='locked_test' for r in rows) == 53
    for stage in ('allinone', 'beats'):
        path = ROOT/'native60_neural_v1'/(stage+'_COMMIT.json')
        check(path)
        commit = read(path)
        assert commit['status'] == 'completed_verified_neural_products' and commit['rows'] == 276
        check(Path(commit['contract']['path']), commit['contract'])
        for name, evidence in commit['products'].items():
            check(Path(name), evidence)
        print('Verified all committed '+stage+' products', flush=True)
    sroot, froot = ROOT/'native60_sdrp_v1', ROOT/'native60_fhm_v1'
    check(sroot/'COMMIT.json')
    scommit = read(sroot/'COMMIT.json')
    assert scommit['status']=='completed_extraction_not_independent_audit' and scommit['rows']==276
    assert set(scommit['items']) == set(ids)
    check(sroot/'contract.json', scommit['contract'])
    contract = read(sroot/'contract.json')
    assert contract['duration']==60 and contract['bias']['sha256']==reference.BIAS_SHA
    assert contract['extractor']['sha256']==reference.OLD_CODE['extract_expanded_four_family.py']
    check(Path(contract['extractor']['path']), contract['extractor'])
    check(Path(contract['bias']['path']), contract['bias'])
    check(froot/'COMMIT.json')
    fcommit = read(froot/'COMMIT.json')
    assert fcommit['status']=='completed_native60_fhm_measurement' and fcommit['rows']==276
    for name, evidence in fcommit['products'].items():
        check(froot/name, evidence)
    with (froot/'features/features.csv').open() as stream:
        frows = list(csv.DictReader(stream))
    assert [r['id'] for r in frows] == ids
    columns = sum(reference.COLUMNS.values(), [])
    results, metadata = [], []
    for row, fr in zip(rows, frows):
        uid = row['id']
        path = sroot/'items'/(uid+'.json')
        check(path, scommit['items'][uid])
        item = read(path)
        assert item['contract_sha256']==scommit['contract']['sha256']
        assert item['row']['group_id']==row['group_id'] and item['row']['split']==row['role']
        sf = item['features']
        assert sf['item_id']==uid and sf['group_id']==row['group_id'] and sf['status']=='complete'
        assert sf['duration_sec']==60 and sf['bias_sha256']==reference.BIAS_SHA
        assert fr['group_id']==row['group_id'] and fr['role']==row['role']
        combined = {'id':uid}
        for family, names in reference.COLUMNS.items():
            for name in names:
                value = sf[name] if family in reference.OLD_COLUMNS else fr[name]
                value = None if value is None or value=='' else float(value)
                assert value is None or math.isfinite(value), (uid, name)
                combined[name] = value
        results.append(combined)
        metadata.append(dict(id=uid,label=1,source_group='YuE2',group_id=row['group_id'],role=row['role']))
    output = ROOT/'native60_feature_package_v1'
    output.mkdir(exist_ok=False)
    payloads = {'features.json':results, 'metadata.json':metadata,
                'input_bindings.json':bound,
                'summary.json':dict(rows=276,columns=len(columns),classifier_fits=0,
                    numerical_independent_audit=False,classification_admitted=False,
                    missing_by_column={name:sum(r[name] is None for r in results) for name in columns})}
    products = {}
    for name, value in payloads.items():
        path = output/name
        with path.open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
        products[name] = dict(bytes=path.stat().st_size,sha256=sha(path))
    with (output/'COMMIT.json').open('x') as stream:
        json.dump(dict(status='assembled_from_hash_verified_measurements_not_numerically_audited',
                       products=products,rows=276,classifier_fits=0),stream,indent=2)
    print('Native60 feature package committed; independent numerical acceptance still required',flush=True)


if __name__=='__main__':
    main()
