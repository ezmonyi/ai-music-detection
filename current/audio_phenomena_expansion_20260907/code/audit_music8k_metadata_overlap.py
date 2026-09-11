#!/usr/bin/env python3
"""Metadata-only acquisition overlap screen; never admits data to a classifier."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import unicodedata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).casefold().split())


def meaningful(value):
    return norm(value) not in {'', 'unknown', 'none', 'nan', 'n/a', 'null'}


def digest(value):
    return hashlib.sha256(norm(value).encode()).hexdigest()


def containment(a, b):
    """Conservative literal screen, not translation or perceptual deduplication."""
    a, b = norm(a), norm(b)
    return min(len(a), len(b)) >= 32 and (a in b or b in a)


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle)) if Path(path).suffix == '.csv' else [json.loads(line) for line in handle]


def screen(new, old, containment_enabled=False):
    hits = []
    old = [(i,f,norm(v)) for i,f,v in old if meaningful(v)]
    for item_id, field, value in new:
        if not meaningful(value):
            continue
        value = norm(value)
        for old_id, old_field, old_value in old:
            exact = value == old_value
            contained = containment_enabled and min(len(value),len(old_value)) >= 32 and (value in old_value or old_value in value)
            if exact or contained:
                hits.append(dict(new_id=item_id, new_field=field, existing_id=old_id,
                    existing_field=old_field, match='normalized_exact' if exact else 'literal_containment_min32',
                    new_text_sha256=digest(value), existing_text_sha256=digest(old_value)))
    return hits


def main():
    root = Path(__file__).resolve().parent.parent
    artifacts = root.parent
    paths = {
        'contract': root/'external_validation/music8k_mureka500_v1/contract.json',
        'reference': root/'external_validation/music8k_metadata_v1/metadata/pop_1000_unique_artist.csv',
        'generation': root/'external_validation/music8k_metadata_v1/metadata/songs_musiccaps.jsonl',
        'registry': artifacts/'source_diversity_expansion_20260905/manifests/final/metadata_10s.csv',
        'diversity': artifacts/'source_diversity_expansion_20260905/manifests/v2/frozen_item_manifest.csv',
        'prompts': artifacts/'open_models_spectral_500_20260901/prompts/prompt_manifest.jsonl',
        'prompt_sources': artifacts/'open_models_spectral_500_20260901/prompts/selected_source_rows.jsonl',
        'external500': artifacts/'external_generator_500_testset_20260904/testset/manifest.jsonl',
        'external100': artifacts/'external_filter_test_20260902/testset/manifest.jsonl',
    }
    initial = {k: sha(p) for k, p in paths.items()}
    contract = json.loads(paths['contract'].read_text())
    selected = contract['selected']
    assert len(selected) == 500 and len({r['id'] for r in selected}) == 500
    assert not contract['classifier_authorized'] and all(r['role'] == 'reserved_unscored' for r in selected)
    assert not set(r['id'] for r in selected) & set(contract['excluded_probe_ids'])
    reference = {r['id']:r for r in rows(paths['reference'])}
    generation = {str(r['id']):r for r in rows(paths['generation'])}
    for field, key in [('reference','pop_1000_unique_artist.csv'), ('generation','songs_musiccaps.jsonl')]:
        assert initial[field] == contract['metadata_sha256'][key]
    registry, diversity, prompts, sources, external500, external100 = [rows(paths[k]) for k in
        ('registry','diversity','prompts','prompt_sources','external500','external100')]
    known_ids = {r['id'] for r in registry}
    assert len(known_ids) == len(registry) == 10141
    new_lyrics, new_captions, new_artists, new_titles = [], [], [], []
    for r in selected:
        item_id = r['id']; a, b = reference[item_id], generation[item_id]
        for field, value in [('reference_lyrics',a['lyrics']),('generation_lyrics',b['lyrics']),
                             ('reference_artist',a['artist']),('caption',b['caption'])]:
            assert r[field+'_hash'] == digest(value)
        new_lyrics.extend([(item_id,'reference_lyrics',a['lyrics']),(item_id,'generation_lyrics',b['lyrics'])])
        new_captions.append((item_id,'caption',b['caption']))
        new_artists.append((item_id,'reference_artist',a['artist']))
        new_titles.append((item_id,'reference_title',a['title']))
    old_captions = [(r['id'],field,r[field]) for r in prompts for field in ('prompt_common','acestep_caption','heartmula_tags')]
    old_captions += [(r['id'],'description',r['description']) for r in external500+external100 if r['id'] in known_ids]
    old_lyrics = [(r['id'],'lyrics_30s',r['lyrics_30s']) for r in prompts]
    old_lyrics += [(r['id'],'joined_source_sections',' '.join(s['text'] for s in r['record']['sections'])) for r in sources]
    old_lyrics += [(r['id'],f'source_section_{i}',s['text']) for r in sources for i,s in enumerate(r['record']['sections'])]
    old_artists = [(r['id'],'artist_or_creator',r['artist_or_creator']) for r in registry]
    old_titles = [(r['item_id'],'title',r['title']) for r in diversity if r['item_id'] in known_ids]
    raw_index = defaultdict(list)
    for r in registry:
        if r['raw_sha256']:
            raw_index[r['raw_sha256']].append(r['id'])
    raw_hits = [dict(new_id=r['id'], existing_ids=raw_index[r['sha256']], sha256=r['sha256'])
                for r in selected if r['sha256'] in raw_index]
    hits = dict(registered_native_byte_hash=raw_hits,
        caption_exact=screen(new_captions,old_captions),
        lyric_exact_or_literal_containment=screen(new_lyrics,old_lyrics,True),
        reference_artist_exact=screen(new_artists,old_artists),
        reference_title_exact=screen(new_titles,old_titles))
    within = {}
    for field in ('sha256','reference_lyrics_hash','generation_lyrics_hash','caption_hash','reference_artist_hash'):
        groups = defaultdict(list)
        for r in selected:
            groups[r[field]].append(r['id'])
        within[field] = [dict(hash=k,ids=v) for k,v in groups.items() if len(v)>1]
    assert initial == {k:sha(p) for k,p in paths.items()}
    result = dict(status='metadata_screen_completed_not_admission', utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(__file__), input_sha256=initial, selected=500, existing_registry_rows=len(registry),
        existing_raw_hash_nonempty=sum(bool(r['raw_sha256']) for r in registry),
        existing_prompt_records=len(prompts), existing_source_lyric_records=len(sources),
        existing_caption_field_instances=len(old_captions), existing_lyric_field_instances=len(old_lyrics),
        existing_known_artist_rows=sum(meaningful(r['artist_or_creator']) for r in registry),
        existing_title_rows=len(old_titles),
        hit_counts={k:len(v) for k,v in hits.items()}, hits=hits, within_selected_duplicate_groups=within,
        classifiers_fitted=0, classifier_admission_authorized=False, physical_audio_rehashed=False,
        limits=[
          'LFS advertised byte hashes were compared with frozen registered native hashes; new raw files are not read by this screen.',
          'No perceptual or decoded-audio fingerprint: re-encoding, alternate performances and excerpts may evade this screen.',
          'Text metadata coverage is incomplete; no match does not prove independent songs, prompts or lyrics.',
          'Lyrics use normalized exact and whole-field literal containment of at least 32 characters; no fuzzy, translation or semantic matching.',
          'Artist/title hits are review flags, not automatic evidence of identical recordings or synthetic performer identity.',
          'Existing historical locked rows were already consumed; this screen does not restore pristine test status.',
          'Reserved acquisition roles remain unchanged; any later split requires an explicit new protocol and resolved grouping.'
        ])
    target=root/'audit/music8k_mureka500_metadata_overlap_v1.json'
    with target.open('x') as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False); handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('hits','input_sha256','limits','within_selected_duplicate_groups')}))
    print('within_duplicate_group_counts', {k:len(v) for k,v in within.items()})


if __name__ == '__main__':
    main()
