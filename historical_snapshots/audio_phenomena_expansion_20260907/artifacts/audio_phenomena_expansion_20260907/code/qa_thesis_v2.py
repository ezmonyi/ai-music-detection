#!/usr/bin/env python3
"""Render the thesis revision and verify text/links; visual review is separate."""
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
from pypdf import PdfReader


def main():
    root = Path(__file__).resolve().parent.parent
    pdf = root/'output/pdf/ai_music_detection_thesis_en_20260907_v2.pdf'
    original = root/'output/pdf/ai_music_detection_thesis_en_20260907.pdf'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == 'f8fb801ca1c0dad29ad25709f3f2be586e125f3496afffc6d2cd397d66df942d'
    reader = PdfReader(pdf)
    texts = [p.extract_text() for p in reader.pages]
    assert len(texts) == 88 and all(len(t.strip()) > 20 for t in texts)
    joined = '\n'.join(texts)
    for value in ('0.028120', '0.018631', '0.036721', '0.6601/0.6167', '5,569,808,264', '33,953,613,152'):
        assert value in joined, value
    assert '\ufffd' not in joined and '(??)' not in joined
    destinations = reader.named_destinations
    links, missing = 0, []
    for page in reader.pages:
        for ref in page.get('/Annots', []):
            annotation = ref.get_object()
            if annotation.get('/Subtype') != '/Link':
                continue
            links += 1
            target = annotation.get('/Dest')
            action = annotation.get('/A')
            if action and action.get('/S') == '/GoTo':
                target = action.get('/D')
            if isinstance(target, str) and target not in destinations:
                missing.append(target)
    assert not missing, missing
    output = root/'tmp/pdfs/thesis_v2_qa'
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pdftoppm', '-r', '45', '-png', str(pdf), str(output/'page')], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    images = sorted(output.glob('page-*.png'))
    for offset in range(0, len(images), 16):
        sheet = Image.new('RGB', (1520, 2240), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, path in enumerate(images[offset:offset+16]):
            image = Image.open(path).convert('RGB')
            image.thumbnail((366, 524))
            x, y = (j % 4)*380, (j // 4)*560
            sheet.paste(image, (x, y+24))
            draw.text((x+5, y+5), f'Page {offset+j+1}', fill='black')
        sheet.save(output/f'contact_{offset//16+1}.png')
    subprocess.run(['pdftoppm', '-f', '69', '-l', '72', '-r', '120', '-png', str(pdf), str(output/'detail')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    receipt = dict(status='automated_pass_visual_review_pending', pages=len(texts), links=links,
                   unresolved_internal_destinations=missing, original_pdf_unchanged=True,
                   pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                   expected_key_numeric_strings_present=True, rendered_all_pages=True,
                   visual_directory=str(output), notes=['Prior caption-number gap is preserved.'])
    (root/'audit/thesis_v2_automated_qa.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
