#!/usr/bin/env python
"""Run the PowerPoint style importer over a file and print what it found.

    ./venv/bin/python scripts/atb_import_report.py ~/Downloads/Some-Pack.pptx

Nothing is stored: this is the reading, not the import. It prints the pack, the layouts it would
emit, the structures it would leave for step 2, every problem, and — when the UI checkout is
reachable — how the derived pack differs from the hand-authored `consulting-navy-16x9.json`.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.atb.pptx_import.import_pack import ImportRefused, import_pack  # noqa: E402

HAND_AUTHORED = os.path.join(
    os.environ.get('BEYOND_PROCWISE_UI', os.path.expanduser('~/PycharmProjects/beyond_procwise_ui')),
    'src/modules/SpendIQ/atb/styles/consulting-navy-16x9.json')


def _rule(title: str) -> None:
    print(f'\n{title}\n' + '-' * len(title))


def main(path: str) -> int:
    with open(path, 'rb') as handle:
        data = handle.read()
    try:
        result = import_pack(data, os.path.basename(path), os.environ.get('USER', 'report'))
    except ImportRefused as exc:
        print(f'REFUSED: {exc}')
        return 1

    pack = result.pack
    print(f'{os.path.basename(path)} — {len(data) // 1024} KB, '
          f'{pack["format"]["kind"]} '
          f'{pack["format"].get("width_in", 8.27)} x {pack["format"].get("height_in", 11.69)} in')

    _rule('Colours')
    for name, value in pack['colours'].items():
        record = result.evidence['values'].get(f'colours.{name}', {})
        counts = ' '.join(f'{k} {record[k]}' for k in ('fills', 'runs', 'lines') if record.get(k))
        print(f'  {name:14s} {value}   {counts}   {record.get("why", "")}')

    _rule('Type scale and fonts')
    for role, size in pack['type_scale_pt'].items():
        record = result.evidence['values'].get(f'type_scale_pt.{role}', {})
        print(f'  {role:12s} {size:>6}pt   runs {record.get("runs", "?")}   {record.get("why", "")}')
    for role, font in pack['fonts'].items():
        print(f'  {role:12s} {font["family"]}   fallback {font["fallback"]}')

    _rule('Grid')
    for key, value in pack['grid'].items():
        record = result.evidence['values'].get(f'grid.{key}', {})
        print(f'  {key:16s} {value:<8} {record.get("why", "")}')

    _rule('Writing')
    writing = pack['writing']
    print(f'  locale {writing["locale"]}'
          + ('  CONTESTED — the spelling suggests ' + writing['locale_suggested']
             if writing['locale_contested'] else ''))
    print(f'  {result.evidence["values"]["writing.locale"]}')

    _rule(f'Layouts it would emit ({len(result.layouts)}, covering '
          f'{sum(len(l["slide_refs"]) for l in result.layouts)} slides)')
    for layout in result.layouts:
        print(f'  {len(layout["slide_refs"]):3d} slides  {layout["proposed_name"][:58]:58s} '
              f'{layout["id"]}')
        print(f'            regions {len(layout["regions"])}  slots {len(layout["slots"])}  '
              f'example from slide {layout["example_source"]["slide"]}  '
              f'problems {len(layout["problems"])}')

    _rule(f'Single-use structures, left for step 2 ({len(result.single_use)})')
    for entry in result.single_use:
        print(f'  slide {entry["slides"][0]:3d}  {entry["structure"][:70]}')

    _rule(f'Problems ({len(result.problems)})')
    for problem in result.problems:
        print(f'  {problem["kind"]:12s} {problem.get("region", "-"):10s} slides '
              f'{problem["slides"]}  {problem["why"][:76]}')

    _rule('Ignored and incidental')
    for what, record in result.evidence['ignored'].items():
        print(f'  ignored {what}: {record["why"]}')
    incidental = result.evidence['incidental']
    print(f'  {len(incidental)} incidental findings; first five:')
    for entry in incidental[:5]:
        print(f'    {entry["kind"]:12s} {str(entry["value"])[:28]:28s} {entry["why"][:60]}')

    if os.path.exists(HAND_AUTHORED):
        _rule('Against the hand-authored pack')
        hand = json.load(open(HAND_AUTHORED, encoding='utf-8'))
        for field in ('colours', 'type_scale_pt', 'grid'):
            for key in sorted(set(hand.get(field, {})) | set(pack.get(field, {}))):
                before, after = hand.get(field, {}).get(key), pack.get(field, {}).get(key)
                if before != after:
                    print(f'  {field}.{key}: authored {before} -> measured {after}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
