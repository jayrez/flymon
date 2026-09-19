"""Deterministic, framebuffer-only capture. ROM remains external via POKEMON_ROM."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from flymon.dataset import DATASET, CAPTURES, ROOT, digest, save_instance, load_dataset, write_json
from flymon.emulator import PokemonEmulator


def capture(recipe):
    with PokemonEmulator() as game:
        state = recipe.get('savestate')
        if state:
            game.load_state(ROOT / state)
        for action in recipe['actions']:
            if 'button' in action:
                game.tap(action['button'], action.get('held', 3), action.get('release', 3))
            else:
                game.tick(action['wait'])
        frames = []
        for i in range(10):
            if i:
                game.tick(4)
            frames.append(game.framebuffer())
    procedure = dict(recipe, frame_offsets=list(range(0,40,4)), emulator='PyBoy 2.7.0',
                     label_source='capture procedure and manual framebuffer inspection; no RAM')
    if state:
        procedure['savestate_sha256'] = digest((ROOT/state).read_bytes())
    return save_instance(recipe['class_label'], recipe['instance_id'], frames, procedure,
                         role=recipe.get('role','primary'))


def recipes():
    # Deterministic capture actions only, never neural action decoding.
    paths = [[], [('left',8)], [('left',16)], [('down',8)], [('down',16)],
             [('right',8)], [('up',8)], [('left',16),('down',16)]]
    for label in ('bedroom','menu'):
        for i, path in enumerate(paths):
            actions=[{'wait':1}]+[dict(button=b,held=n,release=8) for b,n in path]
            if label=='menu':
                actions += [dict(button='start'),dict(wait=30)]
                actions += [dict(button='down',held=1,release=8) for _ in range(i%6)]
            actions += [dict(wait=12)]
            yield dict(class_label=label,instance_id=f'{label}-{i+1:02d}',savestate='states/bedroom.state',actions=actions)
    for label,ticks in [('intro',[700,760,850,900,950,1000,1080,1130]),
                        ('title',[1400,1750,2000,2300,2550,2850,3150,3450])]:
        for i,tick in enumerate(ticks):
            yield dict(class_label=label,instance_id=f'{label}-{i+1:02d}',actions=[dict(wait=tick)])
    for i, advances in enumerate([3,5,6,8,10,11,12,14]):
        actions=[dict(wait=1500)]
        for j in range(advances+1):
            actions += [dict(button='start' if j==0 else 'a'),dict(wait=90)]
        actions += [dict(wait=60)]
        yield dict(class_label='dialogue',instance_id=f'dialogue-{i+1:02d}',actions=actions)

    yield dict(class_label='menu',instance_id='menu-09',savestate='states/bedroom.state',
               actions=[dict(wait=1),dict(button='left',held=8,release=40),dict(button='start'),dict(wait=60)])
    yield dict(class_label='ood',instance_id='ood-options-01',role='ood',savestate='states/bedroom.state',
               actions=[dict(wait=1),dict(button='start'),dict(wait=60),
                        *[dict(button='down',held=1,release=15) for _ in range(4)],dict(button='a'),dict(wait=90)])


def sheet():
    records,seqs=load_dataset()
    manifest=ROOT/'results/experiment-06-generalization/dataset-manifest.json'
    if manifest.exists():
        accepted=set(json.loads(manifest.read_text())['primary_ids'])
        pairs=[(r,s) for r,s in zip(records,seqs) if r['instance_id'] in accepted]
        records,seqs=map(list,zip(*pairs))
    image=Image.new('RGB',(160*8,164*((len(records)+7)//8)),'white');draw=ImageDraw.Draw(image)
    for i,(r,seq) in enumerate(zip(records,seqs)):
        x=i%8*160;y=i//8*164
        image.paste(Image.fromarray(seq[0]).convert('RGB'),(x,y));draw.text((x,y+144),r['instance_id'],fill='black')
    CAPTURES.mkdir(parents=True,exist_ok=True);image.save(CAPTURES/'dataset-contact-sheet.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path);p.add_argument('--collect',action='store_true')
    args=p.parse_args()
    if args.recipe:
        capture(json.loads(args.recipe.read_text()))
    elif args.collect:
        for recipe in recipes():
            if (DATASET/recipe['class_label']/recipe['instance_id']/'metadata.json').exists():
                continue
            try:
                capture(recipe);print('Captured',recipe['instance_id'],flush=True)
            except ValueError as error:
                print('REJECTED',recipe['instance_id'],error,flush=True)
    else:
        p.error('Choose --recipe or --collect')
    sheet()


if __name__=='__main__':
    main()
