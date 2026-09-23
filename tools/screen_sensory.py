"""Screen sensory populations for their effect on the fly_driver motor neurons.

For every candidate population (one side at a time) the whole-brain model is
driven with Poisson input and the rates of the motor groups defined in the
config (P9, DNa01/DNa02 left/right, MDN, ...) are measured.  All candidates
run in parallel as one batch.  The result shows which sensory neurons are
suitable for the left/right distance sensors of the car, e.g. a population
whose left-side activation drives the right-turning DNa neurons.

Candidates are FlyWire annotation queries (Schlegel et al. 2024,
flyconnectome/flywire_annotations, Supplemental_file1_neuron_annotations.tsv).

    python tools/screen_sensory.py [--t-run 0.3] [--rate 200] [--tonic]
                                    [--out screen.csv]
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fly_driver as fd  # noqa: E402

# (label, annotation query without side)
CANDIDATES = [
    ('sugar GRN', {'cell_class': 'gustatory', 'cell_sub_class': 'sugar/water'}),
    ('bitter GRN', {'cell_class': 'gustatory', 'cell_sub_class': 'bitter'}),
    ('JO-CE (wind/gravity)', {'cell_class': 'mechanosensory', 'cell_sub_class': 'wind_gravity'}),
    ('JO-AB (auditory)', {'cell_class': 'mechanosensory', 'cell_sub_class': 'auditory'}),
    ('head bristle', {'cell_class': 'mechanosensory', 'cell_sub_class': 'head bristle'}),
    ('eye bristle', {'cell_class': 'mechanosensory', 'cell_sub_class': 'eye bristle'}),
    ('grooming mechano', {'cell_class': 'mechanosensory', 'cell_sub_class': 'grooming'}),
    ('taste peg mechano', {'cell_class': 'mechanosensory', 'cell_sub_class': 'taste peg'}),
    ('thermosensory', {'cell_class': 'thermosensory'}),
    ('hygrosensory', {'cell_class': 'hygrosensory'}),
    ('LC4 (looming)', {'cell_type': 'LC4'}),
    ('LPLC1', {'cell_type': 'LPLC1'}),
    ('LPLC2 (looming)', {'cell_type': 'LPLC2'}),
    ('LC16 (backing up)', {'cell_type': 'LC16'}),
    ('LC6', {'cell_type': 'LC6'}),
    ('LC9', {'cell_type': 'LC9'}),
    ('LC22', {'cell_type': 'LC22'}),
    ('LC12', {'cell_type': 'LC12'}),
    ('LC15', {'cell_type': 'LC15'}),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=str(fd.DEFAULT_CONFIG_PATH))
    ap.add_argument('--t-run', type=float, default=0.3, help='seconds per condition')
    ap.add_argument('--rate', type=float, default=200.0, help='stimulation rate (Hz)')
    ap.add_argument('--tonic', action='store_true', help='also apply the config tonic_inputs')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out', help='write results as CSV')
    args = ap.parse_args()

    cfg = fd.load_config(args.config)
    ann = cfg['brain'].get('annotations')
    if not ann or not Path(ann).exists():
        sys.exit('brain.annotations must point to the FlyWire annotation TSV')

    conds, all_inputs = [], []
    for label, query in CANDIDATES:
        for side in ('left', 'right'):
            try:
                ids = fd.resolve_neurons({'select': {**query, 'side': side}}, ann)
            except ValueError:
                continue
            conds.append((label, side, ids))
            all_inputs += ids
    tonic = cfg.get('tonic_inputs') or [] if args.tonic else []
    tonic_ids = fd.resolve_neurons([t['neurons'] for t in tonic], ann)
    conds.insert(0, ('(baseline)', '-', []))

    brain = fd.FlyBrain.from_fly_brain(cfg['brain']['fly_brain_dir'], device=args.device,
                                       batch=len(conds), input_neurons=all_inputs + tonic_ids,
                                       seed=0)
    for b, (_, _, ids) in enumerate(conds):
        if ids:
            brain.set_rates(brain.index(ids), args.rate, batch=b)
    for t in tonic:
        brain.set_rates(brain.index(fd.resolve_neurons(t['neurons'], ann)), t['rate_hz'])

    groups = cfg['motor']['groups']
    group_ids = {g: fd.resolve_neurons(v['neurons'], ann) for g, v in groups.items()}
    rec = torch.cat([brain.index(ids) for ids in group_ids.values()])

    print(f'{len(conds)} conditions x {args.t_run}s on {brain.device} ...', flush=True)
    t0 = perf_counter()
    counts = brain.run(args.t_run * 1000.0, rec).cpu() / args.t_run
    print(f'done in {perf_counter() - t0:.0f}s\n')

    rows, k = [], 0
    for b, (label, side, ids) in enumerate(conds):
        row = {'population': label, 'side': side, 'n': len(ids)}
        k = 0
        for g, ids_g in group_ids.items():
            row[g] = round(counts[b, k:k + len(ids_g)].mean().item(), 1)
            k += len(ids_g)
        rows.append(row)
    df = pd.DataFrame(rows)
    if {'steer_left', 'steer_right'} <= set(df.columns):
        # > 0: population drives right-turning DNs more than left-turning ones
        df['turn_right_bias'] = df['steer_right'] - df['steer_left']
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
