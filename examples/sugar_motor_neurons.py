"""Small example: activate gustatory sugar neurons, watch motor neurons.

Stimulates the 21 labellar sugar GRNs of the fly-brain 'sugar' experiment
with 200 Hz Poisson input and reports the firing rates of the proboscis motor
neuron MN9 (feeding) and the descending neurons used by fly_driver
(P9/DNp09, oDN1, DNa01, DNa02, MDN, Giant Fiber).

Expected result (Shiu et al. 2024): MN9 is strongly recruited, the walking
related descending neurons stay silent.

    python examples/sugar_motor_neurons.py [--t-run 1.0] [--trials 4]
    python examples/sugar_motor_neurons.py --parquet ../fly-brain/data/results/pytorch_t1.0s_n1.parquet

--parquet evaluates a spike file written by `python main.py --pytorch ...`
in fly-brain instead of simulating.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fly_driver as fd  # noqa: E402

# Same IDs as EXPERIMENTS['sugar'] in fly-brain/code/benchmark.py
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

# Sides follow the FlyWire annotation table (Schlegel et al. 2024).  The
# Shiu et al. example notebook labels these cells with the opposite side.
MOTOR_NEURONS = {
    'MN9_a': 720575940660219265,
    'MN9_b': 720575940618238523,
    'P9(DNp09)_left': 720575940635872101,
    'P9(DNp09)_right': 720575940627652358,
    'oDN1(DNg97)_left': 720575940626730883,
    'oDN1(DNg97)_right': 720575940620300308,
    'DNa01_left': 720575940627787609,
    'DNa01_right': 720575940644438551,
    'DNa02_left': 720575940629327659,
    'DNa02_right': 720575940604737708,
    'MDN_left_1': 720575940631082808,
    'MDN_left_2': 720575940616026939,
    'MDN_right_1': 720575940610236514,
    'MDN_right_2': 720575940640331472,
    'GiantFiber_1': 720575940622838154,
    'GiantFiber_2': 720575940632499757,
}


def print_table(rates, std=None, header='rate [Hz]'):
    print(f'{"neuron":>20s}  {header}')
    for name, r in rates.items():
        s = f' +- {std[name]:5.1f}' if std is not None else ''
        print(f'{name:>20s}  {r:7.1f}{s}')


def from_parquet(path, t_run):
    import pandas as pd
    df = pd.read_parquet(path)
    n_trials = df['trial'].nunique() if len(df) else 1
    counts = df.groupby('flywire_id').size()
    rates = {name: counts.get(fid, 0) / n_trials / t_run for name, fid in MOTOR_NEURONS.items()}
    print(f'{path}: {len(df)} spikes, {df["flywire_id"].nunique()} active neurons, '
          f'{n_trials} trial(s)')
    print_table(rates)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=str(fd.DEFAULT_CONFIG_PATH))
    ap.add_argument('--t-run', type=float, default=1.0, help='seconds')
    ap.add_argument('--trials', type=int, default=4, help='batched trials')
    ap.add_argument('--rate', type=float, default=200.0, help='sugar GRN rate (Hz)')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--parquet', help='analyse a fly-brain spike parquet instead')
    args = ap.parse_args()

    if args.parquet:
        return from_parquet(args.parquet, args.t_run)

    cfg = fd.load_config(args.config)
    t0 = perf_counter()
    brain = fd.FlyBrain.from_fly_brain(cfg['brain']['fly_brain_dir'], device=args.device,
                                       batch=args.trials, input_neurons=SUGAR_GRNS, seed=0)
    print(f'loaded {brain.N} neurons on {brain.device} in {perf_counter() - t0:.1f}s')
    brain.set_rates(brain.index(SUGAR_GRNS), args.rate)
    rec = brain.index(list(MOTOR_NEURONS.values()))

    t0 = perf_counter()
    counts = brain.run(args.t_run * 1000.0, rec)
    dt = perf_counter() - t0
    print(f'simulated {args.trials} x {args.t_run:.2f}s in {dt:.1f}s '
          f'({args.trials * args.t_run / dt:.2f}x realtime)\n')
    rates = counts / args.t_run
    mean = rates.mean(0).tolist()
    std = rates.std(0).tolist() if args.trials > 1 else [0.0] * len(mean)
    names = list(MOTOR_NEURONS)
    print_table(dict(zip(names, mean)), dict(zip(names, std)),
                header=f'rate [Hz] (sugar GRNs @ {args.rate:.0f} Hz)')


if __name__ == '__main__':
    main()
