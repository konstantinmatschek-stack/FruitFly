"""
fly_driver -- drive a 2D car with the FlyWire whole-brain LIF model.

Closed loop, repeated every ``brain.step_ms`` (20-50 ms of brain time):

    car distance sensors (left / right)
        -> Poisson rates of selected sensory neurons           (SensorEncoder)
        -> whole-brain LIF simulation for step_ms              (FlyBrain, PyTorch)
        -> firing rates of P9 / DNa01+DNa02 L/R / MDN          (MotorDecoder)
        -> throttle, steering, brake                           (Car)

The brain model is the Shiu et al. (Nature 2024) leaky integrate-and-fire
network on the FlyWire v783 connectome, with the same equations and
parameters as ``code/run_pytorch.py`` in eonsystemspbc/fly-brain.  Only the
recurrent propagation differs: instead of a full sparse mat-mul per 0.1 ms
step, only the outgoing synapses of neurons that actually spiked are
scattered (identical result, much faster because few neurons spike per step).

All neuron assignments (sensor -> sensory neurons, motor neurons -> commands)
live in a config file (YAML or JSON, default ``fly_driver_config.yaml``) and
can be swapped without touching the code.

Usage:
    python fly_driver.py                        # pygame window
    python fly_driver.py --config my.yaml
    python fly_driver.py --headless --ticks 300 --log run.csv
    python fly_driver.py --brain reflex         # no connectome, debug controller
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name('fly_driver_config.yaml')

# Same values as MODEL_PARAMS in fly-brain/code/run_pytorch.py
# (= default_params of the Shiu et al. Brian2 model).
MODEL_PARAMS = {
    'tauSyn': 5.0,        # ms, synaptic time constant
    'tDelay': 1.8,        # ms, synaptic delay
    'v0': -52.0,          # mV, initial potential
    'vReset': -52.0,      # mV
    'vRest': -52.0,       # mV
    'vThreshold': -45.0,  # mV
    'tauMem': 20.0,       # ms, membrane time constant
    'tRefrac': 2.2,       # ms
    'scalePoisson': 250,  # Poisson input weight = scalePoisson * wScale (mV)
    'wScale': 0.275,      # mV per synapse
}


# ============================================================================
# Config
# ============================================================================

def load_config(path=None):
    """Load a YAML (.yaml/.yml) or JSON config file into a dict.

    Relative paths inside the ``brain`` section are resolved against the
    config file's directory; ``FLY_BRAIN_DIR`` overrides ``brain.fly_brain_dir``.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg = cfg or {}
    brain = cfg.setdefault('brain', {})
    base = path.resolve().parent
    fly_dir = os.environ.get('FLY_BRAIN_DIR') or brain.get('fly_brain_dir', '../fly-brain')
    brain['fly_brain_dir'] = str((base / fly_dir).resolve())
    ann = brain.get('annotations')
    if ann:
        brain['annotations'] = str((base / ann).resolve())
    return cfg


def _load_annotations(path):
    import pandas as pd
    cols = ['root_id', 'side', 'super_class', 'cell_class', 'cell_sub_class',
            'cell_type', 'hemibrain_type', 'top_nt']
    return pd.read_csv(path, sep='\t', usecols=cols, low_memory=False)


def resolve_neurons(spec, annotations_path=None, _cache={}):
    """Turn a neuron spec from the config into a list of FlyWire root IDs.

    Accepted forms:
        [720575940627652358, ...]                 plain list of IDs
        {P9_left: 720575940627652358, ...}        name -> ID (or name -> [IDs])
        {select: {cell_type: LC16, side: left}}   query on the FlyWire
                                                  annotation table (brain.annotations)
    """
    if spec is None:
        return []
    if isinstance(spec, (int, np.integer)):
        return [int(spec)]
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(resolve_neurons(s, annotations_path))
        return out
    if isinstance(spec, dict):
        if 'select' in spec:
            if not annotations_path:
                raise ValueError("'select' neuron spec needs brain.annotations in the config")
            if annotations_path not in _cache:
                _cache[annotations_path] = _load_annotations(annotations_path)
            df = _cache[annotations_path]
            mask = np.ones(len(df), dtype=bool)
            for col, val in spec['select'].items():
                vals = val if isinstance(val, list) else [val]
                mask &= df[col].isin(vals).to_numpy()
            ids = df.loc[mask, 'root_id'].astype('int64').tolist()
            if not ids:
                raise ValueError(f'annotation query matched no neurons: {spec["select"]}')
            return ids
        return resolve_neurons(list(spec.values()), annotations_path)
    raise TypeError(f'cannot interpret neuron spec {spec!r}')


# ============================================================================
# Brain
# ============================================================================

def pick_device(pref='auto'):
    import torch
    if pref == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if pref == 'cuda' and not torch.cuda.is_available():
        print('[fly_driver] CUDA requested but not available, using CPU', file=sys.stderr)
        return 'cpu'
    return pref


class FlyBrain:
    """Whole-brain LIF network (PyTorch) that can be advanced in short chunks.

    State is kept between calls to :meth:`run`, so the network evolves
    continuously while input rates change from one control step to the next.
    ``batch`` > 1 simulates independent copies in parallel (used for screening).
    """

    def __init__(self, n_neurons, pre, post, weight, flyids=None, params=None,
                 dt_ms=0.1, device='auto', batch=1, input_neurons=(), seed=None):
        import torch
        self.torch = torch
        self.device = pick_device(device)
        self.p = {**MODEL_PARAMS, **(params or {})}
        self.dt = float(dt_ms)
        self.N = int(n_neurons)
        self.B = int(batch)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.seed()

        # CSR by presynaptic neuron: outgoing synapses of neuron j are
        # post[indptr[j]:indptr[j+1]] with weights w[...]
        pre = np.asarray(pre, dtype=np.int64)
        order = np.argsort(pre, kind='stable')
        counts = np.bincount(pre, minlength=self.N)
        indptr = np.zeros(self.N + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        self.indptr = torch.as_tensor(indptr, device=self.device)
        self.post = torch.as_tensor(np.asarray(post, dtype=np.int64)[order], device=self.device)
        self.w = torch.as_tensor(np.asarray(weight, dtype=np.float32)[order], device=self.device)

        if flyids is not None:
            self.flyids = np.asarray(flyids, dtype=np.int64)
            self.flyid2i = {int(f): i for i, f in enumerate(self.flyids)}
        else:
            self.flyids, self.flyid2i = None, None

        # Integer step counts exactly as in run_pytorch.py
        self.steps_delay = int(self.p['tDelay'] / self.dt)
        self.ring_len = self.steps_delay + 1      # effective delay, see _step
        self.refrac_len = torch.full((self.N,), int(round(self.p['tRefrac'] / self.dt)),
                                     dtype=torch.float32, device=self.device)
        # Neurons that receive Poisson drive get no refractory period (as in
        # the Brian2 / PyTorch reference implementations).
        in_idx = self.index(input_neurons) if len(input_neurons) else []
        if len(in_idx):
            self.refrac_len[in_idx] = 0
        self.rates = torch.zeros(self.B, self.N, device=self.device)
        self.reset()

    # ------------------------------------------------------------------
    @classmethod
    def from_fly_brain(cls, fly_brain_dir, completeness='data/2025_Completeness_783.csv',
                       connectivity='data/2025_Connectivity_783.parquet', **kw):
        """Build the brain from the data files of an eonsystemspbc/fly-brain clone."""
        import pandas as pd
        root = Path(fly_brain_dir)
        comp = pd.read_csv(root / completeness, index_col=0)
        con = pd.read_parquet(root / connectivity, columns=[
            'Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'])
        return cls(len(comp), con['Presynaptic_Index'].to_numpy(),
                   con['Postsynaptic_Index'].to_numpy(),
                   con['Excitatory x Connectivity'].to_numpy(),
                   flyids=comp.index.to_numpy(), **kw)

    def index(self, neurons):
        """FlyWire IDs (or raw indices if the brain has no IDs) -> LongTensor."""
        if self.flyid2i is None:
            idx = [int(n) for n in neurons]
        else:
            missing = [n for n in neurons if int(n) not in self.flyid2i]
            if missing and len(missing) == len(neurons):
                raise KeyError(f'none of the neuron IDs are in the connectome: {missing[:3]}...')
            if missing:
                # e.g. annotation rows whose root ID changed after v783
                print(f'[fly_driver] warning: ignoring {len(missing)} of {len(neurons)} IDs '
                      f'not in the connectome, e.g. {missing[:3]}', file=sys.stderr)
            idx = [self.flyid2i[int(n)] for n in neurons if int(n) in self.flyid2i]
        return self.torch.as_tensor(idx, dtype=self.torch.long, device=self.device)

    def reset(self):
        t, B, N = self.torch, self.B, self.N
        self.v = t.full((B, N), self.p['v0'], device=self.device)
        self.g = t.zeros(B, N, device=self.device)
        self.ring = t.zeros(self.ring_len, B, N, device=self.device)
        self.ring_pos = 0
        self.spikes = t.zeros(B, N, device=self.device)
        self.refrac = self.refrac_len.expand(B, N).clone()
        self.t_ms = 0.0

    def set_rates(self, idx, rate_hz, batch=None):
        """Set Poisson input rate (Hz) for neuron indices ``idx`` (tensor)."""
        if batch is None:
            self.rates[:, idx] = float(rate_hz)
        else:
            self.rates[batch, idx] = float(rate_hz)

    def clear_rates(self):
        self.rates.zero_()

    # ------------------------------------------------------------------
    def _propagate(self, spikes):
        """Recurrent synaptic input: wScale * W @ spikes, event driven."""
        t = self.torch
        out = t.zeros(self.B * self.N, device=self.device)
        b, j = spikes.nonzero(as_tuple=True)
        if j.numel() == 0:
            return out.view(self.B, self.N)
        start = self.indptr[j]
        cnt = self.indptr[j + 1] - start
        seg = t.repeat_interleave(t.arange(j.numel(), device=self.device), cnt)
        seg_start = t.cumsum(cnt, 0) - cnt
        flat = start[seg] + (t.arange(seg.numel(), device=self.device) - seg_start[seg])
        out.index_add_(0, b[seg] * self.N + self.post[flat], self.w[flat])
        return (out * self.p['wScale']).view(self.B, self.N)

    def _step(self):
        t, p = self.torch, self.p
        # Poisson drive into the membrane potential
        poisson = t.bernoulli(self.rates * (self.dt / 1000.0), generator=self.gen)
        v_stim = p['wScale'] * p['scalePoisson'] * poisson
        recurrent = self._propagate(self.spikes)

        # refractory counter and alpha synapse (inputs dropped while refractory)
        self.refrac = t.where(self.spikes > 0, t.zeros_like(self.refrac), self.refrac + 1)
        delayed = self.ring[self.ring_pos]
        g_new = self.g * (1 - self.dt / p['tauSyn']) + delayed * (self.refrac >= self.refrac_len)
        # Ring buffer: the slot just read is refilled and read again ring_len
        # steps later -> same (steps_delay + 1)-step delay as torch.roll version.
        self.ring[self.ring_pos] = recurrent
        self.ring_pos = (self.ring_pos + 1) % self.ring_len

        # LIF membrane (uses conductance of the previous step, like run_pytorch)
        v = self.v + v_stim
        v = v + (self.dt / p['tauMem']) * (self.g - (v - p['vRest']))
        spk = (v > p['vThreshold']).float()
        self.v = t.where(spk > 0, t.full_like(v, p['vReset']), v)
        self.g = g_new * (1 - spk)
        self.spikes = spk
        self.t_ms += self.dt
        return spk

    def run(self, duration_ms, record_idx=None):
        """Advance the network by ``duration_ms``.

        Returns spike counts per recorded neuron, shape (batch, len(record_idx)),
        or for all neurons if ``record_idx`` is None.
        """
        n_steps = int(round(duration_ms / self.dt))
        t = self.torch
        with t.no_grad():
            counts = t.zeros(self.B, self.N if record_idx is None else len(record_idx),
                             device=self.device)
            for _ in range(n_steps):
                spk = self._step()
                counts += spk if record_idx is None else spk[:, record_idx]
        return counts


# ============================================================================
# Sensor encoding / motor decoding
# ============================================================================

class SensorEncoder:
    """Maps normalised proximity (0 = free, 1 = contact) to Poisson rates.

    p    = clip((proximity - threshold) / (1 - threshold), 0, 1)
    rate = min_rate + (max_rate - min_rate) * p ** gamma
    ``threshold`` keeps far-away objects (e.g. the road edges while driving
    in the middle) silent.  Each side ('left', 'right') drives its own set of sensory neurons.
    """

    def __init__(self, cfg, brain, annotations=None):
        enc = cfg.get('encoding', {})
        self.min_rate = float(enc.get('min_rate_hz', 0.0))
        self.max_rate = float(enc.get('max_rate_hz', 200.0))
        self.gamma = float(enc.get('gamma', 1.0))
        self.threshold = float(enc.get('threshold', 0.0))
        self.idx = {side: brain.index(resolve_neurons(cfg[side]['neurons'], annotations))
                    for side in ('left', 'right')}

    def rate(self, proximity):
        p = (float(proximity) - self.threshold) / (1.0 - self.threshold)
        p = min(max(p, 0.0), 1.0)
        return self.min_rate + (self.max_rate - self.min_rate) * p ** self.gamma

    def apply(self, brain, proximity):
        rates = {}
        for side in ('left', 'right'):
            rates[side] = self.rate(proximity[side])
            brain.set_rates(self.idx[side], rates[side])
        return rates


@dataclass
class Commands:
    throttle: float = 0.0   # 0..1
    steer: float = 0.0      # -1 (right) .. +1 (left)
    brake: float = 0.0      # 0..1
    rates: dict = field(default_factory=dict)   # smoothed group rates in Hz


class MotorDecoder:
    """Firing rates of motor/descending neurons -> car commands.

    Each group in ``motor.groups`` is a set of neurons whose mean rate (Hz) is
    low-pass filtered (time constant ``smoothing_ms``) and normalised:

        level = clip((rate - offset_hz) / full_scale_hz, 0, 1)

    throttle = level(throttle) (+ base_throttle), brake = level(brake),
    steer    = steer_gain * (level(steer_left) - level(steer_right)).
    Group names used for the commands are configurable in ``motor.commands``.
    """

    def __init__(self, cfg, brain, annotations=None):
        self.cfg = cfg
        self.groups = {}
        order = []
        for name, g in cfg['groups'].items():
            ids = resolve_neurons(g['neurons'], annotations)
            idx = brain.index(ids)
            self.groups[name] = dict(slice=slice(len(order), len(order) + len(ids)),
                                     offset=float(g.get('offset_hz', 0.0)),
                                     scale=float(g.get('full_scale_hz', 50.0)))
            order.extend(idx.tolist())
        import torch
        self.record_idx = torch.as_tensor(order, dtype=torch.long, device=brain.device)
        self.tau = float(cfg.get('smoothing_ms', 0.0))
        self.smoothed = {name: 0.0 for name in self.groups}
        c = cfg.get('commands', {})
        self.cmd = dict(throttle=c.get('throttle', 'throttle'), brake=c.get('brake', 'brake'),
                        steer_left=c.get('steer_left', 'steer_left'),
                        steer_right=c.get('steer_right', 'steer_right'))
        self.base_throttle = float(cfg.get('base_throttle', 0.0))
        self.steer_gain = float(cfg.get('steer_gain', 1.0))

    def decode(self, counts, window_ms):
        counts = counts.detach().cpu().numpy().reshape(-1) if hasattr(counts, 'detach') \
            else np.asarray(counts).reshape(-1)
        alpha = 1.0 if self.tau <= 0 else 1.0 - math.exp(-window_ms / self.tau)
        levels = {}
        for name, g in self.groups.items():
            rate = counts[g['slice']].mean() / (window_ms / 1000.0)
            self.smoothed[name] += alpha * (rate - self.smoothed[name])
            levels[name] = min(max((self.smoothed[name] - g['offset']) / g['scale'], 0.0), 1.0)

        def lvl(key):
            return levels.get(self.cmd[key], 0.0)

        throttle = min(1.0, self.base_throttle + lvl('throttle'))
        steer = max(-1.0, min(1.0, self.steer_gain * (lvl('steer_left') - lvl('steer_right'))))
        return Commands(throttle=throttle, steer=steer, brake=lvl('brake'),
                        rates=dict(self.smoothed))

    def reset(self):
        self.smoothed = {name: 0.0 for name in self.groups}


# ============================================================================
# 2D car world
# ============================================================================

@dataclass
class Obstacle:
    x: float
    y: float
    w: float
    h: float


class World:
    """Endless straight road along +y with rectangular obstacles.

    Coordinates in metres; x = 0 is the road centre, the road spans
    [-road_width/2, road_width/2].  Obstacles are generated lazily ahead of
    the car and dropped once they are far behind.
    """

    def __init__(self, cfg, seed=None):
        self.road_w = float(cfg.get('road_width', 12.0))
        self.spacing = tuple(cfg.get('obstacle_spacing', [14.0, 24.0]))
        self.size = tuple(cfg.get('obstacle_size', [1.5, 3.5]))
        self.free_start = float(cfg.get('free_start', 20.0))
        self.rng = random.Random(seed)
        self.obstacles: list[Obstacle] = []
        self.next_y = self.free_start

    def update(self, car_y, lookahead=80.0, behind=30.0):
        while self.next_y < car_y + lookahead:
            w = self.rng.uniform(*self.size)
            h = self.rng.uniform(*self.size)
            half = self.road_w / 2
            x = self.rng.uniform(-half + w / 2, half - w / 2)
            self.obstacles.append(Obstacle(x, self.next_y, w, h))
            self.next_y += self.rng.uniform(*self.spacing)
        self.obstacles = [o for o in self.obstacles if o.y + o.h > car_y - behind]

    def raycast(self, ox, oy, angle, max_range):
        """Distance from (ox, oy) along ``angle`` to road edge / obstacle."""
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_range
        half = self.road_w / 2
        for edge in (-half, half):             # road edges x = +-half
            if abs(dx) > 1e-9:
                tt = (edge - ox) / dx
                if 0 < tt < best:
                    best = tt
        for o in self.obstacles:               # slab test for axis-aligned boxes
            if abs(o.y - oy) > max_range + o.h:
                continue
            tmin, tmax = 0.0, best
            hit = True
            for p, d, lo, hi in ((ox, dx, o.x - o.w / 2, o.x + o.w / 2),
                                 (oy, dy, o.y - o.h / 2, o.y + o.h / 2)):
                if abs(d) < 1e-9:
                    if p < lo or p > hi:
                        hit = False
                        break
                else:
                    t1, t2 = (lo - p) / d, (hi - p) / d
                    if t1 > t2:
                        t1, t2 = t2, t1
                    tmin, tmax = max(tmin, t1), min(tmax, t2)
                    if tmin > tmax:
                        hit = False
                        break
            if hit and tmin < best:
                best = tmin
        return best

    def collides(self, x, y, radius):
        half = self.road_w / 2
        if x - radius < -half or x + radius > half:
            return True
        for o in self.obstacles:
            cx = min(max(x, o.x - o.w / 2), o.x + o.w / 2)
            cy = min(max(y, o.y - o.h / 2), o.y + o.h / 2)
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                return True
        return False


class Car:
    """Kinematic car; heading 0 = straight ahead (+y), positive = to the left."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_speed = float(cfg.get('max_speed', 12.0))
        self.accel = float(cfg.get('accel', 6.0))
        self.brake_decel = float(cfg.get('brake_decel', 14.0))
        self.drag = float(cfg.get('drag', 0.4))
        self.max_turn = math.radians(float(cfg.get('max_turn_rate_deg', 120.0)))
        self.radius = float(cfg.get('radius', 0.9))
        self.reset()

    def reset(self, x=0.0, y=0.0):
        self.x, self.y, self.heading, self.speed = x, y, 0.0, 0.0

    @property
    def angle(self):
        """World angle of the heading (math convention, +y = pi/2)."""
        return math.pi / 2 + self.heading

    def update(self, cmd: Commands, dt):
        acc = self.accel * cmd.throttle - self.brake_decel * cmd.brake - self.drag * self.speed
        self.speed = min(max(self.speed + acc * dt, 0.0), self.max_speed)
        # turn rate grows with speed, but allow slow turning on the spot
        turn_factor = 0.25 + 0.75 * min(1.0, self.speed / (0.5 * self.max_speed))
        self.heading += cmd.steer * self.max_turn * turn_factor * dt
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt


class DistanceSensors:
    """Ray fan per side; proximity = 1 - min_distance / range (0 = free)."""

    def __init__(self, cfg):
        self.range = float(cfg.get('range', 15.0))
        self.angles = {side: [math.radians(a) for a in cfg[side]['angles_deg']]
                       for side in ('left', 'right')}

    def read(self, world, car):
        dist, prox, rays = {}, {}, []
        for side, angles in self.angles.items():
            ds = []
            for a in angles:
                d = world.raycast(car.x, car.y, car.angle + a, self.range)
                ds.append(d)
                rays.append((car.angle + a, d))
            dist[side] = min(ds)
            prox[side] = 1.0 - dist[side] / self.range
        return dist, prox, rays


# ============================================================================
# Controllers
# ============================================================================

class BrainController:
    """Sensors -> Poisson -> FlyBrain (step_ms) -> motor rates -> commands."""

    def __init__(self, cfg, brain=None):
        bcfg = cfg['brain']
        ann = bcfg.get('annotations')
        sensor_ids = resolve_neurons([cfg['sensors']['left']['neurons'],
                                      cfg['sensors']['right']['neurons']], ann)
        tonic = cfg.get('tonic_inputs') or []
        tonic_ids = resolve_neurons([t['neurons'] for t in tonic], ann)
        if brain is None:
            t0 = time.perf_counter()
            brain = FlyBrain.from_fly_brain(
                bcfg['fly_brain_dir'], bcfg.get('completeness', 'data/2025_Completeness_783.csv'),
                bcfg.get('connectivity', 'data/2025_Connectivity_783.parquet'),
                params=bcfg.get('model_params'), dt_ms=bcfg.get('dt_ms', 0.1),
                device=bcfg.get('device', 'auto'), seed=bcfg.get('seed'),
                input_neurons=sensor_ids + tonic_ids)
            print(f'[fly_driver] brain: {brain.N} neurons, {brain.w.numel()} synapse rows, '
                  f'device={brain.device}, loaded in {time.perf_counter() - t0:.1f}s')
        self.brain = brain
        self.step_ms = float(bcfg.get('step_ms', 30.0))
        if not 20.0 <= self.step_ms <= 50.0:
            print(f'[fly_driver] warning: step_ms={self.step_ms} outside 20-50 ms', file=sys.stderr)
        self.encoder = SensorEncoder(cfg['sensors'], brain, ann)
        self.decoder = MotorDecoder(cfg['motor'], brain, ann)
        for t in tonic:
            brain.set_rates(brain.index(resolve_neurons(t['neurons'], ann)), t['rate_hz'])
        self.last_rates = {}

    def __call__(self, proximity):
        self.last_rates = self.encoder.apply(self.brain, proximity)
        counts = self.brain.run(self.step_ms, self.decoder.record_idx)
        return self.decoder.decode(counts, self.step_ms)


class ReflexController:
    """Braitenberg-style stand-in (no connectome) for testing the car world."""

    def __init__(self, cfg):
        self.step_ms = float(cfg['brain'].get('step_ms', 30.0))
        self.last_rates = {}

    def __call__(self, proximity):
        pl, pr = proximity['left'], proximity['right']
        near = max(pl, pr)
        return Commands(throttle=max(0.3, 1.0 - near), steer=max(-1.0, min(1.0, 3.0 * (pr - pl))),
                        brake=max(0.0, near - 0.85) * 3.0)


# ============================================================================
# Simulation loop
# ============================================================================

class Simulation:
    def __init__(self, cfg, controller, seed=None):
        self.cfg = cfg
        self.world = World(cfg.get('world', {}), seed=seed)
        self.car = Car(cfg.get('car', {}))
        self.sensors = DistanceSensors(cfg['sensors'])
        self.controller = controller
        self.dt = controller.step_ms / 1000.0     # car time == brain time
        self.t = 0.0
        self.collisions = 0        # number of distinct crashes
        self.colliding = False
        self.cmd = Commands()
        self.rays = []
        self.world.update(self.car.y)

    def tick(self):
        dist, prox, self.rays = self.sensors.read(self.world, self.car)
        self.cmd = self.controller(prox)
        prev = (self.car.x, self.car.y)
        was_colliding = self.colliding
        self.car.update(self.cmd, self.dt)
        self.colliding = self.world.collides(self.car.x, self.car.y, self.car.radius)
        if self.colliding:
            # back to the last free position and stop; the heading change is
            # kept so the controller can steer away from the obstacle
            self.car.x, self.car.y = prev
            self.car.speed = 0.0
            self.collisions += not was_colliding
        self.world.update(self.car.y)
        self.t += self.dt
        return dict(t=round(self.t, 4), x=self.car.x, y=self.car.y,
                    heading_deg=math.degrees(self.car.heading), speed=self.car.speed,
                    dist_left=dist['left'], dist_right=dist['right'],
                    rate_in_left=self.controller.last_rates.get('left', 0.0),
                    rate_in_right=self.controller.last_rates.get('right', 0.0),
                    throttle=self.cmd.throttle, steer=self.cmd.steer, brake=self.cmd.brake,
                    collisions=self.collisions,
                    **{f'rate_{k}': v for k, v in self.cmd.rates.items()})


class Renderer:
    PX_PER_M = 22

    def __init__(self, sim, size=(640, 720)):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('fly_driver - FlyWire brain drives a car')
        self.font = pygame.font.SysFont('monospace', 14)
        self.sim = sim
        self.size = size

    def to_screen(self, x, y):
        w, h = self.size
        car = self.sim.car
        return (int(w / 2 + x * self.PX_PER_M),
                int(h * 0.75 - (y - car.y) * self.PX_PER_M))

    def draw(self, wall_fps=None):
        pg, s, sim = self.pg, self.screen, self.sim
        w, h = self.size
        s.fill((40, 110, 40))
        half = sim.world.road_w / 2
        x0, _ = self.to_screen(-half, 0)
        x1, _ = self.to_screen(half, 0)
        pg.draw.rect(s, (60, 60, 60), (x0, 0, x1 - x0, h))
        # dashed centre line scrolling with the car
        off = int((sim.car.y * self.PX_PER_M) % 40)
        for yy in range(-40 + off, h, 40):
            pg.draw.line(s, (220, 220, 220), (w // 2, yy), (w // 2, yy + 20), 2)
        for o in sim.world.obstacles:
            cx, cy = self.to_screen(o.x, o.y)
            rw, rh = int(o.w * self.PX_PER_M), int(o.h * self.PX_PER_M)
            pg.draw.rect(s, (200, 70, 50), (cx - rw // 2, cy - rh // 2, rw, rh))
        car = sim.car
        cx, cy = self.to_screen(car.x, car.y)
        rng = sim.sensors.range
        for ang, d in sim.rays:
            ex, ey = self.to_screen(car.x + math.cos(ang) * d, car.y + math.sin(ang) * d)
            col = (255, 220, 0) if d < rng else (120, 120, 120)
            pg.draw.line(s, col, (cx, cy), (ex, ey), 1)
        r = int(car.radius * self.PX_PER_M)
        pg.draw.circle(s, (60, 140, 230), (cx, cy), r)
        hx = cx + int(math.cos(car.angle) * r * 1.4)
        hy = cy - int(math.sin(car.angle) * r * 1.4)
        pg.draw.line(s, (255, 255, 255), (cx, cy), (hx, hy), 3)
        c = sim.cmd
        lines = [f't={sim.t:6.2f}s  v={car.speed:5.2f} m/s  crashes={sim.collisions}',
                 f'throttle={c.throttle:4.2f} steer={c.steer:+5.2f} brake={c.brake:4.2f}']
        lines += [f'{k:>12s}: {v:6.1f} Hz' for k, v in c.rates.items()]
        if wall_fps is not None:
            lines.append(f'{wall_fps:4.1f} ticks/s wall-clock')
        pg.draw.rect(s, (0, 0, 0), (0, 0, 330, 18 * len(lines) + 8))
        for i, line in enumerate(lines):
            s.blit(self.font.render(line, True, (255, 255, 255)), (6, 4 + 18 * i))
        pg.display.flip()

    def quit_requested(self):
        for ev in self.pg.event.get():
            if ev.type == self.pg.QUIT or (ev.type == self.pg.KEYDOWN and ev.key == self.pg.K_ESCAPE):
                return True
        return False


def build_controller(cfg, kind='connectome'):
    return ReflexController(cfg) if kind == 'reflex' else BrainController(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument('--brain', choices=['connectome', 'reflex'], default='connectome')
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], help='override brain.device')
    ap.add_argument('--step-ms', type=float, help='override brain.step_ms (20-50)')
    ap.add_argument('--headless', action='store_true', help='no window')
    ap.add_argument('--ticks', type=int, default=0, help='stop after N control steps (0 = endless)')
    ap.add_argument('--log', help='write per-step CSV log')
    ap.add_argument('--seed', type=int, default=0, help='world seed')
    ap.add_argument('--screenshot', help='save the last frame as PNG (works with --headless)')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.device:
        cfg['brain']['device'] = args.device
    if args.step_ms:
        cfg['brain']['step_ms'] = args.step_ms
    if args.headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

    sim = Simulation(cfg, build_controller(cfg, args.brain), seed=args.seed)
    renderer = Renderer(sim) if (not args.headless or args.screenshot) else None
    writer, fh = None, None
    n, t_last = 0, time.perf_counter()
    try:
        while not args.ticks or n < args.ticks:
            if renderer and renderer.quit_requested():
                break
            row = sim.tick()
            n += 1
            now = time.perf_counter()
            fps, t_last = 1.0 / max(now - t_last, 1e-9), now
            if args.log:
                if writer is None:
                    fh = open(args.log, 'w', newline='')
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            if renderer:
                renderer.draw(fps)
            if args.headless and n % 10 == 0:
                print(f"t={row['t']:6.2f}s y={row['y']:7.2f} x={row['x']:+5.2f} v={row['speed']:4.1f} "
                      f"thr={row['throttle']:.2f} steer={row['steer']:+.2f} brk={row['brake']:.2f} "
                      f"crash={row['collisions']} ({fps:.1f} ticks/s)", flush=True)
    finally:
        if fh:
            fh.close()
        if renderer and args.screenshot:
            renderer.pg.image.save(renderer.screen, args.screenshot)
    print(f'done: {n} steps, {sim.t:.2f}s sim time, distance {sim.car.y:.1f} m, '
          f'{sim.collisions} collisions')
    return sim


if __name__ == '__main__':
    main()
