"""Tests for fly_driver.py that run without the 100 MB connectome.

The equivalence test compares FlyBrain against TorchModel from
fly-brain/code/run_pytorch.py on a small random network; it is skipped if the
fly-brain clone is not found (set FLY_BRAIN_DIR or clone it next to this repo).
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fly_driver as fd  # noqa: E402

FLY_BRAIN_DIR = Path(os.environ.get('FLY_BRAIN_DIR') or ROOT.parent / 'fly-brain')


def random_network(n=400, nnz=6000, seed=1):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, nnz)
    post = rng.integers(0, n, nnz)
    w = rng.integers(1, 12, nnz) * rng.choice([-1, 1, 1, 1], nnz)
    return n, pre, post, w.astype(np.float32)


@pytest.mark.skipif(not (FLY_BRAIN_DIR / 'code' / 'run_pytorch.py').exists(),
                    reason='fly-brain clone not found')
def test_matches_reference_pytorch_model():
    sys.path.insert(0, str(FLY_BRAIN_DIR / 'code'))
    from run_pytorch import TorchModel, MODEL_PARAMS, DT

    n, pre, post, w = random_network()
    exc = list(range(0, 40))
    batch = 3
    weights = torch.sparse_coo_tensor(np.stack([post, pre]), w, (n, n)).coalesce().to_sparse_csr()
    ref = TorchModel(batch, n, DT, MODEL_PARAMS, weights, exc_indices=exc)
    state = ref.state_init()
    rates = torch.zeros(batch, n)
    rates[:, exc] = 180.0
    g_ref = torch.Generator().manual_seed(7)

    brain = fd.FlyBrain(n, pre, post, w, dt_ms=DT, device='cpu', batch=batch,
                        input_neurons=exc, seed=7)
    brain.set_rates(torch.tensor(exc), 180.0)

    total_ref, total = 0, 0
    with torch.no_grad():
        for _ in range(1500):
            state = ref(rates, *state, generator=g_ref)
            spk = brain._step()
            assert torch.equal(state[2], spk)
            total_ref += state[2].sum().item()
            total += spk.sum().item()
    assert total > 100            # the network actually did something
    assert torch.allclose(state[3], brain.v, atol=1e-4)


def test_run_counts_and_state_persistence():
    n, pre, post, w = random_network()
    brain = fd.FlyBrain(n, pre, post, w, device='cpu', input_neurons=range(10), seed=0)
    idx = torch.arange(10)
    brain.set_rates(idx, 200.0)
    c1 = brain.run(30.0, idx)
    assert c1.shape == (1, 10)
    assert abs(brain.t_ms - 30.0) < 1e-6
    brain.run(20.0)
    assert abs(brain.t_ms - 50.0) < 1e-6
    # 200 Hz for 30 ms -> ~6 spikes per driven neuron
    assert 2 < c1.mean().item() < 12


def test_resolve_neurons_forms():
    assert fd.resolve_neurons([1, 2]) == [1, 2]
    assert fd.resolve_neurons({'a': 1, 'b': [2, 3]}) == [1, 2, 3]
    assert fd.resolve_neurons([{'a': 5}, [6]]) == [5, 6]
    with pytest.raises(ValueError):
        fd.resolve_neurons({'select': {'cell_type': 'X'}})


def test_encoder_and_decoder():
    n, pre, post, w = random_network()
    brain = fd.FlyBrain(n, pre, post, w, device='cpu')
    enc = fd.SensorEncoder({'encoding': {'min_rate_hz': 10, 'max_rate_hz': 110, 'gamma': 1},
                            'left': {'neurons': [0, 1]}, 'right': {'neurons': [2]}}, brain)
    assert enc.rate(0) == 10 and enc.rate(1) == 110 and enc.rate(0.5) == 60
    enc.apply(brain, {'left': 1.0, 'right': 0.0})
    assert brain.rates[0, 0] == 110 and brain.rates[0, 2] == 10

    dec = fd.MotorDecoder({'smoothing_ms': 0, 'steer_gain': 1.0, 'groups': {
        'throttle': {'neurons': [3], 'full_scale_hz': 100},
        'steer_left': {'neurons': [4, 5], 'full_scale_hz': 100},
        'steer_right': {'neurons': [6], 'full_scale_hz': 100},
        'brake': {'neurons': [7], 'full_scale_hz': 100, 'offset_hz': 50}}}, brain)
    # 50 ms window: 5 spikes = 100 Hz
    counts = torch.tensor([[5., 5., 0., 0., 2.5]])
    cmd = dec.decode(counts, 50.0)
    assert cmd.throttle == pytest.approx(1.0)
    assert cmd.steer == pytest.approx(0.5)       # left mean 50 Hz vs right 0 Hz
    assert cmd.brake == pytest.approx(0.0)       # 50 Hz - 50 Hz offset


def test_world_raycast_and_collision():
    world = fd.World({'road_width': 10, 'free_start': 1000})
    # straight left from centre hits the edge at 5 m
    assert world.raycast(0, 0, math.pi, 20) == pytest.approx(5.0)
    world.obstacles.append(fd.Obstacle(0, 10, 2, 2))
    assert world.raycast(0, 0, math.pi / 2, 20) == pytest.approx(9.0)
    assert world.raycast(3, 0, math.pi / 2, 20) == pytest.approx(20.0)
    assert world.collides(0, 9.5, 0.9)
    assert world.collides(4.5, 0, 0.9)
    assert not world.collides(0, 0, 0.9)


def test_reflex_simulation_runs_headless():
    cfg = fd.load_config()
    sim = fd.Simulation(cfg, fd.ReflexController(cfg), seed=3)
    for _ in range(400):
        row = sim.tick()
    assert row['y'] > 20              # made progress along the road
    assert sim.dt == pytest.approx(cfg['brain']['step_ms'] / 1000)


def test_default_config_needs_no_annotation_file():
    cfg = fd.load_config()
    specs = [cfg['sensors']['left']['neurons'], cfg['sensors']['right']['neurons']]
    specs += [t['neurons'] for t in cfg.get('tonic_inputs') or []]
    specs += [g['neurons'] for g in cfg['motor']['groups'].values()]
    for spec in specs:
        ids = fd.resolve_neurons(spec, annotations_path=None)
        assert ids and all(isinstance(i, int) for i in ids)
    left = set(fd.resolve_neurons(specs[0]))
    right = set(fd.resolve_neurons(specs[1]))
    assert not left & right


def test_json_config(tmp_path):
    p = tmp_path / 'c.json'
    p.write_text('{"brain": {"fly_brain_dir": "fb", "step_ms": 25}}')
    cfg = fd.load_config(p)
    assert cfg['brain']['step_ms'] == 25
    assert Path(cfg['brain']['fly_brain_dir']) == (tmp_path / 'fb').resolve()
