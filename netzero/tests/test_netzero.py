"""Unit tests for the netzero package (fast; episode tests use small_rack, days=1)."""
from __future__ import annotations

import math
import random

import pytest

from netzero.experiment import Event, run_episode
from netzero.facility import (
    Device,
    Facility,
    FUNCTIONS,
    available_facilities,
    load_facility,
    simulator_truth,
)
from netzero.fuzzy import get_crop_model
from netzero.grid import load_grid
from netzero.model import LearnedDynamics
from netzero.simulator import VirtualFarm, available_climates, load_climate


# ----------------------------------------------------------------------------
# 1. load_facility / Device
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", available_facilities())
def test_load_facility_devices_and_idle(name):
    fac = load_facility(name)
    assert fac.devices, f"{name} has no devices"
    ids = [d.id for d in fac.devices]
    assert len(ids) == len(set(ids))
    idle = fac.idle_control()
    assert set(idle) == set(ids)
    for d in fac.devices:
        assert idle[d.id] == d.min_level
        assert min(d.levels) == d.min_level
        assert d.levels == sorted(d.levels)
        assert all(f in FUNCTIONS for f in d.functions)
    # simulator truth is available but not part of the facility devices
    assert isinstance(simulator_truth(name), dict)


@pytest.mark.parametrize("name", available_facilities())
def test_energy_wh_math(name):
    fac = load_facility(name)
    for d in fac.devices:
        top = max(d.levels)
        if d.unit == "min":
            assert d.energy_wh(top) == pytest.approx(d.power_w * top / 60.0)
        elif d.unit == "ml":
            assert d.energy_wh(top) == pytest.approx(d.energy_wh_per_unit * top)
        assert d.energy_wh(0) == 0.0
    top_ctrl = {d.id: max(d.levels) for d in fac.devices}
    assert fac.energy_wh(top_ctrl) == pytest.approx(sum(d.energy_wh(max(d.levels)) for d in fac.devices))


def test_device_from_dict_min_level_and_unknown_function():
    d = Device.from_dict({"id": "x", "levels": [0, 5, 20], "min_level": 10, "power_w": 6})
    assert d.levels == [10.0, 20.0]
    assert d.energy_wh(30) == pytest.approx(3.0)
    ml = Device.from_dict({"id": "y", "unit": "ml", "levels": [0, 4], "energy_wh_per_unit": 0.25,
                           "resource_per_unit": {"nutrient_ml": 1.0}})
    assert ml.energy_wh(4) == pytest.approx(1.0)
    assert ml.resources(4) == {"nutrient_ml": 4.0}
    with pytest.raises(ValueError):
        Device.from_dict({"id": "z", "functions": ["teleport"]})


# ----------------------------------------------------------------------------
# 2. candidates / without
# ----------------------------------------------------------------------------
@pytest.fixture
def rack() -> Facility:
    return load_facility("small_rack")


def test_candidates_pruning(rack):
    cands = rack.candidates(["water_temp_down"])
    movers = {d.id for d in rack.devices if "water_temp_down" in d.functions}
    assert movers  # peltier_bank, vent_fan
    for c in cands:
        for d in rack.devices:
            if d.id not in movers:
                assert c[d.id] == d.min_level
    assert len(cands) == math.prod(len(rack.device(i).levels) for i in movers)


def test_candidates_none_full_product(rack):
    cands = rack.candidates(None)
    assert len(cands) == math.prod(len(d.levels) for d in rack.devices)
    assert len({tuple(sorted(c.items())) for c in cands}) == len(cands)


def test_candidates_empty_needed_is_idle(rack):
    assert rack.candidates([]) == [rack.idle_control()]


def test_without_removes_device(rack):
    f2 = rack.without("vent_fan")
    assert "vent_fan" not in [d.id for d in f2.devices]
    assert "vent_fan" in [d.id for d in rack.devices]  # original untouched
    with pytest.raises(KeyError):
        f2.device("vent_fan")
    for c in f2.candidates(None):
        assert "vent_fan" not in c
    for c in f2.candidates(["water_temp_down"]):
        assert "vent_fan" not in c


# ----------------------------------------------------------------------------
# 3. build_commands
# ----------------------------------------------------------------------------
def test_build_commands_pins_and_protection(rack):
    ctrl = rack.idle_control()
    ctrl.update({"peltier_bank": 30, "main_pump": 10})
    cmds = rack.build_commands(ctrl)
    assert set(cmds) == {d.id for d in rack.devices}
    for d in rack.devices:
        pins = (d.hardware or {}).get("pins")
        if pins:
            assert cmds[d.id]["pins"] == pins
        else:
            assert "pins" not in cmds[d.id]
    assert cmds["main_pump"]["driver"] == rack.device("main_pump").hardware["driver"]

    pb = cmds["peltier_bank"]
    hw = rack.device("peltier_bank").hardware
    prot = hw["protection"]
    assert pb["enabled"] is True
    assert pb["duration_sec"] == 30 * 60
    seq = pb["sequence"]
    support = list(hw["support_pins"].values())
    main = list(hw["pins"].values())
    assert seq["start"][0] == {"pins": support, "action": "ON"}
    assert seq["start"][1] == {"wait_sec": prot["startup_delay_sec"]}
    assert seq["start"][2] == {"pins": main, "action": "ON"}
    assert seq["stop"][0] == {"pins": main, "action": "OFF"}
    assert seq["stop"][1] == {"wait_sec": prot["cooldown_sec"]}
    assert seq["stop"][2] == {"pins": support, "action": "OFF"}
    assert pb["support_duration_sec"] == 30 * 60 + prot["startup_delay_sec"] + prot["cooldown_sec"]
    assert pb["estimated_wh"] == pytest.approx(rack.device("peltier_bank").energy_wh(30), abs=1e-4)


def test_build_commands_disabled_and_ml_units(rack):
    ctrl = rack.idle_control()
    ctrl["nutrient_doser"] = 10
    cmds = rack.build_commands(ctrl)
    for dev_id in ("peltier_bank", "main_pump", "vent_fan"):
        assert cmds[dev_id]["enabled"] is False
        assert cmds[dev_id]["level"] == 0
    # no protection sequence when the protected device is off
    assert "sequence" not in cmds["peltier_bank"]
    assert "support_duration_sec" not in cmds["peltier_bank"]
    # oxygen pump has a non-zero minimum, so it is always on
    assert cmds["oxygen_pump"]["enabled"] is True
    doser = cmds["nutrient_doser"]
    assert doser["enabled"] is True
    assert doser["target_amount"] == 10
    assert "duration_sec" not in doser


# ----------------------------------------------------------------------------
# 4. LearnedDynamics
# ----------------------------------------------------------------------------
def _synthetic_wt_data(fac: Facility, true_coef: dict, n: int, seed: int = 1):
    """Yield (x, u, outdoor, sun, x_next) from a known linear water-temperature model."""
    rng = random.Random(seed)
    for _ in range(n):
        u = {d.id: rng.choice(d.levels) for d in fac.devices}
        x = {"water_temperature": rng.uniform(18, 30), "flow_ratio": 1.0, "EC": 1.5, "turbidity": 0.5}
        outdoor = rng.uniform(15, 35)
        sun = rng.uniform(0, 1)
        wt = x["water_temperature"]
        dwt = (
            true_coef["ambient"] * (outdoor - wt)
            + true_coef["sun"] * sun
            + true_coef["peltier_bank"] * u["peltier_bank"]
            + true_coef["vent_fan"] * u["vent_fan"] * (outdoor - wt)
            + rng.gauss(0, 0.05)
        )
        x_next = dict(x, water_temperature=wt + dwt)
        yield x, u, outdoor, sun, x_next


def test_learned_dynamics_recovers_true_coefficient(rack):
    prior = rack.device("peltier_bank").effects[0].prior
    true = {"ambient": 0.12, "sun": 0.6, "peltier_bank": -0.045, "vent_fan": 0.006}
    assert abs(true["peltier_bank"] - prior) > 0.01
    m = LearnedDynamics(rack, learn=True)
    assert m.device_effect("water_temperature", "peltier_bank") == pytest.approx(prior)
    for x, u, out, sun, xn in _synthetic_wt_data(rack, true, 400):
        m.update(x, u, out, sun, xn)
    learned = m.device_effect("water_temperature", "peltier_bank")
    err_learned = abs(learned - true["peltier_bank"])
    err_prior = abs(prior - true["peltier_bank"])
    assert err_learned < 0.003, (learned, true["peltier_bank"])
    assert err_learned < 0.2 * err_prior
    assert m.n_updates == 400


def test_learned_dynamics_no_learning(rack):
    m = LearnedDynamics(rack, learn=False)
    before = m.coefficients()
    true = {"ambient": 0.12, "sun": 0.6, "peltier_bank": -0.045, "vent_fan": 0.006}
    for x, u, out, sun, xn in _synthetic_wt_data(rack, true, 50):
        m.update(x, u, out, sun, xn)
    assert m.coefficients() == before
    assert m.n_updates == 0


def test_learned_dynamics_rebuild_keeps_other_coefficients(rack):
    m = LearnedDynamics(rack, learn=True)
    true = {"ambient": 0.12, "sun": 0.6, "peltier_bank": -0.045, "vent_fan": 0.006}
    for x, u, out, sun, xn in _synthetic_wt_data(rack, true, 300):
        m.update(x, u, out, sun, xn)
    learned = m.coefficients()
    m.rebuild(rack.without("vent_fan"))
    after = m.coefficients()
    assert m.device_effect("water_temperature", "vent_fan") is None
    assert "vent_fan:exchange" not in after["water_temperature"]
    for var, coefs in after.items():
        for nm, v in coefs.items():
            assert v == learned[var][nm], (var, nm)
    assert m.device_effect("water_temperature", "peltier_bank") == pytest.approx(
        learned["water_temperature"]["peltier_bank"], abs=1e-5
    )


def test_learned_dynamics_predict_respects_bounds(rack):
    m = LearnedDynamics(rack, learn=False)
    x = {"water_temperature": 0.5, "flow_ratio": 0.01, "EC": 0.0, "turbidity": 0.0, "pH": 6.0}
    u = {d.id: max(d.levels) for d in rack.devices}
    nxt = m.predict(x, u, outdoor=-10.0, sun=0.0)
    assert nxt["water_temperature"] >= 0.0
    assert nxt["pH"] == 6.0  # not modeled, passed through
    assert nxt["water_temperature"] < x["water_temperature"] + 1e-9


# ----------------------------------------------------------------------------
# 5. GridProfile
# ----------------------------------------------------------------------------
def test_grid_profile_values_and_factor():
    g = load_grid("kr_stylized")
    assert len(g.emission_factor) == 24 and len(g.price) == 24
    assert g.ef(25) == g.ef(1)
    assert g.price_at(-1) == g.price_at(23)
    for wc, wp in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)):
        avg = sum(g.weighted_cost_factor(h, wc, wp) for h in range(24)) / 24
        assert avg == pytest.approx(1.0, abs=1e-9)
    # carbon-only factor is highest in the evening peak and lowest at midday
    cf = [g.weighted_cost_factor(h, 1.0, 0.0) for h in range(24)]
    assert cf.index(max(cf)) in range(17, 22)
    assert cf.index(min(cf)) in range(10, 16)


# ----------------------------------------------------------------------------
# 6. VirtualFarm
# ----------------------------------------------------------------------------
def _farm(seed=0, initial=None, facility="small_rack"):
    fac = load_facility(facility)
    return VirtualFarm(fac, simulator_truth(facility), load_climate("summer_south", seed=seed),
                       initial_state=initial, seed=seed)


def test_available_climates():
    assert {"summer_south", "spring_mild", "heatwave"} <= set(available_climates())
    c = load_climate("spring_mild")
    assert all(0.0 <= c.sun(t) <= 1.0 for t in range(48))
    assert c.sun(0) == 0.0 and c.sun(12) > 0.5


def test_virtual_farm_deterministic():
    def run():
        f = _farm(seed=3)
        out = []
        for _ in range(10):
            out.append(f.observe())
            f.step(f.facility.idle_control())
        return out, dict(f.state)

    assert run() == run()


def test_cooling_lowers_water_temperature():
    init = {"water_temperature": 28.0}
    idle_f, cool_f = _farm(seed=1, initial=init), _farm(seed=1, initial=init)
    cool = idle_f.facility.idle_control()
    cool["peltier_bank"] = 60
    for _ in range(4):
        idle_f.step(idle_f.facility.idle_control())
        cool_f.step(cool)
    assert cool_f.state["water_temperature"] < idle_f.state["water_temperature"] - 1.0


def test_degrade_device_scales_effects():
    f = _farm()
    before = dict(f.truth["device_effects"]["main_pump"])
    f.degrade_device("main_pump", 0.5)
    for k, v in before.items():
        assert f.truth["device_effects"]["main_pump"][k] == pytest.approx(0.5 * v)
    # truth was deep-copied: the profile on disk is untouched
    assert simulator_truth("small_rack")["device_effects"]["main_pump"] == before

    init = {"water_temperature": 28.0}
    full, half = _farm(seed=2, initial=init), _farm(seed=2, initial=init)
    half.degrade_device("peltier_bank", 0.5)
    c = full.facility.idle_control()
    c["peltier_bank"] = 60
    t0 = full.state["water_temperature"]
    full.step(c)
    half.step(c)
    drop_full = t0 - full.state["water_temperature"]
    drop_half = t0 - half.state["water_temperature"]
    assert drop_half < drop_full
    eff = simulator_truth("small_rack")["device_effects"]["peltier_bank"]["water_temperature"]
    assert drop_full - drop_half == pytest.approx(-0.5 * eff * 60, rel=1e-6)


def test_step_energy_matches_facility_and_power_factor():
    f = _farm()
    ctrl = f.facility.idle_control()
    ctrl.update({"peltier_bank": 30, "main_pump": 10, "nutrient_doser": 5})
    used = f.step(ctrl)
    pf = f.truth.get("power_factor") or {}
    expected = sum(d.energy_wh(ctrl[d.id]) * float(pf.get(d.id, 1.0)) for d in f.facility.devices)
    assert used["energy_wh"] == pytest.approx(expected)
    assert used["energy_wh"] > f.facility.energy_wh(ctrl)  # peltier power factor > 1
    assert used["nutrient_ml"] == pytest.approx(5.0)
    assert f.t == 1


def test_remove_device_from_farm():
    f = _farm()
    f.remove_device("vent_fan")
    assert "vent_fan" not in [d.id for d in f.facility.devices]
    f.step({"peltier_bank": 10})  # missing ids default to min level


# ----------------------------------------------------------------------------
# 7. CropModel
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def basil():
    return get_crop_model("basil")


def test_crop_model_target_is_safe(basil):
    ev = basil.evaluate(basil.targets)
    assert not ev.is_alert
    assert ev.risk_score < 30.0  # 0..100 scale
    assert basil.danger_vars(ev) == []
    assert basil.in_optimal_range(ev)
    for var, (lo, hi) in basil.optimal_ranges.items():
        assert lo <= hi


def test_crop_model_hot_water_alert(basil):
    state = dict(basil.targets, water_temperature=40.0)
    ev = basil.evaluate(state)
    assert ev.is_alert
    assert ev.risk_score > basil.evaluate(basil.targets).risk_score
    assert "water_temperature" in basil.danger_vars(ev)
    assert not basil.in_optimal_range(ev)


def test_crop_model_cache_consistent(basil):
    a = basil.evaluate(basil.targets)
    b = basil.evaluate(dict(basil.targets))
    assert a is b


# ----------------------------------------------------------------------------
# 8. run_episode / Event
# ----------------------------------------------------------------------------
METRIC_KEYS = {
    "in_range_rate", "alert_rate", "danger_hours", "mean_risk", "energy_kwh", "carbon_kg",
    "cost_krw", "carbon_intensity_kg_per_kwh", "nutrient_ml", "pred_mae_wt_first_day",
    "pred_mae_wt_last_day", "adaptation_hours",
}


@pytest.mark.parametrize("label", ["A", "B", "C", "D"])
def test_run_episode_smoke(label):
    r = run_episode("small_rack", label, days=1, seed=0)
    m = r.metrics
    assert METRIC_KEYS <= set(m)
    assert 0.0 <= m["in_range_rate"] <= 1.0
    assert 0.0 <= m["alert_rate"] <= 1.0
    assert 0.0 <= m["danger_hours"] <= 24
    assert m["energy_kwh"] > 0
    assert m["carbon_kg"] > 0 and m["cost_krw"] > 0
    assert len(r.series["hour"]) == 24
    assert r.controller == label
    if label in ("C", "D"):
        assert r.learned_coefficients is not None
    else:
        assert r.learned_coefficients is None
    d = r.to_dict(with_series=False)
    assert "series" not in d and d["metrics"] == m


def test_run_episode_with_remove_event():
    r = run_episode("small_rack", "C", days=2, seed=0, events=[Event.parse("remove:vent_fan@1")])
    assert all("vent_fan" not in c for c in r.series["controls"][24:])
    assert any("vent_fan" in c for c in r.series["controls"][:24])
    assert "vent_fan:exchange" not in r.learned_coefficients["water_temperature"]


def test_event_parse():
    e = Event.parse("degrade:chiller_a:0.5@3")
    assert (e.day, e.type, e.device, e.factor) == (3, "degrade", "chiller_a", 0.5)
    e = Event.parse("remove:vent_fan@2")
    assert (e.day, e.type, e.device, e.factor) == (2, "remove", "vent_fan", 1.0)
    with pytest.raises(ValueError):
        Event.parse("explode:vent_fan@2")


# ----------------------------------------------------------------------------
# 10. dilution valve and pre-cooling
# ----------------------------------------------------------------------------
def test_dilution_valve_lowers_ec_in_simulator():
    fac = load_facility("mid_greenhouse")
    assert "ec_down" in fac.device("fresh_water_valve").functions
    clim = load_climate("summer_south", seed=0)
    truth = simulator_truth("mid_greenhouse")
    truth["process_noise"] = {}
    a = VirtualFarm(fac, truth, clim, initial_state={"EC": 2.2}, seed=1)
    b = VirtualFarm(fac, truth, clim, initial_state={"EC": 2.2}, seed=1)
    idle = fac.idle_control()
    a.step(idle)
    b.step({**idle, "fresh_water_valve": 20.0})
    assert b.state["EC"] < a.state["EC"] - 0.05
    assert fac.resources({**idle, "fresh_water_valve": 20.0})["water_l"] == 20.0


def test_dilution_effect_is_learned():
    fac = load_facility("mid_greenhouse")
    m = LearnedDynamics(fac, learn=True)
    prior = m.device_effect("EC", "fresh_water_valve")
    true_k = -0.004
    rng = random.Random(0)
    for _ in range(300):
        ec = rng.uniform(1.0, 2.5)
        u = {**fac.idle_control(), "fresh_water_valve": rng.choice([0.0, 5.0, 10.0, 20.0])}
        x = {"water_temperature": 22.0, "flow_ratio": 1.0, "EC": ec, "turbidity": 0.5}
        nxt = dict(x, EC=ec + true_k * u["fresh_water_valve"] * ec)
        m.update(x, u, 28.0, 0.0, nxt)
    learned = m.device_effect("EC", "fresh_water_valve")
    assert abs(learned - true_k) < abs(prior - true_k)
    assert abs(learned - true_k) < 0.001


def test_threshold_controller_dilutes_high_ec(basil):
    from netzero.controllers import Context, ThresholdController
    fac = load_facility("mid_greenhouse")
    ctrl = ThresholdController(fac, basil)
    obs = {"water_temperature": 22.5, "flow_ratio": 1.0, "EC": 2.2, "turbidity": 0.5, "pH": 6.15}
    u, _ = ctrl.decide(obs, Context(0, load_climate("summer_south"), load_grid("kr_stylized")))
    assert u["fresh_water_valve"] > 0
    assert u["nutrient_doser"] == 0


def test_precool_plan_only_for_carbon_aware(basil):
    from netzero.controllers import Context, make_controller
    fac = load_facility("mid_greenhouse")
    ctx = Context(10, load_climate("heatwave"), load_grid("kr_stylized"))
    obs = {"water_temperature": 23.5, "flow_ratio": 1.0, "EC": 1.5, "turbidity": 0.5, "pH": 6.15}
    c = make_controller("C", fac, basil)
    d = make_controller("D", fac, basil)
    assert not c.precool and d.precool
    plan = d._precool_plan(obs, ctx)
    assert plan is not None
    value, floor, hi = plan
    lo = basil.optimal_ranges["water_temperature"][0]
    assert value > 0 and lo < floor < hi
