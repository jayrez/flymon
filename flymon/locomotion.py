"""Experiment 9 baseline-relative DNg100/MDN fixed BCI."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class PopulationBaseline:
    dng100_mean_hz: float
    dng100_variance_hz2: float
    mdn_mean_hz: float
    mdn_variance_hz2: float
    windows: int
    dng100_per_neuron_mean_hz: tuple[float,...]
    mdn_per_neuron_mean_hz: tuple[float,...]


@dataclass(frozen=True)
class LocomotionConfig:
    forward_multiplier: float
    backward_multiplier: float
    forward_sd_floor_hz: float
    backward_sd_floor_hz: float
    cooldown_decisions: int=1
    def __post_init__(self):
        if min(self.forward_multiplier,self.backward_multiplier,self.forward_sd_floor_hz,self.backward_sd_floor_hz)<=0:raise ValueError("positive thresholds required")


def estimate_population_baseline(windows, per_cell):
    f=np.array([x["DNg100"] for x in windows],float);b=np.array([x["MDN"] for x in windows],float)
    fc=np.array([x["DNg100"] for x in per_cell],float)/(0.2);bc=np.array([x["MDN"] for x in per_cell],float)/(0.2)
    return PopulationBaseline(float(f.mean()),float(f.var(ddof=1)),float(b.mean()),float(b.var(ddof=1)),len(windows),
      tuple(map(float,fc.mean(axis=0))),tuple(map(float,bc.mean(axis=0))))


class BaselineLocomotionController:
    """UP/DOWN/NONE from summed population rates and a frozen baseline."""
    def __init__(self,baseline:PopulationBaseline,config:LocomotionConfig,ablate:str|None=None):
        self.baseline,self.config,self.ablate=baseline,config,ablate
        self.forward_scale=max(math.sqrt(max(0,baseline.dng100_variance_hz2)),config.forward_sd_floor_hz)
        self.backward_scale=max(math.sqrt(max(0,baseline.mdn_variance_hz2)),config.backward_sd_floor_hz)
        self.forward_threshold=config.forward_multiplier*self.forward_scale;self.backward_threshold=config.backward_multiplier*self.backward_scale;self.cooldown=0
    def decode(self,rates:Mapping[str,float]):
        f=float(rates["DNg100"])-self.baseline.dng100_mean_hz;b=float(rates["MDN"])-self.baseline.mdn_mean_hz
        if self.ablate=="DNg100":f=-math.inf
        if self.ablate=="MDN":b=-math.inf
        fz=f/self.forward_scale;bz=b/self.backward_scale;fu=f>self.forward_threshold;bd=b>self.backward_threshold
        raw="UP" if fu and (not bd or fz>=bz) else "DOWN" if bd else None
        if self.cooldown:action=None;reason="cooldown";self.cooldown-=1
        else:action=raw;reason="threshold" if raw else "deadband";self.cooldown=self.config.cooldown_decisions if action else 0
        return action,{"forward_delta_hz":f,"backward_delta_hz":b,"forward_z":fz,"backward_z":bz,"forward_threshold_hz":self.forward_threshold,
          "backward_threshold_hz":self.backward_threshold,"forward_crossed":fu,"backward_crossed":bd,"raw_action":raw,"selected_action":action,
          "reason":reason,"ablation":self.ablate,"baseline":asdict(self.baseline)}
