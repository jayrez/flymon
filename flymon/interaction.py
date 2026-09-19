"""Frozen baseline population and sparse event decoders for Experiment 11."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
@dataclass(frozen=True)
class FrozenPopulationBaseline:
    means_hz:dict[str,float];variances_hz2:dict[str,float];per_neuron_means_hz:dict[str,tuple[float,...]];windows:int
def estimate_frozen_population_baseline(windows,per_cell_counts,populations,window_seconds=.2):
    if len(windows)<2 or len(windows)!=len(per_cell_counts):raise ValueError("matching baseline windows and per-cell counts required")
    means,variances,cells={},{},{}
    for name in populations:
        values=np.asarray([float(x[name]) for x in windows])
        if not np.all(np.isfinite(values)) or np.any(values<0):raise ValueError("baseline rates must be finite and nonnegative")
        means[name]=float(values.mean());variances[name]=float(values.var(ddof=1));cells[name]=tuple(map(float,(np.asarray([x[name] for x in per_cell_counts],float)/window_seconds).mean(axis=0)))
    return FrozenPopulationBaseline(means,variances,cells,len(windows))
@dataclass(frozen=True)
class PopulationThresholdConfig:
    population:str;multiplier:float;sd_floor_hz:float;cooldown_decisions:int=1
class BaselinePopulationDecoder:
    def __init__(self,baseline,config,action,ablated=False):
        self.baseline,self.config,self.action,self.ablated=baseline,config,action,ablated;self.scale_hz=max(math.sqrt(max(0.,baseline.variances_hz2[config.population])),config.sd_floor_hz);self.threshold_hz=config.multiplier*self.scale_hz;self.cooldown=0
    def decode(self,rates):
        delta=float(rates[self.config.population])-self.baseline.means_hz[self.config.population];crossed=delta>self.threshold_hz and not self.ablated;raw=self.action if crossed else None
        if self.cooldown:action,reason=None,"cooldown";self.cooldown-=1
        else:action,reason=raw,"threshold" if raw else "deadband";self.cooldown=self.config.cooldown_decisions if action else 0
        return action,{"population":self.config.population,"delta_hz":delta,"scale_hz":self.scale_hz,"threshold_hz":self.threshold_hz,"normalized_excess":delta/self.threshold_hz-1.,"crossed":crossed,"selected_action":action,"reason":reason,"ablated":self.ablated,"baseline":asdict(self.baseline)}
@dataclass(frozen=True)
class EventConfig:
    population:str;multiplier:float;sd_floor_hz:float;refractory_decisions:int=5;require_rising_edge:bool=True
class BaselineEventController:
    def __init__(self,baseline,config,ablated=False):
        self.baseline,self.config,self.ablated=baseline,config,ablated;self.scale_hz=max(math.sqrt(max(0.,baseline.variances_hz2[config.population])),config.sd_floor_hz);self.threshold_hz=config.multiplier*self.scale_hz;self.refractory=0;self.was_above=False;self.suppressions=0
    def decode(self,rates):
        delta=float(rates[self.config.population])-self.baseline.means_hz[self.config.population];above=delta>self.threshold_hz;edge=above and (not self.was_above or not self.config.require_rising_edge);blocked=edge and self.refractory>0
        if blocked:self.suppressions+=1
        fire=edge and not blocked and not self.ablated
        if self.refractory:self.refractory-=1
        if fire:self.refractory=self.config.refractory_decisions
        self.was_above=above
        return ("A" if fire else None),{"population":self.config.population,"delta_hz":delta,"scale_hz":self.scale_hz,"threshold_hz":self.threshold_hz,"above_threshold":above,"rising_edge":edge,"refractory_blocked":blocked,"refractory_remaining":self.refractory,"refractory_suppressions":self.suppressions,"selected_action":"A" if fire else None,"ablated":self.ablated,"baseline":asdict(self.baseline)}
