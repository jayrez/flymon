"""Frozen-baseline motor and transparent event decoders."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict,dataclass
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
        means[name]=float(values.mean());variances[name]=float(values.var(ddof=1))
        cells[name]=tuple(map(float,(np.asarray([x[name] for x in per_cell_counts],float)/window_seconds).mean(axis=0)))
    return FrozenPopulationBaseline(means,variances,cells,len(windows))

@dataclass(frozen=True)
class PopulationThresholdConfig:
    population:str;multiplier:float;sd_floor_hz:float;cooldown_decisions:int=1

class BaselinePopulationDecoder:
    def __init__(self,baseline,config,action,ablated=False):
        self.baseline,self.config,self.action,self.ablated=baseline,config,action,ablated
        self.scale_hz=max(math.sqrt(max(0.,baseline.variances_hz2[config.population])),config.sd_floor_hz)
        self.threshold_hz=config.multiplier*self.scale_hz;self.cooldown=0
    def decode(self,rates):
        delta=float(rates[self.config.population])-self.baseline.means_hz[self.config.population]
        crossed=delta>self.threshold_hz and not self.ablated;raw=self.action if crossed else None
        if self.cooldown:action,reason=None,"cooldown";self.cooldown-=1
        else:action,reason=raw,"threshold" if raw else "deadband";self.cooldown=self.config.cooldown_decisions if action else 0
        return action,{"population":self.config.population,"delta_hz":delta,"scale_hz":self.scale_hz,"threshold_hz":self.threshold_hz,"normalized_excess":delta/self.threshold_hz-1.,"crossed":crossed,"selected_action":action,"reason":reason,"ablated":self.ablated,"baseline":asdict(self.baseline)}

@dataclass(frozen=True)
class EventConfig:
    population:str;multiplier:float;sd_floor_hz:float;refractory_decisions:int=5;require_rising_edge:bool=True

class BaselineEventController:
    """Experiment 11 reference decoder; behavior is intentionally unchanged."""
    def __init__(self,baseline,config,ablated=False):
        self.baseline,self.config,self.ablated=baseline,config,ablated
        self.scale_hz=max(math.sqrt(max(0.,baseline.variances_hz2[config.population])),config.sd_floor_hz)
        self.threshold_hz=config.multiplier*self.scale_hz;self.refractory=0;self.was_above=False;self.suppressions=0
    def decode(self,rates):
        delta=float(rates[self.config.population])-self.baseline.means_hz[self.config.population];above=delta>self.threshold_hz
        edge=above and (not self.was_above or not self.config.require_rising_edge);blocked=edge and self.refractory>0
        if blocked:self.suppressions+=1
        fire=edge and not blocked and not self.ablated
        if self.refractory:self.refractory-=1
        if fire:self.refractory=self.config.refractory_decisions
        self.was_above=above
        return ("A" if fire else None),{"population":self.config.population,"delta_hz":delta,"scale_hz":self.scale_hz,"threshold_hz":self.threshold_hz,"above_threshold":above,"rising_edge":edge,"refractory_blocked":blocked,"refractory_remaining":self.refractory,"refractory_suppressions":self.suppressions,"selected_action":"A" if fire else None,"ablated":self.ablated,"baseline":asdict(self.baseline)}

def aggregate_dnp01(cell_rates,method):
    values=tuple(float(x) for x in cell_rates)
    if len(values)!=2 or not all(math.isfinite(x) and x>=0 for x in values):raise ValueError("exactly two finite nonnegative DNp01 rates required")
    if method=="sum":return sum(values)
    if method=="mean":return sum(values)/2
    if method=="max":return max(values)
    if method=="difference":return values[0]-values[1]
    raise ValueError(f"unknown aggregation {method}")

@dataclass(frozen=True)
class EventDecoderConfig:
    strategy:str;aggregation:str;baseline_hz:float;scale_hz:float;threshold_z:float
    window:int=1;hysteresis_z:float=.5;refractory_decisions:int=5
    def __post_init__(self):
        if self.strategy not in {"rising_edge","level","recovery","derivative","cumulative"}:raise ValueError("unknown event strategy")
        if self.aggregation not in {"sum","mean","max","difference"}:raise ValueError("unknown aggregation")
        if self.scale_hz<=0 or self.window<1 or self.refractory_decisions<0 or self.hysteresis_z<0:raise ValueError("invalid event decoder configuration")

class TransparentEventController:
    """Low-capacity event decoder receiving only two DNp01 cell rates."""
    def __init__(self,config:EventDecoderConfig,ablated=False):
        self.config,self.ablated=config,ablated;self.reset()
    def reset(self):
        self.history=deque(maxlen=max(10,4*self.config.window));self.refractory=0;self.armed=True;self.suppressions=0
    def _evidence(self):
        z=list(self.history);w=self.config.window
        if self.config.strategy in ("rising_edge","level"):return z[-1]
        if self.config.strategy=="cumulative":
            return sum(z[-w:])/math.sqrt(w) if len(z)>=w else -math.inf
        if self.config.strategy=="derivative":
            return float(np.mean(z[-w:])-np.mean(z[-2*w:-w])) if len(z)>=2*w else -math.inf
        if self.config.strategy=="recovery":
            if len(z)<2*w:return -math.inf
            return float(np.mean(z[-w:])-min(z[:-w]))
        raise AssertionError
    def decode(self,rates):
        cells=rates["DNp01_cells_hz"];raw=aggregate_dnp01(cells,self.config.aggregation)
        z=(raw-self.config.baseline_hz)/self.config.scale_hz;self.history.append(z);evidence=self._evidence()
        above=evidence>self.config.threshold_z
        edge=above and self.armed
        if self.config.strategy=="rising_edge":
            # Preserve Experiment 11 exactly: any below-threshold decision rearms.
            if not above:self.armed=True
        elif evidence<=self.config.threshold_z-self.config.hysteresis_z:self.armed=True
        blocked=edge and self.refractory>0
        if blocked:self.suppressions+=1
        fire=edge and not blocked and not self.ablated
        if edge:self.armed=False
        if self.refractory:self.refractory-=1
        if fire:self.refractory=self.config.refractory_decisions
        state={"strategy":self.config.strategy,"aggregation":self.config.aggregation,"cell_rates_hz":list(map(float,cells)),"aggregate_rate_hz":raw,"z":z,"evidence":evidence if math.isfinite(evidence) else None,"threshold_z":self.config.threshold_z,"above_threshold":above,"armed":self.armed,"refractory_blocked":blocked,"refractory_remaining":self.refractory,"refractory_suppressions":self.suppressions,"selected_action":"A" if fire else None,"ablated":self.ablated}
        if fire and self.config.strategy=="cumulative":self.history.clear()
        return ("A" if fire else None),state
