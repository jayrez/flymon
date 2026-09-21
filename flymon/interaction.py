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


@dataclass(frozen=True)
class FrozenVectorBaseline:
    """Per-cell firing-rate baseline frozen before active visual input."""
    means_hz: tuple[float, ...]
    variances_hz2: tuple[float, ...]
    windows: int

    def __post_init__(self) -> None:
        if not self.means_hz or len(self.means_hz) != len(self.variances_hz2):
            raise ValueError("matching nonempty vector baseline statistics required")
        if self.windows < 2:
            raise ValueError("at least two baseline windows required")


def estimate_frozen_vector_baseline(per_cell_counts, window_seconds: float = .2) -> FrozenVectorBaseline:
    counts = np.asarray(per_cell_counts, float)
    if counts.ndim != 2 or counts.shape[0] < 2 or window_seconds <= 0:
        raise ValueError("baseline counts must be a windows-by-cells matrix")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("baseline counts must be finite and nonnegative")
    rates = counts / window_seconds
    return FrozenVectorBaseline(
        tuple(map(float, rates.mean(axis=0))),
        tuple(map(float, rates.var(axis=0, ddof=1))),
        int(rates.shape[0]),
    )


def population_z_vector(rates_hz, baseline: FrozenVectorBaseline, sd_floor_hz: float) -> np.ndarray:
    rates = np.asarray(rates_hz, float)
    means = np.asarray(baseline.means_hz, float)
    variances = np.asarray(baseline.variances_hz2, float)
    if rates.shape != means.shape or sd_floor_hz <= 0:
        raise ValueError("population rates and baseline shape must match; floor must be positive")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("population rates must be finite and nonnegative")
    scales = np.maximum(np.sqrt(np.maximum(variances, 0)), sd_floor_hz)
    return (rates - means) / scales


def population_signal(z_history, strategy: str, window: int = 1) -> float | None:
    history = [np.asarray(x, float) for x in z_history]
    if not history or window < 1:
        return None
    current = history[-1]
    if strategy == "mean":
        return float(current.mean())
    if strategy == "norm":
        return float(np.sqrt(np.mean(current * current)))
    if strategy == "change":
        if len(history) < 2:
            return None
        difference = current - history[-2]
        return float(np.sqrt(np.mean(difference * difference)))
    if strategy == "window_change":
        if len(history) < 2 * window:
            return None
        recent = np.mean(history[-window:], axis=0)
        preceding = np.mean(history[-2 * window:-window], axis=0)
        return float(np.sqrt(np.mean((recent - preceding) ** 2)))
    raise ValueError(f"unknown population signal strategy {strategy}")


@dataclass(frozen=True)
class PopulationEventConfig:
    signal: str
    event_rule: str
    threshold: float
    sd_floor_hz: float
    signal_window: int = 1
    evidence_window: int = 1
    hysteresis: float = .5
    refractory_decisions: int = 5

    def __post_init__(self) -> None:
        if self.signal not in {"mean", "norm", "change", "window_change"}:
            raise ValueError("unknown population signal")
        if self.event_rule not in {"level", "rising", "cumulative"}:
            raise ValueError("unknown population event rule")
        if self.threshold < 0 or self.sd_floor_hz <= 0:
            raise ValueError("threshold must be nonnegative and floor positive")
        if min(self.signal_window, self.evidence_window) < 1:
            raise ValueError("window sizes must be positive")
        if self.hysteresis < 0 or self.refractory_decisions < 0:
            raise ValueError("hysteresis and refractory must be nonnegative")


class PopulationEventController:
    """Sparse A-event decoder receiving only a frozen DN population rate vector."""
    def __init__(self, baseline: FrozenVectorBaseline, config: PopulationEventConfig,
                 ablated: bool = False):
        self.baseline, self.config, self.ablated = baseline, config, ablated
        self.reset()

    def reset(self) -> None:
        history = max(2 * self.config.signal_window + 1, self.config.evidence_window + 1, 4)
        self.z_history = deque(maxlen=history)
        self.signal_history = deque(maxlen=max(self.config.evidence_window, 1))
        self.refractory = 0
        self.armed = True
        self.suppressions = 0

    def decode(self, rates):
        if self.ablated:
            return None, {
                "signal": None, "evidence": None, "above_threshold": False,
                "refractory_blocked": False, "refractory_remaining": self.refractory,
                "refractory_suppressions": self.suppressions,
                "selected_action": None, "ablated": True,
            }
        z = population_z_vector(
            rates["event_population_rates_hz"], self.baseline, self.config.sd_floor_hz)
        self.z_history.append(z)
        signal = population_signal(self.z_history, self.config.signal, self.config.signal_window)
        if signal is not None:
            self.signal_history.append(signal)
        if signal is None:
            evidence = None
        elif self.config.event_rule == "cumulative":
            evidence = (float(np.mean(self.signal_history))
                        if len(self.signal_history) >= self.config.evidence_window else None)
        else:
            evidence = signal
        above = evidence is not None and evidence > self.config.threshold
        if self.config.event_rule == "level":
            candidate = above
        else:
            candidate = above and self.armed
            if evidence is not None and evidence <= self.config.threshold - self.config.hysteresis:
                self.armed = True
        blocked = candidate and self.refractory > 0
        if blocked:
            self.suppressions += 1
        fire = candidate and not blocked
        if candidate and self.config.event_rule != "level":
            self.armed = False
        if self.refractory:
            self.refractory -= 1
        if fire:
            self.refractory = self.config.refractory_decisions
            if self.config.event_rule == "cumulative":
                self.signal_history.clear()
        state = {
            "signal_family": self.config.signal,
            "event_rule": self.config.event_rule,
            "signal": signal,
            "evidence": evidence,
            "threshold": self.config.threshold,
            "above_threshold": above,
            "armed": self.armed,
            "refractory_blocked": blocked,
            "refractory_remaining": self.refractory,
            "refractory_suppressions": self.suppressions,
            "selected_action": "A" if fire else None,
            "ablated": False,
        }
        return ("A" if fire else None), state


def permute_population_cells(windows, seed: int) -> np.ndarray:
    """Independently permute cell identity in each decision window."""
    values = np.asarray(windows, float)
    if values.ndim != 2:
        raise ValueError("population windows must be two-dimensional")
    rng = np.random.default_rng(seed)
    return np.stack([row[rng.permutation(values.shape[1])] for row in values])


def permute_population_time(windows, seed: int) -> np.ndarray:
    """Permute decision-window order while preserving each population vector."""
    values = np.asarray(windows, float)
    if values.ndim != 2:
        raise ValueError("population windows must be two-dimensional")
    return values[np.random.default_rng(seed).permutation(values.shape[0])]
