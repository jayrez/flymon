"""Frozen Experiment 10 motor arbitration and visual-only novelty metrics."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,math
import numpy as np
from .spatiotemporal import spatial_grid

@dataclass(frozen=True)
class ExplorationConfig:
    steering_multiplier:float;steering_floor_hz:float;steering_side:str
    forward_multiplier:float;forward_floor_hz:float
    backward_multiplier:float;backward_floor_hz:float
    cooldown_decisions:int=1

class FrozenExplorationController:
    """One-action competition using only frozen-baseline motor-DN rates."""
    def __init__(self,steering_baseline,locomotion_baseline,config:ExplorationConfig,enable_down=False):
        self.sb,self.lb,self.c,self.enable_down=steering_baseline,locomotion_baseline,config,enable_down
        self.ss=max(math.sqrt(max(0,steering_baseline.difference_variance_hz2)),config.steering_floor_hz)
        self.fs=max(math.sqrt(max(0,locomotion_baseline.dng100_variance_hz2)),config.forward_floor_hz)
        self.bs=max(math.sqrt(max(0,locomotion_baseline.mdn_variance_hz2)),config.backward_floor_hz);self.cooldown=0
    def decode(self,rates):
        anatomical=(rates["DNa02_L"]-self.sb.left_mean_hz)-(rates["DNa02_R"]-self.sb.right_mean_hz)
        steer=anatomical if self.c.steering_side=="L" else -anatomical;forward=rates["DNg100"]-self.lb.dng100_mean_hz;backward=rates["MDN"]-self.lb.mdn_mean_hz
        thresholds={"LEFT":self.c.steering_multiplier*self.ss,"RIGHT":self.c.steering_multiplier*self.ss,"UP":self.c.forward_multiplier*self.fs,"DOWN":self.c.backward_multiplier*self.bs}
        raw={"LEFT":steer,"RIGHT":-steer,"UP":forward}
        if self.enable_down:raw["DOWN"]=backward
        excess={a:raw[a]/thresholds[a]-1 for a in raw if raw[a]>thresholds[a]};order=("LEFT","RIGHT","UP","DOWN")
        chosen=max(order,key=lambda a:(excess.get(a,-1),-order.index(a))) if excess else None
        if self.cooldown:action=None;reason="cooldown";self.cooldown-=1
        else:action=chosen;reason="threshold" if chosen else "deadband";self.cooldown=self.c.cooldown_decisions if action else 0
        return action,{"steering_signal_hz":steer,"forward_delta_hz":forward,"backward_delta_hz":backward,"scales_hz":{"steering":self.ss,"forward":self.fs,"backward":self.bs},
          "thresholds_hz":thresholds,"normalized_excess":excess,"raw_action":chosen,"selected_action":action,"reason":reason,"down_enabled":self.enable_down}


class FrozenInterfaceController:
    """Experiment 10 directions with optional normalized DOWN and phasic A."""
    def __init__(self,direction,down=None,event=None):self.direction,self.down,self.event=direction,down,event
    def decode(self,rates):
        ea,es=self.event.decode(rates) if self.event else (None,None);da,ds=self.direction.decode(rates);xa,xs=self.down.decode(rates) if self.down else (None,None)
        if xs and xa and xs["normalized_excess"]>max(ds["normalized_excess"].values(),default=-math.inf):da="DOWN"
        action=ea if ea else da
        return action,{"selected_action":action,"arbitration":"A-event-first; normalized direction excess","direction":ds,"down":xs,"event":es}

def visual_metrics(frame,initial,vector,initial_vector,previous):
    _,grid=spatial_grid(frame,__import__('flymon.motor',fromlist=['E7_CONFIG']).E7_CONFIG);_,igrid=spatial_grid(initial,__import__('flymon.motor',fromlist=['E7_CONFIG']).E7_CONFIG)
    quant=np.rint(np.clip(grid,0,1)*15).astype(np.uint8);coarse=hashlib.sha256(quant.tobytes()).hexdigest()
    return {"frame_sha256":hashlib.sha256(frame.tobytes()).hexdigest(),"coarse_state_sha256":coarse,
      "frame_to_previous_mae":0.0 if previous is None else float(np.mean(np.abs(frame[...,:3].astype(float)-previous[...,:3]))/255),
      "frame_to_initial_mae":float(np.mean(np.abs(frame[...,:3].astype(float)-initial[...,:3]))/255),
      "coarse_to_initial_mae":float(np.mean(np.abs(grid-igrid))),"encoder_sha256":hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest(),
      "encoder_to_initial_rms":float(np.sqrt(np.mean((vector.astype(float)-initial_vector.astype(float))**2)))}
