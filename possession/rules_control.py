"""Rule-based control and stoppage inference. No labels or ground truth are read.

All thresholds below are physical/duration assumptions chosen before match generation.
This model is offline: confirmation backdates only the observed candidate run.
"""
import numpy as np

PARAMETERS = {
    'control_distance_m': 2.5, 'control_advantage_m': 0.75,
    'control_dwell_s': 0.4, 'max_control_ball_speed_mps': 12.0,
    'close_contact_m': 1.2, 'outside_margin_m': 0.3, 'outside_dwell_s': 0.2,
    'motion_window_s': 0.48, 'settled_speed_mps': 0.6,
    'settled_far_distance_m': 3.0, 'settled_far_dwell_s': 1.5,
    'settled_restart_dwell_s': 0.6, 'restart_speed_mps': 1.5,
    'restart_near_m': 4.0, 'restart_motion_dwell_s': 0.12,
    'missing_abstain_s': 2.0,
}

def duration_frames(seconds, fps):
    return max(1, int(round(seconds * fps)))

def motion_speed(ball, fps, pitch, window_seconds):
    """Endpoint displacement over a short fully observed window; never bridge missing data."""
    lag = duration_frames(window_seconds, fps)
    speed = np.full(len(ball), np.nan)
    valid = np.isfinite(ball).all(axis=1)
    if len(ball) <= lag:
        return speed
    cumulative = np.r_[0, np.cumsum(valid)]
    complete = cumulative[lag+1:] - cumulative[:-lag-1] == lag+1
    displacement = np.linalg.norm((ball[lag:]-ball[:-lag])*np.array(pitch), axis=1)
    speed[lag:] = np.where(complete, displacement / (lag / fps), np.nan)
    return speed

def infer_control(ball, nearest, events, windows, fps=25, pitch=(105.3,68.0), parameters=None):
    """Return labels, per-frame evidence, and an audit trail; windows are inclusive.

    No team change follows a one-frame touch. Loose/contested balls retain last control.
    Missing observations alone never create a stoppage. Confirmed stoppages persist until
    observed motion near a player supports a restart. Long observation gaps abstain.
    """
    p = PARAMETERS if parameters is None else {**PARAMETERS, **parameters}
    n = len(ball)
    if nearest.shape != (n,2):
        raise ValueError('Expected one nearest-distance pair per frame')
    observed = np.isfinite(ball).all(axis=1)
    speed = motion_speed(ball, fps, pitch, p['motion_window_s'])
    physical = ball * np.array(pitch)
    margin = p['outside_margin_m']
    outside = observed & ((physical < -margin).any(axis=1) |
                          (physical > np.array(pitch)+margin).any(axis=1))
    inbounds = observed & (physical >= 0).all(axis=1) & (physical <= np.array(pitch)).all(axis=1)
    restart_context = np.zeros(n, dtype=bool)
    for event in events:
        if (event.get('code') or '').strip().upper() == 'SET PIECES':
            a,b = max(0,event['start_frame']), min(n,event['end_frame'])
            if b > a:
                restart_context[a:b] = True
    candidate = np.full(n,-1,dtype=np.int8)
    for team in (0,1):
        own, other = nearest[:,team], nearest[:,1-team]
        # An absent opposing track cannot establish a distance advantage.
        advantage = np.zeros(n)
        both = np.isfinite(own) & np.isfinite(other)
        advantage[both] = other[both]-own[both]
        eligible = (inbounds & both & (own <= p['control_distance_m']) &
                    (advantage >= p['control_advantage_m']) &
                    ((speed <= p['max_control_ball_speed_mps']) | (own <= p['close_contact_m'])))
        candidate[eligible] = team
    far = nearest.min(axis=1) >= p['settled_far_distance_m']
    # Require a visible player to avoid treating wholly absent tracking as a stoppage.
    some_player = np.isfinite(nearest).any(axis=1)
    settled = observed & np.isfinite(speed) & (speed <= p['settled_speed_mps'])
    labels = np.full(n,'X',dtype='<U1')
    dead_evidence = np.zeros(n,dtype=np.int8)  # 1 outside, 2 stationary far, 3 stationary in restart context
    restart_evidence = np.zeros(n,dtype=bool)
    audit = []
    dwell = duration_frames(p['control_dwell_s'],fps)
    outside_dwell = duration_frames(p['outside_dwell_s'],fps)
    far_dwell = duration_frames(p['settled_far_dwell_s'],fps)
    tagged_dwell = duration_frames(p['settled_restart_dwell_s'],fps)
    motion_dwell = duration_frames(p['restart_motion_dwell_s'],fps)
    missing_limit = duration_frames(p['missing_abstain_s'],fps)
    for _,start,end in windows:
        state = 'X'
        pending = -1
        pending_start = start
        pending_count = outside_count = far_count = tagged_count = moving_count = missing_count = 0
        for i in range(start,end+1):
            previous = state
            missing_count = 0 if observed[i] else missing_count+1
            outside_count = outside_count+1 if outside[i] else 0
            far_count = far_count+1 if settled[i] and far[i] and some_player[i] else 0
            tagged_count = tagged_count+1 if settled[i] and restart_context[i] else 0
            moving = (inbounds[i] and np.isfinite(speed[i]) and speed[i] >= p['restart_speed_mps']
                      and nearest[i].min() <= p['restart_near_m'])
            moving_count = moving_count+1 if moving else 0
            dead_kind, dead_start = 0, i
            if outside_count == outside_dwell:
                dead_kind, dead_start = 1,i-outside_count+1
            elif far_count == far_dwell:
                dead_kind, dead_start = 2,i-far_count+1
            elif tagged_count == tagged_dwell:
                dead_kind, dead_start = 3,i-tagged_count+1
            if dead_kind:
                state = 'D'
                labels[dead_start:i+1] = 'D'
                dead_evidence[dead_start:i+1] = dead_kind
                pending, pending_count, moving_count = -1,0,0
                if previous != 'D':
                    audit.append({'frame':dead_start,'confirmed_frame':i,'from':previous,'to':'D',
                                  'reason':{1:'sustained_outside',2:'settled_far_from_players',3:'settled_in_restart_context'}[dead_kind]})
            elif state == 'D':
                if moving_count >= motion_dwell:
                    onset = i-moving_count+1
                    # Restart ownership uses only measured nearby tracks, never a tag's team.
                    team = int(nearest[onset].argmin())
                    own, other = nearest[onset,team],nearest[onset,1-team]
                    if np.isfinite(other) and other-own >= p['control_advantage_m']:
                        state = 'HA'[team]
                    else:
                        state = 'X'
                    labels[onset:i+1] = state
                    restart_evidence[onset:i+1] = True
                    audit.append({'frame':onset,'confirmed_frame':i,'from':'D','to':state,'reason':'observed_restart_motion'})
                    pending,pending_count = -1,0
                    # A new stop requires a fresh run of supporting observations.
                    outside_count = far_count = tagged_count = 0
            else:
                team = int(candidate[i])
                if team < 0:
                    pending,pending_count = -1,0
                else:
                    if team != pending:
                        pending,pending_start,pending_count = team,i,1
                    else:
                        pending_count += 1
                    if pending_count == dwell and state != 'HA'[team]:
                        state = 'HA'[team]
                        labels[pending_start:i+1] = state
                        audit.append({'frame':pending_start,'confirmed_frame':i,'from':previous,
                                      'to':state,'reason':'sustained_close_control'})
            labels[i] = 'X' if missing_count >= missing_limit else state
    evidence = {'control_candidate':candidate,'stoppage_evidence':dead_evidence,
                'restart_evidence':restart_evidence,'restart_context':restart_context,'motion_speed':speed}
    return labels,evidence,audit
