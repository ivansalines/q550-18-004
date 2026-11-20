"""Support-exit right-censoring for v18.38 original Q550 trajectories.

An unobserved exit is NOT a confirmed infinite residence time.
No right-censored trajectory is assigned a fictitious exit at window end.
"""
import math


def checked_support_horizon(status,exit_time,window_end):
    if not math.isfinite(window_end) or window_end<=0.:
        raise ValueError('Invalid observation window')
    if status=='CROSSED_AFTER_START':
        if exit_time is None or not math.isfinite(exit_time) or not (0.<exit_time<=window_end):
            raise ValueError('Invalid observed crossing time')
        return float(exit_time)
    if status=='NO_EXIT_OBSERVED':
        if exit_time is not None:
            raise ValueError('Censored trial must not carry an exit time')
        return float(window_end)
    # Includes OUTSIDE_AT_START and all unknown labels: cannot certify admission.
    raise RuntimeError('Invalid initial diagnostic support or exit status: '+str(status))


def common_inside_time(horizons,window_end):
    if not horizons or any(not math.isfinite(t) or not 0.<t<=window_end for t in horizons):
        raise ValueError('Invalid support horizons')
    return .5*min(horizons)


def summarize_exits(records,horizons):
    if not records or len(records)!=len(horizons):
        raise ValueError('Unmatched support exit records')
    observed=[float(t) for status,t in records if status=='CROSSED_AFTER_START']
    censored=sum(status=='NO_EXIT_OBSERVED' for status,t in records)
    if len(observed)+censored!=len(records):
        raise ValueError('Unrecognized exit status')
    return dict(observed_exit_count=len(observed),right_censored_count=censored,
        min_observed_exit_delta_xi=min(observed) if observed else '',
        max_observed_exit_delta_xi=max(observed) if observed else '')
