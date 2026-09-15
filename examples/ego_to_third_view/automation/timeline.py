"""Duration-safe reference preparation and monotone event time mapping (CPU only)."""
from __future__ import annotations

import math
from .contracts import number, validate_events


def reference_frame_count(output_frames):
    """Pinned H3 VAE accepts 17*n+5: ceil rather than floor the input reference."""
    if isinstance(output_frames, bool) or not isinstance(output_frames, int) or output_frames < 5:
        raise ValueError('Expected integer frame count >=5')
    return 5 + 17 * math.ceil((output_frames - 5) / 17)


def event_anchors(sim, dit, frames, fps, min_confidence=.7):
    duration = frames / fps
    validate_events(sim, duration)
    validate_events(dit, duration)
    by_id = {e['id']: e for e in dit}
    result = [(0., 0.)]
    for event in sim:
        other = by_id.get(event['id'])
        if other is None:
            raise ValueError(f'Missing DiT event {event["id"]}; request review, do not invent a match')
        if min(event['confidence'], other['confidence']) < min_confidence:
            raise ValueError(f'Low confidence event {event["id"]}')
        x, y = event['time_s'] * fps, other['time_s'] * fps
        if x == 0 and y == 0:
            continue
        if x >= frames-1 or y >= frames-1:
            if x == y == frames-1:
                continue
            raise ValueError('Nonterminal events must precede the last frame')
        if (x,y)!=result[-1]:result.append((x, y))
    result.append((float(frames-1), float(frames-1)))
    validate_anchors(result, frames)
    return result


def validate_anchors(anchors, frames):
    if len(anchors) < 2 or tuple(anchors[0]) != (0,0) or tuple(anchors[-1]) != (frames-1,frames-1):
        raise ValueError('Mapping must preserve first and final frames')
    for i, (x,y) in enumerate(anchors):
        number(x,'output anchor',0,frames-1)
        number(y,'source anchor',0,frames-1)
        if i and (x <= anchors[i-1][0] or y <= anchors[i-1][1]):
            raise ValueError('Correspondences must be strictly monotone; cannot reorder/missing actions')


def pchip_map(anchors, frames, min_speed=.2, max_speed=3.):
    """Fritsch-Carlson/PCHIP using only Python, returning source positions and indices."""
    validate_anchors(anchors, frames)
    x,y = map(list,zip(*anchors))
    h = [b-a for a,b in zip(x,x[1:])]
    d = [(b-a)/v for a,b,v in zip(y,y[1:],h)]
    n = len(x)
    if n == 2:
        m = [d[0], d[0]]
    else:
        m = [0.]*n
        for i in range(1,n-1):
            w1,w2 = 2*h[i]+h[i-1],h[i]+2*h[i-1]
            m[i] = (w1+w2)/(w1/d[i-1]+w2/d[i])
        for index,a,b,c,e in [(0,h[0],h[1],d[0],d[1]),(-1,h[-1],h[-2],d[-1],d[-2])]:
            slope = ((2*a+b)*c-a*e)/(a+b)
            m[index] = max(0., min(3*c, slope))
    positions=[]; segment=0
    for frame in range(frames):
        while segment < n-2 and frame > x[segment+1]:
            segment+=1
        t=(frame-x[segment])/h[segment]
        positions.append((2*t**3-3*t**2+1)*y[segment] + (t**3-2*t**2+t)*h[segment]*m[segment]
                         +(-2*t**3+3*t**2)*y[segment+1]+(t**3-t**2)*h[segment]*m[segment+1])
    speeds=[b-a for a,b in zip(positions,positions[1:])]
    if min(speeds) < min_speed or max(speeds) > max_speed:
        raise ValueError(f'Unsafe retime speed range {min(speeds):.3f}..{max(speeds):.3f}; regenerate or review')
    indices=[int(math.floor(v+.5)) for v in positions]
    if any(b<a for a,b in zip(indices,indices[1:])):
        raise ValueError('Time reversal')
    return {'positions':positions,'source_frames':indices,'anchors':anchors,
            'repeat_transitions':sum(a==b for a,b in zip(indices,indices[1:])),
            'speed_range':[min(speeds),max(speeds)],
            'method':'monotone PCHIP, nearest source frame; no synthetic interpolation'}
