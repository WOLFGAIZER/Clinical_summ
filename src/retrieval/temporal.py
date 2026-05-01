def temporal_filter(events, target_time=None, window=2):
    """
    Filters events based on time proximity
    Assumes timestamps like '2026-01-01' or day index
    """

    if target_time is None:
        return events

    filtered = []

    for event, time in events:
        try:
            t = int(time)  # if numeric
            if abs(t - target_time) <= window:
                filtered.append((event, time))
        except:
            filtered.append((event, time))  # fallback

    return filtered