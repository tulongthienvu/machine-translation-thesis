import math
import time
def as_minutes(seconds):
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)

def time_since(since, percent):
    now = time.time()
    seconds = now - since
    elapsed_seconds = seconds / (percent)
    rest_seconds = elapsed_seconds - seconds
    return '%s (- %s)' % (as_minutes(seconds), as_minutes(rest_seconds))
