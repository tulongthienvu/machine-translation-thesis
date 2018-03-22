import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks as regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)