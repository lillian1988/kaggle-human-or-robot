# encoding: UTF-8


def tweak_min(x):
    try:
        return x[0]
    except:
        return 0


def tweak_max(x):
    try:
        return x[-1]
    except:
        return 0
