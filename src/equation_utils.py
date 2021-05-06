import math


def jaeb_basal_equation(tdd, carbs):
    """ Basal equation fitted from Jaeb data """
    a = -0.4554
    b = -0.0015202
    return math.exp(a) * tdd * math.exp(b * carbs)


def traditional_basal_equation(tdd):
    """ Traditional basal equation with constants fit to Jaeb dataset """
    a = 0.5086
    return a * tdd


def traditional_constants_basal_equation(tdd):
    """ Traditional basal equation with constants from ACE consensus """
    a = 0.5
    return a * tdd


def jaeb_isf_equation(tdd, bmi):
    """ ISF equation fitted from Jaeb data """
    a = 11.458
    b = -0.41612
    c = -1.9408
    return math.exp(a) * (tdd ** b) * (bmi ** c)


def traditional_isf_equation(tdd):
    """ Traditional ISF equation with constants fit to Jaeb dataset """
    a = 1800.8
    return a / tdd


def traditional_constants_isf_equation(tdd):
    """ Traditional ISF equation with constants from ACE consensus """
    a = 1700
    return a / tdd


def jaeb_icr_equation(tdd, carbs):
    """ ICR equation fitted from Jaeb data """
    a = 0.39556
    b = 62.762
    c = -0.71148
    return (a * carbs + b) * (tdd ** c)


def traditional_icr_equation(tdd):
    """ Traditional ICR equation with constants fit to Jaeb dataset """
    a = 308.76
    return a / tdd


def traditional_constants_icr_equation(tdd):
    """ Traditional ICR equation with constants from ACE consensus """
    a = 450
    return a / tdd
