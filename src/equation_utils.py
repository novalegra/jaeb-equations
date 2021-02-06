import math


def jaeb_basal_equation(tdd, carbs):
    """ Basal equation fitted from Jaeb data """
    a = -0.36542
    b = -0.0019846
    return math.exp(a) * tdd * math.exp(b * carbs)


def traditional_basal_equation(tdd):
    """ Traditional basal equation """
    a = 0.5013
    return a * tdd


def jaeb_isf_equation(tdd, bmi):
    """ ISF equation fitted from Jaeb data """
    a = 11.612
    b = -0.47864
    c = -1.9088
    return math.exp(a) * (tdd ** b) * (bmi ** c)


def traditional_isf_equation(tdd):
    """ Traditional ISF equation """
    a = 1900.4
    return a / tdd


def jaeb_icr_equation(tdd, carbs):
    """ ICR equation fitted from Jaeb data """
    a = 0.32798
    b = 61.542
    c = -0.68226
    return (a * carbs + b) * (tdd ** c)


def traditional_icr_equation(tdd):
    """ Traditional ICR equation """
    a = 295.3
    return a / tdd
