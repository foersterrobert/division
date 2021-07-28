import math

def divisionHard(i1, i2):
    h1 = AktivierungsFunktionHard(i1 * 0.000001 - 0.000001)
    h2 = AktivierungsFunktionHard(i2 * 0.000001 - 0.000001)
    output = AktivierungsFunktionHard(h1 * 33.3333 + h2 * -33.3333 - 3.912023)
    return output

def AktivierungsFunktionHard(x):
    if x <= 0:
        return 1.359140915 * math.exp(x - 1)
    elif x > 15:
        return 1 - 1/(109.0858178 * x - 1403.359435)
    else:
        return 0.03 * math.log(1000000 * x + 1) + 0.5

