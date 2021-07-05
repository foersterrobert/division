import math
import decimal
import random

def devision(i1, i2):
    h1 = AktivierungsFunktion(i1 * 0.000001 - 0.000001)
    h2 = AktivierungsFunktion(i2 * 0.000001 - 0.000001)
    output = AktivierungsFunktion(h1 * 33.3333 + h2 * -33.3333 - 3.912023)
    return output

def AktivierungsFunktion(x):
    if x <= 0:
        return 1.359140915 * math.exp(x - 1)
    elif x > 15:
        return 1 - 1/(109.0858178 * x - 1403.359435)
    else:
        return 0.03 * math.log(1000000 * x + 1) + 0.5

# print(devision(20, 5)) # 0.04 -> 20 / 5 = 4
# print(devision(40, 8)) # 0.05 -> 40 / 8 = 5

print(devision(1, 18.057) * 18.057 * 100) # 0.05 -> 40 / 8 = 5
# print(devision(2, 8)) # 0.0025 -> 2 / 8 = 0,25
# print(devision(1.02, 6.55)) # 0.0015572 -> 1,02 / 6,55 = 0,15572

