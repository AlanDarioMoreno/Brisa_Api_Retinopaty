


def classDescription(clase):
    switch = {
        0: ["TIENE RETINOPATÍA", "LEVE"],
        1: ["TIENE RETINOPATÍA", "MODERADA"],
        2: ["NO TIENE RETINOPATÍA", "NO_DR"],
        3: ["TIENE RETINOPATÍA", "PROLIFERADA"],
        4: ["TIENE RETINOPATÍA", "SEVERA"]
    }
    return switch.get(clase, ["Clase no válida", "Descripción no válida"])