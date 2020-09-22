from enum import IntFlag


class CurveType(IntFlag):
    """
    Enum representing curve type
    """

    oval = 1
    wigner_caustic = 2