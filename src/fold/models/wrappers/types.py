from finml_utils.enums import ParsableEnum


class ClassWeightingStrategy(ParsableEnum):
    none = "none"
    balanced = "balanced"
    balanced_sqrt = "balanced_sqrt"
