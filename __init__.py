# Делает папку с алгоритмами полноценным Python-модулем
from .ahp import calculate_ahp_weights, build_matrix_from_comparisons
from .compatibility import compute_compatibility

__all__ = ['calculate_ahp_weights', 'build_matrix_from_comparisons', 'compute_compatibility']
