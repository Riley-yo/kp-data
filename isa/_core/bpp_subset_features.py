from fractions import Fraction
from functools import reduce
from math import gcd, lcm, sqrt


FEATURE_NAMES = (
    'subset_smallest_prefix_count',
    'subset_smallest_prefix_fraction',
    'subset_largest_prefix_count',
    'subset_largest_prefix_fraction',
    'subset_feasible_fraction',
    'subset_exact_fill_fraction_of_feasible',
    'subset_feasible_load_mean_ratio',
    'subset_feasible_load_std_ratio',
    'subset_feasible_cardinality_mean_ratio',
    'subset_feasible_cardinality_std_ratio',
    'subset_reachable_positive_load_fraction',
    'subset_maximum_load_ratio',
)


def exact_lattice(data):
    values = [Fraction(str(data['capacity']))]
    values.extend(Fraction(str(value)) for value in data['sizes'])
    if len(values) < 2 or values[0] <= 0:
        raise ValueError('A positive capacity and nonempty item list are required')
    if any(value <= 0 or value > values[0] for value in values[1:]):
        raise ValueError('Every item must have positive size no greater than capacity')
    denominator = lcm(*(value.denominator for value in values))
    integers = [int(value * denominator) for value in values]
    divisor = reduce(gcd, integers)
    return [value // divisor for value in integers[1:]], integers[0] // divisor


def subset_features(data):
    sizes, capacity = exact_lattice(data)
    n = len(sizes)
    counts = [0] * (capacity + 1)
    cardinality_sum = [0] * (capacity + 1)
    cardinality_squared_sum = [0] * (capacity + 1)
    counts[0] = 1
    for size in sizes:
        for load in range(capacity - size, -1, -1):
            if counts[load]:
                target = load + size
                counts[target] += counts[load]
                cardinality_sum[target] += cardinality_sum[load] + counts[load]
                cardinality_squared_sum[target] += (
                    cardinality_squared_sum[load] + 2 * cardinality_sum[load] + counts[load]
                )
    feasible = sum(counts) - 1
    load_sum = sum(load * counts[load] for load in range(1, capacity + 1))
    load_squared_sum = sum(load * load * counts[load] for load in range(1, capacity + 1))
    size_sum = sum(cardinality_sum)
    size_squared_sum = sum(cardinality_squared_sum)
    reachable = [load for load in range(1, capacity + 1) if counts[load]]
    prefix_counts = []
    for reverse in (False, True):
        total, count = 0, 0
        for size in sorted(sizes, reverse=reverse):
            if total + size > capacity:
                break
            total += size
            count += 1
        prefix_counts.append(count)
    small, large = prefix_counts
    values = (
        small,
        float(Fraction(small, n)),
        large,
        float(Fraction(large, n)),
        float(Fraction(feasible, (1 << n) - 1)),
        float(Fraction(counts[capacity], feasible)),
        float(Fraction(load_sum, feasible * capacity)),
        sqrt(float(Fraction(load_squared_sum * feasible - load_sum ** 2,
                            (feasible * capacity) ** 2))),
        float(Fraction(size_sum, feasible * n)),
        sqrt(float(Fraction(size_squared_sum * feasible - size_sum ** 2,
                            (feasible * n) ** 2))),
        float(Fraction(len(reachable), capacity)),
        float(Fraction(reachable[-1], capacity)),
    )
    return dict(zip(FEATURE_NAMES, values))
