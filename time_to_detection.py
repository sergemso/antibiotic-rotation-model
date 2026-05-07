"""
Alternative metric: time to detection at low escape frequencies.
Tests whether rotation delays early-stage escape, even if fixation time is the same.
"""

import numpy as np
from model import Parameters, ThreeRegionModel


def time_to_detection(r_array, threshold=0.01, G=100):
    """
    Time (in years) until escape frequency exceeds threshold
    in at least two regions.
    """
    n_generations = r_array.shape[1]
    for t in range(n_generations):
        if np.sum(r_array[:, t] > threshold) >= 2:
            return t / G
    return n_generations / G


def compare_detection_times():
    """Compare time to detection (1%) and time to fixation (50%)."""
    
    print("=" * 80)
    print("Comparison of detection vs. fixation metrics")
    print(f"Parameters: s_growth={Parameters().s_growth}, "
          f"s_esc={Parameters().s_esc}, gamma={Parameters().gamma}")
    print("=" * 80)
    
    thresholds = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
    
    params = Parameters()
    model = ThreeRegionModel(params)
    
    res_u = model.run_simulation(n_generations=50000, rotation=False)
    res_r = model.run_simulation(n_generations=50000, rotation=True)
    
    print(f"\n{'Threshold':>10s}  {'TTD uniform':>12s}  {'TTD rotation':>12s}  "
          f"{'Delay (rot - uni)':>16s}  {'Ratio':>8s}")
    print("-" * 70)
    
    for thresh in thresholds:
        ttd_u = time_to_detection(res_u['r'], threshold=thresh, G=params.G)
        ttd_r = time_to_detection(res_r['r'], threshold=thresh, G=params.G)
        delay = ttd_r - ttd_u
        ratio = ttd_r / ttd_u if ttd_u > 0 else float('inf')
        
        print(f"{thresh:10.3f}  {ttd_u:12.1f}  {ttd_r:12.1f}  "
              f"{delay:16.1f}  {ratio:8.2f}x")
    
    # Also check: when does r first become distinguishable between strategies?
    print(f"\n\nEarliest generation where |r_rot - r_uni| > 0.001 in any region:")
    r_u = res_u['r']
    r_r = res_r['r']
    for t in range(r_u.shape[1]):
        max_diff = np.max(np.abs(r_r[:, t] - r_u[:, t]))
        if max_diff > 0.001:
            print(f"  Generation {t} (year {t/params.G:.1f}): max diff = {max_diff:.6f}")
            break
    else:
        print("  Never exceeds 0.001 within simulation horizon.")


if __name__ == '__main__':
    compare_detection_times()