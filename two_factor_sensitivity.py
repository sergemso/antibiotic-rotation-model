"""
Two-factor sensitivity analysis: s_esc × s_growth.
Scans a grid of parameter pairs and reports protection factor for each.
"""

import numpy as np
from model import Parameters, ThreeRegionModel


def two_factor_scan():
    """Scan s_esc × s_growth grid and report protection factors."""
    
    s_esc_values = [0, -0.005, -0.010, -0.015, -0.020, -0.030, -0.050]
    s_growth_values = [0, -0.005, -0.010, -0.020, -0.030, -0.050]
    
    print("=" * 80)
    print("Two-factor sensitivity: protection factor for s_esc × s_growth")
    print(f"Fixed: c={Parameters().c}, gamma={Parameters().gamma}, "
          f"m_mig={Parameters().m_mig}, m_inter={Parameters().m_inter}")
    print("=" * 80)
    
    # Header
    header = f"{'s_esc':>10s}"
    for sg in s_growth_values:
        header += f"  {'sg='+str(sg):>10s}"
    print(header)
    print("-" * len(header))
    
    for se in s_esc_values:
        row = f"{se:10.3f}"
        for sg in s_growth_values:
            params = Parameters(s_esc=se, s_growth=sg)
            model = ThreeRegionModel(params)
            
            res_u = model.run_simulation(n_generations=50000, rotation=False)
            res_r = model.run_simulation(n_generations=50000, rotation=True)
            
            ttf_u = model.time_to_failure(res_u['r']) / params.G
            ttf_r = model.time_to_failure(res_r['r']) / params.G
            
            if ttf_u > 0 and ttf_u < 500:
                protection = ttf_r / ttf_u
            elif ttf_r >= 500:
                protection = float('inf')  # rotation prevents failure
            else:
                protection = 0.0  # edge case
            
            if protection == float('inf'):
                row += f"  {'inf':>10s}"
            elif protection >= 1.05:
                row += f"  {protection:10.2f}"
            else:
                row += f"  {protection:10.2f}"
        
        print(row)
    
    # Find where protection > 1.1
    print("\n\nPairs with protection factor > 1.1 (rotation extends trojan life):")
    print("-" * 60)
    found = False
    for se in s_esc_values:
        for sg in s_growth_values:
            params = Parameters(s_esc=se, s_growth=sg)
            model = ThreeRegionModel(params)
            
            res_u = model.run_simulation(n_generations=50000, rotation=False)
            res_r = model.run_simulation(n_generations=50000, rotation=True)
            
            ttf_u = model.time_to_failure(res_u['r']) / params.G
            ttf_r = model.time_to_failure(res_r['r']) / params.G
            
            if ttf_u > 0 and ttf_u < 500:
                protection = ttf_r / ttf_u
                if protection > 1.1:
                    found = True
                    print(f"  s_esc={se:.3f}, s_growth={sg:.3f}: "
                          f"TTF_u={ttf_u:.0f}, TTF_r={ttf_r:.0f}, "
                          f"protection={protection:.2f}x")
            elif ttf_r >= 500 and ttf_u < 500:
                found = True
                print(f"  s_esc={se:.3f}, s_growth={sg:.3f}: "
                      f"TTF_u={ttf_u:.0f}, TTF_r=500+, protection=inf")
    
    if not found:
        print("  None found. Protection does not exceed 1.1 for any tested pair.")


if __name__ == '__main__':
    two_factor_scan()