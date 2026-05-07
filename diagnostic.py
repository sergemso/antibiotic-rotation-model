"""
Diagnostic: find m_inter threshold where rotation protection emerges.

Tests m_inter from 1e-5 to 1e-1 and reports time-to-failure and protection factor
for both uniform and rotation strategies.
"""

import numpy as np
from model import Parameters, ThreeRegionModel


def diagnostic_m_inter():
    """Sweep m_inter to find where protection factor exceeds 1.0."""
    m_inter_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    
    print("=" * 80)
    print("Diagnostic: protection factor vs m_inter")
    print(f"Fixed: s_growth={Parameters().s_growth}, c={Parameters().c}, "
          f"gamma={Parameters().gamma}, s_esc={Parameters().s_esc}")
    print("=" * 80)
    print(f"{'m_inter':>10s}  {'TTF uniform':>12s}  {'TTF rotation':>12s}  "
          f"{'Protection':>10s}  {'Note':>20s}")
    print("-" * 80)
    
    for m_inter in m_inter_values:
        params = Parameters(m_inter=m_inter)
        model = ThreeRegionModel(params)
        
        res_u = model.run_simulation(n_generations=50000, rotation=False)
        res_r = model.run_simulation(n_generations=50000, rotation=True)
        
        ttf_u = model.time_to_failure(res_u['r']) / params.G
        ttf_r = model.time_to_failure(res_r['r']) / params.G
        
        if ttf_u > 0:
            protection = ttf_r / ttf_u
        else:
            protection = float('inf')
        
        # Determine note
        if ttf_u >= 500 or ttf_r >= 500:
            note = "no failure in sim"
        elif protection > 1.5:
            note = "protection emerges"
        elif protection > 1.05:
            note = "weak protection"
        else:
            note = "no protection"
        
        print(f"{m_inter:10.1e}  {ttf_u:12.0f}  {ttf_r:12.0f}  "
              f"{protection:10.2f}x  {note:>20s}")


def diagnostic_s_esc_with_higher_m_inter():
    """Re-run sensitivity_s_esc with m_inter=1e-2 to see if protection emerges."""
    s_esc_values = [0, -0.005, -0.010, -0.020, -0.050]
    
    print("\n" + "=" * 80)
    print("Diagnostic: sensitivity to s_esc with m_inter=1e-2")
    print("=" * 80)
    print(f"{'s_esc':>8s}  {'TTF uniform':>12s}  {'TTF rotation':>12s}  "
          f"{'Protection':>10s}")
    print("-" * 60)
    
    for s_esc in s_esc_values:
        params = Parameters(s_esc=s_esc, m_inter=1e-2)
        model = ThreeRegionModel(params)
        
        res_u = model.run_simulation(n_generations=50000, rotation=False)
        res_r = model.run_simulation(n_generations=50000, rotation=True)
        
        ttf_u = model.time_to_failure(res_u['r']) / params.G
        ttf_r = model.time_to_failure(res_r['r']) / params.G
        
        protection = ttf_r / ttf_u if ttf_u > 0 else float('inf')
        
        print(f"{s_esc:8.3f}  {ttf_u:12.0f}  {ttf_r:12.0f}  "
              f"{protection:10.2f}x")


if __name__ == '__main__':
    diagnostic_m_inter()
    diagnostic_s_esc_with_higher_m_inter()