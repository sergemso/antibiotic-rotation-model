"""
DIAGNOSTIC: Single-region test with asynchronous trojan.
This bypasses all multi-region logic and tests the core question:
Does asynchronous trojan application protect against escape mutations
compared to synchronous application?
"""

import numpy as np

# Parameters (same as model.py defaults)
s_growth = -0.005
c = 0.01
m_mig = 0.01
q_ext = 0.15
gamma = 0.95
mu_esc = 1e-7
s_esc = -0.005
delta_T = 300
G = 100
delta_s_ab = 0.20
s_eff = s_growth + c


def single_region_sync_vs_async():
    """
    Simulate ONE region.
    
    Sync: trojan every 300 generations.
    Async: trojan every 900 generations (simulating one region in a 3-region
           async system where trojan rotates among regions).
    
    Key insight: in an async system, a given region receives trojan only
    once every 900 generations (at the end of its own active phase),
    not every 300 generations.
    """
    
    n_generations = 30000
    
    # --- SYNCHRONOUS ---
    q = 0.10
    r = mu_esc
    
    for t in range(1, n_generations):
        trojan = (t > 0 and t % 300 == 0)
        active = True  # always active for uniform strategy
        
        if trojan:
            q_esc = q * r
            q_non_esc = q * (1 - r) * (1 - gamma)
            q_sens = 1 - q
            total = q_esc + q_non_esc + q_sens
            q = (q_esc + q_non_esc) / total if total > 0 else 0.0
            r = r / (r + (1 - r) * (1 - gamma))
        else:
            s = s_eff + delta_s_ab
            delta = s * q * (1 - q) + m_mig * (q_ext - q)
            q = max(0.0, min(1.0, q + delta))
            if q > 1e-10:
                delta_r = s_esc * r * (1 - r) + mu_esc * (1 - r)
                r = max(0.0, min(1.0, r + delta_r))
        
        if r > 0.50:
            ttf_sync = t / G
            break
    else:
        ttf_sync = n_generations / G
    
    # --- ASYNCHRONOUS (trojan every 900 gen, simulating 1 of 3 regions) ---
    q = 0.10
    r = mu_esc
    
    for t in range(1, n_generations):
        # Async: trojan only at t=300, 1200, 2100... (end of this region's phase)
        cycle_pos = t % 900
        trojan = (cycle_pos == 300)  # end of active phase for this region
        
        active = True  # uniform antibiotic strategy
        
        if trojan:
            q_esc = q * r
            q_non_esc = q * (1 - r) * (1 - gamma)
            q_sens = 1 - q
            total = q_esc + q_non_esc + q_sens
            q = (q_esc + q_non_esc) / total if total > 0 else 0.0
            r = r / (r + (1 - r) * (1 - gamma))
        else:
            s = s_eff + delta_s_ab
            delta = s * q * (1 - q) + m_mig * (q_ext - q)
            q = max(0.0, min(1.0, q + delta))
            if q > 1e-10:
                delta_r = s_esc * r * (1 - r) + mu_esc * (1 - r)
                r = max(0.0, min(1.0, r + delta_r))
        
        if r > 0.50:
            ttf_async = t / G
            break
    else:
        ttf_async = n_generations / G
    
    print("=" * 60)
    print("Single-region diagnostic: sync vs async trojan")
    print(f"s_growth={s_growth}, s_esc={s_esc}, gamma={gamma}")
    print("=" * 60)
    print(f"  Synchronous (every 300 gen):  TTF = {ttf_sync:.0f} years")
    print(f"  Asynchronous (every 900 gen): TTF = {ttf_async:.0f} years")
    
    if ttf_async > 2 * ttf_sync:
        print("\n  => Asynchronous application DRAMATICALLY extends trojan lifespan.")
    elif ttf_async > ttf_sync * 1.1:
        print("\n  => Asynchronous application MODERATELY extends trojan lifespan.")
    else:
        print("\n  => No significant difference.")


if __name__ == '__main__':
    single_region_sync_vs_async()