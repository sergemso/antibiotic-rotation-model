"""
Regional-scale antibiotic rotation and trojan pro-drugs:
A model of reciprocal protection against resistance evolution.

Implements the impulsive difference-equation model described in:
Strebulaev S. (2026) Regional-scale antibiotic rotation and trojan pro-drugs:
a model of reciprocal protection against resistance evolution.

Author: Sergey Strebulaev
Date: 2026
License: CC BY 4.0
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt


@dataclass
class Parameters:
    """Model parameters with default values from Table 3 of the manuscript."""
    
    # Fitness cost of resistance plasmid in absence of antibiotic
    s_growth: float = -0.005
    
    # Horizontal gene transfer coefficient
    c: float = 0.01
    
    # Per-generation immigration rate from outside the region
    m_mig: float = 0.01
    
    # Frequency of resistance allele in external sources
    q_ext: float = 0.15
    
    # Per-generation migration rate between regions
    m_inter: float = 1e-4
    
    # Trojan lethality (fraction of non-escape resistant bacteria killed)
    gamma: float = 0.95
    
    # Escape mutation rate per generation
    mu_esc: float = 1e-7
    
    # Fitness cost of escape mutation
    s_esc: float = -0.005
    
    # Generations between trojan impulses (3 years * 100 gen/year)
    delta_T: int = 300
    
    # Generations per year
    G: int = 100


class ThreeRegionModel:
    """
    Impulsive difference-equation model of resistance allele dynamics
    in a three-region system with staggered antibiotic rotation.
    """
    
    def __init__(self, params: Parameters):
        self.p = params
        
        # Effective selection in absence of antibiotic
        self.s_eff = params.s_growth + params.c
        
        # Selection advantage of resistance under antibiotic pressure
        self.delta_s_ab = 0.20  # typical advantage under active antibiotic
        
        # Initial frequencies
        self.q_init = 0.10  # initial resistance frequency
        self.r_init = params.mu_esc  # initial escape frequency (~mutation rate)
        
    def _rotation_active(self, region: int, generation: int) -> bool:
        """
        Determine if beta-lactams are active in the given region at the given generation.
        
        Rotation schedule (Table 2):
            Region 0 (A): active in first third of each 9-year cycle
            Region 1 (B): active in second third
            Region 2 (C): active in last third
        
        Each full cycle = 9 years * 100 gen/year = 900 generations
        Each phase = 3 years * 100 gen/year = 300 generations
        """
        cycle_position = generation % 900
        active_phase = region * 300
        return active_phase <= cycle_position < active_phase + 300
    
    def _is_trojan_impulse_sync(self, generation: int) -> bool:
        """Check if this generation is a SYNCHRONOUS trojan impulse (all regions)."""
        return generation > 0 and generation % self.p.delta_T == 0
    
    def _is_trojan_impulse_async(self, region: int, generation: int) -> bool:
        """
        Check if this generation is an ASYNCHRONOUS trojan impulse.
        Each region receives trojan at the END of its own active phase.
        
        Region 0: active gens 0-299, trojan at gen 300, 1200, 2100...
        Region 1: active gens 300-599, trojan at gen 600, 1500, 2400...
        Region 2: active gens 600-899, trojan at gen 900, 1800, 2700...
        """
        if generation == 0:
            return False
        # Trojan at end of each region's active phase
        # Active phase for region r: [r*300, (r+1)*300) within each 900-gen cycle
        # End of active phase = (r+1)*300
        cycle_position = generation % 900
        return cycle_position == ((region + 1) * 300) % 900
    
    def _next_q(self, q: float, r: float, active: bool, trojan: bool) -> float:
        """
        Compute next-generation resistance frequency q (Eq. 1, 3).
        """
        if trojan:
            # Trojan impulse (Eq. 3): fraction of non-escape killed
            q_escape = q * r  # escape mutants survive
            q_non_escape = q * (1 - r) * (1 - self.p.gamma)  # non-escape killed
            q_sensitive = (1 - q)  # sensitive unaffected
            total = q_escape + q_non_escape + q_sensitive
            if total > 0:
                return (q_escape + q_non_escape) / total
            return 0.0
        
        # Baseline dynamics (Eq. 1)
        s = self.s_eff
        if active:
            s += self.delta_s_ab
            
        delta = (s * q * (1 - q) + 
                 self.p.m_mig * (self.p.q_ext - q) + 
                 self.p.c * q * (1 - q))
        
        return max(0.0, min(1.0, q + delta))
    
    def _next_r(self, q: float, r: float, r_other: float, trojan: bool) -> float:
        """
        Compute next-generation escape frequency r (Eq. 4, 5).
        """
        if trojan:
            # Enrichment of escape mutants after trojan impulse (Eq. 5)
            if r > 0 or q > 0:
                r_new = r / (r + (1 - r) * (1 - self.p.gamma))
                return max(0.0, min(1.0, r_new))
            return 0.0
        
        # Dynamics without trojan (Eq. 4)
        delta = (self.p.s_esc * r * (1 - r) + 
                 self.p.mu_esc * (1 - r) + 
                 self.p.m_inter * (r_other - r))
        
        return max(0.0, min(1.0, r + delta))
    
    def run_simulation(self, 
                       n_generations: int = 50000,
                       rotation: bool = True,
                       trojan_sync: bool = True) -> dict:
        """
        Run the full three-region simulation.
        
        Args:
            n_generations: number of generations to simulate
            rotation: if True, use staggered rotation; if False, all regions 
                      are synchronised (uniform strategy)
            trojan_sync: if True, trojan applied simultaneously to all regions;
                         if False, trojan applied asynchronously (each region at
                         end of its own active phase)
        
        Returns:
            Dictionary with time series of q and r for each region
        """
        # Initialize arrays
        q = np.full((3, n_generations), np.nan)
        r = np.full((3, n_generations), np.nan)
        
        # Set initial conditions
        for region in range(3):
            q[region, 0] = self.q_init
            r[region, 0] = self.r_init
        
        # Run simulation
        for t in range(1, n_generations):
            for region in range(3):
                # Determine if antibiotic active
                if rotation:
                    active = self._rotation_active(region, t)
                else:
                    # Uniform strategy: all regions synchronised with Region 0
                    active = self._rotation_active(0, t)
                
                # Determine if trojan impulse for this region
                if trojan_sync:
                    trojan = self._is_trojan_impulse_sync(t)
                else:
                    trojan = self._is_trojan_impulse_async(region, t)
                
                # Average escape frequency in other regions
                other_regions = [i for i in range(3) if i != region]
                r_other = np.mean([r[i, t-1] for i in other_regions])
                
                # Update
                q[region, t] = self._next_q(q[region, t-1], r[region, t-1], active, trojan)
                r[region, t] = self._next_r(q[region, t-1], r[region, t-1], r_other, trojan)
        
        return {'q': q, 'r': r, 'generations': np.arange(n_generations)}
    
    def time_to_failure(self, r: np.ndarray, threshold: float = 0.50) -> int:
        """
        Compute time (in generations) until escape frequency exceeds threshold
        in at least two regions.
        
        Returns n_generations if threshold never reached.
        """
        n_generations = r.shape[1]
        for t in range(n_generations):
            if np.sum(r[:, t] > threshold) >= 2:
                return t
        return n_generations


def compare_sync_vs_async(m_inter_values=None):
    """
    Compare synchronous vs. asynchronous trojan at different m_inter values.
    """
    if m_inter_values is None:
        m_inter_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    
    print("=" * 90)
    print("Synchronous vs. Asynchronous Trojan: Protection Factor Comparison")
    print(f"Fixed: s_growth={Parameters().s_growth}, s_esc={Parameters().s_esc}, "
          f"gamma={Parameters().gamma}")
    print("=" * 90)
    print(f"{'m_inter':>10s}  {'Sync TTF':>10s}  {'Async TTF':>10s}  "
          f"{'Sync Prot':>10s}  {'Async Prot':>10s}  {'Async Win':>10s}")
    print("-" * 90)
    
    for m_inter in m_inter_values:
        params = Parameters(m_inter=m_inter)
        model = ThreeRegionModel(params)
        
        # Synchronous trojan
        res_sync_u = model.run_simulation(n_generations=50000, rotation=False, trojan_sync=True)
        res_sync_r = model.run_simulation(n_generations=50000, rotation=True, trojan_sync=True)
        
        ttf_sync_u = model.time_to_failure(res_sync_u['r']) / params.G
        ttf_sync_r = model.time_to_failure(res_sync_r['r']) / params.G
        
        prot_sync = ttf_sync_r / ttf_sync_u if ttf_sync_u > 0 else float('inf')
        
        # Asynchronous trojan
        res_async_u = model.run_simulation(n_generations=50000, rotation=False, trojan_sync=False)
        res_async_r = model.run_simulation(n_generations=50000, rotation=True, trojan_sync=False)
        
        ttf_async_u = model.time_to_failure(res_async_u['r']) / params.G
        ttf_async_r = model.time_to_failure(res_async_r['r']) / params.G
        
        prot_async = ttf_async_r / ttf_async_u if ttf_async_u > 0 else float('inf')
        
        async_wins = prot_async > prot_sync * 1.05  # 5% threshold
        
        sync_str = f"{ttf_sync_r:.0f}" if ttf_sync_r < 500 else "500+"
        async_str = f"{ttf_async_r:.0f}" if ttf_async_r < 500 else "500+"
        prot_sync_str = f"{prot_sync:.2f}x" if prot_sync != float('inf') else "inf"
        prot_async_str = f"{prot_async:.2f}x" if prot_async != float('inf') else "inf"
        
        print(f"{m_inter:10.1e}  {sync_str:>10s}  {async_str:>10s}  "
              f"{prot_sync_str:>10s}  {prot_async_str:>10s}  {str(async_wins):>10s}")
    
    return None


def sensitivity_async_s_esc():
    """Reproduce Table 7: time to trojan failure as function of s_esc (async trojan)."""
    s_esc_values = [0, -0.005, -0.010, -0.020, -0.050]
    results = []
    
    for s_esc in s_esc_values:
        params = Parameters(s_esc=s_esc, m_inter=1e-2)
        model = ThreeRegionModel(params)
        
        res_u = model.run_simulation(n_generations=50000, rotation=False, trojan_sync=False)
        res_r = model.run_simulation(n_generations=50000, rotation=True, trojan_sync=False)
        
        ttf_u = model.time_to_failure(res_u['r'])
        ttf_r = model.time_to_failure(res_r['r'])
        
        # Format TTF
        ttf_u_str = f"{ttf_u / params.G:.0f}" if ttf_u < 50000 else "500+"
        ttf_r_str = f"{ttf_r / params.G:.0f}" if ttf_r < 50000 else "500+"
        
        # Protection factor
        if ttf_u < 50000 and ttf_r < 50000:
            protection = ttf_r / ttf_u
            prot_str = f"{protection:.2f}x"
        elif ttf_u >= 50000 and ttf_r >= 50000:
            prot_str = "---"
        elif ttf_u < 50000 and ttf_r >= 50000:
            prot_str = "inf"
        else:
            prot_str = "---"
        
        results.append((s_esc, ttf_u_str, ttf_r_str, prot_str))
    
    print("\n" + "=" * 80)
    print("Asynchronous Trojan: Sensitivity to s_esc (Table 7)")
    print(f"Fixed: m_inter=1e-2, others default")
    print("=" * 80)
    print(f"{'s_esc':>8s}  {'TTF uniform':>12s}  {'TTF rotation':>12s}  {'Protection':>10s}")
    print("-" * 65)
    
    for s_esc, ttf_u, ttf_r, prot in results:
        print(f"{s_esc:8.3f}  {ttf_u:>12s}  {ttf_r:>12s}  {prot:>10s}")
    
    return results

def plot_async_dynamics():
    """Plot escape dynamics with asynchronous trojan."""
    params = Parameters(m_inter=1e-2)
    model = ThreeRegionModel(params)
    
    # Asynchronous trojan with rotation
    res = model.run_simulation(n_generations=30000, rotation=True, trojan_sync=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for region in range(3):
        ax.plot(res['generations'] / params.G, res['r'][region], 
                label=f'Region {["A","B","C"][region]}', alpha=0.8)
    ax.set_yscale('log')
    ax.set_title('Asynchronous Trojan + Regional Rotation')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Escape frequency $r$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs('out', exist_ok=True)
    plt.tight_layout()
    plt.savefig('out/figure_async_rotation_dynamics.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved: out/figure_async_rotation_dynamics.pdf")


def main():
    """Run all comparisons."""
    compare_sync_vs_async()
    sensitivity_async_s_esc()
    plot_async_dynamics()


if __name__ == '__main__':
    main()