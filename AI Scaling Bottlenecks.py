"""
AI Scaling Bottlenecks Simulator

Simulates large AI model training across compute, memory, budget, energy, data availability,
and network/interconnect constraints.

Inspired by:
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Chinchilla: Compute-Optimal LLMs" (2022)
- MLPerf HPC benchmarks for network efficiency

DISCLAIMER:
This simulator implements known AI scaling laws heuristically to model compute, memory,
data, cost, thermal, verification, and interconnect bottlenecks. While grounded in
established scaling research, it has not been empirically validated against real-world
large-scale training data. Therefore, the framework provides conceptual and predictive
insights for exploratory analysis rather than confirmed empirical results.
"""

import math
from typing import List, Dict

# ----------------------------
# 1. MODEL CLASS
# ----------------------------
class EnhancedScalingModel:
    """
    AI Scaling Bottlenecks Simulator

    All times are in seconds unless otherwise noted.
    
    Heuristic assumptions:
    - FLOPs, memory bandwidth, cost, and energy growth rates are parametric estimates.
    - Data scarcity, verification, and interconnect penalties are modeled conceptually.
    """

    def __init__(self) -> None:
        # Hardware & growth parameters
        self.compute_doubles_per_year: float = 2.0
        self.memory_bandwidth_growth: float = 1.4
        self.cost_reduction_per_year: float = 0.7
        self.energy_efficiency_gain: float = 0.85

        # Base values for year 2020
        self.base_compute_flops: float = 1e15
        self.base_bandwidth_bytes: float = 1e12
        self.base_cost_per_flop: float = 1e-17
        self.base_budget_dollars: float = 1e8
        self.base_energy_per_flop: float = 1e-9

        # System utilization
        self.compute_utilization: float = 0.5
        self.memory_utilization: float = 0.6
        self.ops_per_byte_low: int = 10
        self.ops_per_byte_high: int = 100
        self.datacenter_power_watts: float = 50e6

        # Workload scaling
        self.workload_growth_rate: float = 4.0
        self.base_workload_flops: float = 1e23
        self.verification_coefficient: float = 1e-10
        self.verification_exponent: float = 1.5
        self.model_size_exponent: float = 1.2
        self.budget_growth_rate: float = 1.5

        # Data wall
        self.base_data_tokens: float = 1e13
        self.data_growth_rate: float = 1.15
        self.chinchilla_tokens_per_param: int = 20
        self.flops_per_token_per_param: int = 6

        # Network / interconnect
        self.network_scaling_exponent: float = 0.05
        self.min_network_efficiency: float = 0.2
        self.max_network_efficiency: float = 1.0

    # ----------------------------
    # Core Computations
    # ----------------------------
    def get_network_efficiency(self, total_flops: float) -> float:
        """Network efficiency decreases as cluster scale increases (heuristic)."""
        relative_scale = total_flops / self.base_workload_flops
        raw_efficiency = 1.0 / (relative_scale ** self.network_scaling_exponent)
        return max(self.min_network_efficiency, min(self.max_network_efficiency, raw_efficiency))

    def estimate_model_size(self, total_flops: float) -> float:
        """Estimate number of parameters using simplified Chinchilla-like scaling (heuristic)."""
        params = math.sqrt(total_flops / 120)  # FLOPs ≈ 120 * N^2 (conceptual)
        return params

    # ----------------------------
    # Constraint calculations
    # ----------------------------
    def time_limited_by_compute(self, year: float, total_flops: float) -> float:
        hardware_flops = self.base_compute_flops * (self.compute_doubles_per_year ** year)
        network_eff = self.get_network_efficiency(total_flops)
        effective_utilization = self.compute_utilization * network_eff
        return total_flops / (effective_utilization * hardware_flops)

    def time_limited_by_memory(self, year: float, total_flops: float, ops_per_byte: int) -> float:
        bandwidth = self.base_bandwidth_bytes * (self.memory_bandwidth_growth ** year)
        achievable_flops = self.memory_utilization * bandwidth * ops_per_byte
        return total_flops / achievable_flops

    def time_limited_by_cost(self, year: float, total_flops: float) -> float:
        cost_per_flop = self.base_cost_per_flop * (self.cost_reduction_per_year ** year)
        budget = self.base_budget_dollars * (self.budget_growth_rate ** year)
        total_cost = cost_per_flop * total_flops
        if total_cost <= budget:
            return 0
        return total_cost / budget * 86400  # scaled to seconds

    def time_limited_by_verification(self, year: float, total_flops: float) -> float:
        """Heuristic verification time based on model size."""
        model_size = (total_flops / self.base_workload_flops) ** (1 / self.model_size_exponent)
        verify_time = self.verification_coefficient * (model_size ** self.verification_exponent)
        improvement = 0.9 ** year
        return verify_time * improvement

    def time_limited_by_thermal(self, year: float, total_flops: float) -> float:
        energy_per_flop = self.base_energy_per_flop * (self.energy_efficiency_gain ** year)
        max_compute_from_power = self.datacenter_power_watts / energy_per_flop
        return total_flops / (self.compute_utilization * max_compute_from_power)

    def time_limited_by_data(self, year: float, total_flops: float) -> float:
        """Data wall penalty applied heuristically if tokens are insufficient."""
        available_tokens = self.base_data_tokens * (self.data_growth_rate ** year)
        model_params = self.estimate_model_size(total_flops)
        required_tokens = self.chinchilla_tokens_per_param * model_params
        if required_tokens <= available_tokens:
            return 0
        scarcity_ratio = required_tokens / available_tokens
        penalty_factor = math.sqrt(scarcity_ratio)
        base_time = total_flops / (self.base_compute_flops * self.compute_utilization)
        return base_time * (penalty_factor - 1)

    def time_limited_by_interconnect(self, year: float, total_flops: float) -> float:
        """Heuristic interconnect overhead based on network efficiency decay."""
        efficiency = self.get_network_efficiency(total_flops)
        base_compute_time = self.time_limited_by_compute(year, total_flops)
        if efficiency >= 0.95:
            return 0
        overhead_multiplier = (1.0 / efficiency) - 1.0
        return base_compute_time * overhead_multiplier

    # ----------------------------
    # Bottleneck Analysis
    # ----------------------------
    def compute_bottleneck(self, year: float, total_flops: float) -> Dict[str, float]:
        """Return all constraint times and active bottleneck (heuristic analysis)."""
        times = {
            'compute': self.time_limited_by_compute(year, total_flops),
            'memory': self.time_limited_by_memory(year, total_flops, self.ops_per_byte_low),
            'cost': self.time_limited_by_cost(year, total_flops),
            'verification': self.time_limited_by_verification(year, total_flops),
            'thermal': self.time_limited_by_thermal(year, total_flops),
            'data': self.time_limited_by_data(year, total_flops),
            'interconnect': self.time_limited_by_interconnect(year, total_flops)
        }
        bottleneck = max(times, key=times.get)
        model_params = self.estimate_model_size(total_flops)
        network_eff = self.get_network_efficiency(total_flops)

        return {
            'year': year,
            'total_seconds': times[bottleneck],
            'total_days': times[bottleneck] / 86400,
            'bottleneck': bottleneck,
            'compute_seconds': times['compute'],
            'memory_seconds': times['memory'],
            'cost_seconds': times['cost'],
            'verification_seconds': times['verification'],
            'thermal_seconds': times['thermal'],
            'data_seconds': times['data'],
            'interconnect_seconds': times['interconnect'],
            'network_efficiency': network_eff,
            'model_params': model_params,
            'workload_flops': total_flops
        }

    def simulate(self, years: List[float]) -> List[Dict[str, float]]:
        """Run simulation across multiple years (heuristic predictions)."""
        results = []
        for year in years:
            workload = self.base_workload_flops * (self.workload_growth_rate ** year)
            result = self.compute_bottleneck(year, workload)
            results.append(result)
        return results

# ----------------------------
# 2. REPORTING FUNCTIONS
# ----------------------------
def print_timeline(results: List[Dict[str, float]]) -> None:
    print("\n" + "="*100)
    print("AI SCALING BOTTLENECK TIMELINE")
    print("="*100)
    print(f"{'Year':<6} {'Params':<12} {'Time (days)':<13} {'Net Eff':<9} {'Bottleneck':<13} {'Details':<35}")
    print("-"*100)

    for r in results:
        year_str = f"{r['year']:.1f}"
        params_str = f"{r['model_params']:.1e}"
        days_str = f"{r['total_days']:.1f}"
        net_eff_str = f"{r['network_efficiency']*100:.1f}%"
        bottleneck_str = r['bottleneck'].upper()
        details = (f"C:{r['compute_seconds']/86400:.0f}d "
                   f"M:{r['memory_seconds']/86400:.0f}d "
                   f"D:{r['data_seconds']/86400:.0f}d "
                   f"I:{r['interconnect_seconds']/86400:.0f}d")
        print(f"{year_str:<6} {params_str:<12} {days_str:<13} {net_eff_str:<9} "
              f"{bottleneck_str:<13} {details:<35}")
    print("="*100)

def generate_ascii_plot(results: List[Dict[str, float]]) -> None:
    print("\n" + "="*100)
    print("BOTTLENECK PHASE DIAGRAM")
    print("="*100)
    bottleneck_chars = {
        'compute': '█', 'memory': '▓', 'cost': '▒',
        'verification': '░', 'thermal': '▪', 'data': '◆',
        'interconnect': '◈'
    }
    for i, r in enumerate(results):
        year = r['year']
        bottleneck = r['bottleneck']
        bar_length = 60
        filled = int((i / len(results)) * bar_length)
        char = bottleneck_chars.get(bottleneck, '?')
        bar = char * filled + ' ' * (bar_length - filled)
        net_eff = r['network_efficiency'] * 100
        print(f"Year {year:4.1f} |{bar}| {bottleneck.upper():<13} (Net: {net_eff:4.1f}%)")
    print("="*100)
    print("\nLegend:")
    for key, char in bottleneck_chars.items():
        print(f"  {char} = {key.upper()}")
    print()

# ----------------------------
# 3. MAIN ENTRY
# ----------------------------
def run_simulation() -> None:
    model = EnhancedScalingModel()
    years = [i * 0.1 for i in range(121)]  # 0–12 years
    results = model.simulate(years)

    # Display only every 0.5 years
    display_results = [r for r in results if int(r['year'] * 10) % 5 == 0]

    print_timeline(display_results)
    generate_ascii_plot(display_results)

    # Final summary
    final = results[-1]
    print(f"\nFinal Year ({final['year']:.1f}) State:")
    print(f"  Model Size: {final['model_params']:.2e} parameters")
    print(f"  Network Efficiency: {final['network_efficiency']*100:.1f}%")
    print(f"  Active Bottleneck: {final['bottleneck'].upper()}")
    print(f"  Training Time: {final['total_days']:.1e} days")
    print("\n" + "="*100)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    run_simulation()
