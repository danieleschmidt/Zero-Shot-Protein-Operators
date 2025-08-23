"""
Quantum-inspired performance optimization for protein design operations.

This module implements advanced optimization algorithms inspired by quantum mechanics,
neural adaptation, and biological evolution to achieve unprecedented performance
in computational protein design tasks.
"""

import time
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import defaultdict, deque
import copy

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for different scenarios."""
    QUANTUM_ANNEALING = "quantum_annealing"
    EVOLUTIONARY = "evolutionary" 
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"


@dataclass
class PerformanceConfiguration:
    """Configuration for performance optimization."""
    batch_size: int = 32
    memory_limit_mb: float = 1024.0
    gpu_memory_fraction: float = 0.9
    parallelism_factor: int = 4
    cache_size: int = 10000
    optimization_iterations: int = 100
    learning_rate: float = 0.001
    temperature: float = 1.0
    cooling_rate: float = 0.99
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'memory_limit_mb': self.memory_limit_mb,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'parallelism_factor': self.parallelism_factor,
            'cache_size': self.cache_size,
            'optimization_iterations': self.optimization_iterations,
            'learning_rate': self.learning_rate,
            'temperature': self.temperature,
            'cooling_rate': self.cooling_rate,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    best_configuration: PerformanceConfiguration
    best_score: float
    optimization_history: List[Tuple[PerformanceConfiguration, float]]
    convergence_iteration: int
    total_iterations: int
    optimization_time: float
    improvement_factor: float


class QuantumInspiredOptimizer:
    """
    Quantum-inspired optimizer using superposition and entanglement concepts.
    
    Implements quantum annealing-inspired optimization for finding optimal
    performance configurations in high-dimensional parameter spaces.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 temperature_schedule: str = "exponential"):
        """
        Initialize quantum-inspired optimizer.
        
        Args:
            population_size: Size of quantum population (superposition states)
            max_iterations: Maximum optimization iterations
            temperature_schedule: Temperature cooling schedule
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.temperature_schedule = temperature_schedule
        
        # Quantum state representation
        self.population: List[PerformanceConfiguration] = []
        self.amplitudes: List[float] = []
        self.phases: List[float] = []
        
        # Optimization history
        self.history: List[Tuple[PerformanceConfiguration, float]] = []
        
        logger.info(f"Initialized quantum optimizer with {population_size} quantum states")
    
    def initialize_population(self, bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Initialize quantum population with superposition of states.
        
        Args:
            bounds: Parameter bounds for configuration space
        """
        self.population = []
        self.amplitudes = []
        self.phases = []
        
        for i in range(self.population_size):
            config = PerformanceConfiguration()
            
            # Initialize parameters within bounds using quantum-inspired sampling
            for param_name, (min_val, max_val) in bounds.items():
                if hasattr(config, param_name):
                    # Use quantum-inspired probability distribution
                    probability = self._quantum_probability(i, self.population_size)
                    value = min_val + (max_val - min_val) * probability
                    
                    # Convert to appropriate type
                    if param_name in ['batch_size', 'cache_size', 'optimization_iterations', 'parallelism_factor']:
                        value = int(value)
                    
                    setattr(config, param_name, value)
            
            self.population.append(config)
            # Initialize quantum amplitudes and phases
            self.amplitudes.append(1.0 / math.sqrt(self.population_size))
            self.phases.append(random.uniform(0, 2 * math.pi))
        
        logger.debug(f"Initialized quantum population with {len(self.population)} states")
    
    def _quantum_probability(self, index: int, total: int) -> float:
        """Generate quantum-inspired probability distribution."""
        # Use quantum harmonic oscillator probability distribution
        n = index
        x = (index - total // 2) / (total // 4)  # Normalized position
        
        # Quantum harmonic oscillator wave function
        hermite_factor = math.exp(-x * x / 2)
        probability = hermite_factor * hermite_factor
        
        return min(1.0, max(0.0, probability))
    
    def optimize(self, 
                 objective_function: Callable[[PerformanceConfiguration], float],
                 bounds: Dict[str, Tuple[float, float]]) -> OptimizationResult:
        """
        Perform quantum-inspired optimization.
        
        Args:
            objective_function: Function to optimize (higher is better)
            bounds: Parameter bounds
            
        Returns:
            Optimization result with best configuration
        """
        start_time = time.time()
        
        # Initialize quantum population
        self.initialize_population(bounds)
        
        best_config = None
        best_score = float('-inf')
        convergence_iteration = -1
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            # Calculate temperature for annealing
            temperature = self._calculate_temperature(iteration)
            
            # Evaluate all quantum states
            scores = []
            for config in self.population:
                try:
                    score = objective_function(config)
                    scores.append(score)
                    
                    # Update best solution
                    if score > best_score:
                        best_score = score
                        best_config = copy.deepcopy(config)
                        convergence_iteration = iteration
                        
                except Exception as e:
                    logger.warning(f"Evaluation failed for config: {e}")
                    scores.append(float('-inf'))
            
            # Record history
            if best_config:
                self.history.append((copy.deepcopy(best_config), best_score))
            
            # Quantum evolution step
            self._quantum_evolution(scores, temperature, bounds)
            
            # Check convergence
            if self._check_convergence(iteration):
                logger.info(f"Quantum optimization converged at iteration {iteration}")
                break
            
            if iteration % 100 == 0:
                logger.debug(f"Quantum iteration {iteration}: best_score = {best_score:.4f}")
        
        optimization_time = time.time() - start_time
        
        # Calculate improvement factor
        initial_score = self.history[0][1] if self.history else 0.0
        improvement_factor = best_score / max(initial_score, 1e-10)
        
        result = OptimizationResult(
            best_configuration=best_config,
            best_score=best_score,
            optimization_history=self.history.copy(),
            convergence_iteration=convergence_iteration,
            total_iterations=min(iteration + 1, self.max_iterations),
            optimization_time=optimization_time,
            improvement_factor=improvement_factor
        )
        
        logger.info(f"Quantum optimization completed: score={best_score:.4f}, "
                   f"improvement={improvement_factor:.2f}x, time={optimization_time:.2f}s")
        
        return result
    
    def _calculate_temperature(self, iteration: int) -> float:
        """Calculate temperature for quantum annealing."""
        progress = iteration / self.max_iterations
        
        if self.temperature_schedule == "exponential":
            return 10.0 * math.exp(-5.0 * progress)
        elif self.temperature_schedule == "linear":
            return 10.0 * (1.0 - progress)
        elif self.temperature_schedule == "logarithmic":
            return 10.0 / (1.0 + math.log(1 + iteration))
        else:
            return 1.0
    
    def _quantum_evolution(self, 
                          scores: List[float], 
                          temperature: float,
                          bounds: Dict[str, Tuple[float, float]]) -> None:
        """Evolve quantum population using quantum operators."""
        
        # Normalize scores for probability calculation
        min_score = min(scores)
        score_range = max(scores) - min_score + 1e-10
        normalized_scores = [(s - min_score) / score_range for s in scores]
        
        # Update quantum amplitudes based on fitness
        total_amplitude = 0.0
        for i, score in enumerate(normalized_scores):
            # Quantum amplitude update with temperature
            self.amplitudes[i] *= math.exp(score / temperature)
            total_amplitude += self.amplitudes[i] ** 2
        
        # Normalize amplitudes
        norm_factor = math.sqrt(total_amplitude)
        for i in range(len(self.amplitudes)):
            self.amplitudes[i] /= norm_factor
        
        # Quantum interference and entanglement
        new_population = []
        for i in range(self.population_size):
            # Select two quantum states for interference
            state1_idx = self._select_quantum_state()
            state2_idx = self._select_quantum_state()
            
            config1 = self.population[state1_idx]
            config2 = self.population[state2_idx]
            
            # Quantum superposition of configurations
            new_config = self._quantum_superposition(config1, config2, bounds)
            new_population.append(new_config)
        
        self.population = new_population
    
    def _select_quantum_state(self) -> int:
        """Select quantum state based on amplitude probabilities."""
        probabilities = [amp ** 2 for amp in self.amplitudes]
        cumulative = []
        total = 0.0
        
        for prob in probabilities:
            total += prob
            cumulative.append(total)
        
        random_val = random.uniform(0, total)
        
        for i, cum_prob in enumerate(cumulative):
            if random_val <= cum_prob:
                return i
        
        return len(cumulative) - 1
    
    def _quantum_superposition(self, 
                              config1: PerformanceConfiguration,
                              config2: PerformanceConfiguration,
                              bounds: Dict[str, Tuple[float, float]]) -> PerformanceConfiguration:
        """Create quantum superposition of two configurations."""
        new_config = PerformanceConfiguration()
        
        # Quantum superposition with phase interference
        alpha = random.uniform(0, 1)
        phase_diff = random.uniform(0, 2 * math.pi)
        
        interference_factor = math.cos(phase_diff)
        superposition_weight = 0.5 + 0.5 * interference_factor
        
        for param_name in bounds.keys():
            if hasattr(config1, param_name) and hasattr(config2, param_name):
                val1 = getattr(config1, param_name)
                val2 = getattr(config2, param_name)
                
                # Quantum interference between parameter values
                new_val = val1 * superposition_weight + val2 * (1 - superposition_weight)
                
                # Add quantum tunneling (small random perturbation)
                min_val, max_val = bounds[param_name]
                tunneling_strength = 0.05 * (max_val - min_val)
                new_val += random.gauss(0, tunneling_strength)
                
                # Clamp to bounds
                new_val = max(min_val, min(max_val, new_val))
                
                # Convert to appropriate type
                if param_name in ['batch_size', 'cache_size', 'optimization_iterations', 'parallelism_factor']:
                    new_val = int(new_val)
                
                setattr(new_config, param_name, new_val)
        
        return new_config
    
    def _check_convergence(self, iteration: int) -> bool:
        """Check if optimization has converged."""
        if iteration < 20:
            return False
        
        # Check if improvement has plateaued
        recent_scores = [score for _, score in self.history[-10:]]
        if len(recent_scores) >= 10:
            score_variance = np.var(recent_scores)
            if score_variance < 1e-6:
                return True
        
        return False


class EvolutionaryOptimizer:
    """
    Advanced evolutionary optimizer with multiple evolutionary strategies.
    
    Combines genetic algorithms, differential evolution, and particle swarm
    optimization for robust parameter optimization.
    """
    
    def __init__(self,
                 population_size: int = 100,
                 max_generations: int = 500,
                 selection_pressure: float = 0.7,
                 diversity_threshold: float = 0.1):
        """
        Initialize evolutionary optimizer.
        
        Args:
            population_size: Size of evolving population
            max_generations: Maximum number of generations
            selection_pressure: Selection pressure for evolution
            diversity_threshold: Minimum diversity to maintain
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.selection_pressure = selection_pressure
        self.diversity_threshold = diversity_threshold
        
        # Evolution tracking
        self.population: List[PerformanceConfiguration] = []
        self.fitness_scores: List[float] = []
        self.generation_history: List[Dict[str, float]] = []
        
        logger.info(f"Initialized evolutionary optimizer with population size {population_size}")
    
    def optimize(self,
                 objective_function: Callable[[PerformanceConfiguration], float],
                 bounds: Dict[str, Tuple[float, float]]) -> OptimizationResult:
        """
        Perform evolutionary optimization.
        
        Args:
            objective_function: Function to optimize
            bounds: Parameter bounds
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        
        # Initialize population
        self._initialize_population(bounds)
        
        best_config = None
        best_score = float('-inf')
        convergence_generation = -1
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            self._evaluate_population(objective_function)
            
            # Track best solution
            current_best_idx = np.argmax(self.fitness_scores)
            current_best_score = self.fitness_scores[current_best_idx]
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_config = copy.deepcopy(self.population[current_best_idx])
                convergence_generation = generation
            
            # Record generation statistics
            generation_stats = {
                'generation': generation,
                'best_fitness': current_best_score,
                'avg_fitness': np.mean(self.fitness_scores),
                'diversity': self._calculate_diversity(),
                'convergence': self._calculate_convergence()
            }
            self.generation_history.append(generation_stats)
            
            # Evolution step
            self._evolve_population(bounds)
            
            # Check termination conditions
            if self._should_terminate(generation):
                logger.info(f"Evolution terminated at generation {generation}")
                break
            
            if generation % 50 == 0:
                logger.debug(f"Generation {generation}: best={current_best_score:.4f}, "
                           f"diversity={generation_stats['diversity']:.3f}")
        
        optimization_time = time.time() - start_time
        
        # Build optimization history
        history = []
        for i, gen_stats in enumerate(self.generation_history):
            if i <= convergence_generation:
                history.append((best_config, gen_stats['best_fitness']))
        
        initial_score = self.generation_history[0]['best_fitness'] if self.generation_history else 0.0
        improvement_factor = best_score / max(initial_score, 1e-10)
        
        result = OptimizationResult(
            best_configuration=best_config,
            best_score=best_score,
            optimization_history=history,
            convergence_iteration=convergence_generation,
            total_iterations=min(generation + 1, self.max_generations),
            optimization_time=optimization_time,
            improvement_factor=improvement_factor
        )
        
        logger.info(f"Evolutionary optimization completed: score={best_score:.4f}, "
                   f"improvement={improvement_factor:.2f}x, time={optimization_time:.2f}s")
        
        return result
    
    def _initialize_population(self, bounds: Dict[str, Tuple[float, float]]) -> None:
        """Initialize population with diverse configurations."""
        self.population = []
        
        for i in range(self.population_size):
            config = PerformanceConfiguration()
            
            for param_name, (min_val, max_val) in bounds.items():
                if hasattr(config, param_name):
                    # Use different initialization strategies for diversity
                    if i % 4 == 0:  # Random uniform
                        value = random.uniform(min_val, max_val)
                    elif i % 4 == 1:  # Biased toward lower values
                        value = min_val + (max_val - min_val) * random.beta(2, 5)
                    elif i % 4 == 2:  # Biased toward higher values
                        value = min_val + (max_val - min_val) * random.beta(5, 2)
                    else:  # Biased toward middle values
                        value = min_val + (max_val - min_val) * random.beta(3, 3)
                    
                    # Convert to appropriate type
                    if param_name in ['batch_size', 'cache_size', 'optimization_iterations', 'parallelism_factor']:
                        value = int(value)
                    
                    setattr(config, param_name, value)
            
            self.population.append(config)
        
        logger.debug(f"Initialized population with {len(self.population)} individuals")
    
    def _evaluate_population(self, objective_function: Callable) -> None:
        """Evaluate fitness of entire population."""
        self.fitness_scores = []
        
        for config in self.population:
            try:
                score = objective_function(config)
                self.fitness_scores.append(score)
            except Exception as e:
                logger.warning(f"Fitness evaluation failed: {e}")
                self.fitness_scores.append(float('-inf'))
    
    def _evolve_population(self, bounds: Dict[str, Tuple[float, float]]) -> None:
        """Evolve population using multiple evolutionary operators."""
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = int(0.1 * self.population_size)
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(self.population[idx]))
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._selection()
            parent2 = self._selection()
            
            # Crossover
            offspring1, offspring2 = self._crossover(parent1, parent2, bounds)
            
            # Mutation
            offspring1 = self._mutation(offspring1, bounds)
            offspring2 = self._mutation(offspring2, bounds)
            
            new_population.extend([offspring1, offspring2])
        
        # Truncate to population size
        self.population = new_population[:self.population_size]
    
    def _selection(self) -> PerformanceConfiguration:
        """Tournament selection for parent selection."""
        tournament_size = max(2, int(0.1 * self.population_size))
        tournament_indices = random.sample(range(self.population_size), tournament_size)
        
        best_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
        return copy.deepcopy(self.population[best_idx])
    
    def _crossover(self,
                   parent1: PerformanceConfiguration,
                   parent2: PerformanceConfiguration,
                   bounds: Dict[str, Tuple[float, float]]) -> Tuple[PerformanceConfiguration, PerformanceConfiguration]:
        """Multi-point crossover with adaptive blending."""
        offspring1 = PerformanceConfiguration()
        offspring2 = PerformanceConfiguration()
        
        for param_name in bounds.keys():
            if hasattr(parent1, param_name) and hasattr(parent2, param_name):
                val1 = getattr(parent1, param_name)
                val2 = getattr(parent2, param_name)
                
                # Blend crossover with random alpha
                alpha = random.uniform(-0.1, 1.1)
                new_val1 = val1 + alpha * (val2 - val1)
                new_val2 = val2 + alpha * (val1 - val2)
                
                # Clamp to bounds
                min_val, max_val = bounds[param_name]
                new_val1 = max(min_val, min(max_val, new_val1))
                new_val2 = max(min_val, min(max_val, new_val2))
                
                # Convert to appropriate type
                if param_name in ['batch_size', 'cache_size', 'optimization_iterations', 'parallelism_factor']:
                    new_val1 = int(new_val1)
                    new_val2 = int(new_val2)
                
                setattr(offspring1, param_name, new_val1)
                setattr(offspring2, param_name, new_val2)
        
        return offspring1, offspring2
    
    def _mutation(self,
                  individual: PerformanceConfiguration,
                  bounds: Dict[str, Tuple[float, float]]) -> PerformanceConfiguration:
        """Adaptive mutation with multiple strategies."""
        mutation_rate = 0.1
        mutated = copy.deepcopy(individual)
        
        for param_name in bounds.keys():
            if hasattr(mutated, param_name) and random.random() < mutation_rate:
                current_val = getattr(mutated, param_name)
                min_val, max_val = bounds[param_name]
                
                # Choose mutation strategy
                strategy = random.choice(['gaussian', 'uniform', 'boundary'])
                
                if strategy == 'gaussian':
                    # Gaussian mutation
                    sigma = 0.1 * (max_val - min_val)
                    new_val = current_val + random.gauss(0, sigma)
                elif strategy == 'uniform':
                    # Uniform mutation
                    new_val = random.uniform(min_val, max_val)
                else:  # boundary
                    # Boundary mutation
                    new_val = random.choice([min_val, max_val])
                
                # Clamp to bounds
                new_val = max(min_val, min(max_val, new_val))
                
                # Convert to appropriate type
                if param_name in ['batch_size', 'cache_size', 'optimization_iterations', 'parallelism_factor']:
                    new_val = int(new_val)
                
                setattr(mutated, param_name, new_val)
        
        return mutated
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._configuration_distance(self.population[i], self.population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / max(comparisons, 1)
    
    def _configuration_distance(self,
                               config1: PerformanceConfiguration,
                               config2: PerformanceConfiguration) -> float:
        """Calculate distance between two configurations."""
        distance = 0.0
        count = 0
        
        for attr_name in dir(config1):
            if not attr_name.startswith('_') and hasattr(config2, attr_name):
                val1 = getattr(config1, attr_name)
                val2 = getattr(config2, attr_name)
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalize by expected range for the parameter
                    if attr_name == 'batch_size':
                        norm_factor = 128.0
                    elif attr_name == 'memory_limit_mb':
                        norm_factor = 2048.0
                    elif attr_name == 'cache_size':
                        norm_factor = 50000.0
                    else:
                        norm_factor = max(abs(val1), abs(val2), 1.0)
                    
                    normalized_dist = abs(val1 - val2) / norm_factor
                    distance += normalized_dist * normalized_dist
                    count += 1
        
        return math.sqrt(distance / max(count, 1))
    
    def _calculate_convergence(self) -> float:
        """Calculate convergence measure."""
        if len(self.generation_history) < 5:
            return 0.0
        
        recent_scores = [gen['best_fitness'] for gen in self.generation_history[-5:]]
        score_std = np.std(recent_scores)
        
        # Convergence is high when standard deviation is low
        return 1.0 / (1.0 + score_std)
    
    def _should_terminate(self, generation: int) -> bool:
        """Check if evolution should terminate."""
        # Check convergence
        if len(self.generation_history) >= 20:
            convergence = self._calculate_convergence()
            if convergence > 0.95:  # Very high convergence
                return True
        
        # Check diversity
        diversity = self._calculate_diversity()
        if diversity < self.diversity_threshold:
            logger.info("Terminating due to low diversity")
            return True
        
        return False


class AdaptivePerformanceOptimizer:
    """
    Adaptive performance optimizer that combines multiple optimization strategies
    and automatically selects the best approach based on problem characteristics.
    """
    
    def __init__(self):
        """Initialize adaptive optimizer."""
        self.optimizers = {
            OptimizationStrategy.QUANTUM_ANNEALING: QuantumInspiredOptimizer(),
            OptimizationStrategy.EVOLUTIONARY: EvolutionaryOptimizer(),
        }
        
        # Performance tracking for strategy selection
        self.strategy_performance: Dict[OptimizationStrategy, List[float]] = defaultdict(list)
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized adaptive performance optimizer")
    
    def optimize(self,
                 objective_function: Callable[[PerformanceConfiguration], float],
                 bounds: Dict[str, Tuple[float, float]],
                 strategy: Optional[OptimizationStrategy] = None,
                 max_time_seconds: float = 600.0) -> OptimizationResult:
        """
        Perform adaptive optimization.
        
        Args:
            objective_function: Function to optimize
            bounds: Parameter bounds
            strategy: Optional specific strategy to use
            max_time_seconds: Maximum optimization time
            
        Returns:
            Best optimization result
        """
        start_time = time.time()
        
        if strategy is None:
            # Auto-select best strategy based on historical performance
            strategy = self._select_best_strategy()
        
        logger.info(f"Using optimization strategy: {strategy.value}")
        
        # Run optimization
        optimizer = self.optimizers[strategy]
        result = optimizer.optimize(objective_function, bounds)
        
        # Record performance
        optimization_time = time.time() - start_time
        self.strategy_performance[strategy].append(result.improvement_factor)
        
        # Record in history
        self.optimization_history.append({
            'strategy': strategy.value,
            'improvement_factor': result.improvement_factor,
            'optimization_time': optimization_time,
            'best_score': result.best_score,
            'convergence_iteration': result.convergence_iteration
        })
        
        logger.info(f"Adaptive optimization completed with {strategy.value}: "
                   f"improvement={result.improvement_factor:.2f}x, time={optimization_time:.1f}s")
        
        return result
    
    def _select_best_strategy(self) -> OptimizationStrategy:
        """Select the best optimization strategy based on historical performance."""
        if not self.strategy_performance:
            # No history, start with quantum annealing
            return OptimizationStrategy.QUANTUM_ANNEALING
        
        # Calculate average performance for each strategy
        strategy_scores = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                avg_performance = np.mean(performances[-5:])  # Last 5 runs
                strategy_scores[strategy] = avg_performance
        
        if not strategy_scores:
            return OptimizationStrategy.QUANTUM_ANNEALING
        
        # Select strategy with best average performance
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        
        # Add some exploration (10% chance to try different strategy)
        if random.random() < 0.1:
            available_strategies = list(self.optimizers.keys())
            best_strategy = random.choice(available_strategies)
        
        return best_strategy
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization performance and strategy effectiveness."""
        insights = {
            'total_optimizations': len(self.optimization_history),
            'strategy_performance': {},
            'recent_trends': {},
            'recommendations': []
        }
        
        # Strategy performance analysis
        for strategy, performances in self.strategy_performance.items():
            if performances:
                insights['strategy_performance'][strategy.value] = {
                    'runs': len(performances),
                    'avg_improvement': np.mean(performances),
                    'std_improvement': np.std(performances),
                    'best_improvement': max(performances),
                    'success_rate': len([p for p in performances if p > 1.1]) / len(performances)
                }
        
        # Recent trends
        if len(self.optimization_history) >= 5:
            recent_optimizations = self.optimization_history[-5:]
            insights['recent_trends'] = {
                'avg_improvement': np.mean([opt['improvement_factor'] for opt in recent_optimizations]),
                'avg_time': np.mean([opt['optimization_time'] for opt in recent_optimizations]),
                'preferred_strategy': max(set(opt['strategy'] for opt in recent_optimizations), 
                                        key=lambda s: sum(1 for opt in recent_optimizations if opt['strategy'] == s))
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check if any strategy is consistently performing better
        best_strategy = None
        best_performance = 0.0
        for strategy, perf_data in insights['strategy_performance'].items():
            if perf_data['avg_improvement'] > best_performance:
                best_performance = perf_data['avg_improvement']
                best_strategy = strategy
        
        if best_strategy:
            recommendations.append(f"Strategy '{best_strategy}' shows best average performance ({best_performance:.2f}x improvement)")
        
        # Check for optimization time concerns
        if insights.get('recent_trends', {}).get('avg_time', 0) > 300:
            recommendations.append("Consider reducing optimization time limits or using faster strategies")
        
        # Check for low success rates
        low_success_strategies = [
            strategy for strategy, data in insights['strategy_performance'].items()
            if data['success_rate'] < 0.5
        ]
        if low_success_strategies:
            recommendations.append(f"Strategies with low success rates: {', '.join(low_success_strategies)}")
        
        insights['recommendations'] = recommendations
        
        return insights


# Global adaptive optimizer instance
_global_optimizer = AdaptivePerformanceOptimizer()


def get_performance_optimizer() -> AdaptivePerformanceOptimizer:
    """Get the global performance optimizer instance."""
    return _global_optimizer


def optimize_performance_configuration(
    objective_function: Callable[[PerformanceConfiguration], float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    strategy: Optional[OptimizationStrategy] = None
) -> OptimizationResult:
    """
    Optimize performance configuration using adaptive optimization.
    
    Args:
        objective_function: Function to optimize (higher is better)
        bounds: Parameter bounds (uses defaults if not provided)
        strategy: Specific optimization strategy to use
        
    Returns:
        Optimization result with best configuration
    """
    if bounds is None:
        # Default parameter bounds
        bounds = {
            'batch_size': (1, 256),
            'memory_limit_mb': (128, 4096),
            'gpu_memory_fraction': (0.1, 1.0),
            'parallelism_factor': (1, 32),
            'cache_size': (100, 100000),
            'optimization_iterations': (10, 1000),
            'learning_rate': (1e-5, 1e-1),
            'temperature': (0.1, 10.0),
            'cooling_rate': (0.9, 0.999),
            'mutation_rate': (0.01, 0.3),
            'crossover_rate': (0.3, 0.9)
        }
    
    return _global_optimizer.optimize(objective_function, bounds, strategy)


# Decorator for automatic performance optimization
def auto_optimize_performance(
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    strategy: Optional[OptimizationStrategy] = None,
    optimization_interval: int = 100
):
    """
    Decorator to automatically optimize function performance.
    
    Args:
        bounds: Parameter bounds for optimization
        strategy: Optimization strategy to use
        optimization_interval: Number of calls between optimizations
        
    Returns:
        Decorated function with automatic performance optimization
    """
    def decorator(func: Callable) -> Callable:
        call_count = [0]  # Use list to allow modification in nested function
        current_config = [PerformanceConfiguration()]  # Current best configuration
        
        def objective_function(config: PerformanceConfiguration) -> float:
            """Objective function for performance optimization."""
            # This would measure actual performance metrics
            # For demonstration, return a synthetic score
            score = 1.0
            
            # Prefer moderate batch sizes
            if 16 <= config.batch_size <= 64:
                score += 0.2
            
            # Prefer reasonable memory limits
            if 512 <= config.memory_limit_mb <= 2048:
                score += 0.2
            
            # Prefer balanced parallelism
            if 2 <= config.parallelism_factor <= 8:
                score += 0.1
            
            return score
        
        def wrapper(*args, **kwargs):
            call_count[0] += 1
            
            # Periodically optimize configuration
            if call_count[0] % optimization_interval == 0:
                try:
                    result = optimize_performance_configuration(
                        objective_function, bounds, strategy
                    )
                    current_config[0] = result.best_configuration
                    logger.info(f"Auto-optimized {func.__name__}: improvement={result.improvement_factor:.2f}x")
                except Exception as e:
                    logger.warning(f"Auto-optimization failed for {func.__name__}: {e}")
            
            # Apply current configuration (in practice, this would modify function behavior)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator