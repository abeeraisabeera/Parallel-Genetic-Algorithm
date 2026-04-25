"""
Parallel Genetic Algorithm for Feature Selection
==================================================
An optimized feature selection framework using genetic algorithms with 
multiprocessing support for improved computational efficiency.

"""

import numpy as np
import random
import time
from typing import Tuple, List
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


class GeneticFeatureSelector:
    """
    A genetic algorithm-based feature selector with parallel processing capabilities.
    
    This class implements an evolutionary approach to feature selection, identifying
    optimal feature subsets that maximize classification accuracy while reducing
    dimensionality.
    """
    
    def __init__(self, population_size: int = 50, num_generations: int = 50,
                 crossover_prob: float = 0.5, mutation_prob: float = 0.2):
        """
        Initialize the Genetic Feature Selector.
        
        Args:
            population_size: Number of individuals in each generation
            num_generations: Number of evolutionary iterations
            crossover_prob: Probability of crossover between individuals
            mutation_prob: Probability of mutation in offspring
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Will be set during fit
        self.best_individual_ = None
        self.best_features_ = None
        self.best_accuracy_ = None
        self.fitness_history_ = []
        
    def _evaluate_fitness(self, individual: List[int], 
                         X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray) -> Tuple[float]:
        """
        Evaluate the fitness of an individual (feature subset).
        
        Args:
            individual: Binary representation of feature selection
            X_train, X_test: Training and test feature matrices
            y_train, y_test: Training and test target vectors
            
        Returns:
            Tuple containing accuracy score
        """
        # Extract selected features
        selected_features = [idx for idx, gene in enumerate(individual) if gene == 1]
        
        # Handle edge cases
        if not selected_features or any(idx >= X_train.shape[1] for idx in selected_features):
            return (0.0,)
        
        # Select feature subset
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_selected, y_train)
        
        # Evaluate performance
        y_pred = clf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        return (accuracy,)
    
    def fit(self, X_train: np.ndarray, X_test: np.ndarray,
            y_train: np.ndarray, y_test: np.ndarray,
            parallel: bool = False) -> 'GeneticFeatureSelector':
        """
        Execute the genetic algorithm to find optimal feature subset.
        
        Args:
            X_train, X_test: Training and test feature matrices
            y_train, y_test: Training and test target vectors
            parallel: Whether to use multiprocessing
            
        Returns:
            self: Fitted selector instance
        """
        num_features = X_train.shape[1]
        
        # Setup DEAP framework
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_bool, n=num_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness, 
                        X_train=X_train, X_test=X_test, 
                        y_train=y_train, y_test=y_test)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Enable parallel processing if requested
        if parallel:
            pool = Pool()
            toolbox.register("map", pool.map)
        
        # Initialize population
        population = toolbox.population(n=self.population_size)
        
        # Run evolutionary algorithm
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.num_generations,
            verbose=False
        )
        
        # Extract best solution
        self.best_individual_ = tools.selBest(population, k=1)[0]
        self.best_features_ = [idx for idx, gene in enumerate(self.best_individual_) if gene == 1]
        self.best_accuracy_ = self.best_individual_.fitness.values[0]
        
        # Store fitness evolution
        self.fitness_history_ = [self._evaluate_fitness(ind, X_train, X_test, y_train, y_test)[0] 
                                for ind in population]
        
        if parallel:
            pool.close()
            pool.join()
        
        return self
    
    def get_best_features(self) -> List[int]:
        """Return indices of selected features."""
        return self.best_features_
    
    def get_feature_importance(self) -> np.ndarray:
        """Return binary feature selection mask."""
        return np.array(self.best_individual_)


def load_and_split_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the digits dataset and split into train/test sets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def benchmark_performance(X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray,
                         population_size: int = 50,
                         num_generations: int = 50) -> dict:
    """
    Benchmark sequential vs parallel genetic algorithm performance.
    
    Returns:
        Dictionary containing results from both approaches
    """
    results = {}
    
    print("=" * 70)
    print("BENCHMARKING: Sequential vs Parallel Genetic Algorithm")
    print("=" * 70)
    
    # Sequential execution
    print("\n[1/2] Running Sequential Genetic Algorithm...")
    selector_seq = GeneticFeatureSelector(population_size, num_generations)
    start_time = time.time()
    selector_seq.fit(X_train, X_test, y_train, y_test, parallel=False)
    seq_time = time.time() - start_time
    
    results['sequential'] = {
        'selector': selector_seq,
        'time': seq_time,
        'accuracy': selector_seq.best_accuracy_,
        'num_features': len(selector_seq.best_features_),
        'features': selector_seq.best_features_
    }
    
    print(f"   ✓ Completed in {seq_time:.2f} seconds")
    print(f"   ✓ Best Accuracy: {selector_seq.best_accuracy_:.4f}")
    print(f"   ✓ Features Selected: {len(selector_seq.best_features_)}/{X_train.shape[1]}")
    
    # Parallel execution
    print("\n[2/2] Running Parallel Genetic Algorithm...")
    selector_par = GeneticFeatureSelector(population_size, num_generations)
    start_time = time.time()
    selector_par.fit(X_train, X_test, y_train, y_test, parallel=True)
    par_time = time.time() - start_time
    
    results['parallel'] = {
        'selector': selector_par,
        'time': par_time,
        'accuracy': selector_par.best_accuracy_,
        'num_features': len(selector_par.best_features_),
        'features': selector_par.best_features_
    }
    
    print(f"   ✓ Completed in {par_time:.2f} seconds")
    print(f"   ✓ Best Accuracy: {selector_par.best_accuracy_:.4f}")
    print(f"   ✓ Features Selected: {len(selector_par.best_features_)}/{X_train.shape[1]}")
    
    # Calculate speedup
    speedup = seq_time / par_time
    print("\n" + "=" * 70)
    print(f"SPEEDUP: {speedup:.2f}x faster with parallel processing")
    print(f"Time Saved: {seq_time - par_time:.2f} seconds ({((seq_time - par_time) / seq_time * 100):.1f}%)")
    print("=" * 70)
    
    return results


def visualize_results(results: dict, save_path: str = 'results.png'):
    """
    Create comprehensive visualization of GA results.
    
    Args:
        results: Dictionary containing benchmark results
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Fitness Evolution Comparison
    ax1 = axes[0]
    seq_fitness = results['sequential']['selector'].fitness_history_
    par_fitness = results['parallel']['selector'].fitness_history_
    
    ax1.plot(range(len(seq_fitness)), sorted(seq_fitness, reverse=True), 
             'o-', color='#3498db', label='Sequential', linewidth=2, markersize=4)
    ax1.plot(range(len(par_fitness)), sorted(par_fitness, reverse=True), 
             's-', color='#2ecc71', label='Parallel', linewidth=2, markersize=4)
    ax1.set_xlabel('Population Rank', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Fitness Distribution Across Population', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Execution Time Comparison
    ax2 = axes[1]
    times = [results['sequential']['time'], results['parallel']['time']]
    colors = ['#3498db', '#2ecc71']
    bars = ax2.bar(['Sequential', 'Parallel'], times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add speedup annotation
    speedup = times[0] / times[1]
    ax2.text(0.5, max(times) * 0.8, f'Speedup: {speedup:.2f}x',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Feature Selection Summary
    ax3 = axes[2]
    total_features = 64  # Digits dataset has 64 features
    seq_selected = results['sequential']['num_features']
    par_selected = results['parallel']['num_features']
    
    categories = ['Sequential', 'Parallel']
    selected = [seq_selected, par_selected]
    unselected = [total_features - seq_selected, total_features - par_selected]
    
    x = np.arange(len(categories))
    width = 0.5
    
    bars1 = ax3.bar(x, selected, width, label='Selected', color='#2ecc71', alpha=0.8)
    bars2 = ax3.bar(x, unselected, width, bottom=selected, label='Not Selected', 
                   color='#e74c3c', alpha=0.5)
    
    ax3.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax3.set_title('Feature Selection Results', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=10)
    
    # Add percentage labels
    for i, (bar, sel) in enumerate(zip(bars1, selected)):
        percentage = (sel / total_features) * 100
        ax3.text(bar.get_x() + bar.get_width()/2., sel/2,
                f'{sel}\n({percentage:.1f}%)',
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print(" " * 15 + "PARALLEL GENETIC ALGORITHM")
    print(" " * 18 + "Feature Selection Framework")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading Digits Dataset...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"   Dataset Shape: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"   Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    
    # Run benchmark
    results = benchmark_performance(X_train, X_test, y_train, y_test)
    
    # Visualize results
    print("\n📈 Generating Visualizations...")
    visualize_results(results)
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    print(f"\nSequential GA:")
    print(f"  Selected Features: {results['sequential']['features']}")
    print(f"\nParallel GA:")
    print(f"  Selected Features: {results['parallel']['features']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
