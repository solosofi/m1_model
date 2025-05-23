import time
from expression_nodes import ExpressionNode, VariableNode, OperatorNode, apply_distributivity_factor, apply_distributivity_expand
from solver_strategy import SolverStrategy
from solver import configurable_solver

def get_benchmark_expressions():
    """
    Returns a list of manually constructed ExpressionNode objects for benchmarking.
    """
    a = VariableNode('a')
    b = VariableNode('b')
    c = VariableNode('c')
    d = VariableNode('d')
    x = VariableNode('x')
    y = VariableNode('y')
    z = VariableNode('z')
    # e = VariableNode('e') # Not used, removed for tidiness

    expressions = []

    # 1. Simple factoring: (a*b) + (a*c) -> a*(b+c)
    # Ops: 3 -> 2
    expr1 = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), OperatorNode('*', a.copy(), c.copy()))
    expressions.append(expr1)

    # 2. Simple expansion (but solver should not expand if ops increase): x*(y+z)
    # Ops: 2 (remains 2, as expansion to (x*y)+(x*z) is 3 ops)
    expr2 = OperatorNode('*', x.copy(), OperatorNode('+', y.copy(), z.copy()))
    expressions.append(expr2)

    # 3. Already optimal (simple sum): a+b
    # Ops: 1 (remains 1)
    expr3 = OperatorNode('+', a.copy(), b.copy())
    expressions.append(expr3)

    # 4. Already optimal (simple product): a*b
    # Ops: 1 (remains 1)
    expr4 = OperatorNode('*', a.copy(), b.copy())
    expressions.append(expr4)

    # 5. Complex factoring: ((a*b)+(a*c)) + d -> (a*(b+c)) + d
    # Ops: 4 -> 3
    expr5_inner = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), OperatorNode('*', a.copy(), c.copy()))
    expr5 = OperatorNode('+', expr5_inner, d.copy())
    expressions.append(expr5)

    # 6. Multi-level factoring: (a*b) + ((a*c)+(a*d)) -> (a*b) + (a*(c+d))
    # Original: (a*b) + ((a*c)+(a*d)). Ops: 1 (outer +) + 1 (ab) + 3 (ac+ad) = 5
    # Solved: (a*b) + (a*(c+d)). Ops: 1 (outer +) + 1 (ab) + 2 (a(c+d)) = 4
    expr6_inner_cd = OperatorNode('+', OperatorNode('*', a.copy(), c.copy()), OperatorNode('*', a.copy(), d.copy())) 
    expr6 = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), expr6_inner_cd) 
    expressions.append(expr6)
    
    # 7. No common factor: (a*b) + (c*d)
    # Ops: 3 (remains 3)
    expr7 = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), OperatorNode('*', c.copy(), d.copy()))
    expressions.append(expr7)

    # 8. Right-side expansion (but solver shouldn't if ops increase): (a+b)*c
    # Ops: 2 (remains 2, as expansion to (a*c)+(b*c) is 3 ops)
    expr8 = OperatorNode('*', OperatorNode('+', a.copy(), b.copy()), c.copy())
    expressions.append(expr8)
    
    # 9. Factoring with common term on different sides: (a*b) + (c*a) -> a*(b+c)
    # Ops: 3 -> 2
    expr9 = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), OperatorNode('*', c.copy(), a.copy()))
    expressions.append(expr9)

    # 10. More complex expression: (x*(y+z)) + (a*(b+c))
    # Ops: 2 + 2 + 1 = 5. No factoring/expansion between the two main terms.
    term_xyz = OperatorNode('*', x.copy(), OperatorNode('+', y.copy(), z.copy()))
    term_abc = OperatorNode('*', a.copy(), OperatorNode('+', b.copy(), c.copy()))
    expr10 = OperatorNode('+', term_xyz, term_abc)
    expressions.append(expr10)
    
    # 11. (a*b) + (a*b) -> a*(b+b)
    # Ops: 3 -> 2
    expr11 = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), OperatorNode('*', a.copy(), b.copy()))
    expressions.append(expr11)

    # 12. a*(b*(c+d)) - expansion would increase ops
    # ops a*(b*(c+d)) = 3
    expr12_cd = OperatorNode('+',c.copy(),d.copy())
    expr12_bcd = OperatorNode('*',b.copy(),expr12_cd)
    expr12 = OperatorNode('*',a.copy(),expr12_bcd)
    expressions.append(expr12)
    
    # 13. ((a*b)+(a*c)) * ((x*y)+(x*z)) -> single step factor one side
    # Original ops: 7. One step factor: 6.
    left_sum_e13 = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), OperatorNode('*', a.copy(), c.copy()))
    right_sum_e13 = OperatorNode('+', OperatorNode('*', x.copy(), y.copy()), OperatorNode('*', x.copy(), z.copy()))
    expr13 = OperatorNode('*', left_sum_e13, right_sum_e13)
    expressions.append(expr13)

    # 14. Simple variable, no operations
    # Ops: 0
    expr14 = a.copy()
    expressions.append(expr14)
    
    # 15. (a+b)*(c+d) - expansion would increase ops
    # ops ((a+b)*(c+d)) = 3
    expr15_ab = OperatorNode('+',a.copy(),b.copy())
    expr15_cd = OperatorNode('+',c.copy(),d.copy())
    expr15 = OperatorNode('*',expr15_ab, expr15_cd)
    expressions.append(expr15)

    return expressions

def evaluate_strategy_performance(strategy: SolverStrategy, test_expressions_list: list):
    """
    Evaluates the performance of a given SolverStrategy on a set of test expressions.
    """
    total_initial_ops = 0
    total_final_ops = 0
    problems_improved_count = 0
    total_time_taken = 0.0

    num_problems = len(test_expressions_list)

    if num_problems == 0:
        return {
            'num_problems': 0,
            'total_initial_ops': 0,
            'total_final_ops': 0,
            'total_ops_reduced': 0,
            'average_improvement_per_problem': 0,
            'problems_improved_percentage': 0,
            'average_time_per_problem': 0
        }

    for expr_in in test_expressions_list:
        # Ensure we are using the strategy's metric for op counting
        initial_ops = strategy.evaluation_metric(expr_in)
        
        start_time = time.time()
        expr_out = configurable_solver(expr_in, strategy)
        end_time = time.time()
        
        final_ops = strategy.evaluation_metric(expr_out)
        
        total_initial_ops += initial_ops
        total_final_ops += final_ops
        total_time_taken += (end_time - start_time)
        
        if final_ops < initial_ops:
            problems_improved_count += 1

    total_ops_reduced = total_initial_ops - total_final_ops
    average_improvement = total_ops_reduced / num_problems
    problems_improved_percentage = (problems_improved_count / num_problems) * 100
    average_time = total_time_taken / num_problems

    return {
        'num_problems': num_problems,
        'total_initial_ops': total_initial_ops,
        'total_final_ops': total_final_ops,
        'total_ops_reduced': total_ops_reduced,
        'average_improvement_per_problem': average_improvement,
        'problems_improved_percentage': problems_improved_percentage,
        'average_time_per_problem': average_time
    }

if __name__ == '__main__':
    benchmark_expressions = get_benchmark_expressions()

    # Define evaluation metric (count_ops)
    ops_metric = lambda node: node.count_ops()

    # Create Default Strategy
    default_strategy = SolverStrategy(
        transformations_list=[apply_distributivity_factor, apply_distributivity_expand],
        evaluation_metric_func=ops_metric
    )

    # Create Factor-Only Strategy
    factor_only_strategy = SolverStrategy(
        transformations_list=[apply_distributivity_factor],
        evaluation_metric_func=ops_metric
    )
    
    # Create Expand-Only Strategy (for more robust testing)
    expand_only_strategy = SolverStrategy(
        transformations_list=[apply_distributivity_expand],
        evaluation_metric_func=ops_metric
    )

    print("--- Evaluating Default Strategy ---")
    default_perf = evaluate_strategy_performance(default_strategy, benchmark_expressions)
    for key, value in default_perf.items():
        print(f"{key}: {value}")

    print("\n--- Evaluating Factor-Only Strategy ---")
    factor_only_perf = evaluate_strategy_performance(factor_only_strategy, benchmark_expressions)
    for key, value in factor_only_perf.items():
        print(f"{key}: {value}")
        
    print("\n--- Evaluating Expand-Only Strategy ---")
    expand_only_perf = evaluate_strategy_performance(expand_only_strategy, benchmark_expressions)
    for key, value in expand_only_perf.items():
        print(f"{key}: {value}")

    # Assertions
    print("\n--- Assertions ---")
    
    # 1. Default strategy should reduce ops at least as much as factor_only or expand_only
    assert default_perf['total_ops_reduced'] >= factor_only_perf['total_ops_reduced'], \
        "Default strategy should reduce ops at least as much as factor-only."
    # For expansion, it's possible it reduces 0 ops if all expansions lead to more ops.
    # So default_perf['total_ops_reduced'] >= expand_only_perf['total_ops_reduced'] is also true.
    # (as expand_only_perf['total_ops_reduced'] will likely be 0 or negative, and default_perf will be positive or zero)
    
    # 2. Factor-only strategy should have non-negative ops reduction (it should not increase ops).
    assert factor_only_perf['total_ops_reduced'] >= 0, \
        "Factor-only strategy should not increase total ops."

    # 3. Expand-only strategy might not reduce ops, as expansion often increases them.
    #    The configurable_solver logic only accepts changes if metric score is lower.
    #    So, expand_only_perf['total_ops_reduced'] should be >= 0.
    assert expand_only_perf['total_ops_reduced'] >= 0, \
        "Expand-only strategy (with current solver logic) should not increase total ops."

    # 4. Check if some problems were improved by the default strategy (expected for this benchmark)
    assert default_perf['problems_improved_percentage'] > 0, \
        "Default strategy should improve some problems in this benchmark set."
        
    # 5. Factor-only should also improve some problems.
    assert factor_only_perf['problems_improved_percentage'] > 0, \
        "Factor-only strategy should improve some problems in this benchmark set."
        
    # 6. Expand-only likely won't improve any problems given the metric is count_ops.
    assert expand_only_perf['problems_improved_percentage'] == 0, \
        "Expand-only strategy with count_ops metric is unlikely to improve problems."

    print("All assertions passed.")

```

I have implemented `evaluate_strategy_performance` and added test cases to `evaluation.py`.
- `get_benchmark_expressions` provides 15 test expressions.
- `evaluate_strategy_performance` calculates various metrics including op counts, reduction, improvement percentage, and time. It correctly uses `strategy.evaluation_metric`.
- The `if __name__ == '__main__':` block now:
    - Creates a `default_strategy` (factor & expand, ops_metric).
    - Creates a `factor_only_strategy` (factor only, ops_metric).
    - Creates an `expand_only_strategy` (expand only, ops_metric) for more thorough testing.
    - Evaluates these strategies on the benchmark expressions.
    - Prints the performance dictionaries.
    - Includes assertions to check for expected behaviors (e.g., `total_ops_reduced` comparisons, improvement percentages).

I'm ready to run this file.
