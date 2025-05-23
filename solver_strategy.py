class SolverStrategy:
    """
    Defines a strategy for the configurable solver.
    """
    def __init__(self, transformations_list, evaluation_metric_func):
        """
        Initializes the solver strategy.

        Args:
            transformations_list: A list of function handles 
                                  (e.g., [apply_distributivity_factor, apply_distributivity_expand]).
                                  These functions take an ExpressionNode and return a (potentially)
                                  transformed ExpressionNode.
            evaluation_metric_func: A function handle that takes an ExpressionNode 
                                    and returns a numerical score (lower is better).
        """
        self.transformations = transformations_list
        self.evaluation_metric = evaluation_metric_func

if __name__ == '__main__':
    # Example usage (not strictly necessary for the class definition but good for illustration)
    from expression_nodes import VariableNode, OperatorNode, apply_distributivity_factor, apply_distributivity_expand

    # Define a simple metric (e.g., count_ops, but could be anything)
    def simple_metric(node):
        return node.count_ops()

    # Create a strategy instance
    strategy1 = SolverStrategy(
        transformations_list=[apply_distributivity_factor, apply_distributivity_expand],
        evaluation_metric_func=simple_metric
    )
    print(f"Strategy 1: Uses {len(strategy1.transformations)} transformations and metric: {strategy1.evaluation_metric.__name__}")

    strategy2 = SolverStrategy(
        transformations_list=[apply_distributivity_factor],
        evaluation_metric_func=simple_metric
    )
    print(f"Strategy 2: Uses {len(strategy2.transformations)} transformation and metric: {strategy2.evaluation_metric.__name__}")

    # Example of using the metric
    a = VariableNode('a')
    b = VariableNode('b')
    expr = OperatorNode('+', a, b)
    print(f"Metric score for '{str(expr)}': {strategy1.evaluation_metric(expr)}")
