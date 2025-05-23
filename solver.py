from expression_nodes import ExpressionNode, VariableNode, OperatorNode, apply_distributivity_expand, apply_distributivity_factor
from solver_strategy import SolverStrategy

# Helper functions

def get_all_nodes(node):
    """
    Performs a traversal of the expression tree and returns a list of all nodes.
    """
    nodes = [node]
    if isinstance(node, OperatorNode):
        nodes.extend(get_all_nodes(node.left_child))
        nodes.extend(get_all_nodes(node.right_child))
    return nodes

def find_paths_recursive(current_node, current_path_list_val, node_path_list_ref):
    """
    Recursively finds all paths to nodes in the tree.
    current_path_list_val: The current path to current_node (list of 'L'/'R').
    node_path_list_ref: A list reference to append (path, node) tuples to.
    """
    node_path_list_ref.append((current_path_list_val, current_node))
    if isinstance(current_node, OperatorNode):
        find_paths_recursive(current_node.left_child, current_path_list_val + ['L'], node_path_list_ref)
        find_paths_recursive(current_node.right_child, current_path_list_val + ['R'], node_path_list_ref)


def configurable_solver(expression_tree: ExpressionNode, strategy: SolverStrategy):
    """
    Applies transformations to an expression tree based on a given strategy
    to find an equivalent expression with a lower score according to the
    strategy's evaluation metric. This version performs a single pass of transformations.
    """
    original_metric_score = strategy.evaluation_metric(expression_tree)
    best_expression = expression_tree.copy() # Start with a copy of the original
    min_metric_score = original_metric_score
    
    candidate_expressions = []
    node_path_list = [] # Using a local list
    
    find_paths_recursive(expression_tree, [], node_path_list) # Pass list to be populated

    for path, original_node_at_path in node_path_list:
        if not isinstance(original_node_at_path, ExpressionNode): 
            continue

        for transformation_func in strategy.transformations:
            # Apply transformation to a copy of the specific sub-node
            transformed_sub_node = transformation_func(original_node_at_path.copy())
            
            if not transformed_sub_node.equals(original_node_at_path): # If transformation occurred
                new_full_tree_candidate = expression_tree.copy() # Fresh copy of entire original tree
                
                if not path: # Root node was transformed
                    new_full_tree_candidate = transformed_sub_node
                else:
                    parent_path = path[:-1]
                    direction = path[-1]
                    
                    node_to_attach_to = new_full_tree_candidate
                    for p_dir in parent_path:
                        if p_dir == 'L':
                            node_to_attach_to = node_to_attach_to.left_child
                        else: # 'R'
                            node_to_attach_to = node_to_attach_to.right_child
                    
                    if direction == 'L':
                        node_to_attach_to.left_child = transformed_sub_node
                    else: # 'R'
                        node_to_attach_to.right_child = transformed_sub_node
                
                candidate_expressions.append(new_full_tree_candidate)

    for candidate in candidate_expressions:
        current_metric_score = strategy.evaluation_metric(candidate)
        if current_metric_score < min_metric_score:
            min_metric_score = current_metric_score
            best_expression = candidate

    return best_expression


if __name__ == '__main__':
    print("--- Testing configurable_solver ---")

    a = VariableNode('a')
    b = VariableNode('b')
    c = VariableNode('c')
    d = VariableNode('d')
    e_var = VariableNode('e') 
    x = VariableNode('x')
    y = VariableNode('y')
    z = VariableNode('z')
    f = VariableNode('f')

    default_metric = lambda node: node.count_ops()
    
    default_strategy = SolverStrategy(
        transformations_list=[apply_distributivity_factor, apply_distributivity_expand],
        evaluation_metric_func=default_metric
    )

    print("\nTest 1: Factor (a*b) + (a*c) [Default Strategy]")
    term1_t1 = OperatorNode('*', a.copy(), b.copy())
    term2_t1 = OperatorNode('*', a.copy(), c.copy())
    expr_t1 = OperatorNode('+', term1_t1, term2_t1)
    print(f"Original T1: {str(expr_t1)}, Metric: {default_strategy.evaluation_metric(expr_t1)}")
    solved_t1 = configurable_solver(expr_t1, default_strategy)
    print(f"Solved T1: {str(solved_t1)}, Metric: {default_strategy.evaluation_metric(solved_t1)}")
    expected_t1_inner = OperatorNode('+', b.copy(), c.copy())
    expected_t1 = OperatorNode('*', a.copy(), expected_t1_inner)
    if solved_t1.equals(expected_t1) and default_strategy.evaluation_metric(solved_t1) < default_strategy.evaluation_metric(expr_t1):
        print("Test 1 PASSED")
    else:
        print("Test 1 FAILED")

    print("\nTest 2: Expand x*(y+z) (no change - default strategy)")
    inner_t2 = OperatorNode('+', y.copy(), z.copy())
    expr_t2 = OperatorNode('*', x.copy(), inner_t2)
    print(f"Original T2: {str(expr_t2)}, Metric: {default_strategy.evaluation_metric(expr_t2)}")
    solved_t2 = configurable_solver(expr_t2, default_strategy)
    print(f"Solved T2: {str(solved_t2)}, Metric: {default_strategy.evaluation_metric(solved_t2)}")
    if solved_t2.equals(expr_t2) and default_strategy.evaluation_metric(solved_t2) == default_strategy.evaluation_metric(expr_t2):
        print("Test 2 PASSED")
    else:
        print("Test 2 FAILED")

    print("\nTest 3: Complex Factor ((a*b)+(a*c))+d [Default Strategy]")
    term1_t3 = OperatorNode('*', a.copy(), b.copy())
    term2_t3 = OperatorNode('*', a.copy(), c.copy())
    sum_t3 = OperatorNode('+', term1_t3, term2_t3)
    expr_t3 = OperatorNode('+', sum_t3, d.copy())
    print(f"Original T3: {str(expr_t3)}, Metric: {default_strategy.evaluation_metric(expr_t3)}")
    solved_t3 = configurable_solver(expr_t3, default_strategy)
    print(f"Solved T3: {str(solved_t3)}, Metric: {default_strategy.evaluation_metric(solved_t3)}")
    expected_inner_sum_t3 = OperatorNode('+', b.copy(), c.copy())
    expected_factor_part_t3 = OperatorNode('*', a.copy(), expected_inner_sum_t3)
    expected_t3 = OperatorNode('+', expected_factor_part_t3, d.copy())
    if solved_t3.equals(expected_t3) and default_strategy.evaluation_metric(solved_t3) < default_strategy.evaluation_metric(expr_t3):
        print("Test 3 PASSED")
    else:
        print("Test 3 FAILED")

    print("\nTest 4: Simple a+b (no change) [Default Strategy]")
    expr_t4 = OperatorNode('+', a.copy(), b.copy())
    print(f"Original T4: {str(expr_t4)}, Metric: {default_strategy.evaluation_metric(expr_t4)}")
    solved_t4 = configurable_solver(expr_t4, default_strategy)
    print(f"Solved T4: {str(solved_t4)}, Metric: {default_strategy.evaluation_metric(solved_t4)}")
    if solved_t4.equals(expr_t4) and default_strategy.evaluation_metric(solved_t4) == default_strategy.evaluation_metric(expr_t4):
        print("Test 4 PASSED")
    else:
        print("Test 4 FAILED")

    print("\nTest 5: Expand (a+b)*c (no change) [Default Strategy]")
    inner_t5 = OperatorNode('+', a.copy(), b.copy())
    expr_t5 = OperatorNode('*', inner_t5, c.copy())
    print(f"Original T5: {str(expr_t5)}, Metric: {default_strategy.evaluation_metric(expr_t5)}")
    solved_t5 = configurable_solver(expr_t5, default_strategy)
    print(f"Solved T5: {str(solved_t5)}, Metric: {default_strategy.evaluation_metric(solved_t5)}")
    if solved_t5.equals(expr_t5) and default_strategy.evaluation_metric(solved_t5) == default_strategy.evaluation_metric(expr_t5):
        print("Test 5 PASSED")
    else:
        print("Test 5 FAILED")

    print("\nTest 6: Factor (x*y) + (z*x) [Default Strategy]")
    term1_t6 = OperatorNode('*', x.copy(), y.copy())
    term2_t6 = OperatorNode('*', z.copy(), x.copy())
    expr_t6 = OperatorNode('+', term1_t6, term2_t6)
    print(f"Original T6: {str(expr_t6)}, Metric: {default_strategy.evaluation_metric(expr_t6)}")
    solved_t6 = configurable_solver(expr_t6, default_strategy)
    print(f"Solved T6: {str(solved_t6)}, Metric: {default_strategy.evaluation_metric(solved_t6)}")
    expected_t6_v1_inner = OperatorNode('+', y.copy(), z.copy())
    expected_t6_v1 = OperatorNode('*', x.copy(), expected_t6_v1_inner)
    expected_t6_v2_inner = OperatorNode('+', y.copy(), z.copy())
    expected_t6_v2 = OperatorNode('*', expected_t6_v2_inner, x.copy())
    if (solved_t6.equals(expected_t6_v1) or solved_t6.equals(expected_t6_v2)) and \
       default_strategy.evaluation_metric(solved_t6) < default_strategy.evaluation_metric(expr_t6):
        print("Test 6 PASSED")
    else:
        print("Test 6 FAILED")
        
    print("\nTest 7: Nested expansion a*(b*(c+d)) (no change) [Default Strategy]")
    inner_cd_t7 = OperatorNode('+', c.copy(), d.copy())
    inner_bcd_t7 = OperatorNode('*', b.copy(), inner_cd_t7)
    expr_t7 = OperatorNode('*', a.copy(), inner_bcd_t7)
    print(f"Original T7: {str(expr_t7)}, Metric: {default_strategy.evaluation_metric(expr_t7)}")
    solved_t7 = configurable_solver(expr_t7, default_strategy)
    print(f"Solved T7: {str(solved_t7)}, Metric: {default_strategy.evaluation_metric(solved_t7)}")
    if solved_t7.equals(expr_t7) and default_strategy.evaluation_metric(solved_t7) == default_strategy.evaluation_metric(expr_t7):
        print("Test 7 PASSED")
    else:
        print("Test 7 FAILED")

    print("\nTest 8: Double Factor (single step improvement) [Default Strategy]")
    term_ab_t8 = OperatorNode('*', a.copy(), b.copy())
    term_ac_t8 = OperatorNode('*', a.copy(), c.copy())
    left_sum_t8 = OperatorNode('+', term_ab_t8, term_ac_t8)
    term_xy_t8 = OperatorNode('*', x.copy(), y.copy())
    term_xz_t8 = OperatorNode('*', x.copy(), z.copy())
    right_sum_t8 = OperatorNode('+', term_xy_t8, term_xz_t8)
    expr_t8 = OperatorNode('*', left_sum_t8, right_sum_t8)
    original_metric_t8 = default_strategy.evaluation_metric(expr_t8)
    print(f"Original T8: {str(expr_t8)}, Metric: {original_metric_t8}")
    solved_t8 = configurable_solver(expr_t8, default_strategy)
    solved_metric_t8 = default_strategy.evaluation_metric(solved_t8)
    print(f"Solved T8: {str(solved_t8)}, Metric: {solved_metric_t8}")
    exp_left_bc_t8 = OperatorNode('+', b.copy(), c.copy())
    exp_left_factor_t8 = OperatorNode('*', a.copy(), exp_left_bc_t8)
    expected_t8_v1 = OperatorNode('*', exp_left_factor_t8, right_sum_t8.copy()) 
    exp_right_yz_t8 = OperatorNode('+', y.copy(), z.copy())
    exp_right_factor_t8 = OperatorNode('*', x.copy(), exp_right_yz_t8)
    expected_t8_v2 = OperatorNode('*', left_sum_t8.copy(), exp_right_factor_t8) 
    if (solved_t8.equals(expected_t8_v1) or solved_t8.equals(expected_t8_v2)) and solved_metric_t8 < original_metric_t8:
        print(f"Test 8 PASSED (achieved {solved_metric_t8} vs original {original_metric_t8})")
    else:
        print(f"Test 8 FAILED. Solved: {str(solved_t8)}, Metric: {solved_metric_t8}. Expected v1: {str(expected_t8_v1)} or v2: {str(expected_t8_v2)}")

    print("\nTest 9: (a*b)+(c*d) (no change) [Default Strategy]")
    expr_t9 = OperatorNode('+', OperatorNode('*', a.copy(), b.copy()), OperatorNode('*', c.copy(), d.copy()))
    print(f"Original T9: {str(expr_t9)}, Metric: {default_strategy.evaluation_metric(expr_t9)}")
    solved_t9 = configurable_solver(expr_t9, default_strategy)
    print(f"Solved T9: {str(solved_t9)}, Metric: {default_strategy.evaluation_metric(solved_t9)}")
    if solved_t9.equals(expr_t9) and default_strategy.evaluation_metric(solved_t9) == default_strategy.evaluation_metric(expr_t9):
        print("Test 9 PASSED")
    else:
        print("Test 9 FAILED")

    print("\nTest 10: Factor a*(b+c) + a*(d+e) [Default Strategy]")
    bc_t10 = OperatorNode('+', b.copy(), c.copy())
    de_t10 = OperatorNode('+', d.copy(), e_var.copy()) 
    term1_t10 = OperatorNode('*', a.copy(), bc_t10)
    term2_t10 = OperatorNode('*', a.copy(), de_t10)
    expr_t10 = OperatorNode('+', term1_t10, term2_t10)
    print(f"Original T10: {str(expr_t10)}, Metric: {default_strategy.evaluation_metric(expr_t10)}")
    solved_t10 = configurable_solver(expr_t10, default_strategy)
    print(f"Solved T10: {str(solved_t10)}, Metric: {default_strategy.evaluation_metric(solved_t10)}")
    sum_bc_de_t10 = OperatorNode('+', bc_t10.copy(), de_t10.copy())
    expected_t10 = OperatorNode('*', a.copy(), sum_bc_de_t10)
    if solved_t10.equals(expected_t10) and default_strategy.evaluation_metric(solved_t10) < default_strategy.evaluation_metric(expr_t10):
        print("Test 10 PASSED")
    else:
        print("Test 10 FAILED")

    print("\nTest 11: (a+b)*(c+d) (no change) [Default Strategy]")
    ab_t11 = OperatorNode('+', a.copy(), b.copy())
    cd_t11 = OperatorNode('+', c.copy(), d.copy())
    expr_t11 = OperatorNode('*', ab_t11, cd_t11)
    print(f"Original T11: {str(expr_t11)}, Metric: {default_strategy.evaluation_metric(expr_t11)}")
    solved_t11 = configurable_solver(expr_t11, default_strategy)
    print(f"Solved T11: {str(solved_t11)}, Metric: {default_strategy.evaluation_metric(solved_t11)}")
    if solved_t11.equals(expr_t11) and default_strategy.evaluation_metric(solved_t11) == default_strategy.evaluation_metric(expr_t11):
        print("Test 11 PASSED")
    else:
        print("Test 11 FAILED")
        
    print("\nTest 12: Complex factor with trailing term [Default Strategy]")
    expr_t12_original_sum = OperatorNode('+', OperatorNode('*', a.copy(), OperatorNode('+', b.copy(), c.copy())), OperatorNode('*', a.copy(), OperatorNode('+', d.copy(), e_var.copy())))
    expr_t12 = OperatorNode('+', expr_t12_original_sum, f.copy()) 
    print(f"Original T12: {str(expr_t12)}, Metric: {default_strategy.evaluation_metric(expr_t12)}")
    solved_t12 = configurable_solver(expr_t12, default_strategy)
    print(f"Solved T12: {str(solved_t12)}, Metric: {default_strategy.evaluation_metric(solved_t12)}")
    
    expected_sum_bc_de_t12 = OperatorNode('+', OperatorNode('+',b.copy(),c.copy()), OperatorNode('+',d.copy(),e_var.copy()))
    expected_factored_part_t12 = OperatorNode('*', a.copy(), expected_sum_bc_de_t12)
    expected_t12 = OperatorNode('+', expected_factored_part_t12, f.copy())
    if solved_t12.equals(expected_t12) and default_strategy.evaluation_metric(solved_t12) < default_strategy.evaluation_metric(expr_t12):
        print("Test 12 PASSED")
    else:
        print("Test 12 FAILED")

    print("\nTest 13: Root is a variable [Default Strategy]")
    expr_t13 = a.copy()
    print(f"Original T13: {str(expr_t13)}, Metric: {default_strategy.evaluation_metric(expr_t13)}")
    solved_t13 = configurable_solver(expr_t13, default_strategy)
    print(f"Solved T13: {str(solved_t13)}, Metric: {default_strategy.evaluation_metric(solved_t13)}")
    if solved_t13.equals(expr_t13) and default_strategy.evaluation_metric(solved_t13) == default_strategy.evaluation_metric(expr_t13):
        print("Test 13 PASSED")
    else:
        print("Test 13 FAILED")
        
    print("\nTest 14: Factor (y*x) + (z*x) -> (y+z)*x [Default Strategy]")
    term1_t14 = OperatorNode('*', y.copy(), x.copy())
    term2_t14 = OperatorNode('*', z.copy(), x.copy())
    expr_t14 = OperatorNode('+', term1_t14, term2_t14)
    print(f"Original T14: {str(expr_t14)}, Metric: {default_strategy.evaluation_metric(expr_t14)}")
    solved_t14 = configurable_solver(expr_t14, default_strategy)
    print(f"Solved T14: {str(solved_t14)}, Metric: {default_strategy.evaluation_metric(solved_t14)}")
    expected_t14_inner = OperatorNode('+', y.copy(), z.copy())
    expected_t14 = OperatorNode('*', expected_t14_inner, x.copy())
    if solved_t14.equals(expected_t14) and default_strategy.evaluation_metric(solved_t14) < default_strategy.evaluation_metric(expr_t14):
        print("Test 14 PASSED")
    else:
        print("Test 14 FAILED")
        
    print("\nTest 15: Factor (a*b) + (a*b) [Default Strategy]")
    term_ab1_t15 = OperatorNode('*', a.copy(), b.copy())
    term_ab2_t15 = OperatorNode('*', a.copy(), b.copy())
    expr_t15 = OperatorNode('+', term_ab1_t15, term_ab2_t15)
    print(f"Original T15: {str(expr_t15)}, Metric: {default_strategy.evaluation_metric(expr_t15)}")
    solved_t15 = configurable_solver(expr_t15, default_strategy)
    print(f"Solved T15: {str(solved_t15)}, Metric: {default_strategy.evaluation_metric(solved_t15)}")
    expected_b_plus_b_t15 = OperatorNode('+', b.copy(), b.copy())
    expected_t15 = OperatorNode('*', a.copy(), expected_b_plus_b_t15)
    if solved_t15.equals(expected_t15) and default_strategy.evaluation_metric(solved_t15) < default_strategy.evaluation_metric(expr_t15):
        print("Test 15 PASSED")
    else:
        print("Test 15 FAILED")

    print("\n--- Testing Configurability ---")

    print("\nTest 16: x*(y+z) with Factor-Only Strategy (no change)")
    factor_only_strategy = SolverStrategy(
        transformations_list=[apply_distributivity_factor], 
        evaluation_metric_func=default_metric
    )
    # expr_t2 is x*(y+z)
    print(f"Original T16 (expr_t2): {str(expr_t2)}, Metric: {factor_only_strategy.evaluation_metric(expr_t2)}")
    solved_t16 = configurable_solver(expr_t2.copy(), factor_only_strategy)
    print(f"Solved T16: {str(solved_t16)}, Metric: {factor_only_strategy.evaluation_metric(solved_t16)}")
    if solved_t16.equals(expr_t2) and factor_only_strategy.evaluation_metric(solved_t16) == factor_only_strategy.evaluation_metric(expr_t2):
        print("Test 16 PASSED")
    else:
        print("Test 16 FAILED")

    print("\nTest 17: (a*b)+(a*c) with Expand-Only Strategy (no change)")
    expand_only_strategy = SolverStrategy(
        transformations_list=[apply_distributivity_expand], 
        evaluation_metric_func=default_metric
    )
    # expr_t1 is (a*b)+(a*c)
    print(f"Original T17 (expr_t1): {str(expr_t1)}, Metric: {expand_only_strategy.evaluation_metric(expr_t1)}")
    solved_t17 = configurable_solver(expr_t1.copy(), expand_only_strategy)
    print(f"Solved T17: {str(solved_t17)}, Metric: {expand_only_strategy.evaluation_metric(solved_t17)}")
    if solved_t17.equals(expr_t1) and expand_only_strategy.evaluation_metric(solved_t17) == expand_only_strategy.evaluation_metric(expr_t1):
        print("Test 17 PASSED")
    else:
        print("Test 17 FAILED")
        
    def count_unique_vars_metric(node):
        vars_found = set()
        q = [node]
        while q:
            curr = q.pop(0)
            if isinstance(curr, VariableNode):
                vars_found.add(curr.name)
            elif isinstance(curr, OperatorNode):
                q.append(curr.left_child)
                q.append(curr.right_child)
        return -len(vars_found) 

    custom_metric_strategy = SolverStrategy(
        transformations_list=[apply_distributivity_factor, apply_distributivity_expand],
        evaluation_metric_func=count_unique_vars_metric
    )
    
    print("\nTest 18: x*(y+z) with Custom Metric (prefer more unique variables)")
    print(f"Original T18 (expr_t2): {str(expr_t2)}, Custom Metric: {custom_metric_strategy.evaluation_metric(expr_t2)}")
    solved_t18 = configurable_solver(expr_t2.copy(), custom_metric_strategy)
    print(f"Solved T18: {str(solved_t18)}, Custom Metric: {custom_metric_strategy.evaluation_metric(solved_t18)}")
    if solved_t18.equals(expr_t2): 
        print("Test 18 PASSED (custom metric correctly used, no change preferred as scores are equal)")
    else:
        print("Test 18 FAILED")

print("--- Finished testing configurable_solver ---")
