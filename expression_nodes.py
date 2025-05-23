class ExpressionNode:
    def __str__(self):
        raise NotImplementedError

    def count_ops(self):
        raise NotImplementedError

    def equals(self, other):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class VariableNode(ExpressionNode):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def count_ops(self):
        return 0

    def equals(self, other):
        if not isinstance(other, VariableNode):
            return False
        return self.name == other.name

    def copy(self):
        return VariableNode(self.name)


class OperatorNode(ExpressionNode):
    def __init__(self, operator_char, left_child, right_child):
        self.operator_char = operator_char
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return f"({str(self.left_child)}{self.operator_char}{str(self.right_child)})"

    def count_ops(self):
        return 1 + self.left_child.count_ops() + self.right_child.count_ops()

    def equals(self, other):
        if not isinstance(other, OperatorNode):
            return False
        return (self.operator_char == other.operator_char and
                self.left_child.equals(other.left_child) and
                self.right_child.equals(other.right_child))

    def copy(self):
        return OperatorNode(self.operator_char, self.left_child.copy(), self.right_child.copy())

if __name__ == '__main__':
    # Test cases
    a = VariableNode('a')
    b = VariableNode('b')
    c = VariableNode('c')
    x = VariableNode('x')
    y = VariableNode('y')
    z = VariableNode('z')

    # Expression: a*(b+c)
    expr1_inner = OperatorNode('+', b, c)
    expr1 = OperatorNode('*', a, expr1_inner)
    print(f"Expression 1: {str(expr1)}")
    print(f"Ops in Expression 1: {expr1.count_ops()}")

    # Expression: (a*b)+(a*c)
    expr2_left = OperatorNode('*', a, b)
    expr2_right = OperatorNode('*', a, c)
    expr2 = OperatorNode('+', expr2_left, expr2_right)
    print(f"Expression 2: {str(expr2)}")
    print(f"Ops in Expression 2: {expr2.count_ops()}")
    
    # Test equals
    a1 = VariableNode('a')
    a2 = VariableNode('a')
    b1 = VariableNode('b')
    print(f"a.equals(a1): {a.equals(a1)}") # True
    print(f"a1.equals(a2): {a1.equals(a2)}") # True
    print(f"a.equals(b1): {a.equals(b1)}") # False

    op1 = OperatorNode('+', VariableNode('a'), VariableNode('b'))
    op2 = OperatorNode('+', VariableNode('a'), VariableNode('b'))
    op3 = OperatorNode('*', VariableNode('a'), VariableNode('b'))
    op4 = OperatorNode('+', VariableNode('a'), VariableNode('c'))
    print(f"op1.equals(op2): {op1.equals(op2)}") # True
    print(f"op1.equals(op3): {op1.equals(op3)}") # False
    print(f"op1.equals(op4): {op1.equals(op4)}") # False

    # Test complex equals
    # expr1_copy = a*(b+c)
    expr1_copy_inner = OperatorNode('+', VariableNode('b'), VariableNode('c'))
    expr1_copy = OperatorNode('*', VariableNode('a'), expr1_copy_inner)
    print(f"expr1.equals(expr1_copy): {expr1.equals(expr1_copy)}") # True
    print(f"expr1.equals(expr2): {expr1.equals(expr2)}") # False

    # Test string representation for order of operations
    # (a*(b+c))
    node1 = OperatorNode('+', VariableNode('b'), VariableNode('c')) # (b+c)
    node2 = OperatorNode('*', VariableNode('a'), node1) # (a*(b+c))
    print(f"Order of ops 1: {str(node2)}")

    # ((a*b)+c)
    node3 = OperatorNode('*', VariableNode('a'), VariableNode('b')) # (a*b)
    node4 = OperatorNode('+', node3, VariableNode('c')) # ((a*b)+c)
    print(f"Order of ops 2: {str(node4)}")

    # (a+(b*c))
    node5 = OperatorNode('*', VariableNode('b'), VariableNode('c')) # (b*c)
    node6 = OperatorNode('+', VariableNode('a'), node5) # (a+(b*c))
    print(f"Order of ops 3: {str(node6)}")

    # ((a+b)*c)
    node7 = OperatorNode('+', VariableNode('a'), VariableNode('b')) # (a+b)
    node8 = OperatorNode('*', node7, VariableNode('c')) # ((a+b)*c)
    print(f"Order of ops 4: {str(node8)}")


def apply_distributivity_expand(node):
    # a*(b+c) -> (a*b)+(a*c)
    if isinstance(node, OperatorNode) and node.operator_char == '*':
        # Left distributivity: node = X * (Y+Z)
        if isinstance(node.right_child, OperatorNode) and node.right_child.operator_char == '+':
            x = node.left_child
            y = node.right_child.left_child
            z = node.right_child.right_child
            # (X*Y) + (X*Z)
            return OperatorNode('+', OperatorNode('*', x, y), OperatorNode('*', x, z))
        # Right distributivity: node = (X+Y) * Z
        elif isinstance(node.left_child, OperatorNode) and node.left_child.operator_char == '+':
            x = node.left_child.left_child
            y = node.left_child.right_child
            z = node.right_child
            # (X*Z) + (Y*Z)
            return OperatorNode('+', OperatorNode('*', x, z), OperatorNode('*', y, z))
    return node


def apply_distributivity_factor(node):
    # (a*b)+(a*c) -> a*(b+c)
    if isinstance(node, OperatorNode) and node.operator_char == '+':
        term1 = node.left_child
        term2 = node.right_child

        # Check term1 is '*' and term2 is '*'
        if not (isinstance(term1, OperatorNode) and term1.operator_char == '*' and
                isinstance(term2, OperatorNode) and term2.operator_char == '*'):
            return node

        # Pattern (X*Y) + (X*Z) -> X*(Y+Z)
        # term1 = X*Y, term2 = X*Z
        if term1.left_child.equals(term2.left_child):
            x = term1.left_child
            y = term1.right_child
            z = term2.right_child
            return OperatorNode('*', x, OperatorNode('+', y, z))
        
        # Pattern (Y*X) + (Z*X) -> (Y+Z)*X
        # term1 = Y*X, term2 = Z*X
        if term1.right_child.equals(term2.right_child):
            x = term1.right_child # The common factor
            y = term1.left_child
            z = term2.left_child
            return OperatorNode('*', OperatorNode('+', y, z), x)
        
        # Pattern (X*Y) + (Z*X) where X is common, but on different sides
        # term1 = X*Y, term2 = Z*X. If X is common, factor it to the left: X*(Y+Z)
        if term1.left_child.equals(term2.right_child):
            x = term1.left_child # The common factor
            y = term1.right_child
            z = term2.left_child
            return OperatorNode('*', x, OperatorNode('+', y, z))

        # Pattern (Y*X) + (X*Z) where X is common, but on different sides
        # term1 = Y*X, term2 = X*Z. If X is common, factor it to the left: X*(Y+Z)
        if term1.right_child.equals(term2.left_child):
            x = term1.right_child # The common factor
            y = term1.left_child
            z = term2.right_child
            return OperatorNode('*', x, OperatorNode('+', y, z))

    return node


if __name__ == '__main__':
    # Test cases
    a = VariableNode('a')
    b = VariableNode('b')
    c = VariableNode('c')
    x = VariableNode('x')
    y = VariableNode('y')
    z = VariableNode('z')

    # Expression: a*(b+c)
    expr1_inner = OperatorNode('+', b, c)
    expr1 = OperatorNode('*', a, expr1_inner)
    print(f"Original Expression 1: {str(expr1)}")
    print(f"Ops in Expression 1: {expr1.count_ops()}")

    expanded_expr1 = apply_distributivity_expand(expr1)
    print(f"Expanded Expression 1: {str(expanded_expr1)}")
    print(f"Ops in Expanded Expression 1: {expanded_expr1.count_ops()}")
    
    # Expression: (a+b)*c
    expr3_inner = OperatorNode('+', a, b)
    expr3 = OperatorNode('*', expr3_inner, c)
    print(f"Original Expression 3: {str(expr3)}")
    print(f"Ops in Expression 3: {expr3.count_ops()}")
    
    expanded_expr3 = apply_distributivity_expand(expr3)
    print(f"Expanded Expression 3: {str(expanded_expr3)}")
    print(f"Ops in Expanded Expression 3: {expanded_expr3.count_ops()}")

    # Expression: (x*y) + (x*z)
    expr2_left = OperatorNode('*', x, y)
    expr2_right = OperatorNode('*', x, z)
    expr2 = OperatorNode('+', expr2_left, expr2_right)
    print(f"Original Expression 2: {str(expr2)}")
    print(f"Ops in Expression 2: {expr2.count_ops()}")

    factored_expr2 = apply_distributivity_factor(expr2)
    print(f"Factored Expression 2: {str(factored_expr2)}")
    print(f"Ops in Factored Expression 2: {factored_expr2.count_ops()}")

    # Expression: (y*x) + (z*x)
    expr4_left = OperatorNode('*', y, x)
    expr4_right = OperatorNode('*', z, x)
    expr4 = OperatorNode('+', expr4_left, expr4_right)
    print(f"Original Expression 4: {str(expr4)}") # ((y*x)+(z*x))
    print(f"Ops in Expression 4: {expr4.count_ops()}") # 3

    factored_expr4 = apply_distributivity_factor(expr4)
    print(f"Factored Expression 4: {str(factored_expr4)}") # ((y+z)*x)
    print(f"Ops in Factored Expression 4: {factored_expr4.count_ops()}") # 2
    
    # Test equals
    a1 = VariableNode('a')
    a2 = VariableNode('a')
    b1 = VariableNode('b')
    print(f"a.equals(a1): {a.equals(a1)}") 
    print(f"a1.equals(a2): {a1.equals(a2)}")
    print(f"a.equals(b1): {a.equals(b1)}") 

    op1 = OperatorNode('+', VariableNode('a'), VariableNode('b'))
    op2 = OperatorNode('+', VariableNode('a'), VariableNode('b'))
    op3 = OperatorNode('*', VariableNode('a'), VariableNode('b'))
    op4 = OperatorNode('+', VariableNode('a'), VariableNode('c'))
    print(f"op1.equals(op2): {op1.equals(op2)}") 
    print(f"op1.equals(op3): {op1.equals(op3)}") 
    print(f"op1.equals(op4): {op1.equals(op4)}") 

    # Test complex equals
    # expr1_copy = a*(b+c)
    expr1_copy_inner = OperatorNode('+', VariableNode('b'), VariableNode('c'))
    expr1_copy = OperatorNode('*', VariableNode('a'), expr1_copy_inner)
    print(f"expr1.equals(expr1_copy): {expr1.equals(expr1_copy)}") 
    # This was: print(f"expr1.equals(expr2): {expr1.equals(expr2)}") # False
    # expr2 is (x*y) + (x*z) and expr1 is a*(b+c), so this should be false.
    # expanded_expr1 is (a*b)+(a*c)
    print(f"expanded_expr1.equals(expr2): {expanded_expr1.equals(expr2)}") # False, unless a=x,b=y,c=z

    # Test string representation for order of operations
    # (a*(b+c))
    node1 = OperatorNode('+', VariableNode('b'), VariableNode('c')) # (b+c)
    node2 = OperatorNode('*', VariableNode('a'), node1) # (a*(b+c))
    print(f"Order of ops 1: {str(node2)}")

    # ((a*b)+c)
    node3 = OperatorNode('*', VariableNode('a'), VariableNode('b')) # (a*b)
    node4 = OperatorNode('+', node3, VariableNode('c')) # ((a*b)+c)
    print(f"Order of ops 2: {str(node4)}")

    # (a+(b*c))
    node5 = OperatorNode('*', VariableNode('b'), VariableNode('c')) # (b*c)
    node6 = OperatorNode('+', VariableNode('a'), node5) # (a+(b*c))
    print(f"Order of ops 3: {str(node6)}")

    # ((a+b)*c)
    node7 = OperatorNode('+', VariableNode('a'), VariableNode('b')) # (a+b)
    node8 = OperatorNode('*', node7, VariableNode('c')) # ((a+b)*c)
    print(f"Order of ops 4: {str(node8)}")

    # Test cases for factoring with common factor on different sides
    # (a*b) + (c*a) -> a*(b+c)
    expr5_left = OperatorNode('*', a, b)  # a*b
    expr5_right = OperatorNode('*', c, a) # c*a
    expr5 = OperatorNode('+', expr5_left, expr5_right) # (a*b)+(c*a)
    print(f"Original Expression 5: {str(expr5)}")
    print(f"Ops in Expression 5: {expr5.count_ops()}")
    factored_expr5 = apply_distributivity_factor(expr5)
    print(f"Factored Expression 5: {str(factored_expr5)}") # Should be (a*(b+c))
    print(f"Ops in Factored Expression 5: {factored_expr5.count_ops()}")

    # (b*a) + (a*c) -> a*(b+c)
    expr6_left = OperatorNode('*', b, a)  # b*a
    expr6_right = OperatorNode('*', a, c) # a*c
    expr6 = OperatorNode('+', expr6_left, expr6_right) # (b*a)+(a*c)
    print(f"Original Expression 6: {str(expr6)}")
    print(f"Ops in Expression 6: {expr6.count_ops()}")
    factored_expr6 = apply_distributivity_factor(expr6)
    print(f"Factored Expression 6: {str(factored_expr6)}") # Should be (a*(b+c))
    print(f"Ops in Factored Expression 6: {factored_expr6.count_ops()}")

    # Test no expansion/factoring if not applicable
    no_op_expand = apply_distributivity_expand(op1) # op1 is (a+b)
    print(f"No op expand on (a+b): {str(no_op_expand)}, equals op1: {no_op_expand.equals(op1)}")

    no_op_factor = apply_distributivity_factor(op1) # op1 is (a+b)
    print(f"No op factor on (a+b): {str(no_op_factor)}, equals op1: {no_op_factor.equals(op1)}")
    
    # Test (a*b) + (c*d) -> should not factor
    expr_no_factor_left = OperatorNode('*', a,b)
    expr_no_factor_right = OperatorNode('*', c,x) # using x instead of d for variety
    expr_no_factor = OperatorNode('+', expr_no_factor_left, expr_no_factor_right)
    print(f"Original No Factor Expr: {str(expr_no_factor)}")
    factored_expr_no_factor = apply_distributivity_factor(expr_no_factor)
    print(f"Factored No Factor Expr: {str(factored_expr_no_factor)}, equals original: {factored_expr_no_factor.equals(expr_no_factor)}")

    # Test copy method
    print("\n--- Testing Copy ---")
    a_orig = VariableNode('a')
    a_copy = a_orig.copy()
    print(f"a_orig: {str(a_orig)}, a_copy: {str(a_copy)}")
    print(f"a_orig is a_copy: {a_orig is a_copy}") # False
    print(f"a_orig.equals(a_copy): {a_orig.equals(a_copy)}") # True

    # expr1 = a*(b+c)
    expr1_copy_full = expr1.copy()
    print(f"expr1: {str(expr1)}, expr1_copy_full: {str(expr1_copy_full)}")
    print(f"expr1 is expr1_copy_full: {expr1 is expr1_copy_full}") # False
    print(f"expr1.equals(expr1_copy_full): {expr1.equals(expr1_copy_full)}") # True
    # Check for deep copy: modify original, copy should not change
    # Original expr1: OperatorNode('*', a, OperatorNode('+', b, c))
    if isinstance(expr1.left_child, VariableNode):
        expr1.left_child.name = 'new_a' # Modify part of the original expr1
    if isinstance(expr1.right_child, OperatorNode) and isinstance(expr1.right_child.left_child, VariableNode):
         expr1.right_child.left_child.name = 'new_b'

    print(f"expr1 modified: {str(expr1)}") # (new_a*(new_b+c))
    print(f"expr1_copy_full (should be unchanged): {str(expr1_copy_full)}") # (a*(b+c))
    print(f"expr1.equals(expr1_copy_full) after mod: {expr1.equals(expr1_copy_full)}") # False

    # Restore expr1 for other tests if necessary (or just use fresh variables)
    a = VariableNode('a') # re-init 'a'
    b = VariableNode('b') # re-init 'b'
    expr1_inner_re = OperatorNode('+', b, c) # re-init inner part of expr1
    expr1 = OperatorNode('*', a, expr1_inner_re) # re-init expr1
    print(f"expr1 restored for subsequent tests: {str(expr1)}")

