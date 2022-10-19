"""
Environment used to handle spinor helicity expressions
"""

import numpy as np
import random
import sympy as sp
from environment.utils import reorder_expr, generate_random_bk, get_scaling_expr, random_scale_factor
from sympy import Function, latex


class ab(Function):
    """Angle Bracket class"""
    def _latex(self, printer, exp=1):
        """Overwriting the latex outputs to get something nicer"""
        a, b = [printer._print(i) for i in self.args]
        a = a.replace('p_{', "").replace('}', "")
        b = b.replace('p_{', "").replace('}', "")
        if exp == 1:
            return r"\langle %s %s \rangle" % (a, b)
        else:
            return r"\langle %s %s \rangle^{%s}" % (a, b, exp)


class sb(Function):
    """Square Bracket class"""
    def _latex(self, printer, exp=1):
        """Overwriting the latex outputs to get something nicer"""
        a, b = [printer._print(i) for i in self.args]
        a = a.replace('p_{', "").replace('}', "")
        b = b.replace('p_{', "").replace('}', "")
        if exp == 1:
            return r"\left[ %s %s \right]" % (a, b)
        else:
            return r"\left[ %s %s \right]^{%s}" % (a, b, exp)


class SpinHelExpr:
    """
    Class that handles dealing with Spinor Helicity amplitudes
    """

    def __init__(self, expr: str, n_pt=None):
        """Initialize the expression starting from a string"""
        self.str_expr = expr
        self.functions = [ab, sb]
        self.func_dict = {'ab': ab, 'sb': sb}
        self.sp_expr = sp.parse_expr(self.str_expr, local_dict=self.func_dict)

        if n_pt is None:
            self.n_point = np.max(np.array([list(f.args) for f in self.sp_expr.atoms(Function)]))
        else:
            self.n_point = n_pt

    def __str__(self):
        """Nicer printout"""
        return_str = ''
        curr_bk = ''
        for char in self.str_expr:
            if return_str != '' and return_str[-1] == '*':
                return_str = return_str[:-1]
                if char == '*':
                    return_str += '^'
            if char == 'a':
                curr_bk = 'a'
            elif char == 's':
                curr_bk = 's'
            elif char in ["b", ",", " "]:
                continue
            elif char in ["(", ")"]:
                if curr_bk == "a":
                    return_str += "<" if char == "(" else ">"
                if curr_bk == "s":
                    return_str += "[" if char == "(" else "]"
                if curr_bk == '':
                    return_str += char
                if char == ")":
                    curr_bk = ''
            else:
                return_str += char

        return return_str

    def _latex(self, printer):
        """Can output latex directly"""
        return latex(self.sp_expr)

    def antisymm(self, bk, pi, pj):
        """ Apply the antisymmetry identity for square and angle brackets"""
        self.sp_expr = self.sp_expr.subs(self.func_dict[bk](pi, pj), -self.func_dict[bk](pj, pi))
        self.str_expr = str(self.sp_expr)

    def schouten(self, bk, pi, pj, pk, pl):
        """ Apply the Schouten identity to a selected combination of four variables
        e.g: apply it on <1,2><3,4> where the pattern has to be explicitly present in the expression"""
        ret_expr = self.func_dict[bk](pi, pl)*self.func_dict[bk](pk, pj) + \
                   self.func_dict[bk](pi, pk)*self.func_dict[bk](pj, pl)
        self.sp_expr = self.sp_expr.subs(self.func_dict[bk](pi, pj)*self.func_dict[bk](pk, pl), ret_expr)
        self.str_expr = str(self.sp_expr)

    def schouten2(self, bk, pi, pj, pk, pl, canonical=False):
        """ Apply the Schouten to a selected pair of two variables
        e.g apply it to <1,2> where the pattern has to be explicitly present in the expression"""

        ret_expr = self.func_dict[bk](pi, pl)*self.func_dict[bk](pk, pj)/self.func_dict[bk](pk, pl) + \
                   self.func_dict[bk](pi, pk)*self.func_dict[bk](pj, pl)/self.func_dict[bk](pk, pl)

        if canonical:
            ret_expr = reorder_expr(ret_expr)

        self.sp_expr = self.sp_expr.subs(self.func_dict[bk](pi, pj), ret_expr)
        self.str_expr = str(self.sp_expr)

    def momentum(self, pi, pj, pk, max_label):
        """Implement momentum conservation on the given pattern
        For example we use <1,2>[2,3] = - <1,4>[4,3] if we have 4 labels at most
        Can have pi==pk which instead implies <1,2>[2,1] = - <1,3>[3,1] - <1,4>[4,1]"""
        ret_expr = 0

        for j in range(1, max_label+1):
            if j != pi and j != pk and j != pj:
                ret_expr -= self.func_dict['ab'](pi, j)*self.func_dict['sb'](j, pk)

        replace_expr = self.func_dict['ab'](pi, pj)*self.func_dict['sb'](pj, pk)

        self.sp_expr = self.sp_expr.subs(replace_expr, ret_expr)
        self.str_expr = str(self.sp_expr)

    def momentum2(self, bk, pi, pj, pk, canonical=False):
        """Implement momentum conservation on the given pattern
        For example we use <1,2> = - <1,4>[4,3]/[2,3] if we have 4 labels at most
        Can have pi==pk which instead implies <1,2> = - <1,3>[3,1]/[2,1] - <1,4>[4,1]/[2,1]"""
        ret_expr = 0

        for j in range(1, self.n_point+1):
            if j != pi and j != pk and j != pj:
                ratio = self.func_dict['ab'](pi, pj) if bk == 'sb' else self.func_dict['sb'](pj, pk)
                ret_expr -= self.func_dict['ab'](pi, j)*self.func_dict['sb'](j, pk)/ratio

        if canonical:
            ret_expr = reorder_expr(ret_expr)

        replace_expr = self.func_dict['ab'](pi, pj) if bk == 'ab' else self.func_dict['sb'](pj, pk)

        self.sp_expr = self.sp_expr.subs(replace_expr, ret_expr)
        self.str_expr = str(self.sp_expr)

    def momentum2b(self, bk, pi, pj, pk, canonical=False):
        """Implement momentum conservation on the given pattern in combination with antisymmetry
        For example we use <1,2> =  <2,4>[4,3]/[1,3] if we have 4 labels at most
        Can have pi==pk which instead implies <1,2> = - <1,3>[3,1]/[2,1] - <1,4>[4,1]/[2,1]"""
        ret_expr = 0

        for j in range(1, self.n_point + 1):
            if j != pi and j != pk and j != pj:
                ratio = self.func_dict['ab'](pi, pj) if bk == 'sb' else self.func_dict['sb'](pj, pk)
                ret_expr += self.func_dict['ab'](pi, j) * self.func_dict['sb'](j, pk) / ratio

        if canonical:
            ret_expr = reorder_expr(ret_expr)

        replace_expr = self.func_dict['ab'](pj, pi) if bk == 'ab' else self.func_dict['sb'](pk, pj)

        self.sp_expr = self.sp_expr.subs(replace_expr, ret_expr)
        self.str_expr = str(self.sp_expr)

    def identity_mul(self, rng, new_bk, order, canonical=False):
        """Multiply a given random part of the sympy expression by inserting the identity in
        a non trivial way. """

        # Choose a random term in the expression that contains multiplicative factors
        mul_expr = [expr for expr in self.sp_expr.atoms(sp.Mul)]
        if len(mul_expr) == 0:
            mul_expr_in = self.sp_expr
        else:
            mul_expr_in = rng.choice(mul_expr)

        # Choose the corresponding identity (we might be adding a new bracket)
        bk_expr_env = SpinHelExpr(str(new_bk), self.n_point)
        bk_expr_env.random_scramble(rng, max_scrambles=1, canonical=canonical, reduced=True)
        bk_expr_env.cancel()

        # Choose whether to add the new bracket as <>/ID or ID/<>
        bk_shuffle_expr = bk_expr_env.sp_expr
        replace_expr = bk_shuffle_expr/new_bk if order == 0 else new_bk/bk_shuffle_expr

        self.sp_expr = self.sp_expr.subs(mul_expr_in, replace_expr*mul_expr_in)
        self.str_expr = str(self.sp_expr)

    def zero_add(self, rng, bk_base, sign, canonical=False):
        """ Add zero randomly to an expression"""

        # Choose a random term in the expression that contains additive factors
        add_expr = [expr for expr in self.sp_expr.atoms(sp.Add)]
        if len(add_expr) == 0:
            add_expr_in = self.sp_expr
        else:
            add_expr_in = rng.choice(add_expr)

        # Generate a zero identity for the given bracket
        bk_expr_env = SpinHelExpr(str(bk_base), self.n_point)
        bk_expr_env.random_scramble(rng, max_scrambles=1, canonical=canonical, reduced=True)
        bk_expr_env.cancel()

        # Get the scaling necessary to correct it
        scale_list = get_scaling_expr(add_expr_in, [ab, sb])
        scale_factor = random_scale_factor(scale_list, ab, sb, bk_expr_env.n_point, rng, canonical=canonical)

        add_expr = sign*(bk_base - bk_expr_env.sp_expr) * scale_factor

        self.sp_expr = self.sp_expr.subs(add_expr_in, add_expr_in + add_expr)
        self.str_expr = str(self.sp_expr)

    def together(self):
        """Join the fractions"""
        self.sp_expr = sp.together(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def expand(self):
        """Expand the fractions"""
        self.sp_expr = sp.expand(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def cancel(self):
        """Expand the fractions"""
        self.sp_expr = sp.cancel(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def select_random_bracket(self, rng):
        """Select at random on of the brackets in the full expression"""
        fct_list = [f for f in self.sp_expr.atoms(Function)]
        rdm_fct = rng.choice(fct_list)

        return rdm_fct

    def random_scramble(self, rng=None, max_scrambles=5, verbose=False, out_info=False, canonical=False, reduced=False):
        """ Choose a random number of scrambling moves """
        if rng is None:
            rng = np.random.RandomState()
        scr_num = rng.randint(1, max_scrambles + 1)
        if out_info:
            if canonical:
                info_s = self.scramble_canonical(scr_num, rng, verbose=verbose, out_info=True, reduced=reduced)
            else:
                info_s = self.scramble(scr_num, rng, verbose=verbose, out_info=True)
            return info_s
        else:
            if canonical:
                self.scramble_canonical(scr_num, rng, verbose=verbose, reduced=reduced)
            else:
                self.scramble(scr_num, rng, verbose=verbose)

    def scramble(self, num_scrambles, rng=None, verbose=False, out_info=False):
        """ Scramble an expression with the identities at hand """

        info_s = []
        if rng is None:
            rng = np.random.RandomState()
        for i in range(num_scrambles):
            rdm_bracket = self.select_random_bracket(rng)
            bk = rdm_bracket.func.__name__
            args = list(rdm_bracket.args)

            act_num = rng.randint(1, 4)
            # Identity number 1 is antisymmetry
            if act_num == 1:
                info_s.append(['A', str(rdm_bracket)])
                if verbose:
                    print('Using Antisymmetry on {}'.format(str(rdm_bracket)))
                self.antisymm(bk, args[0], args[1])
            # Apply the Schouten identity where we randomly select the other momenta (avoid null brackets)
            elif act_num == 2:
                arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0], args[1]]])
                arg4 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0], args[1], arg3]])
                info_s.append(['S', str(rdm_bracket), str(arg3), str(arg4)])
                if verbose:
                    print('Using Schouten on {} with args({},{})'.format(str(rdm_bracket), arg3, arg4))
                self.schouten2(bk, args[0], args[1], arg3, arg4)
            # Apply the momentum conservation where we randomly select the other momenta (avoid null brackets)
            elif act_num == 3:
                if bk == 'ab':
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[1]]])
                    self.momentum2(bk, args[0], args[1], arg3)
                else:
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0]]])
                    self.momentum2(bk, arg3, args[0], args[1])
                info_s.append(['M', str(rdm_bracket), str(arg3)])
                if verbose:
                    print('Using Momentum conservation on {} with arg{}'.format(str(rdm_bracket), arg3))
        if out_info:
            return info_s

    def scramble_canonical(self, num_scrambles, rng=None, verbose=False, out_info=False, reduced=False):
        """Perform a scrambling procedure where the expressions are kept in canonical form at all times"""
        info_s = []
        if rng is None:
            rng = np.random.RandomState()
        for i in range(num_scrambles):
            rdm_bracket = self.select_random_bracket(rng)
            bk = rdm_bracket.func.__name__
            args = list(rdm_bracket.args)

            if reduced:
                act_num = rng.randint(1, 4)
            else:
                act_num = rng.randint(1, 6)

            # Apply the Schouten identity where we randomly select the other momenta (avoid null brackets)
            if act_num == 1:
                arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0], args[1]]])
                arg4 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0], args[1], arg3]])
                info_s.append(['S', str(rdm_bracket), str(arg3), str(arg4)])
                if verbose:
                    print('Using Schouten on {} with args({},{})'.format(str(rdm_bracket), arg3, arg4))
                self.schouten2(bk, args[0], args[1], arg3, arg4, canonical=True)

            # Apply the momentum conservation where we randomly select the other momenta (avoid null brackets)
            elif act_num == 2:
                if bk == 'ab':
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[1]]])
                    self.momentum2(bk, args[0], args[1], arg3, canonical=True)
                else:
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0]]])
                    self.momentum2(bk, arg3, args[0], args[1], canonical=True)
                info_s.append(['M+', str(rdm_bracket), str(arg3)])
                if verbose:
                    print('Using Momentum conservation + on {} with arg{}'.format(str(rdm_bracket), arg3))

            # Apply momentum conservation with the antisymmetric version of the identity
            elif act_num == 3:
                if bk == 'ab':
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0]]])
                    self.momentum2b(bk, args[1], args[0], arg3, canonical=True)
                else:
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[1]]])
                    self.momentum2b(bk, arg3, args[1], args[0], canonical=True)
                info_s.append(['M-', str(rdm_bracket), str(arg3)])
                if verbose:
                    print('Using Momentum conservation - on {} with arg{}'.format(str(rdm_bracket), arg3))

            elif act_num == 4:
                # Choose the new random bracket to add
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                new_bk = generate_random_bk(bk_type, self.n_point, rng, canonical=True)
                order_bk = rng.randint(0, 2)
                tok = 'ID-' if order_bk == 0 else 'ID+'
                self.identity_mul(rng, new_bk, order_bk, canonical=True)
                info_s.append([tok, str(new_bk)])

            elif act_num == 5:
                # Choose the new random bracket to use as a base
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                base_bk = generate_random_bk(bk_type, self.n_point, rng, canonical=True)
                sign_bk = rng.randint(0, 2)
                tok = 'Z-' if sign_bk == 0 else 'Z+'
                self.zero_add(rng, base_bk, int(2*(sign_bk - 0.5)), canonical=True)
                info_s.append([tok, str(base_bk)])

        if out_info:
            return info_s


if __name__ == '__main__':
    test1 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    print(test1)
    print("\n")
    print("Start test 2 " + "\n")
    test2 = SpinHelExpr("ab(1,2)*sb(2,3) + ab(1,4)*sb(4,3)")
    print(test2.sp_expr)
    test2.schouten2('ab', 1, 2, 3, 4)
    print(test2.sp_expr)
    test2.expand()
    print(test2.sp_expr)
    test2.expand()
    test2.together()
    print(test2)
    test2.antisymm('ab', 2, 4)
    print(test2)
    test2.momentum(4, 2, 3, 4)
    print(test2)
    test2.schouten2('sb', 2, 3, 1, 4)
    test2.expand()
    test2.together()
    print(test2)
    test2.momentum(4, 1, 3, 4)
    test2.expand()
    test2.together()
    print(test2)
    test2.momentum(4, 1, 3, 4)
    test2.expand()
    test2.together()
    print("\n")
    print("Start test 3 " + "\n")
    test3 = SpinHelExpr("ab(1,2)*sb(2,3) + ab(1,4)*sb(4,3)")
    print(test3)
    test3.antisymm('ab', 1, 2)
    test3.momentum2('ab', 2, 1, 4)
    test3.expand()
    test3.together()
    print(test3)
    test3.antisymm('sb', 3, 4)
    test3.antisymm('sb', 2, 3)
    test3.together()
    print(test3)

    print("\n")
    print("Start test 4 " + "\n")
    test4 = SpinHelExpr("ab(1,2)/(ab(1,3)*ab(1,4))")
    print(test4)
    test4.schouten2('ab', 1, 2, 3, 4)
    test4.expand()
    test4.antisymm('ab', 2, 4)
    print(test4)
    test4.together()
    test4.schouten('ab', 1, 4, 3, 2)
    print(test4)

    test4.scramble(3, verbose=True)
    print(test4)
    test4.together()
    print(test4)

    print("\n")
    print("Start test 5 " + "\n")
    test5 = SpinHelExpr("ab(1,3)*ab(5,4)/(ab(4,2)*ab(5,3)*sb(4,3))")
    test5.scramble(3, verbose=True)
    print(test5)
    print(latex(test5.sp_expr))
    test5.together()
    print(test5)
    print(latex(test5.sp_expr))

    print("\n")
    print("Start test 5b " + "\n")
    test5 = SpinHelExpr("ab(1,3)*ab(5,4)/(ab(4,2)*ab(5,3)*sb(4,3))")
    test5.schouten2('ab', 5, 4, 1, 2)
    test5.expand()
    test5.together()
    print(test5)
    print(latex(test5.sp_expr))

    print("\n")
    print("Start test 6 " + "\n")
    test6 = SpinHelExpr("(ab(1,2)*ab(5,2)*sb(2,5)+ab(1,3)*ab(5,2)*sb(3,5)-ab(4,2)*ab(5,1)*sb(4,5))*ab(1,3)/(ab(1,2)*ab(4,2)*ab(5,3)*sb(3,4)*sb(4,5))")
    print(test6)
    test6.antisymm('sb', 3, 4)
    test6.momentum2('sb', 1, 4, 5)
    test6.cancel()
    print(test6)
    test6.schouten2('ab', 1, 2, 5, 4)
    test6.antisymm('ab', 1, 5)
    test6.antisymm('ab', 2, 4)
    test6.cancel()
    print(test6)