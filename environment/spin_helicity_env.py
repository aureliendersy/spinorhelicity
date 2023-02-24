"""
Environment used to handle spinor helicity expressions
"""

import numpy as np
import random, time
import sympy as sp
from environment.utils import reorder_expr, generate_random_bk, get_scaling_expr,\
    random_scale_factor, get_scaling_expr_detail, build_scale_factor
from sympy import Function, latex
from environment.helicity_generator import generate_random_fraction_unbounded
from add_ons.mathematica_utils import solve_diophantine_system
from environment.bracket_env import ab, sb


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

    def schouten2(self, bk, pi, pj, pk, pl, canonical=False, numerator_only=False):
        """ Apply the Schouten to a selected pair of two variables
        e.g apply it to <1,2> where the pattern has to be explicitly present in the expression"""

        ret_expr = self.func_dict[bk](pi, pl)*self.func_dict[bk](pk, pj)/self.func_dict[bk](pk, pl) + \
                   self.func_dict[bk](pi, pk)*self.func_dict[bk](pj, pl)/self.func_dict[bk](pk, pl)

        if canonical:
            ret_expr = reorder_expr(ret_expr)

        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(self.func_dict[bk](pi, pj), ret_expr)
            self.sp_expr = num/denom
        else:
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

    def momentum2(self, bk, pi, pj, pk, canonical=False, numerator_only=False):
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

        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(replace_expr, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(replace_expr, ret_expr)

        self.str_expr = str(self.sp_expr)

    def momentum2b(self, bk, pi, pj, pk, canonical=False, numerator_only=False):
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

        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(replace_expr, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(replace_expr, ret_expr)

        self.str_expr = str(self.sp_expr)

    def identity_mul(self, rng, new_bk, order, canonical=False, numerator_only=False):
        """Multiply a given random part of the sympy expression by inserting the identity in
        a non trivial way. """

        # Choose a random term in the expression that contains multiplicative factors
        if numerator_only:
            func_in, denom = sp.fraction(self.sp_expr)
        else:
            func_in = self.sp_expr
            denom = None
        mul_expr = [expr for expr in func_in.atoms(sp.Mul)]
        if len(mul_expr) == 0:
            mul_expr_in = func_in
        else:
            mul_expr_in = rng.choice(mul_expr)

        # Choose the corresponding identity (we might be adding a new bracket)
        bk_expr_env = SpinHelExpr(str(new_bk), self.n_point)
        info_add = bk_expr_env.random_scramble(rng, max_scrambles=1, canonical=canonical, reduced=True,
                                               numerator_only=numerator_only, out_info=True)
        bk_expr_env.cancel()

        # Choose whether to add the new bracket as <>/ID or ID/<>
        bk_shuffle_expr = bk_expr_env.sp_expr
        replace_expr = bk_shuffle_expr/new_bk if order == 0 or numerator_only else new_bk/bk_shuffle_expr

        if numerator_only:
            self.sp_expr = func_in.subs(mul_expr_in, replace_expr*mul_expr_in) / denom
        else:
            self.sp_expr = self.sp_expr.subs(mul_expr_in, replace_expr*mul_expr_in)

        self.str_expr = str(self.sp_expr)
        return info_add

    def zero_add(self, rng, bk_base, sign, session, canonical=False, numerator_only=False):
        """ Add zero randomly to an expression"""

        # Choose a random term in the expression that contains additive factors
        if numerator_only:
            func_in, denom = sp.fraction(self.sp_expr)
        else:
            func_in = self.sp_expr
            denom = None
        add_expr = [expr for expr in func_in.atoms(sp.Add)]
        if len(add_expr) == 0:
            add_expr_in = func_in
        else:
            add_expr_in = rng.choice(add_expr)

        # Generate a zero identity for the given bracket
        bk_expr_env = SpinHelExpr(str(bk_base), self.n_point)
        info_add = bk_expr_env.random_scramble(rng, max_scrambles=1, canonical=canonical, reduced=True,
                                               numerator_only=numerator_only, out_info=True)
        bk_expr_env.cancel()

        if add_expr_in == 0:
            scale_factor = generate_random_fraction_unbounded(0.75, self.n_point, 2*self.n_point, rng,
                                                              canonical_form=canonical, zero_allowed=False)
        else:
            # Get the scaling necessary to correct it
            num_scales, denom_scales = get_scaling_expr_detail(add_expr_in, [ab, sb], self.n_point)
            bk_scales, _ = get_scaling_expr_detail(bk_base, [ab, sb], self.n_point)
            num_scales = np.array(num_scales) - np.array(bk_scales)

            coeff_add_num = solve_diophantine_system(self.n_point, num_scales, session)
            coeff_add_denom = solve_diophantine_system(self.n_point, denom_scales, session)

            # If no correct scaling exists the identity is not applied
            if coeff_add_num is None or coeff_add_denom is None:
                return False, None

            scale_factor = build_scale_factor(coeff_add_num, ab, sb, self.n_point)\
                           / build_scale_factor(coeff_add_denom, ab, sb, self.n_point)

        add_expr = sign*(bk_base - bk_expr_env.sp_expr) * scale_factor

        if numerator_only:
            self.sp_expr = func_in.subs(add_expr_in, add_expr_in + add_expr)/denom
        else:
            self.sp_expr = self.sp_expr.subs(add_expr_in, add_expr_in + add_expr)
        self.str_expr = str(self.sp_expr)
        return True, info_add

    def together(self):
        """Join the fractions"""
        self.sp_expr = sp.together(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def apart(self):
        """Apart the fractions"""
        self.sp_expr = sp.apart(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def expand(self):
        """Expand the fractions"""
        self.sp_expr = sp.expand(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def cancel(self):
        """Expand the fractions"""
        # self.sp_expr = sp.cancel(self.sp_expr, list(self.sp_expr.atoms(sp.Function)), polys=False)
        self.sp_expr = sp.cancel(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def select_random_bracket(self, rng, numerator_only=False):
        """Select at random on of the brackets in the full expression"""

        if numerator_only:
            function_probed, _ = sp.fraction(self.sp_expr)
        else:
            function_probed = self.sp_expr
        fct_list = [f for f in function_probed.atoms(Function)]
        rdm_fct = rng.choice(fct_list)

        return rdm_fct

    def random_scramble(self, rng=None, max_scrambles=5, verbose=False, out_info=False, canonical=False,
                        reduced=False, session=None, numerator_only=False):
        """ Choose a random number of scrambling moves """
        if rng is None:
            rng = np.random.RandomState()
        scr_num = rng.randint(1, max_scrambles + 1)
        if out_info:
            if canonical:
                info_s = self.scramble_canonical(scr_num, session, rng, verbose=verbose, out_info=True, reduced=reduced,
                                                 numerator_only=numerator_only)
            else:
                info_s = self.scramble(scr_num, rng, verbose=verbose, out_info=True, numerator_only=numerator_only)
            return info_s
        else:
            if canonical:
                self.scramble_canonical(scr_num, session, rng, verbose=verbose, reduced=reduced,
                                        numerator_only=numerator_only)
            else:
                self.scramble(scr_num, rng, verbose=verbose, numerator_only=numerator_only)

    def scramble(self, num_scrambles, rng=None, verbose=False, out_info=False, numerator_only=False):
        """ Scramble an expression with the identities at hand """

        info_s = []
        if rng is None:
            rng = np.random.RandomState()
        for i in range(num_scrambles):
            rdm_bracket = self.select_random_bracket(rng, numerator_only)
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

    def scramble_canonical(self, num_scrambles, session, rng=None, verbose=False, out_info=False, reduced=False,
                           numerator_only=False):
        """Perform a scrambling procedure where the expressions are kept in canonical form at all times"""
        info_s = []
        if rng is None:
            rng = np.random.RandomState()
        for i in range(num_scrambles):

            # If we start with 0 we can only use the addition identity
            if self.sp_expr == 0:
                act_num = 5

            # If we start with a 1 in the numerator we can only use the multiplication identity
            elif isinstance(sp.fraction(self.sp_expr)[0], sp.Integer) and numerator_only:
                act_num = 4

            # If we want to use an actual Schouten or momentum conservation identity
            elif reduced:
                act_num = rng.randint(1, 4)

            # If we want to use any identity
            else:
                act_num = rng.randint(1, 6)

            if act_num < 4:
                rdm_bracket = self.select_random_bracket(rng, numerator_only)
                bk = rdm_bracket.func.__name__
                args = list(rdm_bracket.args)
            else:
                rdm_bracket = None
                bk = None
                args = None

            # start_time = time.time()

            # Apply the Schouten identity where we randomly select the other momenta (avoid null brackets)
            if act_num == 1:

                arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0], args[1]]])
                arg4 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0], args[1], arg3]])
                info_s.append(['S', str(rdm_bracket), str(arg3), str(arg4)])
                if verbose:
                    print('Using Schouten on {} with args({},{})'.format(str(rdm_bracket), arg3, arg4))
                self.schouten2(bk, args[0], args[1], arg3, arg4, canonical=True, numerator_only=numerator_only)
                # print("--- %s seconds for Schouten ---" % (time.time() - start_time))

            # Apply the momentum conservation where we randomly select the other momenta (avoid null brackets)
            elif act_num == 2:
                if bk == 'ab':
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[1]]])
                    self.momentum2(bk, args[0], args[1], arg3, canonical=True, numerator_only=numerator_only)
                else:
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0]]])
                    self.momentum2(bk, arg3, args[0], args[1], canonical=True, numerator_only=numerator_only)
                info_s.append(['M+', str(rdm_bracket), str(arg3)])
                if verbose:
                    print('Using Momentum conservation + on {} with arg{}'.format(str(rdm_bracket), arg3))
                # print("--- %s seconds for Momentum ---" % (time.time() - start_time))

            # Apply momentum conservation with the antisymmetric version of the identity
            elif act_num == 3:
                if bk == 'ab':
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[0]]])
                    self.momentum2b(bk, args[1], args[0], arg3, canonical=True, numerator_only=numerator_only)
                else:
                    arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in [args[1]]])
                    self.momentum2b(bk, arg3, args[1], args[0], canonical=True, numerator_only=numerator_only)
                info_s.append(['M-', str(rdm_bracket), str(arg3)])
                if verbose:
                    print('Using Momentum conservation - on {} with arg{}'.format(str(rdm_bracket), arg3))
                # print("--- %s seconds for Momentum ---" % (time.time() - start_time))

            elif act_num == 4:
                # Choose the new random bracket to add
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                new_bk = generate_random_bk(bk_type, self.n_point, rng, canonical=True)
                order_bk = rng.randint(0, 2)
                tok = 'ID-' if order_bk == 0 or numerator_only else 'ID+'
                tok_add = self.identity_mul(rng, new_bk, order_bk, canonical=True, numerator_only=numerator_only)
                info_s.append([tok, str(new_bk)])
                info_s.append(tok_add[0])
                # print("--- %s seconds for Multiplication ---" % (time.time() - start_time))

            elif act_num == 5:
                # Choose the new random bracket to use as a base
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                base_bk = generate_random_bk(bk_type, self.n_point, rng, canonical=True)
                sign_bk = rng.randint(0, 2)
                tok = 'Z-' if sign_bk == 0 else 'Z+'
                success, tok_add = self.zero_add(rng, base_bk, int(2*(sign_bk - 0.5)), session, canonical=True,
                                                 numerator_only=numerator_only)
                if success:
                    info_s.append([tok, str(base_bk)])
                    info_s.append(tok_add[0])
                else:
                    i = i - 1
                # print("--- %s seconds for Addition ---" % (time.time() - start_time))

            if numerator_only:

                # start_time = time.time()
                self.cancel()
                if isinstance(sp.fraction(self.sp_expr)[1], sp.Add):
                    print("Denominator has add term")
                # print("--- %s seconds for Cancel ---" % (time.time() - start_time))
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
