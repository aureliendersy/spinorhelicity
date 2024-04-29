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

    def schouten_single(self, bk_in, pk, pl, canonical=False, numerator_only=False):
        """ Apply the Schouten to a selected pair of two variables
        e.g apply it to <1,2> where the pattern has to be explicitly present in the expression"""

        # Unpack the bracket information (name, arguments)
        bk_type = bk_in.func.__name__
        pi, pj = bk_in.args

        # Construct the Shouten identity
        ret_expr = self.func_dict[bk_type](pi, pl)*self.func_dict[bk_type](pk, pj)/self.func_dict[bk_type](pk, pl) + \
                   self.func_dict[bk_type](pi, pk)*self.func_dict[bk_type](pj, pl)/self.func_dict[bk_type](pk, pl)

        # Reorder in canonical form if required
        if canonical:
            ret_expr = reorder_expr(ret_expr)

        # Only do the replacement on the numerators if desired
        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(bk_in, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(bk_in, ret_expr)
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

    def momentum_single(self, bk_in, pout1, pout2, canonical=False, numerator_only=False):
        """Implement momentum conservation on the given pattern
        For example we use <1,2> = - <1,4>[4,3]/[2,3] if we have 4 labels at most
        Can have pi==pk which instead implies <1,2> = - <1,3>[3,1]/[2,1] - <1,4>[4,1]/[2,1]"""

        # Unpack the bracket information (name, arguments)
        bk_type = bk_in.func.__name__
        bk_type_opp = 'ab' if bk_type == 'sb' else 'sb'

        # Assert which arguments are at the left or the right of the identity (not iterated over)
        larg = pout1 if bk_type == 'ab' else pout2
        rarg = pout2 if bk_type == 'ab' else pout1

        # Assert which is the middle argument for the input bracket. Also figure out the sign if antisymmetry is used
        in_arg = [arg for arg in bk_in.args if arg != pout1][0]
        sign = 1 if ((bk_type == 'ab' and pout1 == bk_in.args[0]) or (bk_type == 'sb' and pout1 == bk_in.args[1]))\
            else -1

        # Add all the terms of the identity one by one and divide by the appropriate bracket term
        ret_expr = 0
        denom_bk = self.func_dict[bk_type_opp](in_arg, pout2) if bk_type == 'ab' \
            else self.func_dict[bk_type_opp](pout2, in_arg)
        for j in range(1, self.n_point+1):
            if j != in_arg and j != larg and j != rarg:
                ret_expr -= self.func_dict['ab'](larg, j)*self.func_dict['sb'](j, rarg)/denom_bk

        ret_expr = sign * ret_expr

        # Reorder the momenta if we keep canonical ordering
        if canonical:
            ret_expr = reorder_expr(ret_expr)

        # Only do the replacement on the numerators if desired
        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(bk_in, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(bk_in, ret_expr)
        self.str_expr = str(self.sp_expr)

    def momentum_sq(self, bk_in, plist_lsh_add, canonical=False, numerator_only=False):
        """Implement momentum conservation squared on the given pattern
        We feed in the desired bracket, along with the additional momenta that are to be multiplied
        So e.g (5pt) if <12> is the input and 3 is also given then the identity is
        (p1+p2+p3)^2=(p4+p5)^2" -> <12>[12]+<13>[13]+<23>[23]=<45>[45]"""

        # Unpack the bracket information (name, arguments)
        bk_type = bk_in.func.__name__
        bk_type_opp = 'ab' if bk_type == 'sb' else 'sb'
        bk_args = list(bk_in.args)

        # Isolate the momenta that appear on each side of the equation
        plist_lhs = plist_lsh_add + bk_args
        plist_rhs = [i for i in range(1, self.n_point + 1) if i not in plist_lhs]

        # Construct the bracket denominator and initialize the return expression
        ret_expr = 0
        denom_bk = self.func_dict[bk_type_opp](bk_args[0], bk_args[1])

        # Construct the positive RHS
        for j1 in plist_rhs:
            for j2 in plist_rhs:
                if j2 > j1:
                    ret_expr += (self.func_dict['ab'](j1, j2)*self.func_dict['sb'](j1, j2))/denom_bk

        # Construct the negative RHS
        for l1 in plist_lhs:
            for l2 in plist_lhs:
                if l2 > l1 and (l1 not in bk_args or l2 not in bk_args):
                    ret_expr -= (self.func_dict['ab'](l1, l2)*self.func_dict['sb'](l1, l2))/denom_bk

        if canonical:
            ret_expr = reorder_expr(ret_expr)

        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(bk_in, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(bk_in, ret_expr)

        self.str_expr = str(self.sp_expr)

    def identity_mul(self, rng, new_bk, order, canonical=False, numerator_only=False):
        """Multiply a given random part of the sympy expression by inserting the identity in
        a non-trivial way. """

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
                        reduced=False, session=None, numerator_only=False, min_scrambles=1):
        """ Choose a random number of scrambling moves """
        if rng is None:
            rng = np.random.RandomState()
        scr_num = rng.randint(min_scrambles, max_scrambles + 1)
        if out_info:
            info_s = self.scramble(scr_num, session, rng, verbose=verbose, out_info=True, reduced=reduced,
                                   numerator_only=numerator_only, canonical=canonical)
            return info_s
        else:

            self.scramble(scr_num, session, rng, verbose=verbose, reduced=reduced, numerator_only=numerator_only,
                          canonical=canonical)

    def scramble(self, num_scrambles, session, rng=None, verbose=False, out_info=False, reduced=False,
                 numerator_only=False, canonical=False):
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
                args_bk = list(rdm_bracket.args)
            else:
                rdm_bracket = None
                bk = None
                args_bk = None

            # start_time = time.time()

            # Apply the Schouten identity where we randomly select the other momenta (avoid null brackets)
            if act_num == 1:
                arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in args_bk])
                arg4 = rng.choice([i for i in range(1, self.n_point + 1) if i not in args_bk + [arg3]])
                info_s.append(['S', str(rdm_bracket), str(arg3), str(arg4)])
                if verbose:
                    print('Using Schouten on {} with args({},{})'.format(str(rdm_bracket), arg3, arg4))
                self.schouten_single(rdm_bracket, arg3, arg4, canonical=canonical, numerator_only=numerator_only)
                # print("--- %s seconds for Schouten ---" % (time.time() - start_time))

            # Apply the momentum conservation where we randomly select the other momenta (avoid null brackets)
            elif act_num == 2:

                # Choose randomly the momenta that will be fixed
                out_arg = rng.choice(args_bk)
                in_arg = [arg for arg in args_bk if arg != out_arg][0]
                out2_arg = rng.choice([i for i in range(1, self.n_point + 1) if i != in_arg])

                # For the additional string indicate whether antisymmetry is used implicitly
                str_label = 'M+' if ((args_bk[0] == out_arg and bk == 'ab')
                                     or (args_bk[1] == out_arg and bk == 'sb')) else 'M-'

                info_s.append([str_label, str(rdm_bracket), str(out2_arg)])
                if verbose:
                    print('Using {} conservation + on {} with arg{}'.format(str_label, str(rdm_bracket), out2_arg))
                self.momentum_single(rdm_bracket, out_arg, out2_arg, canonical=canonical, numerator_only=numerator_only)
                # print("--- %s seconds for Momentum ---" % (time.time() - start_time))

            # Apply momentum conservation squared
            elif act_num == 3:

                # Choose randomly the number of additional arguments and pick them
                arg_add = rng.randint(0, self.n_point - 3)
                mom_add = list(rng.choice([i for i in range(1, self.n_point + 1) if i != args_bk], arg_add,
                                          replace=False))

                info_s.append(['M', str(rdm_bracket)] + [str(el) for el in mom_add])
                if verbose:
                    print('Using Momentum conservation on {} with arg(s) {}'.format(str(rdm_bracket), mom_add))
                self.momentum_sq(rdm_bracket, mom_add, canonical=canonical, numerator_only=numerator_only)

            elif act_num == 4:
                # Choose the new random bracket to add
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                new_bk = generate_random_bk(bk_type, self.n_point, rng, canonical=canonical)
                order_bk = rng.randint(0, 2)
                tok = 'ID-' if order_bk == 0 or numerator_only else 'ID+'
                tok_add = self.identity_mul(rng, new_bk, order_bk, canonical=canonical, numerator_only=numerator_only)
                info_s.append([tok, str(new_bk)])
                info_s.append(tok_add[0])
                if verbose:
                    print('Using Identity Multiplication')
                # print("--- %s seconds for Multiplication ---" % (time.time() - start_time))

            elif act_num == 5:
                # Choose the new random bracket to use as a base
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                base_bk = generate_random_bk(bk_type, self.n_point, rng, canonical=canonical)
                sign_bk = rng.randint(0, 2)
                tok = 'Z-' if sign_bk == 0 else 'Z+'
                success, tok_add = self.zero_add(rng, base_bk, int(2*(sign_bk - 0.5)), session, canonical=canonical,
                                                 numerator_only=numerator_only)
                if success:
                    info_s.append([tok, str(base_bk)])
                    info_s.append(tok_add[0])
                    if verbose:
                        print('Using Zero Addition')
                else:
                    i = i - 1
                    if verbose:
                        print('Failed Zero Addition')
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
    print("Start test 1 ")
    test1 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected1 = SpinHelExpr("ab(2,1)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    test1.antisymm('ab', 1, 2)
    test1.together()
    expected1.together()
    print('Test 1 {}'.format('passed' if expected1.sp_expr == test1.sp_expr else 'failed'))
    print(test1)
    print("\n")

    print("Start test 2 ")
    test2 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected2 = SpinHelExpr("ab(1,2)*(ab(3,1)*ab(4, 2)+ab(1,4)*ab(3,2))/(sb(2,3)*sb(4,3))")
    test2.schouten_single(ab(3, 4), 1, 2)
    test2.together()
    expected2.together()
    print('Test 2 {}'.format('passed' if expected2.sp_expr == test2.sp_expr else 'failed'))
    print(test2)
    print("\n")

    print("Start test 3 ")
    test3 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected3 = SpinHelExpr("-ab(1,2)**2*ab(3,2)*sb(2,1)/(sb(2,3)*sb(4,3)*sb(4,1))")
    test3.momentum_single(ab(3, 4), 3, 1)
    test3.together()
    expected3.together()
    print('Test 3 {}'.format('passed' if expected3.sp_expr == test3.sp_expr else 'failed'))
    print(test3)
    print("\n")

    print("Start test 4 ")
    test4 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected4 = SpinHelExpr("ab(1,2)**2*ab(4,2)*sb(2,1)/(sb(2,3)*sb(4,3)*sb(3,1))")
    test4.momentum_single(ab(3, 4), 4, 1)
    test4.together()
    expected4.together()
    print('Test 4 {}'.format('passed' if expected4.sp_expr == test4.sp_expr else 'failed'))
    print(test4)
    print("\n")

    print("Start test 5 ")
    test5 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected5 = SpinHelExpr("-ab(1,2)**3*ab(3,4)/(ab(1,4)*sb(4,3)**2)")
    test5.momentum_single(sb(2, 3), 3, 1)
    test5.together()
    expected5.together()
    print('Test 5 {}'.format('passed' if expected5.sp_expr == test5.sp_expr else 'failed'))
    print(test5)
    print("\n")

    print("Start test 6 ")
    test6 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected6 = SpinHelExpr("ab(1,2)**2*ab(3,4)*ab(1,3)/(ab(1,4)*sb(4,3)*sb(4,2))")
    test6.momentum_single(sb(2, 3), 2, 1)
    test6.together()
    expected6.together()
    print('Test 6 {}'.format('passed' if expected6.sp_expr == test6.sp_expr else 'failed'))
    print(test6)
    print("\n")

    print("Start test 7 ")
    test7 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected7 = SpinHelExpr("ab(1,2)**2*(-ab(3,1)*sb(1,3)-ab(3,2)*sb(2,3))/(sb(2,3)*sb(4,3)**2)")
    test7.momentum_single(ab(3, 4), 3, 3)
    test7.together()
    expected7.together()
    print('Test 7 {}'.format('passed' if expected7.sp_expr == test7.sp_expr else 'failed'))
    print(test7)
    print("\n")

    print("Start test 8 ")
    test8 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected8 = SpinHelExpr("ab(1,2)**2*(ab(1,3)*sb(1,3)+ab(2,3)*sb(2,3))/(sb(2,3)*sb(3,4)**2)")
    test8.sp_expr = reorder_expr(test8.sp_expr)
    test8.str_expr = str(test8.sp_expr)
    test8.momentum_single(ab(3, 4), 3, 3, canonical=True)
    test8.expand()
    test8.together()
    expected8.together()
    print('Test 8 {}'.format('passed' if expected8.sp_expr == test8.sp_expr else 'failed'))
    print(test8)
    print("\n")

    print("Start test 9 ")
    test9 = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(4,3))")
    expected9 = SpinHelExpr("ab(1,2)**2*(ab(1,2)*sb(1,2)/sb(3,4))/(-sb(2,3)*sb(3,4))")
    test9.sp_expr = reorder_expr(test9.sp_expr)
    test9.str_expr = str(test9.sp_expr)
    test9.momentum_sq(ab(3, 4), [], canonical=True)
    test9.expand()
    test9.together()
    expected9.together()
    print('Test 9 {}'.format('passed' if expected9.sp_expr == test9.sp_expr else 'failed'))
    print(test9)
    print("\n")

    print("Start test 10 ")
    test10 = SpinHelExpr("ab(1,2)**2*ab(3,4)*ab(1,5)/(sb(2,3)*sb(4,3))")
    expected10 = SpinHelExpr("ab(1,2)**2*ab(1,5)*(ab(1,2)*sb(1,2)/sb(3,4) + ab(1,5)*sb(1,5)/sb(3,4) + ab(2,5)*sb(2,5)/sb(3,4))/(-sb(2,3)*sb(3,4))")
    test10.sp_expr = reorder_expr(test10.sp_expr)
    test10.str_expr = str(test10.sp_expr)
    test10.momentum_sq(ab(3, 4), [], canonical=True)
    test10.expand()
    test10.together()
    expected10.expand()
    expected10.together()
    print('Test 10 {}'.format('passed' if expected10.sp_expr == test10.sp_expr else 'failed'))
    print(test10)
    print("\n")

    print("Start test 11 ")
    test11 = SpinHelExpr("ab(1,2)**2*ab(3,4)*ab(1,5)/(sb(2,3)*sb(4,3))")
    expected11 = SpinHelExpr("ab(1,2)**2*ab(1,5)*(-ab(1,3)*sb(1,3)/sb(3,4) - ab(1,4)*sb(1,4)/sb(3,4) + ab(2,5)*sb(2,5)/sb(3,4))/(-sb(2,3)*sb(3,4))")
    test11.sp_expr = reorder_expr(test11.sp_expr)
    test11.str_expr = str(test11.sp_expr)
    test11.momentum_sq(ab(3, 4), [1], canonical=True)
    test11.expand()
    test11.together()
    expected11.expand()
    expected11.together()
    print('Test 11 {}'.format('passed' if expected11.sp_expr == test11.sp_expr else 'failed'))
    print(test11)
    print("\n")

    print("Start scramble test 1a ")
    test1a = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(3,4))")
    test1a.scramble(2, 'NotRequired', numerator_only=True, verbose=True, reduced=False)
    test1a.together()
    print(test1a)
    print("\n")

    print("Start scramble test 2a ")
    test1a = SpinHelExpr("ab(1,2)**2*ab(3,4)/(sb(2,3)*sb(3,4))")
    test1a.scramble(2, 'NotRequired', numerator_only=True, verbose=True, reduced=False, canonical=True)
    test1a.together()
    print(test1a)
    print("\n")

    print('Start from amplitude')
    testamp = SpinHelExpr("ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,5)*ab(5,1))")
    print(testamp)
    print(latex(testamp))
    testamp.scramble(1, 'NotRequired', numerator_only=True, verbose=True, reduced=False, canonical=True)
    testamp.together()
    print(testamp)
    print(latex(testamp))
    print(testamp.str_expr)

