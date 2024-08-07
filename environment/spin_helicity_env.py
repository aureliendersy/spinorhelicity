"""
Environment used to handle spinor helicity expressions
"""

import numpy as np
import random, time
import sympy as sp
from environment.utils import (reorder_expr, generate_random_bk, build_scale_factor,
                               get_expression_detail_lg_scaling)
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

    def schouten_single(self, bk_in, pk, pl, numerator_only=False):
        """
        Apply the Schouten to a selected pair of two variables
        e.g apply it to <1,2> where the pattern has to be explicitly present in the expression
        :param bk_in: Bracket on which to apply the replacement rule
        :param pk: First additional momentum label for the Schouten identity
        :param pl: Second additional momentum label for the Schouten identity
        :param numerator_only: Whether to only scramble the numerator terms
        :return:
        """
        # Unpack the bracket information (name, arguments)
        bk_type = bk_in.func.__name__
        pi, pj = bk_in.args

        # Construct the Shouten identity
        ret_expr = self.func_dict[bk_type](pi, pl)*self.func_dict[bk_type](pk, pj)/self.func_dict[bk_type](pk, pl) + \
                   self.func_dict[bk_type](pi, pk)*self.func_dict[bk_type](pj, pl)/self.func_dict[bk_type](pk, pl)

        # Reorder in canonical form if required
        ret_expr = reorder_expr(ret_expr)

        # Only do the replacement on the numerators if desired
        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(bk_in, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(bk_in, ret_expr)
        self.str_expr = str(self.sp_expr)

    def momentum_single(self, bk_in, pout1, pout2, numerator_only=False):
        """
        Implement momentum conservation on the given pattern
        For example we use <1,2> = - <1,4>[4,3]/[2,3] if we have 4 labels at most
        Can have pi==pk which instead implies <1,2> = - <1,3>[3,1]/[2,1] - <1,4>[4,1]/[2,1]
        :param bk_in: Bracket on which to apply the replacement rule
        :param pout1: First additional momentum label for the Momentum identity
        :param pout2: Second additional momentum label for the Momentum identity
        :param numerator_only: Whether to only scramble the numerator terms
        :return:
        """
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
        ret_expr = reorder_expr(ret_expr)

        # Only do the replacement on the numerators if desired
        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(bk_in, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(bk_in, ret_expr)
        self.str_expr = str(self.sp_expr)

    def momentum_sq(self, bk_in, plist_lsh_add, numerator_only=False):
        """
        Implement momentum conservation squared on the given pattern
        We feed in the desired bracket, along with the additional momenta that are to be multiplied
        So e.g (5pt) if <12> is the input and 3 is also given then the identity is
        (p1+p2+p3)^2=(p4+p5)^2" -> <12>[12]+<13>[13]+<23>[23]=<45>[45]
        :param bk_in: Bracket on which to apply the replacement rule
        :param plist_lsh_add: List of momenta labels for the LHS of the momentum squared identity
        :param numerator_only: Whether to only scramble the numerator terms
        :return:
        """
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
                # Make sure that we don't include the bracket that we are replacing
                if l2 > l1 and (l1 not in bk_args or l2 not in bk_args):
                    ret_expr -= (self.func_dict['ab'](l1, l2)*self.func_dict['sb'](l1, l2))/denom_bk

        ret_expr = reorder_expr(ret_expr)

        if numerator_only:
            num, denom = sp.fraction(self.sp_expr)
            num = num.subs(bk_in, ret_expr)
            self.sp_expr = num/denom
        else:
            self.sp_expr = self.sp_expr.subs(bk_in, ret_expr)

        self.str_expr = str(self.sp_expr)

    def identity_mul(self, rng, new_bk, order, numerator_only=False):
        """
        Multiply a given random part of the sympy expression by inserting the identity in
        a non-trivial way.
        :param rng: Numpy random state
        :param new_bk: Bracket on which to apply the replacement rule
        :param order: Whether to apply the replacement rule in the numerator or denominator of the identity fraction
        :param numerator_only: Whether to only scramble the numerator terms
        :return:
        """
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
        info_add = bk_expr_env.random_scramble(rng, max_scrambles=1, reduced=True, numerator_only=numerator_only,
                                               out_info=True)
        bk_expr_env.cancel()

        # Choose whether to add the new bracket as <>/ID or ID/<>
        # If we replace the numerator then the second option is always chosen
        bk_shuffle_expr = bk_expr_env.sp_expr
        replace_expr = bk_shuffle_expr/new_bk if order == 0 or numerator_only else new_bk/bk_shuffle_expr

        if numerator_only:
            self.sp_expr = func_in.subs(mul_expr_in, replace_expr*mul_expr_in) / denom
        else:
            self.sp_expr = self.sp_expr.subs(mul_expr_in, replace_expr*mul_expr_in)

        self.str_expr = str(self.sp_expr)
        return info_add

    def zero_add(self, rng, bk_base, sign, session, numerator_only=False):
        """
        Add zero randomly to an expression in a non trivial way.
        :param rng: Numpy random state
        :param bk_base: Bracket on which to apply the replacement rule
        :param sign: Overall sign of the replacement rule
        :param session: Mathematica session
        :param numerator_only: Whether to only scramble the numerator terms
        :return:
        """
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
        info_add = bk_expr_env.random_scramble(rng, max_scrambles=1, reduced=True, numerator_only=numerator_only,
                                               out_info=True)
        bk_expr_env.cancel()

        # If we are adding to an amplitude which is 0, then we generate a new amplitude from scratch
        if add_expr_in == 0:
            scale_factor = generate_random_fraction_unbounded(0.75, self.n_point, 2*self.n_point, rng,
                                                              zero_allowed=False)
        else:
            # Get the scaling that we need to correct for
            # Try to match the scaling at the numerator and denominator level (irrelevant if numerator only scrambling)
            num_scales, denom_scales = get_expression_detail_lg_scaling(add_expr_in, [ab, sb], self.n_point)
            bk_scales, _ = get_expression_detail_lg_scaling(bk_base, [ab, sb], self.n_point)
            num_scales = np.array(num_scales) - np.array(bk_scales)

            coeff_add_num = solve_diophantine_system(self.n_point, num_scales, session)
            coeff_add_denom = solve_diophantine_system(self.n_point, denom_scales, session)

            # If no correct scaling exists the identity is not applied
            if coeff_add_num is None or coeff_add_denom is None:
                return False, None

            # Build the appropriate scale factor from the coefficients
            scale_factor = build_scale_factor(coeff_add_num, ab, sb, self.n_point)\
                           / build_scale_factor(coeff_add_denom, ab, sb, self.n_point)

        # Build the identity
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
        self.sp_expr = sp.cancel(self.sp_expr)
        self.str_expr = str(self.sp_expr)

    def select_random_bracket(self, rng, numerator_only=False):
        """Select at random on of the brackets in the full expression"""

        # If numerator only then we isolate just the brackets in the numerator
        if numerator_only:
            function_probed, _ = sp.fraction(self.sp_expr)
        else:
            function_probed = self.sp_expr
        fct_list = [f for f in function_probed.atoms(Function)]
        rdm_fct = rng.choice(fct_list)

        return rdm_fct

    def random_scramble(self, rng=None, max_scrambles=5, verbose=False, out_info=False,
                        reduced=False, session=None, numerator_only=False, min_scrambles=1):
        """
        Choose a random number of scrambling moves and apply those
        :param rng: Numpy random state
        :param max_scrambles: Maximum number of scrambling moves
        :param verbose: Whether to print scrambling moves information
        :param out_info: Whether to return the scrambling moves utilized
        :param reduced: If reduced then we don't use the addition of zero or the multiplication by unity identitites
        :param session: Mathematica session
        :param numerator_only: Whether to only scramble the numerator terms
        :param min_scrambles: Minimum number of scrambling moves
        :return:
        """

        # Generate the random state if it is not given
        if rng is None:
            rng = np.random.RandomState()

        # Sample the number of scrambling moves to use and perform them
        scr_num = rng.randint(min_scrambles, max_scrambles + 1)
        if out_info:
            info_s = self.scramble(scr_num, session, rng, verbose=verbose, out_info=True, reduced=reduced,
                                   numerator_only=numerator_only)
            return info_s
        else:

            self.scramble(scr_num, session, rng, verbose=verbose, reduced=reduced, numerator_only=numerator_only)

    def scramble(self, num_scrambles, session, rng=None, verbose=False, out_info=False, reduced=False,
                 numerator_only=False):
        """
        Perform a scrambling procedure where the expressions are kept in canonical form at all times
        :param num_scrambles: Number of scrambling moves
        :param rng: Numpy random state
        :param verbose: Whether to print scrambling moves information
        :param out_info: Whether to return the scrambling moves utilized
        :param reduced: If reduced then we don't use the addition of zero or the multiplication by unity identitites
        :param session: Mathematica session
        :param numerator_only: Whether to only scramble the numerator terms
        :return:
        """
        info_s = []
        # Generate the random numpy state if it is not given
        if rng is None:
            rng = np.random.RandomState()

        # Repeat for each scrambling move
        for i in range(num_scrambles):

            # If we start with 0 we can only use the addition identity
            if self.sp_expr == 0:
                act_num = 5

            # If we start with a 1 in the numerator we can only use the multiplication identity
            elif isinstance(sp.fraction(self.sp_expr)[0], sp.Integer) and numerator_only:
                act_num = 4

            # If we want to use a Schouten or momentum conservation identity
            elif reduced:
                act_num = rng.randint(1, 4)

            # If we want to use any identity
            else:
                act_num = rng.randint(1, 6)

            # For replacement rule identities we randomly sample a bracket to replace
            if act_num < 4:
                rdm_bracket = self.select_random_bracket(rng, numerator_only)
                bk = rdm_bracket.func.__name__
                args_bk = list(rdm_bracket.args)
            else:
                rdm_bracket = None
                bk = None
                args_bk = None

            # Apply the Schouten identity where we randomly select the other momenta (avoid null brackets)
            if act_num == 1:
                arg3 = rng.choice([i for i in range(1, self.n_point + 1) if i not in args_bk])
                arg4 = rng.choice([i for i in range(1, self.n_point + 1) if i not in args_bk + [arg3]])
                info_s.append(['S', str(rdm_bracket), str(arg3), str(arg4)])
                if verbose:
                    print('Using Schouten on {} with args({},{})'.format(str(rdm_bracket), arg3, arg4))
                self.schouten_single(rdm_bracket, arg3, arg4, numerator_only=numerator_only)

            # Apply the momentum conservation where we randomly select the other momenta (avoid null brackets)
            elif act_num == 2:

                # Choose randomly the momenta that will be fixed ("outer" momenta) - the other one is the inner momenta
                out_arg = rng.choice(args_bk)
                in_arg = [arg for arg in args_bk if arg != out_arg][0]

                # Sample the other "outer" momenta
                out2_arg = rng.choice([i for i in range(1, self.n_point + 1) if i != in_arg])

                # For the additional string indicate whether antisymmetry is used implicitly
                str_label = 'M+' if ((args_bk[0] == out_arg and bk == 'ab')
                                     or (args_bk[1] == out_arg and bk == 'sb')) else 'M-'

                info_s.append([str_label, str(rdm_bracket), str(out2_arg)])
                if verbose:
                    print('Using {} conservation + on {} with arg{}'.format(str_label, str(rdm_bracket), out2_arg))
                self.momentum_single(rdm_bracket, out_arg, out2_arg, numerator_only=numerator_only)

            # Apply momentum conservation squared
            elif act_num == 3:

                # Choose randomly the number of additional arguments and pick them
                arg_add = rng.randint(0, self.n_point - 3)
                mom_add = list(rng.choice([i for i in range(1, self.n_point + 1) if i != args_bk], arg_add,
                                          replace=False))

                info_s.append(['M', str(rdm_bracket)] + [str(el) for el in mom_add])
                if verbose:
                    print('Using Momentum conservation on {} with arg(s) {}'.format(str(rdm_bracket), mom_add))
                self.momentum_sq(rdm_bracket, mom_add, numerator_only=numerator_only)

            # Apply the multiplication by unity
            elif act_num == 4:

                # Choose the new random bracket to use as a base
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                new_bk = generate_random_bk(bk_type, self.n_point, rng)
                order_bk = rng.randint(0, 2)
                tok = 'ID-' if order_bk == 0 or numerator_only else 'ID+'

                # Apply a scrambling move to generate a non-trivial replacement rule
                tok_add = self.identity_mul(rng, new_bk, order_bk, numerator_only=numerator_only)
                info_s.append([tok, str(new_bk)])
                info_s.append(tok_add[0])
                if verbose:
                    print('Using Identity Multiplication')

            # Apply the addition of zero
            elif act_num == 5:

                # Choose the new random bracket to use as a base
                bk_type = ab if rng.randint(0, 2) == 0 else sb
                base_bk = generate_random_bk(bk_type, self.n_point, rng)
                sign_bk = rng.randint(0, 2)
                tok = 'Z-' if sign_bk == 0 else 'Z+'

                # Apply a scrambling move to generate a non-trivial replacement rule
                success, tok_add = self.zero_add(rng, base_bk, int(2*(sign_bk - 0.5)), session,
                                                 numerator_only=numerator_only)
                if success:
                    info_s.append([tok, str(base_bk)])
                    info_s.append(tok_add[0])
                    if verbose:
                        print('Using Zero Addition')
                # If we fail the identity (could not find correct factors respecting LG scaling) then we continue
                # and we do not update the identity counter
                else:
                    i = i - 1
                    if verbose:
                        print('Failed Zero Addition')

            if numerator_only:
                self.cancel()
                # Sanity check
                if isinstance(sp.fraction(self.sp_expr)[1], sp.Add):
                    print("Denominator has add term")
        if out_info:
            return info_s
