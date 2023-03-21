"""
Module for testing a trained model for contrastive learning
"""
import sympy as sp
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from environment.utils import AttrDict, reorder_expr, to_cuda
from add_ons.slurm import init_signal_handler, init_distributed_mode
import environment
from environment.utils import initialize_exp, get_numerator_lg_scaling
from environment import build_env
from model.contrastive_learner import build_modules_contrastive


def test_expression_factors(envir, module_transfo, input_equation, params, factor_mask=False):

    # load the transformer encoder
    encoder_c = module_transfo['encoder_c']
    encoder_c.eval()

    f = sp.parse_expr(input_equation, local_dict=envir.func_dict)
    if params.canonical_form:
        f = reorder_expr(f)
    f = f.cancel()
    numerator, _ = sp.fraction(f)

    scales = get_numerator_lg_scaling(numerator, list(envir.func_dict.values()))

    if isinstance(numerator, sp.Add):
        terms = numerator.args
    else:
        terms = [numerator]

    # Similarity
    cossim = nn.CosineSimilarity(dim=-1)

    if not factor_mask:
        t_prefix = [envir.sympy_to_prefix(term) for term in terms]
        t_in = [to_cuda(torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in t_pre] + [envir.eos_index]).view(-1, 1))[0]
                for t_pre in t_prefix]

        len_in = [to_cuda(torch.LongTensor([len(term)]))[0] for term in t_in]

        # Forward
        with torch.no_grad():

            encoded_terms = [encoder_c('fwd', x=t, lengths=len_in[i], causal=False) for i, t in enumerate(t_in)]

            # Similarity
            similarity_mat = cossim(torch.stack(encoded_terms), torch.transpose(torch.stack(encoded_terms), 0, 1))
    else:
        similarity_mat = torch.zeros(len(terms), len(terms))
        for i, term in enumerate(terms):
            for j, term2 in enumerate(terms[i+1:]):
                newterm, newterm2 = sp.fraction(sp.cancel(term/term2))

                t_prefix1, t_prefix2 = envir.sympy_to_prefix(newterm), envir.sympy_to_prefix(newterm2)
                t_in1, t_in2 = to_cuda(torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in t_prefix1] + [envir.eos_index]).view(-1, 1))[0],\
                               to_cuda(torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in t_prefix2] + [envir.eos_index]).view(-1, 1))[0]

                len_in1, len_in2 = to_cuda(torch.LongTensor([len(t_in1)]))[0],\
                                   to_cuda(torch.LongTensor([len(t_in2)]))[0]

                # Forward
                with torch.no_grad():
                    encoded1, encoded2 = encoder_c('fwd', x=t_in1, lengths=len_in1, causal=False),\
                                         encoder_c('fwd', x=t_in2, lengths=len_in2, causal=False)

                    similarity_mat[i, i+j+1] = cossim(encoded1, encoded2)
                    similarity_mat[i+j+1, i] = cossim(encoded1, encoded2)
            similarity_mat[i, i] = 1
    return similarity_mat.numpy(), terms


if __name__ == '__main__':

    # Example with 5 pt amplitudes
    #input_eq = '(-ab(1,2)*ab(2,3)*ab(2,4)*ab(3,5)*ab(4,5)*sb(1,4)**2*sb(2,3)*sb(2,5)**2 + 2*ab(1,2)*ab(2,3)*ab(2,4)*ab(3,5)*ab(4,5)*sb(1,4)*sb(1,5)*sb(2,3)*sb(2,4)*sb(2,5) - ab(1,2)*ab(2,3)*ab(2,4)*ab(3,5)*ab(4,5)*sb(1,5)**2*sb(2,3)*sb(2,4)**2)/sb(4,5)**2 + (ab(3, 4)**2*sb(1, 2)*sb(2, 5)/(sb(3, 5)))'
    input_eq = '(ab(1,2)**2*(-(ab(1,2)**2*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(2,5)*sb(3,4)) - ab(1,2)**3*ab(3,4)*sb(1,2)**2*sb(2,3)*sb(2,5)*sb(3,4) - ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4) - ab(1,2)**2*ab(1,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4) - 2*ab(1,2)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(2,5)*sb(3,4) - 2*ab(1,2)**2*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(2,5)*sb(3,4) - ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,3)*sb(2,3)**2*sb(2,5)*sb(3,4) - ab(1,4)*ab(2,3)**3*sb(2,3)**3*sb(2,5)*sb(3,4) - ab(1,2)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(2,5)*sb(3,4) - ab(1,2)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,2)**2*ab(2,4)*ab(3,4)*sb(1,2)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,3)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,4)*ab(2,3)**2*ab(2,4)*sb(2,3)**2*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,2)*ab(2,3)*ab(2,4)*ab(3,4)*sb(2,3)**2*sb(2,4)*sb(2,5)*sb(3,4) + ab(1,2)*ab(1,4)**2*ab(2,3)*sb(1,2)**2*sb(2,5)*sb(3,4)**2 + ab(1,2)**2*ab(1,4)*ab(3,4)*sb(1,2)**2*sb(2,5)*sb(3,4)**2 + ab(1,3)*ab(1,4)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,5)*sb(3,4)**2 + ab(1,2)*ab(1,3)*ab(1,4)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,5)*sb(3,4)**2 + ab(1,4)**2*ab(2,3)**2*sb(1,2)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,2)**2*ab(3,4)**2*sb(1,2)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,3)*ab(1,4)*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,4)*ab(2,3)**2*ab(3,4)*sb(2,3)**2*sb(2,5)*sb(3,4)**2 - ab(1,2)*ab(2,3)*ab(3,4)**2*sb(2,3)**2*sb(2,5)*sb(3,4)**2 + ab(1,2)**2*ab(1,3)*ab(2,3)*sb(1,2)**2*sb(2,3)**2*sb(3,5) + ab(1,2)*ab(1,3)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)**2*sb(3,5) + 2*ab(1,2)*ab(1,3)*ab(2,3)**2*sb(1,2)*sb(2,3)**3*sb(3,5) + ab(1,3)**2*ab(2,3)**2*sb(1,3)*sb(2,3)**3*sb(3,5) + ab(1,3)*ab(2,3)**3*sb(2,3)**4*sb(3,5) + ab(1,2)**2*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)**3*ab(3,4)*sb(1,2)**2*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)**2*ab(1,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,5) + 2*ab(1,2)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + 2*ab(1,2)**2*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,3)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,3)**2*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,4)*ab(2,3)**3*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,3)*ab(2,3)**2*ab(2,4)*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,2)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,2)**2*ab(2,4)*ab(3,4)*sb(1,2)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,4)*ab(2,3)**2*ab(2,4)*sb(2,3)**2*sb(2,4)**2*sb(3,5) + ab(1,2)*ab(2,3)*ab(2,4)*ab(3,4)*sb(2,3)**2*sb(2,4)**2*sb(3,5) - ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(3,4)*sb(3,5) - ab(1,3)**2*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(3,4)*sb(3,5) - ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,3)**2*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,3)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(3,4)*sb(3,5) - ab(1,2)*ab(1,4)**2*ab(2,3)*sb(1,2)**2*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,2)**2*ab(1,4)*ab(3,4)*sb(1,2)**2*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,3)*ab(1,4)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,2)*ab(1,3)*ab(1,4)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,4)**2*ab(2,3)**2*sb(1,2)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,2)**2*ab(3,4)**2*sb(1,2)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,4)*ab(2,3)**2*ab(3,4)*sb(2,3)**2*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,2)*ab(2,3)*ab(3,4)**2*sb(2,3)**2*sb(2,4)*sb(3,4)*sb(3,5)))/(ab(1,3)*ab(1,4)*ab(1,5)*ab(2,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*(ab(1,2)*sb(1,2) + ab(1,3)*sb(1,3) + ab(2,3)*sb(2,3))*(ab(2,3)*sb(2,3) + ab(2,4)*sb(2,4) + ab(3,4)*sb(3,4)))'
    input_eq = '(ab(1,2)**3*sb(1,3)**2*sb(1,5)*sb(2,4) - ab(1,2)**3*sb(1,3)**2*sb(1,4)*sb(2,5) - ab(1,2)**2*ab(2,3)*sb(1,3)*sb(1,5)*sb(2,3)*sb(3,4) + ab(1,2)**2*ab(2,3)*sb(1,3)**2*sb(2,5)*sb(3,4) + ab(1,2)**2*ab(2,3)*sb(1,3)*sb(1,4)*sb(2,3)*sb(3,5)-ab(1,2)**2*ab(2,3)*sb(1,3)**2*sb(2,4)*sb(3,5))/(2*ab(2,3)*ab(2,4)*ab(2,5)*ab(4,5)*sb(1,2)**2*sb(2,3)*sb(4,5))'
    input_eq = 'sb(1, 5)*sb(2, 4)*sb(1,2) - sb(1, 4)*sb(2, 5)*sb(1, 2)'


    # Parke Taylor Example

    # Full input
    input_eq = '(ab(1,2)**2*(ab(1,2)*ab(1,5)*ab(3,4)*sb(1,3)**2*sb(1,5)**2*sb(2,4) - ab(1,2)*ab(1,5)*ab(3,4)*sb(1,3)**2*sb(1,4)*sb(1,5)*sb(2,5) - ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)*sb(1,5)**2*sb(2,3)*sb(3,4) + ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)**2*sb(1,5)*sb(2,5)*sb(3,4) + ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)*sb(1,4)*sb(1,5)*sb(2,3)*sb(3,5) - ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)**2*sb(1,5)*sb(2,4)*sb(3,5) - ab(2,3)**2*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,3)**2*sb(4,5) - ab(2,3)*ab(2,4)*ab(4,5)*sb(1,4)*sb(1,5)*sb(2,3)**2*sb(4,5) - ab(2,3)*ab(2,4)*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,3)*sb(2,4)*sb(4,5) - ab(2,4)**2*ab(4,5)*sb(1,4)*sb(1,5)*sb(2,3)*sb(2,4)*sb(4,5) + ab(2,3)**2*ab(4,5)*sb(1,3)**2*sb(2,3)*sb(2,5)*sb(4,5) + 2*ab(2,3)*ab(2,4)*ab(4,5)*sb(1,3)*sb(1,4)*sb(2,3)*sb(2,5)*sb(4,5) + ab(2,4)**2*ab(4,5)*sb(1,4)**2*sb(2,3)*sb(2,5)*sb(4,5) - ab(2,3)*ab(3,4)*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,3)*sb(3,4)*sb(4,5) - ab(2,4)*ab(3,4)*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,4)*sb(3,4)*sb(4,5) + ab(2,3)*ab(3,4)*ab(4,5)*sb(1,3)**2*sb(2,5)*sb(3,4)*sb(4,5) + ab(2,4)*ab(3,4)*ab(4,5)*sb(1,3)*sb(1,4)*sb(2,5)*sb(3,4)*sb(4,5)))/(ab(1,5)*ab(2,3)*ab(2,4)*ab(2,5)*ab(3,4)*ab(4,5)*sb(1,2)**2*sb(1,5)*sb(2,3)*sb(4,5))'

    # Get the simplified forms
    simple1 = '-ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 4)*sb(3, 5) + ab(1, 2)**2*ab(1, 5)*ab(2, 5)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 5)*sb(4, 5)+ab(1, 2)**2*ab(2, 3)*ab(2, 5)*ab(4, 5)*sb(1, 5)**2*sb(2, 3)**2*sb(4, 5)-ab(1, 2)**2*ab(2, 3)*ab(2, 4)*ab(4, 5)*sb(1, 3)*sb(1, 5)*sb(2, 3)*sb(2, 4)*sb(4, 5) + ab(1, 2)**2*ab(2, 3)*ab(3, 4)*ab(4, 5)*sb(1, 3)**2*sb(2, 5)*sb(3, 4)*sb(4, 5) + ab(1, 2)**2*ab(2, 5)**2*ab(4, 5)*sb(1, 5)**2*sb(2, 3)*sb(2, 5)*sb(4, 5)'
    rest1 = 'ab(1, 2)**3*ab(1, 5)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)**2*sb(2, 4) + ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)*sb(1, 4)*sb(1, 5)*sb(2, 3)*sb(3, 5) - ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)*sb(1, 5)**2*sb(2, 3)*sb(3, 4) - ab(1, 2)**2*ab(2, 3)*ab(3, 4)*ab(4, 5)*sb(1, 3)*sb(1, 5)*sb(2, 3)*sb(3, 4)*sb(4, 5) - ab(1, 2)**2*ab(2, 4)**2*ab(4, 5)*sb(1, 4)*sb(1, 5)*sb(2, 3)*sb(2, 4)*sb(4, 5) + ab(1, 2)**2*ab(2, 4)*ab(3, 4)*ab(4, 5)*sb(1, 3)*sb(1, 4)*sb(2, 5)*sb(3, 4)*sb(4, 5) - ab(1, 2)**2*ab(2, 4)*ab(3, 4)*ab(4, 5)*sb(1, 3)*sb(1, 5)*sb(2, 4)*sb(3, 4)*sb(4, 5)'

    # Simplified input1
    #input_eq = '-ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 4)*sb(3, 5) + ab(1, 2)**2*ab(1, 5)*ab(2, 5)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 5)*sb(4, 5)+ab(1, 2)**2*ab(2, 3)*ab(2, 5)*ab(4, 5)*sb(1, 5)**2*sb(2, 3)**2*sb(4, 5)-ab(1, 2)**2*ab(2, 3)*ab(2, 4)*ab(4, 5)*sb(1, 3)*sb(1, 5)*sb(2, 3)*sb(2, 4)*sb(4, 5) + ab(1, 2)**2*ab(2, 3)*ab(3, 4)*ab(4, 5)*sb(1, 3)**2*sb(2, 5)*sb(3, 4)*sb(4, 5) + ab(1, 2)**2*ab(2, 5)**2*ab(4, 5)*sb(1, 5)**2*sb(2, 3)*sb(2, 5)*sb(4, 5)+ab(1, 2)**3*ab(1, 5)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)**2*sb(2, 4) + ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)*sb(1, 4)*sb(1, 5)*sb(2, 3)*sb(3, 5) - ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)*sb(1, 5)**2*sb(2, 3)*sb(3, 4) - ab(1, 2)**2*ab(2, 3)*ab(3, 4)*ab(4, 5)*sb(1, 3)*sb(1, 5)*sb(2, 3)*sb(3, 4)*sb(4, 5) - ab(1, 2)**2*ab(2, 4)**2*ab(4, 5)*sb(1, 4)*sb(1, 5)*sb(2, 3)*sb(2, 4)*sb(4, 5) + ab(1, 2)**2*ab(2, 4)*ab(3, 4)*ab(4, 5)*sb(1, 3)*sb(1, 4)*sb(2, 5)*sb(3, 4)*sb(4, 5) - ab(1, 2)**2*ab(2, 4)*ab(3, 4)*ab(4, 5)*sb(1, 3)*sb(1, 5)*sb(2, 4)*sb(3, 4)*sb(4, 5)'

    simple2 = 'ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 3)*sb(4, 5) + ab(1, 2)**2*ab(1, 5)*ab(2, 4)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 4)*sb(4, 5)+ab(1, 2)**3*ab(3, 4)*ab(4, 5)*sb(1, 2)*sb(1, 3)*sb(1, 5)*sb(3, 4)*sb(4, 5)+ab(1, 2)**2*ab(2, 4)*ab(2, 5)*ab(4, 5)*sb(1, 5)**2*sb(2, 3)*sb(2, 4)*sb(4, 5)'
    rest2 = 'ab(1, 2)**2*ab(1, 5)*ab(2, 5)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 5)*sb(4, 5) + ab(1, 2)**2*ab(2, 3)*ab(2, 5)*ab(4, 5)*sb(1, 5)**2*sb(2, 3)**2*sb(4, 5) + ab(1, 2)**2*ab(2, 5)**2*ab(4, 5)*sb(1, 5)**2*sb(2, 3)*sb(2, 5)*sb(4, 5)'

    # Simplified input 2
    #input_eq = 'ab(1, 2)**2*ab(1, 5)*ab(2, 3)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 3)*sb(4, 5) + ab(1, 2)**2*ab(1, 5)*ab(2, 4)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 4)*sb(4, 5)+ab(1, 2)**3*ab(3, 4)*ab(4, 5)*sb(1, 2)*sb(1, 3)*sb(1, 5)*sb(3, 4)*sb(4, 5)+ab(1, 2)**2*ab(2, 4)*ab(2, 5)*ab(4, 5)*sb(1, 5)**2*sb(2, 3)*sb(2, 4)*sb(4, 5)+ab(1, 2)**2*ab(1, 5)*ab(2, 5)*ab(3, 4)*sb(1, 3)**2*sb(1, 5)*sb(2, 5)*sb(4, 5) + ab(1, 2)**2*ab(2, 3)*ab(2, 5)*ab(4, 5)*sb(1, 5)**2*sb(2, 3)**2*sb(4, 5) + ab(1, 2)**2*ab(2, 5)**2*ab(4, 5)*sb(1, 5)**2*sb(2, 3)*sb(2, 5)*sb(4, 5)'

    #simple3 = '-ab(1, 2)**3*ab(2, 5)*ab(4, 5)*sb(1, 2)*sb(1, 5)**2*sb(2, 3)*sb(4, 5)+-ab(1, 2)**3*ab(1, 5)*ab(3, 4)*sb(1, 2)*sb(1, 3)**2*sb(1, 5)*sb(4, 5)'
    #rest3 = 'ab(1, 2)**3*ab(3, 4)*ab(4, 5)*sb(1, 2)*sb(1, 3)*sb(1, 5)*sb(3, 4)*sb(4, 5)'

    # Simplified input 3
    #input_eq = '-ab(1, 2)**3*ab(2, 5)*ab(4, 5)*sb(1, 2)*sb(1, 5)**2*sb(2, 3)*sb(4, 5)+-ab(1, 2)**3*ab(1, 5)*ab(3, 4)*sb(1, 2)*sb(1, 3)**2*sb(1, 5)*sb(4, 5)+ab(1, 2)**3*ab(3, 4)*ab(4, 5)*sb(1, 2)*sb(1, 3)*sb(1, 5)*sb(3, 4)*sb(4, 5)'
    #final_sol ='-ab(1, 2)**3*ab(2, 4)*ab(2, 5)*sb(1, 2)**2*sb(1, 5)*sb(2, 3)*sb(4, 5)'

    parameters_c_dict = {
        # Name
        'exp_name': 'Test_train_contrastive',
        'dump_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/dumped/',
        'exp_id': 'test_model',
        'save_periodic': 0,
        'tasks': 'contrastive',

        # environment parameters
        'env_name': 'char_env',
        'npt_list': [5],
        'max_scale': 2,
        'max_terms': 1,
        'max_scrambles': 5,
        'save_info_scr': True,
        'save_info_scaling': True,
        'int_base': 10,
        'numeral_decomp': True,
        'max_len': 2048,
        'canonical_form': True,
        'bracket_tokens': True,
        'generator_id': 2,
        'l_scale': 0.75,
        'numerical_check': False,
        'numerator_only': True,
        'reduced_voc': True,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'n_heads': 8,
        'dropout': 0,
        'head_layers': 2,
        'n_max_positions': 384,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,
        'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments_contrastive/npt5_contrastive2/checkpoint.pth',

        # Data param
        'export_data': False,
        'prefix_path': None,
        'mma_path': None,

        # Trainer param
        'reload_data': 'contrastive,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data_contrastive.prefix.counts.train,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data_contrastive.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data_contrastive.prefix.counts.test',
        'reload_size': '',
        'epoch_size': 50000,
        'max_epoch': 500,
        'amp': -1,
        'fp16': False,
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': '',
        'reload_checkpoint': '',
        'env_base_seed': -1,
        'batch_size': 32,
        'temp_contrastive': 0.10,
        'batch_scalings': True,

        # Evaluation
        'eval_only': True,

        # SLURM/GPU param
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 0,
        'debug_slurm': False,
    }

    parameters_c = AttrDict(parameters_c_dict)

    parameters_s_dict = deepcopy(parameters_c_dict)
    parameters_s = AttrDict(parameters_s_dict)
    parameters_s.tasks = 'spin_hel'
    parameters_s.max_terms = 3
    parameters_s.max_scrambles = 3
    parameters_s.reduced_voc = False

    parameters_s.n_enc_layers = 3
    parameters_s.n_dec_layers = 3
    parameters_s.n_max_positions = 2560

    parameters_s.reload_model = '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5_c/checkpoint.pth'

    # Start the logger
    init_distributed_mode(parameters_c)
    logger = initialize_exp(parameters_c)
    init_signal_handler()

    environment.utils.CUDA = not parameters_c.cpu

    # Load the model and environment
    env_c = build_env(parameters_c)
    env_s = build_env(parameters_s)

    from model.contrastive_simplifier import load_modules, load_equation, encode_term, masked_similarity_term, similarity_terms

    encoder_c, encoder_s, decoder_s = load_modules(env_c, env_s, parameters_c, parameters_s)
    terms_num, denom = load_equation(env_c, input_eq, parameters_c)
    encoded1 = encode_term(env_c, terms_num[0], encoder_c)
    sim_term1 = masked_similarity_term(env_c, terms_num[0], terms_num, encoder_c)
    sim_term_all = similarity_terms(env_c, terms_num, encoder_c)

    modules = build_modules_contrastive(env_c, parameters_c)

    cosin_sim, ref_terms = test_expression_factors(env_c, modules, input_eq, parameters_c, factor_mask=True)

    print('done')
