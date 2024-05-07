"""
Test desired model on a given input expression
"""


from statistics import mean
import torch
import streamlit as st
from add_ons.slurm import init_signal_handler, init_distributed_mode
from environment.utils import AttrDict, to_cuda, initialize_exp, convert_sp_forms, reorder_expr
from environment import build_env
import environment
from model import build_modules, check_model_params
from add_ons.mathematica_utils import *
import sympy as sp
from sympy import latex
import gdown
import os, csv
import user_args as args
from pathlib import Path
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


@st.cache
def load_model():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("model/5pt.pth")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            download_id = '1iyTEhhbvBw1W3cFls9jhnQzFiDtAhIMS'
            download_file_from_google_drive(download_id, f_checkpoint)

    return f_checkpoint


def test_model_expression(envir, module_transfo, input_equation, params, verbose=True, latex_form=False):
    """
    Test the capacity of the transformer model to resolve a given input
    :param envir:
    :param module_transfo:
    :param input_equation:
    :param params:
    :param verbose:
    :param latex_form:
    :return:
    """

    # load the transformer
    encoder = module_transfo['encoder']
    decoder = module_transfo['decoder']
    encoder.eval()
    decoder.eval()

    f = sp.parse_expr(input_equation, local_dict=envir.func_dict)
    if params.canonical_form:
        f = reorder_expr(f)
    f = f.cancel()
    f_prefix = envir.sympy_to_prefix(f)
    x1_prefix = f_prefix
    x1 = torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in x1_prefix] + [envir.eos_index]).view(-1, 1)
    len1 = torch.LongTensor([len(x1)])
    x1, len1 = to_cuda(x1, len1)

    # forward
    encoded = encoder('fwd', x=x1, lengths=len1, causal=False)

    # Beam decoding
    beam_size = params.beam_size
    with torch.no_grad():
        _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_size,
                                           length_penalty=params.beam_length_penalty,
                                           early_stopping=params.beam_early_stopping,
                                           max_len=params.max_len,
                                           stochastic=params.nucleus_sampling,
                                           nucl_p=params.nucleus_p,
                                           temperature=params.temperature)
        assert len(beam) == 1
    hypotheses = beam[0].hyp
    assert len(hypotheses) == beam_size

    if verbose:
        print(f"Input function f: {f}")
        if latex_form:
            print(latex(f))
            print(latex(f.together()))
        print("")

    first_valid_num = None

    data_out = []

    # Print out the scores and the hypotheses
    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):

        # parse decoded hypothesis
        ids = sent[1:].tolist()  # decoded token IDs
        tok = [envir.id2word[wid] for wid in ids]  # convert to prefix
        hyp_disp = ''
        # Parse the identities if required
        try:
            hyp = envir.prefix_to_infix(tok)
            if '&' in tok:
                prefix_info = tok[tok.index('&'):]
                info_infix = envir.scr_prefix_to_infix(prefix_info)
            else:
                info_infix = ''

            # convert to infix
            hyp = envir.infix_to_sympy(hyp)  # convert to SymPy
            hyp_disp = convert_sp_forms(hyp, env.func_dict)

            # When integrating the symbol
            if params.numerical_check:
                hyp_mma = sp_to_mma(hyp, envir.npt_list, params.bracket_tokens, envir.func_dict)
                f_sp = envir.infix_to_sympy(envir.prefix_to_infix(envir.sympy_to_prefix(f)))
                tgt_mma = sp_to_mma(f_sp, envir.npt_list, params.bracket_tokens, envir.func_dict)
                matches, error = check_numerical_equiv(envir.session, hyp_mma, tgt_mma)
            else:
                matches = None
                error = None

            res = "Unknown" if matches is None else ("OK" if (matches or error == -1) else "NO")
            # remain = "" if matches else " | {} remaining".format(remaining_diff)
            remain = ""

            if (matches or error == -1) and first_valid_num is None:
                first_valid_num = num + 1

        except:
            res = "INVALID PREFIX EXPRESSION"
            hyp = tok
            remain = ""
            info_infix = ''

        if verbose:
            # print result
            if latex_form:
                print("%.5f  %s %s %s %s" % (score, res, hyp, info_infix, latex(hyp_disp)))
            else:
                print("%.5f  %s %s  %s %s" % (score, res, hyp, info_infix, remain))

        if res == "INVALID PREFIX EXPRESSION":
            data_out.append(['INVALID EXPR', 'NO', score, ''])
        else:
            hyp_mma = sp_to_mma(hyp, envir.npt_list, params.bracket_tokens, envir.func_dict)
            data_out.append([hyp, res, score, hyp_mma])

    if verbose:
        if first_valid_num is None:
            print('Could not solve')
        else:
            print('Solved in beam search')
        print("")
        print("")

    return first_valid_num


path_mod1 = load_model()


parameters = AttrDict({
    'tasks': 'spin_hel',

    # environment parameters
    'env_name': 'char_env',
    'npt_list': [5],
    'max_scale': 2,
    'max_terms': 3,
    'max_scrambles': 3,
    'min_scrambles': 1,
    'save_info_scr': False,
    'save_info_scaling': False,
    'numeral_decomp': True,
    'int_base': 10,
    'max_len': 2048,
    'canonical_form': True,
    'bracket_tokens': True,
    'generator_id': 2,
    'l_scale': 0.75,
    'numerator_only': True,
    'reduced_voc': True,
    'all_momenta': True,

    # model parameters
    'emb_dim': 512,
    'n_enc_layers': 3,
    'n_dec_layers': 3,
    'n_heads': 8,
    'dropout': 0,
    'n_max_positions': 2560,
    'attention_dropout': 0,
    'sinusoidal_embeddings': False,
    'share_inout_emb': True,
    'positional_encoding': True,

    # Specify the path to the simplifier model
    'reload_model': path_mod1,

    # Evaluation
    'beam_eval': True,
    'beam_size': 10,
    'beam_length_penalty': 1,
    'beam_early_stopping': True,
    'nucleus_sampling': True,
    'nucleus_p': 0.95,
    'temperature': 2,

    # SLURM/GPU param
    'cpu': True,

    # Specify the path to Spinors Mathematica Library
    'lib_path': None,
    'mma_path': None,
    'numerical_check': False,
})

# Start the logger
check_model_params(parameters)

environment.utils.CUDA = not parameters.cpu

# Load the model and environment
env = build_env(parameters)
#modules = build_modules(env, parameters)

input_eq = st.text_input("Input Equation", "ab(1,2)*sb(2,3)-ab(1,4)*sb(3,4))/ab(2, 3)")
f = sp.parse_expr(input_eq, local_dict=env.func_dict)
if parameters.canonical_form:
    f = reorder_expr(f)
f = f.cancel()
st.latex(r'''{}'''.format(latex(f)))

#test_model_expression(env, modules, input_eq, parameters, verbose=True, latex_form=True)
