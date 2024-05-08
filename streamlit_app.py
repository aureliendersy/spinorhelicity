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


@st.cache_resource
def load_model():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("model/5pt.pth")
    f_checkpoint = f_checkpoint.resolve()
    f_checkpoint_path = '/'.join(list(f_checkpoint.parts))
    if not f_checkpoint.exists():
        download_path_simplifier = 'https://drive.google.com/uc?export=download&id=1iyTEhhbvBw1W3cFls9jhnQzFiDtAhIMS'
        gdown.download(download_path_simplifier, f_checkpoint_path, quiet=False)

    return f_checkpoint_path


def test_model_expression(envir, module_transfo, f_eq, params, verbose=True, latex_form=False):
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

    f_prefix = envir.sympy_to_prefix(f_eq)
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

    out_hyp = []
    # Print out the scores and the hypotheses
    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):

        # parse decoded hypothesis
        ids = sent[1:].tolist()  # decoded token IDs
        tok = [envir.id2word[wid] for wid in ids]  # convert to prefix

        # Parse the identities if required
        try:
            hyp = envir.prefix_to_infix(tok)

            # convert to infix
            hyp = envir.infix_to_sympy(hyp)  # convert to SymPy
            hyp_disp = convert_sp_forms(hyp, env.func_dict)
            out_hyp.append(hyp_disp)
        except:
            pass

    return list(set(out_hyp))


with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
    path_mod1 = load_model()

beam_size = st.sidebar.slider('Beam Size', min_value=1, max_value=50, step=1, value=10)
nucleus_p = st.sidebar.slider('Nucleus Cutoff (Nucleus Sampling)', min_value=0.8, max_value=1.0, step=0.05, value=0.95)
temperature = st.sidebar.slider('Temperature (Nucleus Sampling)', min_value=0.5, max_value=4.0, step=0.5, value=1.5)
sample_method = st.selectbox("Sampling Method", ("Nucleus Sampling", "Beam Search"))


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
    'reduced_voc': False,
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
    'beam_size': beam_size,
    'beam_length_penalty': 1,
    'beam_early_stopping': True,
    'nucleus_sampling': sample_method == "Nucleus Sampling",
    'nucleus_p': nucleus_p,
    'temperature': temperature,

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
modules = build_modules(env, parameters)

input_eq = st.text_input("Input Equation", "(-ab(1,2)**2*sb(1,2)*sb(1,5)-ab(1,3)*ab(2,4)*sb(1,3)*sb(4,5)+ab(1,3)*ab(2,4)*sb(1,4)*sb(3,5)-ab(1,3)*ab(2,4)*sb(1,5)*sb(3,4))*ab(1,2)/(ab(1,5)*ab(2,3)*ab(3,4)*ab(4,5)*sb(1,2)*sb(1,5))")
f = sp.parse_expr(input_eq, local_dict=env.func_dict)
if parameters.canonical_form:
    f = reorder_expr(f)
f = f.cancel()
st.latex(r'''{}'''.format(latex(f)))

if st.button("Click Here to Simplify"):
    hyp_found = test_model_expression(env, modules, f, parameters, verbose=True, latex_form=True)
    st.write("Generated List of Unique Hypotheses")
    for i, hyp in enumerate(hyp_found):
        st.write(f"Hypothesis {i+1} : ${latex(hyp)}$")
