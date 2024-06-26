"""
Test desired model on a given input expression
"""


import torch
import streamlit as st
from environment.utils import AttrDict, to_cuda, convert_sp_forms, reorder_expr
from environment import build_env
import environment
from model import build_modules, MODULE_REGISTRAR
from add_ons.numerical_evaluations import check_numerical_equiv_local
import sympy as sp
from sympy import latex
import gdown
from pathlib import Path


@st.cache_resource
def load_model(module_npt_name):
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint_in = Path("model/{}.pth".format(module_npt_name))
    f_checkpoint = f_checkpoint_in.resolve()
    f_checkpoint_path = '/'.join(list(f_checkpoint.parts))
    if not f_checkpoint_in.exists():
        download_path_simplifier = MODULE_REGISTRAR[module_npt_name]
        gdown.download(download_path_simplifier, f_checkpoint_path, quiet=False)

    return f_checkpoint_path


@st.cache_data
def create_base_env(path_model, module_npt_name):
    parameters = AttrDict({
        'tasks': 'spin_hel',

        # environment parameters
        'env_name': 'char_env',
        'npt_list': [int(module_npt_name[0])],
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
        'all_momenta': False,

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
        'reload_model': path_model,

        # Evaluation
        'beam_eval': True,
        'beam_size': 5,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,
        'nucleus_sampling': True,
        'nucleus_p': 0.95,
        'temperature': 1.5,

        # SLURM/GPU param
        'cpu': True,

        # Specify the path to Spinors Mathematica Library
        'lib_path': None,
        'mma_path': None,
        'numerical_check': 2,
    })
    environment.utils.CUDA = not parameters.cpu

    return parameters


@st.cache_resource
def load_models(base_parameters):
    # Load the model and environment
    envir = build_env(base_parameters)
    modules_ml = build_modules(envir, base_parameters)
    # load the transformer
    encoder = modules_ml['encoder']
    decoder = modules_ml['decoder']
    encoder.eval()
    decoder.eval()

    return envir, (encoder, decoder)


def test_model_expression(envir, module_transfo, f_eq, params_in):
    """
    Test the capacity of the transformer model to resolve a given input
    :param envir:
    :param module_transfo:
    :param input_equation:
    :param params:
    :return:
    """

    encoder, decoder = module_transfo
    f_prefix = envir.sympy_to_prefix(f_eq)
    x1_prefix = f_prefix
    x1 = torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in x1_prefix] + [envir.eos_index]).view(-1, 1)
    len1 = torch.LongTensor([len(x1)])
    x1, len1 = to_cuda(x1, len1)

    # forward
    encoded = encoder('fwd', x=x1, lengths=len1, causal=False)

    # Beam decoding
    beam_sz, nucleus_sample, nucleus_prob, temp = params_in
    with torch.no_grad():
        _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_sz,
                                           length_penalty=1,
                                           early_stopping=True,
                                           max_len=2048,
                                           stochastic=nucleus_sample,
                                           nucl_p=nucleus_prob,
                                           temperature=temp)
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
            f_sp = envir.infix_to_sympy(envir.prefix_to_infix(envir.sympy_to_prefix(f_eq)))
            npt = envir.npt_list[0] if len(envir.npt_list) == 1 else None
            matches, _ = check_numerical_equiv_local(envir.special_tokens, hyp, f_sp,  npt=npt)
            out_hyp.append((matches, hyp_disp))
        except:
            pass

    return list(set(out_hyp))


module_npt = st.selectbox("Amplitude Type", ("4-pt", "5-pt", "6-pt"))

with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
    path_mod1 = load_model(module_npt)

base_params = create_base_env(path_mod1, module_npt)
env, modules = load_models(base_params)

beam_size = st.sidebar.slider('Beam Size', min_value=1, max_value=10, step=1, value=5)
nucleus_p = st.sidebar.slider('Nucleus Cutoff (Nucleus Sampling)', min_value=0.8, max_value=1.0, step=0.01, value=0.95)
temperature = st.sidebar.slider('Temperature (Nucleus Sampling)', min_value=0.5, max_value=4.0, step=0.1, value=1.5)
sample_method = st.selectbox("Sampling Method", ("Nucleus Sampling", "Beam Search"))


input_eq = st.text_input("Input Equation", "(-ab(1,2)**2*sb(1,2)*sb(1,5)-ab(1,3)*ab(2,4)*sb(1,3)*sb(4,5)+ab(1,3)*ab(2,4)*sb(1,4)*sb(3,5)-ab(1,3)*ab(2,4)*sb(1,5)*sb(3,4))*ab(1,2)/(ab(1,5)*ab(2,3)*ab(3,4)*ab(4,5)*sb(1,2)*sb(1,5))")
f = sp.parse_expr(input_eq, local_dict=env.func_dict)
if base_params.canonical_form:
    f = reorder_expr(f)
f = f.cancel()
st.latex(r'''{}'''.format(latex(convert_sp_forms(f, env.func_dict))))


if st.button("Click Here to Simplify"):
    nucleus_sampling = sample_method == 'Nucleus Sampling'
    params_input = (beam_size, nucleus_sampling, nucleus_p, temperature)
    hyp_found = test_model_expression(env, modules, f, params_input)
    st.write("Generated List of Unique Hypotheses")
    for i, (match, hyp) in enumerate(hyp_found):
        str_match = "(Valid)" if match else "(Invalid)"
        st.write(f"Hypothesis {i+1} {str_match}: ${latex(hyp)}$")

#"Link: https://spinorhelicity-itaxwjut6ymapaze8phteq.streamlit.app/"
