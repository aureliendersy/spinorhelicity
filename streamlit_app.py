"""
Test desired model on a given input expression
"""


import torch
import streamlit as st
from environment.utils import AttrDict, to_cuda, convert_sp_forms, reorder_expr
from environment import build_env
import environment
from model import build_modules, MODULE_REGISTRAR
from model.simplifier_methods import test_model_expression
from add_ons.mathematica_utils import mma_to_sp_string
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


st.title("Spinor Helicity Simplification")
st.caption('This app simplifies spinor-helicity amplitudes which are expressed as combinations of square'
           ' and angle brackets. The simplification is done using transformer models trained on amplitude'
           ' data with 4,5 or 6 massless external particles. The models are trained on expressions that simplify to'
           ' simple linear combinations of rational functions without any spurious poles. We do not explicitly train on'
           ' expressions with arbitrary constants but offer an option to blind constants and attempt a simplification'
           ' regardless.')
module_npt = st.selectbox("Amplitude Type", ("4-pt", "5-pt", "6-pt"), index=1)

if 'module_npt' not in st.session_state:
    st.session_state['module_npt'] = module_npt

if st.session_state['module_npt'] != module_npt:
    # Clear the resource cache if we change the number of n-pts
    st.cache_resource.clear()
    create_base_env.clear()
    st.session_state['module_npt'] = module_npt

with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
    path_mod1 = load_model(module_npt)

base_params = create_base_env(path_mod1, module_npt)
env, modules = load_models(base_params)

sample_method = st.sidebar.selectbox("Sampling Method", ("Nucleus Sampling", "Beam Search", "Greedy Decoding"))
beam_size = st.sidebar.slider('Beam Size', min_value=1, max_value=10, step=1, value=5)
nucleus_p = st.sidebar.slider('Nucleus Cutoff (Nucleus Sampling)', min_value=0.8, max_value=1.0, step=0.01, value=0.95)
temperature = st.sidebar.slider('Temperature (Nucleus Sampling)', min_value=0.5, max_value=4.0, step=0.1, value=1.5)
blind_constants = st.sidebar.checkbox("Blind Constants", value=False)

input_eq = st.text_input("Input Equation", "(-ab(1,2)**2*sb(1,2)*sb(1,5)-ab(1,3)*ab(2,4)*sb(1,3)*sb(4,5)+ab(1,3)*ab(2,4)*sb(1,4)*sb(3,5)-ab(1,3)*ab(2,4)*sb(1,5)*sb(3,4))*ab(1,2)/(ab(1,5)*ab(2,3)*ab(3,4)*ab(4,5)*sb(1,2)*sb(1,5))")
if "Spaa" in input_eq or "Spbb" in input_eq:
    input_eq = mma_to_sp_string(input_eq)
f = sp.parse_expr(input_eq, local_dict=env.func_dict)
if base_params.canonical_form:
    f = reorder_expr(f)
f = f.cancel()
st.latex(r'''{}'''.format(latex(convert_sp_forms(f, env.func_dict))))

st.divider()

if st.button("Click Here to Simplify"):

    params_input = (beam_size, sample_method, nucleus_p, temperature)
    try:
        hyp_found = test_model_expression(env, modules, f, params_input, blind_const=blind_constants)
        st.write("Generated List of Unique Hypotheses")
        for i, (match, hyp) in enumerate(hyp_found):
            str_match = "(Valid)" if match else "(Invalid)"
            st.write(f"Hypothesis {i+1} {str_match}: ${latex(hyp)}$")
    except AssertionError as e:
        st.write("Error: {}".format(e))

#"Link: https://spinorhelicity-nnwpdnyaaulfzizcmbfhjs.streamlit.app/"
