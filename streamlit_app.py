"""
Test desired model on a given input expression
"""


import torch
import streamlit as st
import logging
from streamlit.logger import get_logger
from environment.utils import AttrDict, convert_sp_forms
from environment import build_env
import environment
from model import MODULE_REGISTRAR
from model import contrastive_simplifier
from model.simplifier_methods import all_one_shot_simplify, load_modules, load_equation
from model.contrastive_simplifier import total_simplification
from add_ons.mathematica_utils import mma_to_sp_string, create_response_frame
import sympy as sp
import numpy as np
from sympy import latex
import gdown
from pathlib import Path
from copy import deepcopy


class StreamlitLogHandler(logging.Handler):
    """
    Define a custom logger
    """
    def __init__(self, widget_update_func):
        super().__init__()
        self.widget_update_func = widget_update_func

    def emit(self, record):
        msg = self.format(record)
        self.widget_update_func(msg)


@st.cache_resource
def load_model(module_npt_name):
    """
    If we cannot find the ML models we download them from GDrive
    :param module_npt_name:
    :return:
    """
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)

    # Resolve the Path to the Simplifier Model
    path_in_simplifier = Path("model/{}.pth".format(module_npt_name))
    path_simplifier_res = path_in_simplifier.resolve()
    path_simplifier = '/'.join(list(path_simplifier_res.parts))

    # Resolve the Path to the Contrastive Model
    path_in_contrastive = Path("model/{}-contrastive.pth".format(module_npt_name))
    path_contrastive_res = path_in_contrastive.resolve()
    path_contrastive = '/'.join(list(path_contrastive_res.parts))

    # If the Simplifier model is not present we download it from Drive
    if not path_in_simplifier.exists():
        download_path_simplifier = MODULE_REGISTRAR[module_npt_name]
        gdown.download(download_path_simplifier, path_simplifier, quiet=False)

    # If the Contrastive model is not present we download it from Drive
    if not path_in_contrastive.exists():
        download_path_contrastive = MODULE_REGISTRAR[module_npt_name+'-contrastive']
        gdown.download(download_path_contrastive, path_contrastive, quiet=False)

    return path_simplifier, path_contrastive


@st.cache_data
def create_base_parameters(path_model_simplifier, path_model_contrastive, module_npt_name):
    """
    Create a base environment for the both the simplifier and contrastive models
    :param path_model_simplifier:
    :param path_model_contrastive:
    :param module_npt_name:
    :return:
    """
    parameters_simplifier = {
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
        'reload_model': path_model_simplifier,

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
    }
    environment.utils.CUDA = not parameters_simplifier['cpu']

    # Create the parameters for the contrastive model
    parameters_contrastive = deepcopy(parameters_simplifier)
    parameters_simplifier = AttrDict(parameters_simplifier)
    parameters_contrastive['tasks'] = 'contrastive'
    parameters_contrastive['n_enc_layers'] = 2
    parameters_contrastive['n_dec_layers'] = 2
    parameters_contrastive['head_layers'] = 2
    parameters_contrastive['n_max_positions'] = 256
    parameters_contrastive['norm_ffn'] = 'None'
    parameters_contrastive['reload_model'] = path_model_contrastive
    parameters_contrastive = AttrDict(parameters_contrastive)

    return parameters_simplifier, parameters_contrastive


@st.cache_resource
def load_models_env(params_simplifier, params_contrastive):
    """
    Create and load the Simplifier and Contrastive models along with the necessary environment
    :param params_simplifier:
    :param params_contrastive:
    :return:
    """
    # Build the environments
    envir_simplifier = build_env(params_simplifier)
    envir_contrastive = build_env(params_contrastive)

    # Build the Simplifier and contrastive modules
    encoder_contrastive, encoder_simplifier, decoder_simplifier = load_modules(envir_contrastive, envir_simplifier,
                                                                               params_contrastive, params_simplifier)

    return envir_simplifier, envir_contrastive, (encoder_simplifier, decoder_simplifier), encoder_contrastive


# Front End - Title and Disclaimer
st.title("Spinor Helicity Simplification")
st.caption('This app simplifies spinor-helicity amplitudes which are expressed as combinations of square'
           ' and angle brackets. The simplification is done using transformer models trained on amplitude'
           ' data with 4,5 or 6 massless external particles. The models are trained on expressions that simplify to'
           ' simple linear combinations of rational functions without any spurious poles. We do not explicitly train on'
           ' expressions with arbitrary constants but offer an option to blind constants and attempt a simplification'
           ' regardless.')

# Select the N-pt of the amplitude considered and update the session (for cache purposes)
module_npt = st.selectbox("Amplitude Type", ("4-pt", "5-pt", "6-pt"), index=1)

if 'module_npt' not in st.session_state:
    st.session_state['module_npt'] = module_npt

if st.session_state['module_npt'] != module_npt:
    # Clear the resource cache if we change the number of n-pts
    st.cache_resource.clear()
    create_base_parameters.clear()
    st.session_state['module_npt'] = module_npt

# Download the models if required
with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
    path_mod_simplifier, path_mod_contrastive = load_model(module_npt)

# Create the environments and load the models
base_params_simplifier, base_params_contrastive = create_base_parameters(path_mod_simplifier, path_mod_contrastive,
                                                                         module_npt)
env_s, env_c, modules_s, module_c = load_models_env(base_params_simplifier, base_params_contrastive)

# Create the random seeds
rng_np = np.random.default_rng(42)
rng_torch = torch.Generator(device='cuda' if not base_params_simplifier.cpu else 'cpu')
rng_torch.manual_seed(42)

# Create multiple selection boxes (sidebar) for choosing the inference method - Also choose how to deal with constants
st.sidebar.write("Sampling Method")
sample_methods = [False, False, False]
labels_methods = ["Nucleus Sampling", "Beam Search", "Greedy Decoding"]
for i in range(3):
    sample_methods[i] = st.sidebar.checkbox(labels_methods[i], value=i == 0)
inference_methods = [label for label, sample in zip(labels_methods, sample_methods) if sample]
st.sidebar.divider()

# Choose the inference parameters (sidebar)
beam_size = st.sidebar.slider('Beam Size (Beam Search and Nucleus Sampling)', min_value=1, max_value=20, step=1, value=5)
nucleus_p = st.sidebar.slider('Nucleus Cutoff (Nucleus Sampling)', min_value=0.8, max_value=1.0, step=0.005, value=0.925)
temperature = st.sidebar.slider('Temperature (Nucleus Sampling)', min_value=0.5, max_value=4.0, step=0.05, value=1.25)

# Parameters for the iterative simplification
st.sidebar.divider()
st.sidebar.write("Iterative simplification parameters")
init_cutoff = st.sidebar.slider('Initial Similarity Cutoff', min_value=0.5, max_value=1.0, step=0.01, value=0.95)
power_decay = st.sidebar.slider('Similarity Cutoff Decay', min_value=0.0, max_value=2.5, step=0.25, value=0.0)

# Field for the input equation (accepts sympy strings or S@M Mathematica syntax)
input_eq = st.text_input("Input Equation", "(-ab(1,2)**2*sb(1,2)*sb(1,5)-ab(1,3)*ab(2,4)*sb(1,3)*sb(4,5)+ab(1,3)*ab(2,4)*sb(1,4)*sb(3,5)-ab(1,3)*ab(2,4)*sb(1,5)*sb(3,4))*ab(1,2)/(ab(1,5)*ab(2,3)*ab(3,4)*ab(4,5)*sb(1,2)*sb(1,5))")
if "Spaa" in input_eq or "Spbb" in input_eq:
    input_eq = mma_to_sp_string(input_eq)

# Put the equation in canonical ordering and display its tex version
f = load_equation(input_eq, env_s, base_params_simplifier)
f = f.cancel()
st.latex(r'''{}'''.format(latex(convert_sp_forms(f, env_s.func_dict))))
st.divider()

# Choose which simplification method to apply
simplification_method = st.selectbox("Simplification Method", ("One-shot simplification", "Iterative simplification"),
                                     index=0)
blind_constants = st.checkbox("Blind Constants", value=True)

# Allow the user the option to simplify the equation
if st.button("Click Here to Simplify") and any(sample_methods):

    if simplification_method == "One-shot simplification":

        hyps_found = []
        params_input = (beam_size, nucleus_p, temperature)
        try:
            # Go through each selected inference method and generate the candidates
            hyps_found = all_one_shot_simplify(inference_methods, env_s, modules_s, f, params_input,
                                               blind_const=blind_constants, rng=rng_torch)

            # Display the list of generated candidate solutions
            st.write("Generated List of Unique Hypotheses")
            for i, (match, hyp, diff) in enumerate(hyps_found):
                str_match = "(Valid)" if match else "(Invalid)"
                st.write(f"Hypothesis {i+1} {str_match}: ${latex(hyp)}$")

            # Create a download option for the generated responses
            response_frame = create_response_frame(hyps_found, env_s)
            st.download_button(label="Download Data", data=response_frame.to_csv().encode('utf-8'),
                               file_name='hypothesis.csv', mime='text/csv')
        except AssertionError as e:
            st.write("Error: {}".format(e))

    elif simplification_method == "Iterative simplification":
        # Initialize the loggers
        streamlit_logger = get_logger(contrastive_simplifier.__name__)
        streamlit_logger2 = get_logger(contrastive_simplifier.__name__+'2')
        streamlit_logger.handlers.clear()
        streamlit_logger2.handlers.clear()
        handler = StreamlitLogHandler(st.empty().text)
        handler2 = StreamlitLogHandler(st.empty().text)
        streamlit_logger.addHandler(handler)
        streamlit_logger2.addHandler(handler2)

        envs = (env_c, env_s)
        params = (base_params_contrastive, base_params_simplifier)
        modules = (module_c,) + modules_s
        simplified_eq, out_frame = total_simplification(envs, params, f, modules, (rng_np, rng_torch),
                                                        inf_method=inference_methods, const_blind=blind_constants,
                                                        init_cutoff=init_cutoff, power_decay=power_decay, verbose=True)
        st.write(f"Simplified form: ${latex(simplified_eq)}$")
        st.download_button(label="Download Simplification Summary", data=out_frame.to_csv().encode('utf-8'),
                           file_name='hypothesis.csv', mime='text/csv')

