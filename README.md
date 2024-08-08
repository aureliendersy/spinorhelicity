# Spinorhelicity

In this repository we provide an environment for simplifying spinor-helicity amplitudes, which are mathemtical expressions written in terms of square and angle brackets. 
Provided an input amplitude $\mathcal{M}$ our machine learning pipeline reduces it to its simplified form $\overline{M}$



## Requirements

We make use of different librarires in this project (torch, numpy, sympy, streamlit, ...)

To install requirements:

```setup
pip install -r requirements.txt
```

## Trying out the model
To interact directly with the trained model please clone the repository, making sure that all of the required dependencies are correctly installed.
Then simply use 
```setup
streamlit run streamlit_app.py 
```
which will allow you to view the Streamlit app in your browser. If your browser does not support the app, copy the local url and past it into another browser like Chrome. If you have a GPU in your machine you can also enable it by setting the 'cpu' flag to False. This parameter is in the `create_base_parameters()` method of `streamlit_app.py`.

## Datasets and trained models
We provide here a set of datasets and trained models


Dataset | Link 
--- | --- 
4pt (3 identities max)| [Link](https://drive.google.com/uc?export=download&id=18idYRLv1Kzt4dcGS7s5YYONAbMsQTxD9)
5pt (3 identities max) | [Link](https://drive.google.com/uc?export=download&id=1eLbgC9F5j0zPcN3xV8-e8fH-BK1t3utF)
6pt (3 identities max) | [Link](https://drive.google.com/uc?export=download&id=150oi6ov9dSzeyCdevCNEg1cdSYzOWEgx)
4pt (numerators) | [Link](https://drive.google.com/uc?export=download&id=1iyqkbEbK_id280Qa2Mg-s0PB7AR3ZmEd)
5pt (numerators) | [Link](https://drive.google.com/uc?export=download&id=1_EwvLjoxqZB1xY6r9BVbgzhkWzUce_lg)
6pt (numerators) | [Link](https://drive.google.com/uc?export=download&id=1pENQIZhbwRKn4yl3VwvG0L-zdN0z5_Bo)



Model | Link 
--- | --- 
4pt Simplifier Model | [Link](https://drive.google.com/uc?export=download&id=1lq46Hc_eF8khsoC4k-lsMjZpaY0Pco29)
5pt Simplifier Model | [Link](https://drive.google.com/uc?export=download&id=1hSyRKcsMjxgVZ3DB1EkwD0OFHxoId4FZ)
6pt Simplifier Model | [Link](https://drive.google.com/uc?export=download&id=1nN8-nCTd08poS1ZCye0HnyLSxhs2SGCb)
4pt Embedding Model | [Link](https://drive.google.com/uc?export=download&id=1g_YAeH1v_hWDPTiSZnwrbqGwnicLRRhz)
5pt Embedding Model | [Link](https://drive.google.com/uc?export=download&id=10hZ2GhrNNOkWp-Z3Z5iQw3jbVEh5N4ei)
6pt Embedding Model| [Link](https://drive.google.com/uc?export=download&id=1AvdizkQkhtCcDdsZwh13vuKcd00HceYA)

## Data generation
Data generation can be done through the main.py script and main_contrastive.py scripts.
For instance setting trainer and environment parameters as
```
        # trainer parameters
        'export_data': True,
        'reload_data': '',
        # environment parameters
        'env_name': 'char_env',
        'npt_list': [5],
        'max_scale': 2,
        'max_terms': 3,
        'max_scrambles': 3,
        'min_scrambles': 1,
        'save_info_scr': True,
        'save_info_scaling': True,
        'numerator_only': True,
```
will generate 5-pt amplitudes with at most 3 numerators terms and scramble them using 1-3 identities (applied only on the numerators). The identities used and the little group scaling are also saved. The various parameters can be adjusted to generate different amplitudes.


The data generation part outputs a data.prefix file that needs to be converted to the correct format and split into test/valid/train following the guidelines detailed in the Readme of the environment folder.

## Training
To train the model we set 
```
        'export_data': False,
        'reload_data': 'task,path_train,path_valid,path_test',
        'reload_model': '',
```
        
and can tune the model hyperparameters (and similarly for the parameters in contrastive_main.py)
```
        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 8,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
```

## Evaluation
To evaluate the model (on the test set for instance) we can set 
```
        'eval_only': True
        'test_file': True,
        'numerical_check': 2,
        'eval_verbose': 2,
        'eval_verbose_print': True,
        'beam_eval': True,
        'beam_size': 5,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,
```
        'nucleus_sampling': False,
        'nucleus_p': 0.95,
        'temperature': 1.5,
        'scaling_eval': False,
which will use beam search with a size of 5. Activate the 'nucleus_sampling' flag to use nucleus sampling. Note that the numerical equivalence is checked locally here. If 'numerical_check' is set to 1 the numerical equivalence will be done through a mathematica session.
