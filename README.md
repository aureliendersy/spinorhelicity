# Spinorhelicity

We provide the necessary environment for dealing with spinor helicity amplitudes in Python. We allow simple manipulations, generation and shuffling of tree level amplitudes. We provide a script for training transformer models on those amplitudes, with the goal of recovering the simplified expression.

Our approach follows closely the work of https://github.com/facebookresearch/SymbolicMathematics where we keep a similar pipeline.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Data generation
Data generation can be done through the main.py script.
For instance setting environment parameters as
        # environment parameters
        'env_name': 'char_env',
        'max_npt': 8,
        'max_scale': 0.5,
        'max_terms': 1,
        'max_scrambles': 5,
        'save_info_scr': True,
        'int_base': 10,
        'max_len': 512,
        'export_data': True,
        'reload_data': '',

The amplitudes are created to correspond to n-pt tree level amplitudes and are then shuffled with momentum conservation / Schouten / Antisymmetry up to a given maximum number of scrambles. We can also choose whether to keep the information on the applied identities in the output. 

The data generation part outputs a data.prefix file that needs to be converted to the correct format following the guidelines of  https://github.com/facebookresearch/SymbolicMathematics.

## Training
To train the model we set 
        'export_data': False,
        'reload_data': 'task,path_train,path_valid,path_test',
        
and can tune the model hyperparameters
        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 4,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'reload_model': '',
