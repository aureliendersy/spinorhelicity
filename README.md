# Spinorhelicity

In this repository we provide an environment for simplifying spinor-helicity amplitudes, which are mathemtical expressions written in terms of square and angle brackets. 
Provided an input amplitude $\mathcal{M}$ our machine learning pipeline reduces it to its simplified form $\overline{M}$



## Requirements

We make use of different librarires in this project

To install requirements:

```setup
pip install -r requirements.txt
```

We provide here a set of datasets and trained models


Dataset | Link 
--- | --- 
5pt with identities (3 terms max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1oM1uzud_VzVyUwsCznrga9dfZUtqdqoz)
5pt with identities (1 term max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1kmvl9N-c1b76DiP1D_bDZQfZIQc55_yi)
5pt no identities (3 terms max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1cTt9tCWW7lCe_gnR9z2mLp-AbhxrnNXW)
New 5pt no identities (3 terms max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1SAWiqo9gQYsT1yf5VI0D-QJ1oVeb5sYK)
5pt no identities (1 term max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1MBeoNhD02mgVsDC2WvDXjPrlHPS95VHz)
5pt contrastive learning | [Link](https://drive.google.com/uc?export=download&id=1TMLYbcrRBBk662M7qkeLUmnO1SQtdon4)
New 5pt contrastive learning | [Link](https://drive.google.com/uc?export=download&id=1w2w5ECf3yKY08i2ehedPeD5E8NNteMV8)

New datasets and models have momentum squared identities like $(p_1+p_2)^2=(p_3+p_4+p_5)^2$.


Model | Link 
--- | --- 
Simplifier with identities (3 terms max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1RRASnvXHtoeTLD0MTwOMEtCndAvwzmfX)
Simplifier with identities (1 term max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1EQdIeEJA9BHQhu6ZXNZn9vU2CFa-SONM)
Simplifier no identities (3 terms max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1EpMKQUjTguISkNJXng7KLVzaDKBj627V)
New Simplifier no identities (3 terms max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1iyTEhhbvBw1W3cFls9jhnQzFiDtAhIMS)
Simplifier no identities (1 term max in numerator) | [Link](https://drive.google.com/uc?export=download&id=1gV53-rn4yVLh0S5-Eo1aRW67-sYFixYR)
Contrastive grouping | [Link](https://drive.google.com/uc?export=download&id=1bdPMiAFsvD33XftHMQiojxzJlG72LnN9)
New Contrastive grouping (1) | [Link](https://drive.google.com/uc?export=download&id=1zrOkJsfERGMiK6fdadnItC8eM-33VAMP)
New Contrastive grouping (2)| [Link](https://drive.google.com/uc?export=download&id=1eopg7hqbIU56Uf3o8LzKs6w2bPXQx7aE)

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
