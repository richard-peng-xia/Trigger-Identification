

# Intruction for The Implementation of Trigger Identification

## File Structure

├── `config.py`: file to hold parameters
├── `data.py`: file to process data
├── `data_trigger` : directory to hold raw data
│   ├── test1.csv
│   ├── test2.csv25nl25jklkjkl0
0
│   ├── train.csv
│   └── val.csv
├── `main.py`: file to run the main program
├── `model_encoder`: directory of encoder module code
│   └── bert.py
├── `model_integrate`: directory of integration module code
│   ├── max.py
│   ├── mean.py
│   └── source.py
├── `model_interact`: directory of interaction module code
│   └── none.py
├── `model_pipeline.py`: file of code to assemble encoder, interaction, integration modules
├── `model_wrapper.py`: file of wrapping pytorch models into pytorch_lightning implementation
├── `README.md`: file of implementation instruction
├── `test_result.json`: example of test result output for baseline
├── `tools`: directory to reserve minor tools
│   └── abbreviation.json
├── `trigger_identification.ipynb`: jupyter implementation to show how the code works
└── `utils.py`: file of utilities such as pytorch_lightning tools and evaluation tools

## Dependencies

```
nltk==3.6.5
numpy==1.21.3
pandas==1.3.4
pytorch-lightning==1.5.0
torchmetrics==0.6.0
scikit-learn==1.0.1
tensorboard==2.7.0
tqdm==4.62.3
transformers==4.12.3
wandb==0.12.6
```

You can also install these packages by runing:

```bash
pip install -r requirements.txt
```

`pytorch` we be automatically installed along with `pytoch_lightning`, or try this command:

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Runing Tips

To implement training process, run `main.py` in the directory.

```bash
python main.py
```

Parameters can be adjusted in `config.py`.

Usually, trigger identification contains 3 teps: (1) encode sentences into embeddings, (2) update message representation according to the propagation structure and (3) integrate message representation to form cascade representation. Therefore, we sperate these modules into 3 parts for the convenience of model modification.
