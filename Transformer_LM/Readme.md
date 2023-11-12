## How to Run a BERT classification model?

Step 1: Create a pickle (or .csv) file with at least two columns: **text** and **label**. The label needs to be binary

Step 2: Use python3 to run the **BERT_generic.py** code

## Example Usage

```python3 BERT_generic.py --log_dir /path/to/log/ --data_dir /path/to/data/ --batch_size 8 --device cuda:1 --epochs 4 --hidden_size 128 --metric_avg macro```

To know more about the argument options:

```python3 BERT_generic.py --help```
