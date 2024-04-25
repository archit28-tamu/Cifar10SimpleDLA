# DLA on CIFAR-10 dataset

## Prequisits:
- Python 3.6+
- PyTorch 1.0+

`conda env create requirements.yml`

## Command to run on train mode:
`python main.py <train path of training data>`

Replace `<path of training data>` with the path to the directory containing the training data.

## Command to run on test mode:
`python main.py test <path of testing data>`

Replace `<path of testing data>` with the path to the directory containing the testing data.

## Command to run on predict mode:
`python main.py predict <path of training data> --pred_data_dir <path to private test data> --result_dir <path to save predictions data>`

Replace `<path of training data>` with the path to the directory containing the training data,
`<path to private test data>` with the path to the directory containing the private test data, 
and `<path to save predictions data>`
