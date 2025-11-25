# dnn-assignment-example
Example of how to complete an assignment for CSE 5526 Neural Nets at Ohio State.

The final submissions are `baseline.ipynb` and `baseline-dropout.ipynb`. Both of
these are Jupyter notebook. For ease of grading the assignment requires two standalone
submissions, which means that the cells for basic setup are repeated.

We actually developed this code using a more version-control-friendly setup where most 
of the code is in `.py` files. The notebook that works this way is `dnn_modular.ipynb`.
For transparency, we also show `dnn_no_imports.ipynb`, which is `dnn_modular.ipynb` with
the imports commented out and the corresponding content copied into cells.

This notebook is public, it exists because it is the basis for an assignment that uses convolutional 
networks to solve the same problem.

```
.
├── baseline-dropout.ipynb: the full submission file for an extension that uses dropout
├── baseline.ipynb: the full submission file for a baseline DNN system
├── confusions.py: accuracy calculations and confusion matrix support. Usable unchanged.
├── data.py: a Python module that starts the data loading process. Usable unchanged.
├── dnn_modular.ipynb: a version-control-friendly notebook using imports instead of cells.
├── dnn_no_imports.ipynb: a notebook that pulls the imports into cells, basis for the full submissions
├── LICENSE
├── mnist_activations.py: a Python module that displays activations for mnist.
├── model.py: a DNN with switchable dropout.
├── plots.py: learning curve plots. Usable unchanged.
├── pyproject.toml: dependency declarations and project organization.
├── README.md
├── train_eval.py: code for the training and testing loop. Usable unchanged.
└── uv.lock
```
