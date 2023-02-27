# Confound debiasing methods in Deep Learning applied to medical imaging

The code has been separated into two directories, each of which encompasses the pipelines for raw data training and reweighing, as well as the PMDN layer approach.
- `Directory raw and reweighing`: contains the pipeline to train the raw ResNet50 and the reweighing approach. Additionally, it contains the notebooks and related python files to process and create the training and test set files.
- `Directory PMDN`: contains the pipeline to train the PMDNResNet50 model.
- `Directory nitorch`: contains files needed to run the ‘CNNpipeline.py‘ such as the trainer
or the inference function.
