# Confound debiasing methods in Deep Learning applied to medical imaging

The following repository comprises the code for the project of the Bachelor's thesis `Confound debiasing methods in Deep Learning applied to medical imaging`. 
The code has been separated into two directories, each of which encompasses the pipelines for raw data training and reweighing, as well as the PMDN layer approach.
- `Directory raw and reweighing`: contains the pipeline to train the raw ResNet50 and the reweighing approach. Additionally, it contains the notebooks and related python files to process and create the training and test set files.
- `Directory PMDN`: contains the pipeline to train the PMDNResNet50 model.
- `Directory nitorch`: contains files needed to run the ‘CNNpipeline.py‘ such as the trainer
or the inference function.


## How to train the models

As mentioned in the section above, this repository contains two directories which contain the pipeline to train the ResNet50 model with two confound debiasing techniques. Also, the directory `raw and reweighing` contains the config files to train the raw model without any debiasing technique during the training. 

### Training a raw model

To train a raw model, you have to forward to the path where the python file that will run the CNN training pipeline is located. 
```
cd /ritter/share/archive/Thesis_GonzaloCardenal/thesis_code/raw_and_reweighing/CNNpipeline/
```
Once there, you have to run the python file `runCNNpipeline.py` with the config file that contains the training parameters in which you want to train the raw model. Config files for the different training are available in the `config` folder. For example we take the config file `confounds_ukbb_sex-final` that will train the model for sex prediction with our h5file. The code to run the training would be:
```
nohup python runCNNpipeline.py confounds_ukbb_sex-final  &> confounds_ukbb_sex-final &
```
The command `nohup` allows to keep the training running in the GPU even when we disconnect from the server.

Now the training will start in the GPU. 

### Training a model with Reweighing approach

To train a model with the reweighing debiasing technique, the steps are similar to the raw model. You have to forward to the path where the python file that will run the CNN training pipeline is located. 
```
cd /ritter/share/archive/Thesis_GonzaloCardenal/thesis_code/raw_and_reweighing/CNNpipeline/
```
Once there, you have to run the python file `runCNNpipeline.py` with the config file that contains the training parameters for the Reweighing configuration. Config files for the different trainings are available in the `config` folder. For example we take the config file `Reweighing-Sex-brain-volume` that will train the model for sex prediction and control for Total Brain Volume as the main confounder. The code to run the training would be:
```
nohup python runCNNpipeline.py Reweighing-Sex-brain-volume  &> Reweighing-Sex-brain-volume &
```

Now the training will start in the GPU. 

### Training a model with PMDN approach

To train a model with the reweighing debiasing technique, the steps are similar to the raw model, but this time you have to be located in the 'PMDN` Directory. You have to forward to the path where the python file that will run the CNN training pipeline is located, which now is:
```
cd /ritter/share/archive/Thesis_GonzaloCardenal/thesis_code/PMDN/MLPipeline/CNNPipeline/
```
Once there, you have to run the python file `runCNNpipeline.py` with any of the config files that contains the training parameters for the PMDN configuration in the pipeline. Config files for the different trainings are available in the `config` folder. For example we take the config file `PMDNResNet-sex` that will train the model for sex prediction and control for Total Brain Volume as the main confounder. The code to run the training would be:
```
nohup python runCNNpipeline.py PMDNResNet-sex  &> PMDNResNet-sex &
```

Now the training will start in the GPU. 
