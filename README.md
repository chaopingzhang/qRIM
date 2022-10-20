# qRIM
qRIM: quantitative Recurrent Inference Machine 

Pytorch Code for the paper:
Chaoping Zhang, Dimitrios Karkalousos, Pierre-Louis Bazin, Bram F. Coolen, Hugo Vrenken, Jan-Jakob Sonke, Birte U. Forstmann, Dirk H.J. Poot, Matthan W.A. Caan.
A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent inference machine. NeuroImage, Volume 264, 2022, 119680, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2022.119680.

## Requirements
Please refer to env.yml for dependencies of qRIM.

For the RIM image reconstruction network, please refer to https://github.com/wdika/mridc.


## Data preprocessing
The raw data is in NIfTI format. It includes 3D complex images, coil sensitivity, and brain mask (needed in constraining the loss in training). The data should be converted into 2D slices and saved in h5 format. Please run preprocess.dataprocess.py with the datapath modified accordingly. Note that in the paper, the RIM image reconstruction + least squares fitting is performed as initialization of the parameters for qRIM. The qRIM code can process the least squares fitting, however, the reconstructed images are expected to be provided for data loading.

The link to the open source raw data: https://doi.org/10.34894/IHZGQM.
## Run
#### training
`
python -m scripts.train_model
--sequence
MEGRE
--data-path
$DATA_PATH
--use_rim
--recurrent_layer
gru
--n_steps
6
--num_epochs
200
--num_workers
4
--sample-rate
1
--n_slices
1
--accelerations
9
9
--center-fractions
.02
.02
--batch_size
1
--device
cuda
--resume
--checkpoint
$CHECKPOINT_DIR/best_model.pt 
--exp_dir
$OUTPUT_DIR
--resolution
290
234
--loss
ssim
`
#### Run the model
`
python -m scripts.train_model 
--sequence
MEGRE
--data-path
$DATA_PATH
--num_workers
0
--sample-rate
1
--accelerations
9
9
--center-fractions
.02
.02
--device
cuda
--resolution
290
234
--loss
mse
--checkpoint
$CHECKPOINT_DIR/best_model.pt 
--out-dir
$OUTPUT_DIR
`
