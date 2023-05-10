# MoCoGAN code map
## use MoCoGAN to generate 3D geological realizations conditioning on log data.
## Key parts to understand model: 
- Dataset
  - Preprocess: path to data, normalize, container
  - Postprocess
- Train
  - optimizer/schedule (type, tunable variables, learining rate, decay schedule)
  - dataloader: batch number, batch size, normalize
  - loss function: BCE, KL
  - validation: intervels, validation data size
  - metrics: same to loss
- Model
  - Input
  - Output
  - Model structure: input size, layers or modules, output size
  - save and load (checkpoints): what need to be save, any indicators of model structure?

Model contains Generator (***G***) and video Discriminator (***D<sub>v</sub>***) and image Discriminator (***D<sub>i</sub>***)

### *G*:
Model structure
```python
 class VideoGenerator(
   output_size: (Int,Int), # H X W
   n_output_channels: Int, # number of facies
   dim_z_content: Int, # dimension of z content, 50 in config.yaml
   dim_z_motion: Int, # dimension of z motion, 100 in config.yaml
   video_len: Int, # D
   ngf=64, # the hiddent dimension
   output_tanh=True # flag of using Tanh activitation function at output layer
 )
 ```

input includes a content vector ***z<sub>c</sub>*** and a motion vector ***z<sub>m</sub>***, which are given to ```forward''' function in order 

```python
   input_content: torch.float64   # B x dim_z_content, on device, e.g., input_content = torch.randn(batch_size, self.dim_z_content, device=device)
   input_motion: torch.float64    # B x dim_z_motion
   VideoGenerator.forward(input_content, input_motion)
```

