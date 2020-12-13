# EECE7311_Project_NST

## Our results 
Log and visualization images are available at [this google drive](https://drive.google.com/file/d/1oJKmuVe2ExwQwCKBUVPBwNt32-aQESbE/view?usp=sharing). Visualization images are put under the `gen_img` folder of each experiment.


## Reproducing
The code is checked on Ubuntu1604 and Mac OS. The following is a step-by-step guidance to reproduce our course report on **Mac OS**. By our experimence, it takes about 10-15 mins to finish the following pipeline. 

1. Download and install [Anaconda for mac](https://docs.anaconda.com/anaconda/install/mac-os/), which is very easy following their guide.
2. After installing Anaconda, open the Mac terminal, type the following snippet one by one:
   1. `sudo conda create --name pt1.7.1 python=3.6 --no-default-packages` -- This may need the password since `sudo` is used.
   2. `conda activate pt1.7.1`
   3. `git clone git@github.com:MingSun-Tse/EECE7311_Project_NST.git` -- This will download our code.
   4. `cd EECE7311_Project_NST`
   5. `pip install -r requirements.txt` -- This will install *all* the necessary libraries. Then you are all set with the environment.
3. Run: In our report, we evaluate 4 settings (2 networks: VGG19 or AlexNet, 2 training schemes: with or without FFT)
   1. AlexNet, without FFT: `CUDA_VISIBLE_DEVICES=0 python neural_style_tutorial.py --screen --content content/in1.jpg --style style/in1.jpg --net alexnet --plot_filter --plot_feat --project in1_in1__alexnet`
   2. AlexNet, with FFT: `CUDA_VISIBLE_DEVICES=0 python neural_style_tutorial.py --screen --content content/in1.jpg --style style/in1.jpg --net alexnet --plot_filter --plot_feat --project in1_in1__alexnet__fft --fft`
   3. VGG19, without FFT: `CUDA_VISIBLE_DEVICES=0 python neural_style_tutorial.py --screen --content content/in1.jpg --style style/in1.jpg --net vgg19 --plot_filter --plot_feat --project in1_in1__vgg19`
   4. VGG19, with FFT: `CUDA_VISIBLE_DEVICES=0 python neural_style_tutorial.py --screen --content content/in1.jpg --style style/in1.jpg --net vgg19 --plot_filter --plot_feat --project in1_in1__vgg19__fft --fft`

The results will be saved into a newly created folder `Experiments`. The visualization figures will be saved in the `Experiments/<project_name>/gen_img/` (a path example is `Experiments/in1_in1__alexnet_SERVER138-20201211-191245/gen_img`).