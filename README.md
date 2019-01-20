### How to setup an eGUP NVIDIA GEFORCE TITAN Xp, CUDA 10.0 and Tensorflow 1.12 on a 2018 MacBook Pro.

Hardware:
* MacBook Pro 2018.
* NVIDIA GEFORCE TITAN Xp
* AKiTiO Node Pro Thunderbolt3 PCIe 3.0

Software:
* macOS High Sierra Version 10.13.6 Build 17G3025
* Xcode 9.4
* Bazel 0.19.2
* CUDA 10.0
* cuDNN 7.4.1.5
* NCCL 2.3.7
* TensorFlow 1.12

These are the steps that worked for me to get fully settled using tensorflow 1.12 on the external GPU. 
Hopefully this will help other people too :)

## Enable MacBook Pro compatibility with NVIDIA Titan Xp.

1. Disable [Secure Boot](https://support.apple.com/en-us/HT208330):

	The 2018 MacBook Pro includes the T2 chip, This chip enables certain features like SSD encription, control the Touch ID and Touch Bar, but also the Secure Boot which controls every step in the boot proccess as being secure, it helps to prevent low-level attacks.

	This [short article](https://www.computerworld.com/article/3290415/apple-mac/the-macbook-pro-s-t2-chip-boosts-enterprise-security.html) helped me to get the picture.

	Reboot holding Cmd+R. Change the 'Secure Boot' and 'External Booth' to 'No Security' and 'Allowing boothing from external media'.

2. Disable System Integrity Protection (SIP) on macOS:

	Reboot holding Cmd+R. Open the terminal and run:
	
```
csrutil disable; reboot
```

3. Download and install NVIDIA GEFORCE Web Drivers:

	macOS High Sierra Version 10.13.6 Build 17G3025: [387.10.10.10.40.108](http://www.macvidcards.com/drivers.html#)

4. Run [PurgeWrangler](https://github.com/mayankk2308/purge-wrangler): 

	Enables unsupported external GPU configurations on macOS for almost all macs.

5. By now it should recognize the GPU :)


## Install CUDA 10.0 and TensforFlow 1.12.

1. Download and install [CUDA drivers](https://www.nvidia.com/object/macosx-cuda-410.130-driver.html)

2. Download [Xcode 9.4](https://developer.apple.com/download/more/) and set as default:
```
sudo xcode-select -s /Applications/Xcode9.4.app
```

3. Install [Bazel 0.19.2](https://docs.bazel.build/versions/master/install-os-x.html)

4. Install [CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-downloads?target_os=MacOSX&target_arch=x86_64&target_version=1013&target_type=dmgnetwork)

5. Install Deep Learning Library [cuDNN 7.4.1 for CUDA 10.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.4.1.5/prod/10.0_20181108/cudnn-10.0-osx-x64-v7.4.1.5.tgz):

```
tar -xzvf cudnn-10.0-osx-x64-v7.4.1.5.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib/libcudnn*
```

5. Install [NCCL 2.3.7 for CUDA 10.0](https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.3/prod3/CUDA10.0/txz/nccl_2.3.7-1%2Bcuda10.0_x86_64.txz): O/S agnostic local installer.

```
brew install xz
xz -d nccl_2.3.7-1+cuda10.0_x86_64.txz
tar xvf nccl_2.3.7-1+cuda10.0_x86_64.tar

sudo mkdir -p /usr/local/nccl
cd nccl_2.3.7-1+cuda10.0_x86_64
sudo mv * /usr/local/nccl
sudo mkdir -p /usr/local/include/third_party/nccl
sudo ln -s /usr/local/nccl/include/nccl.h /usr/local/include/third_party/nccl
```

6. Edit and source ~/.bash_profile:

```
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
PATH=$DYLD_LIBRARY_PATH:$PATH:/Developer/NVIDIA/CUDA-10.0/bin
```

7. Install dependencies:
```
sudo pip install six numpy wheel
brew install coreutils
brew install cliutils/apple/libomp
```

8. Download TensorFlow source code:
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.12
```

9. macOS doesn't support align(sizeof(T)), remove them:
```
sed -i -e "s/ __align__(sizeof(T))//g" tensorflow/core/kernels/concat_lib_gpu_impl.cu.cc
sed -i -e "s/ __align__(sizeof(T))//g" tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc
sed -i -e "s/ __align__(sizeof(T))//g" tensorflow/core/kernels/split_lib_gpu.cu.cc
```

10. Configure the build:

```
./configure

You have bazel 0.19.2 installed.
Please specify the location of python. [Default is /usr/bin/python]: /Library/Frameworks/Python.framework/Versions/3.6/bin/python3


Found possible Python library paths:
  /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages]

Do you wish to build TensorFlow with Apache Ignite support? [Y/n]: n
No Apache Ignite support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10.0


Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 7


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,7.0]: 6.1


Do you want to use clang as CUDA compiler? [y/N]: 
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
Configuration finished
```

11. Sometimes Bazel won't inclue the Bazel.rc, make sure it's included in '.tf_configure.bazelrc':
'WARNING: The following rc files are no longer being read, please transfer their contents or import their path into one of the standard rc files'

```
import 'PATH TO THE TENSORFLOW CHECKOUT'/tensorflow/tensorflow/tools/bazel.rc
```

12. Bazel Build:
```
bazel build --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```
13. Build the PIP package:
```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

14. Install TensorFlow Wheel:
```
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.12.0-cp36-cp36m-macosx_10_6_intel.whl
```

Hopefully this would have worked and it's running properly. Enjoy :)