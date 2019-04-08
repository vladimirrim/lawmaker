#Pytorch
pip3.6 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip3.6 install torchvision

#Ray -- will segfault in .backward with wrong version
#pip uninstall -y ray
#pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/68b11c8251dc41e79dc9dc4b1da31630708a6bd8/ray-0.4.0-cp36-cp36m-manylinux1_x86_64.whl

#Ray
pip3.6 install ray
pip3.6 install setproctitle
pip3.6 install service_identity

#Basics
pip3.6 install numpy
pip3.6 install scipy
pip3.6 install matplotlib


#Tiled map loader
pip install pytmx

#jsonpickle
pip install jsonpickle
pip install opensimplex
pip install twisted
