pip install numpy scipy matplotlib pillow jupyter
pip install easydict opencv-python keras h5py lmdb mahotas PyYAML
pip install cython==0.24

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ #添加国内源
conda config --set show_channel_urls yes

# for gpu
pip install tensorflow-gpu==1.3.0
conda install pytorch=0.1.12 cuda80 torchvision -c soumith
chmod +x ./ctpn/lib/utils/make.sh
cd ./ctpn/lib/utils/ && ./make.sh

# for cpu
# pip install tensorflow==1.3.0
# conda install pytorch-cpu torchvision-cpu -c jjh_pytorch
# chmod +x ./ctpn/lib/utils/make_cpu.sh
# cd ./ctpn/lib/utils/ && ./make_cpu.sh
