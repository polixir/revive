if [ ! -d "./data" ];then
    mkdir -p data
fi
cd data
wget http://datasets.polixir.site/ib/ib-medium-99-train.npz