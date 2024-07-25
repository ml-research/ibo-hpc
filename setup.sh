#!/bin/sh
pip uninstall -y scikit-learn pandas
pip install pandas scikit-learn
pip uninstall -y scikit-learn
pip install scikit-learn==1.1.3
curl -sS https://starship.rs/install.sh | sh
echo "eval $(starship init bash)" >> ~/.bashrc
mkdir -p /usr/local/lib/python3.9/site-packages/Cython-3.0.10.dist-info/
touch /usr/local/lib/python3.9/site-packages/Cython-3.0.10.dist-info/METADATA
pip install hypermapper
bash