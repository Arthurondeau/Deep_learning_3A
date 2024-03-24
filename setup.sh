wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -bu
rm Miniconda3-py38_4.12.0-Linux-x86_64.sh


source /home/arthur/miniconda3/etc/profile.d
source ~/.bashrc
conda init
conda create -n DL3A python=3.10 -y
conda activate DL3A
pip install -r requirements.txt



# pipe storage cp -rfi */csv/* s3://cloud-pipeline-aida-mihc-storage/TMP/IMMUcan/IMC data/raw_data/IMMUcan/IMC
# pipe storage cp -rfi *PDL1*Detections.txt s3://cloud-pipeline-aida-mihc-storage/TMP/MOSCATO_mIF/export/ data/raw_data/MOSCATO/mIF/
