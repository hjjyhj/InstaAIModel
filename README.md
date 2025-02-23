# EECS 487 Project

### Note
Everything in the shared directory "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data" will be deleted in 60 days if the files were not used.
Make sure to save files that you need to your home directory.

### Setup
#### Basic Environment
Clone this repo:
```sh
git clone https://github.com/hjjyhj/InstaAIModel.git
cd InstaAIModel
```
Make sure to clone on your home directory, not on the shared directory mentioned below.

### Conda Environment
Miniconda3 installed. You can make that conda environment as default by
```sh
cd /scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/
export PATH="/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/miniconda3/bin:$PATH"
source ~/.bashrc
```
Check with 
```sh
which conda
```
This should show the path within johnkimm_dir/miniconda3.
Note that is where all the data that we need will be shared. 

You can activate the conda environment by 
```sh
conda activate my_env
```
Make sure to activate "my_env", since all the dependencies are already downloaded there.

### Run the pre-trained LLama-3.1-8B-Instruct model
```sh
cd /home/<your_directory>/InstaAIModel/run_models
python run_post_generation.py
```
Change the prompt to what you want to ask the model.

### SBERT & FAISS
IN PROGRESS. Will be updated


