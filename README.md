# Alien clustering for BCDI data

## Introduction 

This package use clustering algorithms to find aliens in BCDI data with minimal users input with the following steps :
- **preprocessing** : filter and rescale BCDI data in custom log scale
- **mask creation** : intensity threshold mask and possible smoothing
- **clustering** : clustering using sklearn DBSCAN algorithm
- **filtering** : filter out clusters
- **user cluster selection** : widget selection of the alien cluster from the user
- **alien mask creation** : create the final alien mask and save result 


## Files description
- **plot_utiltities.py** : Some general plotting functions
- **alien_removal_3D_utilities.py** : All functions used for the alien mask creation
- **Alien_removal_notebook.ipynb** : jupyter notebook with a step-by-step procedure to make the alien mask
- **example_data** : folder with BCDi data as a tutorial
- **requirements.txt** : file containing required python module versions (ipywidgets CheckBox can be broken in recent versions)

## Get started
- clone the repository

    `git clone https://github.com/ewbellec/alienclustering.git`

- open the jupyter notebook Alien_removal_notebook.ipynb
- follow the instructions
- If you encounter issues using this code, email ewen.bellec@esrf.fr

## Video tutorial
https://github.com/ewbellec/alienclustering/assets/45454083/5b72897b-35a2-4864-aa8f-d9c27bdf527e

## Illustrations
![alien_masking_illustration](https://github.com/ewbellec/alienclustering/assets/45454083/933741c2-93a2-4b40-9f39-db454f1942df)
![scatter3d_colors](https://github.com/ewbellec/alienclustering/assets/45454083/195f84e6-a6eb-46ca-b8bd-4f1085e3445b)
