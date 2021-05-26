# hiv1pr-nacs

Repository of Jupyter Notebook and python analysis functions for reproducing study of HIV-1 protease near attack conformation analysis from the papers: 

- S.K. Sadiq‡ (2020) Catalysts, Fine-tuning of sequence-specificity by near attack conformations in enzyme-catalyzed peptide hydrolysis, 10 (6) 684 
- Sadiq, S.K. ‡ and Coveney P.V. (2015). J Chem Theor Comput. Computing the role of near attack conformations in an enzyme-catalyzed nucleophilic bimolecular reaction. 11 (1), pp 316–324

## Download of Analysis scripts from Github

```
git clone https://github.com/kashifsadiq/hiv1pr-nacs
```

The Jupyter notebook: HIV1pr-substrates-NAC-analysis.ipynb contains the analysis script. It imports a python module nac.py, which contains several functions involved in the analysis. The corresponding datasets can be downloaded from Zenodo. 

## Download of MD postprocessed data set from Zenodo

From https://zenodo.org/record/4808575#.YK5MqiaxUlU  - download the data.tar file into the directory in which the Jupyter notebook is cloned, then untar it.

```
tar xvf data.tar
```

This will create data/apo and data/enz subdirectories containing the apo-substrate and enzyme-substrate systems respectively. 


