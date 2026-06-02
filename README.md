# premier-league-stability
Repo housing the data/code used to investigate factors associated with sustainable Premier League clubs.

## Repo structure

`data/` - Contains all of the data files used in the production of the models.
- `raw/` - Contains landing files as they were received at source.
- `filtered/` - Contains raw files filtered on club/season.
- `mapping` - Contains tables used to map IDs from disparate sources.
- `master` - Contains joined data used for analysis and modelling. Contains `modelling_features`, which was the main dataset used for producing models.
- `eda` - Any data snippets used for EDA.
- `outputs` - Used to house model output data.

`eda` - Notebooks used for exploratory data analysis.

`modelling` - Scripts used to run logistic regression models on the data.

`transformations` - Scripts used for joins/feature engineering on the data.

`utils` - Contains functions shared across the repo.

## Package dependencies

All package dependencies can be found in `requirements.txt`. If using UV, these can be installed in your terminal using `uv pip install -r requirements.txt`
