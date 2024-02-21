## generating config.ini file. Alternatively , edit directly into config.ini file

import os

from configparser import ConfigParser

config = ConfigParser()

## Use config to parse different environment variables, creates a config.ini file and saves it to same file_path

config ['DEFAULT'] = {
    'AIAP Project Name' : 'AIAP',
    ## data for Training
    'Train_URL' : 'https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db',
    ## data for Prediction
    'Predict_URL' : 'https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db',
    ## Prompt User for Input : file_path for saving Prediction results
    'file_path' : 'C:\...prediction.csv',
}
    ## Model_choice default is RF. To change this, assign your chosen value to 'user model'
config ['Model_Choice'] = {
    'RF' : 'RandomForestRegressor(max_depth=10)',
    'KNN' : 'KNeighborsRegressor',
    'GBA' : 'GradientBoostingRegressor',
    'user_model' : 'rf'

}


# Write the configuration to a file
with open('config.ini', 'w') as configfile:
    config.write(configfile)

print("config.ini file has been generated successfully.")

# config.read
#%%
