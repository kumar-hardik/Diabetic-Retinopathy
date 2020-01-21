import pandas as pd
import os


#2015 data
test_2015 = '/project/DRDLM/hardik/Kaggle_data/resized_test_15'
train_2015 = '/project/DRDLM/hardik/Kaggle_data/resized_train_15'
test_df_2015 = pd.read_csv('/project/DRDLM/hardik/Kaggle_data/testLabels15.csv')
train_df_2015 = pd.read_csv('/project/DRDLM/hardik/Kaggle_data/trainLabels15.csv')

dir_2015 = '/project/DRDLM/hardik/Kaggle_data/alldata_15' #test+train 2015 data

train_df_2015.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)
test_df_2015.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)
test_df_2015.drop(columns=["Usage"], inplace=True)

#test_df_2015['path'] = test_df_2015['id_code'].map(lambda x: os.path.join(test_2015, '{}.jpg'.format(x))) #adding image extension
#train_df_2015['path'] = train_df_2015['id_code'].map(lambda x: os.path.join(train_2015, '{}.jpg'.format(x)))

alldata_2015 = pd.concat([train_df_2015, test_df_2015], ignore_index=True) #test+train Dataframe
alldata_2015.to_csv('/home/sgar2405/Hardik/df_2015.csv', index=False) #updated dataframe 2015

#2019 Data
train_2019 = '/project/DRDLM/hardik/Kaggle_data/resized_train_19'
train_df_2019 = pd.read_csv('/project/DRDLM/hardik/Kaggle_data/trainLabels19.csv')
#train_df_2019['path'] = train_df_2019['id_code'].map(lambda x: os.path.join(train_2019, '{}.jpg'.format(x))) #adding image extension

train_df_2019.to_csv('/home/sgar2405/Hardik/df_2019.csv', index=False) #updated dataframe 2019
