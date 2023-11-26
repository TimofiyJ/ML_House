import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder



class DataLoader(object):

    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):

        # Fill na 
        self.dataset['Car'] = self.dataset['Car'].fillna(self.dataset['Car'].mean())

        # Fill na 
        self.dataset['BuildingArea'] = self.dataset['BuildingArea'].fillna(self.dataset['BuildingArea'].mean())



        # Encode
        le = LabelEncoder()
        le.fit(self.dataset['Car'])
        self.dataset['Car'] = le.transform(self.dataset['Car'])

        le.fit(self.dataset['Suburb'])
        self.dataset['Suburb'] = le.transform(self.dataset['Suburb'])

        le.fit(self.dataset['Method'])
        self.dataset['Method'] = le.transform(self.dataset['Method'])

        le.fit(self.dataset['Type'])
        self.dataset['Type'] = le.transform(self.dataset['Type'])

        le.fit(self.dataset['Regionname'])
        self.dataset['Regionname'] = le.transform(self.dataset['Regionname'])

        # drop columns
        drop_elements = ['Address', 'SellerG', 'Date', 'Landsize', 'YearBuilt',
                         'Lattitude', 'Longtitude','Propertycount','CouncilArea']
        self.dataset = self.dataset.drop(drop_elements, axis=1)

        return self.dataset