# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:38:56 2018

@author: ahmer
"""

import numpy as np
import lightfm.datasets as fetch_movielens
import lightfm as LightFM


# fetch datasets from lightfm
data = fetch_movielens(min_rating=4.0)

#Print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#Create model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# Get recommendationns function
def get_recommendations(model, coo_mtrx, users_ids):

    n_items = coo_mtrx.shape[1]

    for user in users_ids:

        # TODO create known positives
        # Artists the model predicts they will like
        scores = model.predict(user, np.arange(n_items))
        top_scores = np.argsort(-scores)[:3]

        print 'Recomendations for user %s:' % user

        for x in top_scores.tolist():
            for artist, values in data['artists'].iteritems():
                if int(x) == values['id']:
                    print '   - %s' % values['name']

        print '\n' # Get it pretty


user_1 = raw_input('Select user_1 (0 to %s): ' % data['users'])
user_2 = raw_input('Select user_2 (0 to %s): ' % data['users'])
user_3 = raw_input('Select user_3 (0 to %s): ' % data['users'])
print '\n' # Get it pretty

get_recommendations(model, data['matrix'], [user_1, user_2, user_3])