# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:09:20 2022

@author: bwb16179
"""

import pandas as pd, os, warnings, joblib, numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn import decomposition
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# =============================================================================
# # =============================================================================
# # Topic Modelling with NMF (Non-negative Matrix Factorisation) and SVD (Singular VAlue Decomposition)
# # =============================================================================
# =============================================================================

# =============================================================================
# Set up the data
# =============================================================================

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
remove = ('headers,', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

#print(newsgroups_train.filenames.shape, newsgroups_test.filenames.shape)

#print("\n".join(newsgroups_train.data[:3]))

#print(np.array(newsgroups_train.target_names)[newsgroups_train.target[:3]])

num_topics, num_top_words = 6, 8

# =============================================================================
# Can extract all word counts through sk.learn
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(newsgroups_train.data).todense()

vocab = np.array(vectorizer.get_feature_names())

# =============================================================================
# Data is now set up 
# =============================================================================


# =============================================================================
# Moving on to SVD
# =============================================================================

u, s, vh = np.linalg.svd(vectors, full_matrices=True)

#print(u.shape, s.shape, vh.shape)

# =============================================================================
# Confirm that U, s and Vh is a decomposition of the var vectors
# =============================================================================

# =============================================================================
# reconstructed_vectors = u @ np.diag(s) @ vh #diag give it a matrix it returns a vector, give it a vector it returns a matrix
# # (contd.) of the diagonal
# print(np.linalg.norm(reconstructed_vectors - vectors))
# =============================================================================

# =============================================================================
# Confirm that u and v are orthogonal
# =============================================================================

print(np.allclose(u.T @ u, np.eye(u.shape[0])))
print(np.allclose(vh @ vh.T, np.eye(vh.shape[0])))