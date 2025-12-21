import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from fmae import forward_selection, FmaeExplainer
from fmae import plot_feature_salience, print_rules
from sklearn import model_selection


"""
Test case for the FMAE Python library of the paper
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI" [1]
https://ieeexplore.ieee.org/document/10731553
Authors: Faliang Yin, Hak-Keung Lam, David Watson
"""

def load_dataset_from_csv(file_path, num_features=5, random_state=1):
    dataset = np.loadtxt(file_path, delimiter=",", skiprows=0)
    sam, label = dataset[:, :-1], dataset[:, -1].reshape(-1, 1)
    fea_idx = forward_selection(sam, label, weights=None, num_features=5, random_state=random_state)
    sam = sam[:, fea_idx]
    train, test, labels_train, _ = \
        model_selection.train_test_split(sam, label, train_size=0.80, random_state=random_state)
    return train, test, labels_train, fea_idx


#  Load Breast Cancer Wisconsin (Original) [2] dataset
dataset_name = 'WBC'
mode = 'classification'
feature_names = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
class_names = ['Benign', 'Malignant']

np.random.seed(0)
train, test, labels_train, fea_idx = load_dataset_from_csv('WBC.csv', random_state=0)
selected_fea_names = [feature_names[i] for i in fea_idx]

# Train a closed box model
if mode == 'classification':
    model = MLPClassifier(max_iter=1000)  # closed box (black box) classifier
else:
    model = MLPRegressor(max_iter=1000)
model.fit(train, labels_train.ravel())

# Generate local explanations for instance test[0] by FMAE
if mode == 'classification':
    explainer = FmaeExplainer(model.predict_proba, np.concatenate((train, test), axis=0), mode=mode)
else:
    explainer = FmaeExplainer(model.predict, np.concatenate((train, test), axis=0), mode=mode)
feature_weights = explainer.explain(np.expand_dims(test[0], axis=0))

# Show explanation results
print(f'R2 score on training set: {explainer.score}')
print_rules(explainer.rule_bases[0], 3, explainer.d_scores[0], feature_list=selected_fea_names, class_list=class_names,)  # save_path='../doc/WBC_Rules.txt'
plot_feature_salience(feature_weights[0, :, 0], absolute_mode=True, title_note=f'for Class {class_names[0]}', feature_names=np.array(selected_fea_names),)  # save_path='../doc/WBC_Class0.png'
plot_feature_salience(feature_weights[0, :, 1], absolute_mode=True, title_note=f'for Class {class_names[1]}', feature_names=np.array(selected_fea_names),)  # save_path='../doc/WBC_Class1.png'

"""
References
[1] F. Yin, H. -K. Lam and D. Watson, "Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms, and 
Interface for XAI," in IEEE Transactions on Fuzzy Systems, vol. 33, no. 2, pp. 549-558, Feb. 2025, 
doi: 10.1109/TFUZZ.2024.3485212.
[2] W. Wolberg, O. Mangasarian, N. Street, and W. Street. "Breast Cancer Wisconsin (Diagnostic)," 
UCI Machine Learning Repository, 1993. [Online]. Available: https://doi.org/10.24432/C5DW2B.
"""
pass

