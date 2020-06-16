# only svm


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage
import csv
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import  cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
import pickle


import time
start_time = time.time()
def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


image_dataset = load_image_files("")

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
# svc = svm.SVC()
# clf = GridSearchCV(svc, param_grid)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,class_weight='balanced')

clf.fit(X_train, y_train)
filename = 'random_forest_binary.sav'
pickle.dump(clf, open(filename, 'wb'))


y_pred = clf.predict(X_test)
time = time.time() - start_time
print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))

print(time)


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn,fp,fn,tp)
Accuracy = accuracy_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred,average='weighted')
Cohen_kappa =  cohen_kappa_score(y_test, y_pred)
Precision = precision_score(y_test, y_pred,average='weighted')
Recall = recall_score(y_test, y_pred,average='weighted')
MCC = matthews_corrcoef(y_test, y_pred)

print(classification_report)
with open('evaluation.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['random_forest_binary',Accuracy,F1,Cohen_kappa,Precision,Recall,MCC])
file.close()



