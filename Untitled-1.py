import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

import utils

data = np.load('dados.npz')
y_test = data['y_test']

dict_prediction = np.load('prediction.npz')
prediction = dict_prediction['arr_0']


from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, prediction, average=None)
print(roc_auc)



# ------------- Teste de precision-recall curve ----------------

# --- The average precision score in multi-label settings
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

n_classes = 5

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], prediction[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], prediction[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(
    y_test.ravel(), prediction.ravel()
)
average_precision["micro"] = average_precision_score(y_test, prediction, average="micro")


# --- Plot the micro-averaged Precision-Recall curve
from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Micro-averaged over all classes")


# --- Plot Precision-Recall curve for each class and iso-f1 curves
import matplotlib.pyplot as plt
from itertools import cycle

# setup plot details
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

_, ax = plt.subplots(figsize=(7, 8))

f_scores = np.linspace(0.2, 0.8, num=4)
lines, labels = [], []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

# add the legend for the iso-f1 curves
handles, labels = display.ax_.get_legend_handles_labels()
handles.extend([l])
labels.extend(["iso-f1 curves"])
# set the legend and the axes
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Extension of Precision-Recall curve to multi-class")

plt.show()



# -------------- Teste de confusion matrix --------------------
label_string = ['NORM','MI','CD','STTC','HYP']

# Confusion matrix
cm = multilabel_confusion_matrix(y_test, prediction)
cm_perc = utils.get_cm_percent(cm=cm, total=len(prediction))

# Plot confusion matrix
fig = plt.figure(figsize = (14, 8))

for i, (label, matrix) in enumerate(zip(label_string, cm_perc)):
    plt.subplot(f'23{i+1}')
    labels = [f'Not {label}', label]
    ax = sns.heatmap(matrix, annot = True, square = True, fmt = '.2f', cbar = False, cmap = 'Blues',
                xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)
    for t in ax.texts: t.set_text(t.get_text() + "%")
    plt.title(label)
plt.tight_layout()
plt.show()




