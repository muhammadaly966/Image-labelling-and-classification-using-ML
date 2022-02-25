#!/usr/bin/env python
# coding: utf-8

# **First we will import the necessary libraries and modules , in addition to the that a script is written to explore the files and load the images with labeling each image regarding its class.**

# In[163]:


import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score,roc_curve,plot_roc_curve,auc

def img_process(path):
    img_read=skio.imread(path) # to read each img
    img_grayscale = rgb2gray(img_read)# to convert to gray scale
    img_resized= resize(img_grayscale,(40, 60)).reshape(1,40*60) #to resize the images to 40,60 and the make them 1D
    return img_resized
    

image_directory="./EnglishImg/English/Img/GoodImg/Bmp"
files=os.listdir(image_directory)
y=[]# to store targets
x=[] # to store imapges as arrays
for i in files : #to open each file and read images
    path=os.path.join(image_directory,i)
    if i == "Sample009":
        for img in os.listdir(path):
            y.append(8)#label 8 for class 8
            x.append(img_process(os.path.join(path,img)))# to add img to the examples list
             
    elif i == "Sample017":
        for img in os.listdir(path):
            y.append(1)#1 is for class G
            x.append(img_process(os.path.join(path,img)))
            
    elif i == "Sample051":
        for img in os.listdir(path):
            y.append(2)#2 is for class o
            x.append(img_process(os.path.join(path,img)))
            
            
    elif i == "Sample053":
        for img in os.listdir(path):
            y.append(3)#3 is for class q
            x.append(img_process(os.path.join(path,img)))
  
    
    
        
X=np.array(x)#converting lists to array
X=X.reshape(-1,2400)# make it 2D matrix with rows are the examples and cloumns are features (pixels)
Y=np.array(y)#to convert the list into array
print("No of the images in all classes is {}".format(len(Y)))





        
        
        
        
    
    
    
    
        
            
        
        


# **seperating each class from the others to be ready if needed**

# In[32]:


G_examples = np.where(Y == 1)
index_G=np.array(G_examples).reshape(-1,1)
first_G=int(index_G[0])
last_G=int(index_G[len(index_G)-1])
G_samples=X[first_G:last_G+1,:]
G_Targets=Y[first_G:last_G+1]
print("No of the images in G class is {}".format(len(G_Targets)))
################# (8) sampels
eight_examples = np.where(Y == 8)
index_eight=np.array(eight_examples).reshape(-1,1)
first_eight=int(index_eight[0])
last_eight=int(index_eight[len(index_eight)-1])
eight_samples=X[first_eight:last_eight+1,:]
eight_Targets=Y[first_eight:last_eight+1]
print("No of the images in 8 class is {}".format(len(eight_Targets)))
################### (q) sampels
q_examples = np.where(Y == 3)
index_q=np.array(q_examples).reshape(-1,1)
first_q=int(index_q[0])
last_q=int(index_q[len(index_q)-1])
q_samples=X[first_q:last_q+1,:]
q_Targets=Y[first_q:last_q+1]
print("No of the images in q class is {}".format(len(q_Targets)))
################### (o) sampels
o_examples = np.where(Y == 2)
index_o=np.array(o_examples).reshape(-1,1)
first_o=int(index_o[0])
last_o=int(index_o[len(index_o)-1])
o_samples=X[first_o:last_o+1,:]
o_Targets=Y[first_o:last_o+1]
print("No of the images in o class is {}".format(len(o_Targets)))


# In[8]:





# Merging the data of q and o together
# and 8 with G

# In[33]:


Y_o_q=np.hstack((o_Targets,q_Targets))
Y_o_q=(Y_o_q==2)
X_o_q=np.vstack((o_samples,q_samples))
Y_G_8=np.hstack((eight_Targets,G_Targets))
Y_G_8=(Y_G_8==1)
X_G_8=np.vstack((eight_samples,G_samples))


# splitting the data into training and testing (there is no need for seperat cross validation set,especially because the data set is small and there is no hyperparameter tuning or evalution for different classifier types) <br>
# -Folding cross validation will use training data 

# In[170]:


#for the o and q classifier
X_train_oq, X_test_oq, y_train_oq, y_test_oq = train_test_split(X_o_q, Y_o_q, test_size=0.33, random_state=42)
print("for the (o-q) classifier the size of training data is {} and the size of testing data is {}".format(len(y_train_oq),len(y_test_oq)))
bin_clf_oq = LogisticRegression(solver='liblinear')
bin_clf_oq.fit(X_train_oq, y_train_oq)


# -the data set of the o and q letters are merged and then splitted using (train_test_split function from scikit learn ) the training data are 135 examples and testing data are 67 examples.<br>
# -For fitting the model we use the training set only as in the cell above.<br>
# -The splitting is based on random process becaused the data was organized and each class was seperated from the other ,so we needed to shuffle to get a mixture of all the classes to train and test the classifier.<br>
# when looking at the whole traning data one can notice that (o) examples is more than (q)  q ratio is =(54/54+148)*100 =26.7%<br>
# -After sampling nearly the same ratio is maintained (around 27.4% in training and  25.3% in testing) which indicates to apprpriate random sampling results.<br>
# 
# -the precision and recall functions defined below returns the **average** precision for all classes when it operates on a multi-class problem, and it returns only the **positive** class when operating on binary classification problem.<br>
# 
# 
# 

# In[216]:


y_pred_test_oq = bin_clf_oq.predict(X_test_oq)
cm_test_oq=confusion_matrix(y_test_oq, y_pred_test_oq)
def accuracy(v):
    a=v.diagonal()
    a=float(np.sum(a))
    
    summ=float(np.sum(np.sum(v)))
    return a/summ
def precision(v):
    a=v.diagonal()
    precision_1=[]
    if v.shape[0]==2:
        return a[1]/np.sum(v[:,1])
    else:
        for i in range(v.shape[0]):
            s=np.sum(v[:,i])
            precision_1.append(a[i]/s)
        return sum(precision_1)/len(precision_1)
    
def recall(v):
    a=v.diagonal()
    recall_1=[]
    if v.shape[0]==2:
        return a[1]/np.sum(v[1,:])
    else:
        for i in range(v.shape[0]):
            s=np.sum(v[i,:])
            recall_1.append(a[i]/s)
        return sum(recall_1)/len(recall_1)
    
def F1_score(pre,rec):
    return (2*pre*rec)/(pre+rec)
    
    

print("the accuracy for testing data :",accuracy(cm_test_oq))
print("the precision for testing data :",precision(cm_test_oq))
print("the recall for testing data :",recall(cm_test_oq))
print("the F1-score for testing data :",F1_score(precision(cm_test_oq),recall(cm_test_oq)))
print('The confusion matrix of (o-q) binary classifier based on testing data: \n {0}'.format(cm_test_oq ))
print("first cloumn/row is for 'q' predictions and second column/row is for 'o'")
print("----------------------------------------")
y_pred_train_oq = bin_clf_oq.predict(X_train_oq)
cm_train_oq=confusion_matrix(y_train_oq, y_pred_train_oq)
print("the accuracy for training data :",accuracy(cm_train_oq))
print("the precision for training data :",precision(cm_train_oq))
print("the recall for training data :",recall(cm_train_oq))
print("the F1-score for training data :",F1_score(precision(cm_train_oq),recall(cm_train_oq)))
print( 'The confusion matrix of (o-q) binary classifier based on training data: \n {0}'.format(cm_train_oq))


# Considering the confusion matrix of testing data  in the above cell , It's noticable that for the True class (predicting "o") the model is working well but some of the "q" sampels is missclassified as "o" , on the other hand no "o" is missclassifed as "q"<br>  
# It can also be observed from the recall value from the testing dataset<br>
# 
# One of the main reasons for that to happen is the small number of dataset in generall and  the small amount of "q" exampels compared to "o" exampels and what supports that is the confusion matrix from training data , the model is performing well on training data but the data are not enough to teach the model to work on any new data

# In[27]:


plot_roc_curve(bin_clf_oq, X_train_oq, y_train_oq)


# In[167]:


plot_roc_curve(bin_clf_oq, X_test_oq, y_test_oq)


# In[172]:


#for the 8 and G classifier
X_train_G8, X_test_G8, y_train_G8, y_test_G8 = train_test_split(X_G_8, Y_G_8, test_size=0.33, random_state=42)
print("for the (G-8) classifier the size of training data is {} and the size of testing data is {}".format(len(y_train_G8),len(y_test_G8)))
bin_clf_G8 = LogisticRegression(solver='liblinear')
bin_clf_G8.fit(X_train_G8, y_train_G8)


# **The same procedure used on "o"and "q" is used on "8" and "G" classes <br>**
# The data is splitted and shuffeled randomly and a classifier is trained using the training set 

# In[218]:


y_pred_test_G8 = bin_clf_G8.predict(X_test_G8)
cm_test_G8=confusion_matrix(y_test_G8, y_pred_test_G8)
print("the accuracy for testing data :",accuracy(cm_test_G8))
print("the precision for testing data :",precision(cm_test_G8))
print("the recall for testing data :",recall(cm_test_G8))
print("the F1-score for testing data :",F1_score(precision(cm_test_G8),recall(cm_test_G8)))
print( 'The confusion matrix of (G-8) binary classifier based on testing data: \n {0}'.format(cm_test_G8 ))
print("first cloumn/row is for '8' predictions and second column/row is for 'G'")
print("----------------------------------------")
y_pred_train_G8 = bin_clf_G8.predict(X_train_G8)
cm_train_G8=confusion_matrix(y_train_G8, y_pred_train_G8)
print("the accuracy for training data :",accuracy(cm_train_G8))
print("the precision for training data :",precision(cm_train_G8))
print("the recall for training data :",recall(cm_train_G8))
print("the F1-score for training  data :",F1_score(precision(cm_train_G8),recall(cm_train_G8)))
print( 'The confusion matrix of (G-8) binary classifier based on training data: \n {0}'.format(cm_train_G8))


# **Everything was computed the same as "o" and "q"**<br>
# One can notice that the classfier's performance on training dataset is outstanding (without any missclassifications)<br>
# but when tested the numbers are low ,which is due to insufficient number of data , with the data used being unbalanced as we have 143 example for "G" and only 38 for "8"<br>
# 
# **in order to have stronger classification process more data is needed to train the models in a better way and to balnce both classes examples.**
# 

# In[38]:


plot_roc_curve(bin_clf_G8, X_train_G8, y_train_G8)


# In[39]:


plot_roc_curve(bin_clf_G8, X_test_G8, y_test_G8)


# **The code below is used for Multi-class classification.** <br>Now the data is treated as a whole and the target y is used with labels=[1,2,3,8]<br> 
# ["8" is labeled 8,'G' is labeled 1,'o'is labeled 2,'q' is labeled 3]<br>
# The logistic classifier is trained withe the **training split** of the data and when it's used in multiclass problems it has different options ,the one whic is used here is **one vs. rest**<br> 
# 

# In[158]:



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
clf = LogisticRegression(solver='liblinear',multi_class="ovr")
clf.fit(X_train, y_train)
#given that "8" is labeled 8 
#'G' is labeled 1
#'o'is labeled 2
# 'q' is labeled 3


# **The predictions were made using training data and testing data to find the metrics for both cases**

# In[219]:


y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)




cm_test=confusion_matrix(y_test, y_pred_test)
print("the accuracy for testing data :",accuracy(cm_test))                                            
print("the precision for testing data :",precision(cm_test))
print("the recall for testing data :",recall(cm_test))
print("the F1-score for testing data :",F1_score(precision(cm_test),recall(cm_test)))
print( 'The confusion matrix of multi-class  classifier based on testing data: \n {0}'.format(cm_test ))
print("columns/rows are of order 'G','o','q','8'")
print("----------------------------------------")
cm_train=confusion_matrix(y_train, y_pred_train)
print("the accuracy for training data :",accuracy(cm_train))                                            
print("the precision for training data :",precision(cm_train))
print("the recall for training data :",recall(cm_train))
print("the F1-score for training data :",F1_score(precision(cm_train),recall(cm_train)))
print( 'The confusion matrix of multi-class  classifier based on training data: \n {0}'.format(cm_train))
print("columns/rows are of order 'G','o','q','8'")
print("----------------------------------------")


# **The metrics shows that most of the performance metrics when working on multiclassication is so low on testing data and the model isn't reliable** <br>
# Incresing the data set amount may help in making the model better but apparently using binary classification is more efficient.

# In order to plot the ROC curves we will seperate each label in one column instead of having one column vector with all labels<br>
# the problem will be addressed with the same approch ,but this time the **one Vs rest classifier** will be used explicitly to get seperate vector predictions and scores<br>

# In[155]:


#data_labels seperation in different columns
y = label_binarize(Y, classes=[1,2,3,8])
n_classes = y.shape[1]
#re-splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,
                                                    random_state=42)
classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
y_score = classifier.fit(X_train, y_train).decision_function(X_train)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_train[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



plt.figure()
lw = 2
colors = ['aqua', 'darkorange', 'cornflowerblue','red']
for i, color in zip(range(n_classes), colors):
    x=fpr[i]
    y=tpr[i]
    if i ==3 :
        c=7
    else:
        c=i
    plt.plot(x, y, color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(c+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of multi-class')
plt.legend(loc="lower right")
plt.show()


# The performance of all the classfieres on the **training data** is near ideal

# In[156]:


classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



plt.figure()
lw = 2
colors = ['aqua', 'darkorange', 'cornflowerblue','red']
for i, color in zip(range(n_classes), colors):
    x=fpr[i]
    y=tpr[i]
    if i ==3 :
        c=7
    else:
        c=i
    plt.plot(x, y, color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(c+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of multi-class')
plt.legend(loc="lower right")
plt.show()


# The above figure shows the ROC for each class using **Testing data**<br>
# it can be observed that class 3 is giving better charateristics (more precise in general) relative to other classes

# In[199]:


image = X_test_oq[20,:].reshape(40, 60)
plt.imshow(image, cmap = matplotlib.cm.binary,
           interpolation="nearest")

