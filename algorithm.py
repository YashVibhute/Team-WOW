from tkinter import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import os

# from gui_stuff import *

l1=['t15','t16','t17','t18','t19','t20','t21','t22','t23','t24','t25','t26',
    't27','t28','t29','t30']

disease=['yes','no']

l2=[]
for i in range(0,len(l1)):
    l2.append(0)


# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("tempTrain.csv")
DF = pd.read_csv("tempTrain.csv", index_col='prognosis')
df.replace({'prognosis':{'yes':0,'no':1}},inplace=True)
DF.head()

X = df[l1]
y = df[["prognosis"]]
np.ravel(y)
# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("tempTest.csv")
tr.replace({'prognosis':{'yes':0,'no':1}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------
def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    #print(accuracy_score(y_test, y_pred))
    #print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])

        #t2.get("0",END)
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")

    #temperature = StringVar [t2.get("1.0",END)]
    #print(temperature)

# gui_stuff------------------------------------------------------------------------------------

root = Tk()
root.configure()

# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)



S1Lb = Label(root, text="Temperature 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Temperature 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Temperature 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Temperature 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Temperature 5", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)



# entries
OPTIONS = sorted(l1)


S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)




rnf = Button(root, text="Calculate", command=randomforest,bg="green",
             fg="yellow")
rnf.grid(row=17, column=0, pady=10, sticky=W)
t2 = Text(root, height=1, width=40,bg="orange",fg="black")
t2.grid(row=17, column=1 , padx=10)
#temperature = t2.get()


root.mainloop()
