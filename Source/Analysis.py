import numpy as np

def per_class_f1(model,vor,validate_batches,classLabels):
    print("Truth in rows, prediction in columns")

    #confusion = np.zeros((len(classLabels.keys(),len(classLabels.keys()))
    TP = np.zeros(len(classLabels.keys()))
    FN = np.zeros(len(classLabels.keys()))
    FP = np.zeros(len(classLabels.keys()))
    TN = np.zeros(len(classLabels.keys()))

    for i in range(0,int(validate_batches*3)):
        X,y = vor.next()
        pred = model.predict(X).round()
        for row in range(pred.shape[0]):
            for col in range(pred.shape[1]):
                if y[row,col]==1:
                    if pred[row,col]==1:
                        TP[col] = TP[col]+1
                    else:
                    	FN[col] = FN[col]+1
                else:
                    if pred[row,col]==1:
                        FP[col] = FP[col]+1
                    else:
                        TN[col] = FP[col]+1                        


    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*recall*precision/(recall+precision)

    results={"precision":precision,"recall":recall,"f1":f1,"TP":TP,"FP":FP,"TN":TN,"FN":FN}
    return results


precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = recall*precision/(recall+precision)

