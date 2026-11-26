from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

def metrics(y_true,y_pred,class_names):
    return {
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "macro_f1":float(f1_score(y_true,y_pred,average="macro")),
        "weighted_f1":float(f1_score(y_true,y_pred,average="weighted")),
        "confusion_matrix":confusion_matrix(y_true,y_pred,labels=list(range(len(class_names)))).tolist(),
        "classification_report":classification_report(y_true,y_pred,labels=list(range(len(class_names))),target_names=class_names,output_dict=True,zero_division=0)
    }
