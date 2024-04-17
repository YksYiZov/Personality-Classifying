from Data import getData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

def get_config(config):
    return (config["DataFilePath"],
            config["ValidFliter"],
            config["Weight"],
            config["Number"],
            config["Tqdm"])

def get_acc(model, X_valid, y):
    return model.score(X_valid, y)

def get_predict(model, X_valid):
    return model.predict(X_valid)

def train_and_test(config):
    path, is_flit, weight, number, is_tqdm = get_config(config)
    features_train, labels_train, features_valid, labels_valid = getData(path, is_flit, number, is_tqdm)
    
    knn0 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[0], labels_train[0])
    knn1 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[1], labels_train[1])
    knn2 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[2], labels_train[2])
    knn3 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[3], labels_train[3])
    
    joblib.dump(knn0, "knn0")
    joblib.dump(knn1, "knn1")
    joblib.dump(knn2, "knn2")
    joblib.dump(knn3, "knn3")
    
    IE_acc = get_acc(knn0, features_valid[0], labels_valid[0])
    SN_acc = get_acc(knn1, features_valid[1], labels_valid[1])
    TF_acc = get_acc(knn2, features_valid[2], labels_valid[2])
    JP_acc = get_acc(knn3, features_valid[3], labels_valid[3])
    
    print(f"IE的准确度{IE_acc}, SN的准确度{SN_acc}, TF的准确度{TF_acc}, JP的准确度:{JP_acc}")
    
    IE_pred = get_predict(knn0, features_valid[0])
    SN_pred = get_predict(knn1, features_valid[1])
    TF_pred = get_predict(knn2, features_valid[2])
    JP_pred = get_predict(knn3, features_valid[3])
    
    total = 0
    true = 0
    for ie, sn, tf, jp, t_ie, t_sn, t_tf, t_jp in zip(IE_pred, SN_pred, TF_pred, JP_pred, labels_valid[0], labels_valid[1], labels_valid[2], labels_valid[3]):
        if ie == t_ie and sn == t_sn and tf == t_tf and jp == t_jp:
            true += 1
        total += 1
        
    print(f"综合的准确度:{true / total}")
    
    
