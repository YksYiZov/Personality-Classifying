from Data import getData
from sklearn.neighbors import KNeighborsClassifier

def get_config(config):
    return (config["DataFilePath"],
            config["ValidFliter"],
            config["Weight"],
            config["Number"],
            config["Tqdm"])

def get_acc(model, X_valid, y):
    return model.score(X_valid, y)

def train(config):
    path, is_flit, weight, number, is_tqdm = get_config(config)
    features_train, labels_train, features_valid, labels_valid = getData(path, is_flit, number, is_tqdm)
    
    knn0 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[0], labels_train[0])
    knn1 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[1], labels_train[1])
    knn2 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[2], labels_train[2])
    knn3 = KNeighborsClassifier(n_neighbors=3, weights=weight).fit(features_train[3], labels_train[3])
    
    IE_acc = get_acc(knn0, features_valid[0], labels_valid[0])
    SN_acc = get_acc(knn1, features_valid[1], labels_valid[1])
    TF_acc = get_acc(knn2, features_valid[2], labels_valid[2])
    JP_acc = get_acc(knn3, features_valid[3], labels_valid[3])
    
    print(IE_acc, SN_acc, TF_acc, JP_acc)
    
