from sklearn import datasets
from annoy import AnnoyIndex
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def main():
    # Iris データセットを読み込む
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target

    # 3-fold CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        # データを学習用と検証用に分ける
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ユークリッド距離で探索するモデルを用意する
        annoy_index = AnnoyIndex(X.shape[1], metric='euclidean')

        # モデルを学習させる
        for i, x in enumerate(X_train):
            print(X_train.shape)
            annoy_index.add_item(i, x)

        # k-d tree をビルドする
        annoy_index.build(n_trees=10)

        # 近傍 1 点を探索して、そのクラスを返す
        y_pred = [y_train[annoy_index.get_nns_by_vector(v, 1)[0]] for v in X_test]

        # 精度 (Accuracy) を計算する
        score = accuracy_score(y_test, y_pred)
        print('acc:', score)
if __name__ == '__main__':
    main()