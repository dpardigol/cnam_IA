from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def ACP(data,thresh=False, plot=False):
    # Fit PCA with different numbers of components
    num_components = data.shape[1]
    explained_variances = []
    count=1
    for i in range(1, num_components + 1):
        pca = PCA(n_components=i)
        pca.fit(data)#X_train
        explained_variances.append(np.sum(pca.explained_variance_ratio_))
        if thresh:
            if explained_variances[-1] > thresh:
                X_reduced = pca.fit_transform(data)
                count=i
                break
        else:   
            if explained_variances[-1] > 0.95:
                X_reduced = pca.fit_transform(data)
                count=i
                break
    if plot:
        # Plot the cumulative explained variance
        plt.plot(range(1, count + 1), explained_variances, marker='o')
        if thresh:
            plt.axhline(thresh,color='r', linestyle='--', linewidth=1)
        else:
            plt.axhline(0.9,color='r', linestyle='--', linewidth=1)
        plt.axvline(count,color='r', linestyle='--', linewidth=1)
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.show()
    return X_reduced