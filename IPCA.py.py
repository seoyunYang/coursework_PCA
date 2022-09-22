import numpy as np 
import matplotlib.pyplot as plt
import math
import time

from scipy import io
from scipy.linalg import block_diag
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA as PCA_n
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def div_4train_test(data, label):

    train1_indices = []
    train2_indices = []
    train3_indices = []
    train4_indices = []
    test_indices = []

    for i in range(len(data)):
        if (i % 10) <= 1:
            train1_indices.append(i)
        elif (i % 10) <= 3:
            train2_indices.append(i)
        elif (i % 10) <= 5:
            train3_indices.append(i)
        elif (i % 10) <= 7:
            train4_indices.append(i)
        else:
            test_indices.append(i)

    train1 = data[train1_indices, :]
    train2 = data[train2_indices, :]
    train3 = data[train3_indices, :]
    train4 = data[train4_indices, :]
    test = data[test_indices, :]

    train1_label = label[train1_indices]
    train2_label = label[train2_indices]
    train3_label = label[train3_indices]
    train4_label = label[train4_indices]
    test_label = label[test_indices]  
    
    return [train1, train2, train3, train4], test, [train1_label, train2_label, train3_label, train4_label],test_label
def show_img(data):
    fig = plt.figure()
    axes = []
    for i in range(104):
        face = data[i].reshape((46, 56))
        face = face.real
        axes.append(fig.add_subplot(8, 13, i+1))
        plt.imshow(face.T, cmap = 'gray')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad = 0.1)
    plt.show()
    
def eigen_decomp_rank(S, m):
    # print(S.shape)
    start = time.time()
    w, v = np.linalg.eig(S)
    end = time.time()
    # print(end-start)
    sorted_index = np.argsort(w)[::-1]
    eigen_val = w[sorted_index]
    eigen_vec = v[:,sorted_index]
    
    eigen_val = eigen_val[0:m]
    eigen_vec = eigen_vec[:, 0:m]
    return eigen_val, eigen_vec, (end-start)

def PCA(data, m_pca):
    n = len(data)
    mean = np.mean(data, axis = 0)
    phi = np.subtract(data, mean)
    phi = phi.T
    cov = (phi.T @ phi) / n
    eigen_val, eigen_vec, t = eigen_decomp_rank(cov, m_pca)
    eigen_vec = phi @ eigen_vec
    norm = np.linalg.norm(eigen_vec, axis = 0)
    eigen_vec = eigen_vec/norm
    return eigen_vec, t 
    
    
def iPCA_eig_model(data, d):
    n= len(data)
    glob_mean = np.mean(data, axis = 0)
    phi = np.subtract(data, glob_mean)
    phi = phi.T
    cov = (phi @ phi.T) / n
    cov_red = (phi.T @ phi) /n
    w, v, t = eigen_decomp_rank(cov_red, d)
    v = phi @ v
    norm = np.linalg.norm(v, axis = 0)
    v = v/norm
    return glob_mean, n, w, v, cov, t

def iPCA_combine(glob_mean1, n1, v1, cov1, glob_mean2, n2, v2, cov2):
    n3 = n1+n2 
    glob_mean3 = (n1*glob_mean1 + n2*glob_mean2)/n3
    cov3 = (n1/n3) * cov1 + (n2/n3) * cov2 + (n1*n2/n3) * (glob_mean1 - glob_mean2) @ (glob_mean1 - glob_mean2).T
    h = np.concatenate((v1, v2, (glob_mean1 - glob_mean2).reshape(-1, 1)), axis = 1)
    span_set, r = np.linalg.qr(h)
    new_cov = span_set.T @ cov3 @ span_set
    # print(new_cov.shape)
    start = time.time()
    w3, rotat_mat = np.linalg.eig(new_cov)
    end = time.time()
    v3 = span_set@rotat_mat
    # print(end-start)
    return glob_mean3, n3, w3, v3, cov3, (end-start)
    
def projection(data, mean, w):
    return np.subtract(data, mean) @ w
    
def reconstruct(glob_mean, proj_coef, eigen_vec):
    recon = glob_mean + proj_coef @ eigen_vec.T
    return recon

def rec_err(original, recon):
    return mean_squared_error(original.real, recon.real, squared = False)
    
def NN(proj_train, proj_test, train_label):
    label = np.array([])
    for i in range(len(proj_test)):
        err = np.inf
        for j in range(len(train_label)):
            # new_err = mean_squared_error(proj_test[i].real, proj_train[j].real, squared = False)
            new_err = np.linalg.norm(proj_test[i] - proj_train[j])
            if new_err < err:
                index = train_label[j]
                err = new_err
        label = np.append(label, index)
    return label

def accuracy(pred_label, true_label):
    err = 0
    tot_len = len(true_label)
    for i in range(tot_len):
        if pred_label[i] != true_label[i]:
            err += 1
    return ((tot_len - err) / tot_len) * 100


def main(train, test, train_label, test_label, m_pca):
    ##batch PCA
    
    m_pca_set = np.arange(30, 111, 10)
    X = train[0]
    Y = train_label[0]
    time = list()
    recon_tr = list()
    recon_te = list()
    acc_list = list()
    
    time_1 = list()
    recon_tr_1 = list()
    recon_te_1 = list()
    acc_list_1 = list()
    
    time_i = list()
    recon_tr_i = list()
    recon_te_i = list()
    acc_list_i = list()
    for i in range(3):
            X = np.append(X, train[i+1], axis = 0)
            Y = np.append(Y, train_label[i+1])
    mean_X = np.mean(X, axis = 0)
    for m_pca in m_pca_set:
        w_pca_b, t = PCA(X, m_pca)
        # print("batch PCA time: ", t)
        train_proj_b = projection(X, mean_X, w_pca_b)
        test_proj_b = projection(test, mean_X, w_pca_b)

        recon_train = reconstruct(mean_X, train_proj_b, w_pca_b)
        recon_error = rec_err(X, recon_train)

        recon_test = reconstruct(mean_X, test_proj_b, w_pca_b)
        recon_error_test = rec_err(test, recon_test)
        
        # print("recon_train: ", recon_error)
        # print("recon_test: ", recon_error_test)   
        pred_label_b =  NN(train_proj_b, test_proj_b, Y)
        # show_img(recon_test)
        acc_b = accuracy(pred_label_b, test_label)
        time.append(t)
        recon_tr.append(recon_error)
        recon_te.append(recon_error_test)
        acc_list.append(acc_b)
        # print(acc_b)
        # print("batch PCA acc: ", acc_b)
        
        ##1 subset
        if m_pca < 101:
            w_pca_b, t = PCA(train[0], m_pca)
            # print("batch PCA time: ", t)
            mean_X = np.mean(train[0], axis = 0)
            train_proj_b = projection(train[0], mean_X, w_pca_b)
            test_proj_b = projection(test, mean_X, w_pca_b)
            
            recon_train = reconstruct(mean_X, train_proj_b, w_pca_b)
            recon_error = rec_err(train[0], recon_train)

            recon_test = reconstruct(mean_X, test_proj_b, w_pca_b)
            recon_error_test = rec_err(test, recon_test)
            # print("recon_train: ", recon_error)
            # print("recon_test: ", recon_error_test)   
            
            pred_label_b = NN(train_proj_b, test_proj_b, train_label[0])
            acc_b = accuracy(pred_label_b, test_label)
            # print(acc_b)
            
            time_1.append(t)
            recon_tr_1.append(recon_error)
            recon_te_1.append(recon_error_test)
            acc_list_1.append(acc_b)
        
    
        #inc PCA
        if m_pca < 101:
            glob_mean, n, w, v, cov, t = iPCA_eig_model(train[0], m_pca)
            # X = train[0]
            # Y = train_label[0]
            inc_t = t

            for i in range(len(train)-1):
            #     X = np.append(X, train[i+1], axis = 0)
            #     Y = np.append(Y, train_label[i+1])

                glob_mean2, n2, w2, v2, cov2, t2 = iPCA_eig_model(train[i+1], m_pca)
                inc_t += t2
                
                glob_mean, n, w, v, cov, t =iPCA_combine(glob_mean, n, v, cov, glob_mean2, n2, v2, cov2)
                inc_t += t
            train_proj = projection(X, glob_mean, v)
            test_proj = projection(test, glob_mean, v)
            
            recon_train = reconstruct(glob_mean, train_proj,v)
            recon_error = rec_err(X, recon_train)
            # print("recon_train: ", recon_error)

            pred_label = NN(train_proj, test_proj, Y)
            acc= accuracy(pred_label, test_label)
            recon_test = reconstruct(glob_mean, test_proj, v)
            recon_error_test = rec_err(test, recon_test)
            # print("recon_test: ", recon_error_test)
            
            time_i.append(inc_t)
            recon_tr_i.append(recon_error)
            recon_te_i.append(recon_error_test)
            acc_list_i.append(acc)
            # print("inc")
            # print(acc)
            # show_img(recon_train)
            # print("recongnition accuracy: ", acc)

    plt.plot(np.arange(30, 111, 10), np.array(time) * 1000, 'r--', label = 'batch PCA')
    plt.plot(np.arange(30, 101, 10), np.array(time_1) * 1000, 'g--', label = '1- PCA')
    plt.plot(np.arange(30, 101, 10), np.array(time_i) * 1000, 'b--', label = 'IPCA')
    plt.xlabel('Number of Principal Components (m)')
    plt.xticks(np.arange(30, 111, 20))
    plt.ylabel('Training Time (ms)')
    plt.legend()
    plt.show()
    
    # plt.plot(np.arange(30, 401, 10), recon_tr, 'r--', label = 'batch PCA')
    # plt.plot(np.arange(30, 101, 10), recon_tr_1, 'g--', label = '1- PCA')
    # plt.plot(np.arange(30, 101, 10), recon_tr_i, 'b--', label = 'IPCA')
    # plt.xlabel('Number of Principal Components (m)')
    # plt.xticks(np.arange(30, 401, 40))
    # plt.ylabel('Reconstruction Error (train)')
    # plt.legend()
    # plt.show()
    
    # plt.plot(np.arange(30, 401, 10), recon_te, 'r--', label = 'batch PCA')
    # plt.plot(np.arange(30, 101, 10), recon_te_1, 'g--', label = '1- PCA')
    # plt.plot(np.arange(30, 101, 10), recon_te_i, 'b--', label = 'IPCA')
    # plt.xlabel('Number of Principal Components (m)')
    # plt.xticks(np.arange(30, 401, 40))
    # plt.ylabel('Reconstruction Error (test)')
    # plt.legend()
    # plt.show()
    
    plt.plot(np.arange(30, 111, 10), acc_list, 'r--', label = 'batch PCA')
    # plt.plot(np.arange(30, 101, 10), acc_list_1, 'g--', label = '1- PCA')
    plt.plot(np.arange(30, 101, 10), acc_list_i, 'b--', label = 'IPCA')
    plt.xlabel('Number of Principal Components (m)')
    plt.xticks(np.arange(30, 111, 20))
    plt.ylabel('Face Recognition Accuracy (%)')
    plt.legend()
    plt.show()
    
    plt.plot(np.arange(30, 101, 10), acc_list_1, 'g--', label = '1- PCA')
    plt.xlabel('Number of Principal Components (m)')
    plt.xticks(np.arange(30, 101, 20))
    plt.ylabel('Face Recognition Accuracy (%)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    m_pca = 100
    d = 104
    # print("m_pca, d: ", m_pca, d)
    mat_content = io.loadmat('./face.mat')
    data = mat_content['X']
    data = data.T 
    label = mat_content['l'].flatten()
    train, test, train_label, test_label = div_4train_test(data, label)
    main(train, test, train_label, test_label, m_pca)
    