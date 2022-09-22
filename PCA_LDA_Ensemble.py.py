import numpy as np 
import matplotlib.pyplot as plt
import math
import timeit

from sklearn.metrics import confusion_matrix, mean_squared_error, plot_confusion_matrix
from scipy import io 
from cw1 import reconstruct, show_img, NN, accuracy, projection

def div_train_test(data, label):
    train_indices = []
    test_indices = []

    for i in range(len(data)):
        if (i % 10) <= 7:
            train_indices.append(i)
        else:
            test_indices.append(i)
    return data[train_indices, :], data[test_indices, :], label[train_indices], label[test_indices]

def eigen_decomp_rank(S, m):
    start = timeit.default_timer()
    w, v = np.linalg.eig(S)
    end = timeit.default_timer()
    sorted_index = np.argsort(w)[::-1]
    eigen_val = w[sorted_index]
    eigen_vec = v[:,sorted_index]
    eigen_val = eigen_val[0:m]
    eigen_vec = eigen_vec[:, 0:m]
    return eigen_val, eigen_vec, (end - start)

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
    return eigen_vec#, t

def LDA(data, label, w_pca, m_lda):
    classes = np.unique(label)
    glob_mean = np.mean(data, axis = 0)
    d = data.shape[1]
    SW = np.zeros((d, d))
    SB = np.zeros((d, d))
    for c in classes:
        x = data[label == c]
        c_mean = np.mean(x, axis = 0)
        SW += (x - c_mean).T @ (x - c_mean)
        mean_diff = (c_mean - glob_mean).reshape(d, 1)
        SB += x.shape[0] * (mean_diff @ mean_diff.T)
    WSBW = w_pca.T @ SB @ w_pca 
    WSWW = w_pca.T @ SW @ w_pca
    h, w = WSWW.shape
    # I = np.eye(h,h)
    # eigen_val, eigen_vec, t = eigen_decomp_rank(np.linalg.lstsq(WSWW, I, rcond = None)[0] @ WSBW, m_lda)
    eigen_val, eigen_vec, t = eigen_decomp_rank(np.linalg.pinv(WSWW) @ WSBW, m_lda)
    return eigen_vec#, t

def pca_lda(data, label, m_pca, m_lda, mean):
    w_pca, t_pca = PCA(data, m_pca)
    w_lda, t_lda= LDA(data, label, w_pca, m_lda)
    lda_proj = projection(data, mean, w_pca @ w_lda)
    return lda_proj, w_pca, w_lda#, t_pca, t_lda

def test_pca_lda(data, w_pca, w_lda, mean):
    proj = projection(data, mean, w_pca @ w_lda)
    return proj

def NN_rank(proj_train, proj_test, train_label):
    x, y = proj_test.shape
    w = len(train_label)
    c = np.unique(train_label)
    label = np.empty((x, w))
    rank_class = np.zeros((x, len(c)))
    for i in range(x):
        for j in range(w):
            label[i][j] =  mean_squared_error(proj_test[i].real, proj_train[j].real, squared = False)
    for i in range(x):
        for j in range(w):
            a = train_label[j]
            rank_class[i][a-1] += label[i][j]
    # rank = np.argsort(rank_class, axis=1)
    # rank = rank / np.sum(rank[0])
    return rank_class

def train_ensemble(data, label, num_models, m0, m1, m_lda, mean):
    n, d = data.shape
    w_pca = PCA(data, n-1)
    eigen_vec_m0 = w_pca[:, 0 : m0]
    c = np.arange(m0, n-1)
    pca_proj_coef = np.empty((0, d, m0+m1))
    lda_proj_coef = np.empty((0, m0+m1, m_lda))
    train_proj = np.empty((0, n, m_lda))
    for i in range(num_models):
        random_indices = np.random.choice(c, m1, replace = False)
        random_subspace = np.append(eigen_vec_m0, w_pca[:, random_indices], axis = 1)
        w_lda = LDA(data, label, random_subspace, m_lda)
        
        pca_proj_coef = np.append(pca_proj_coef, random_subspace.reshape(1, d, m0+m1), axis = 0)
        # pca_proj = projection(data, mean, random_subspace)
        lda_proj_coef = np.append(lda_proj_coef, w_lda.reshape(1, m0+m1, m_lda), axis = 0)
        # mean_lda = np.mean(pca_proj, axis = 0)
        # train_proj = np.append(train_proj, projection(pca_proj, mean_lda, w_lda).reshape(1, n, m_lda), axis = 0)
        
        train_proj = np.append(train_proj, projection(data, mean, random_subspace @ w_lda).reshape(1, n, m_lda), axis = 0)
        
    return train_proj, pca_proj_coef, lda_proj_coef

def test_ensemble(data, train_proj, pca_proj_coef, lda_proj_coef, train_label, test_label, mean):
    l = len(pca_proj_coef)
    tot_pred_label = np.empty((len(data), 0))
    avg_acc = 0
    for i in range(l):
        # proj_mid = projection(data, mean, pca_proj_coef[i])
        # test_proj = projection(proj_mid, mean_lda, lda_proj_coef[i])
        test_proj = projection(data, mean, pca_proj_coef[i] @ lda_proj_coef[i])
        pred_label = NN(train_proj[i], test_proj, train_label)
        tot_pred_label = np.append(tot_pred_label, pred_label.reshape(-1, 1), axis = 1)
        avg_acc += accuracy(pred_label, test_label)
    return tot_pred_label, avg_acc/l

def test_ensemble_rank(data, train_proj, pca_proj_coef, lda_proj_coef, train_label, mean):
    l = len(pca_proj_coef)
    c = np.unique(train_label)
    fin_label = np.empty((len(data), len(c)))
    for i in range(l):
        # proj_mid = projection(data, mean, pca_proj_coef[i])
        # test_proj = projection(proj_mid, mean_lda, lda_proj_coef[i])
        test_proj = projection(data, mean, pca_proj_coef[i] @ lda_proj_coef[i])
        pred_label = NN_rank(train_proj[i], test_proj, train_label)
        fin_label = fin_label + pred_label
    max_indices = np.argmin(fin_label, axis = 1)
    return max_indices + 1

def fusion_majority_voting(label):
    label = label.astype(int)
    fin_label = np.array([])
    for l in label:
        l = np.bincount(l).argmax()
        fin_label = np.append(fin_label, l)
    return fin_label
        

def main(data, label, num_models, m0, m1):
    train, test, train_label, test_label = div_train_test(data, label)
    ##pca-lda
    # m_pca_set = [70] #[100,150,200, 250]
    # m_lda_set = [50]#np.arange(10, 51, 2)
    # # time = []
    # # acc_total = list()
    mean = np.mean(train, axis = 0)
    # for m_pca in m_pca_set:
    # #     acc_list = list()
    #     for m_lda in m_lda_set:
    #         train_proj, w_pca, w_lda, t_pca, t_lda = pca_lda(train, train_label, m_pca, m_lda, mean)
    # #         # print("t_pca, t_lda: ", t_pca, t_lda)
    #         test_proj = test_pca_lda(test, w_pca, w_lda, mean)
    #         pred_label = NN(train_proj, test_proj, train_label)
            # print(pred_label)
            # print(test_label)
            # show_img(test)
            # print(w_pca.shape)
            # print(w_lda.shape)
            # print((w_pca@w_lda).shape)
            # recon_test = reconstruct(mean, test_proj,w_pca @ w_lda)
            # recon_test = reconstruct(mean, projection(w_pca.T, mean, w_pca @ w_lda), w_pca @ w_lda)
            # show_img(recon_test)
            # print("confusion matrix: ", confusion_matrix(test_label, pred_label))
            # acc = accuracy(pred_label, test_label)
            # print(acc)
            # print(acc, t_pca, t_lda)
            # con_mat = confusion_matrix(test_label, pred_label)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # cax = ax.matshow(con_mat, cmap = plt.get_cmap('Greys'))
            # fig.colorbar(cax)
            # plt.xticks(np.arange(0, len(test_label)/2 , 1), rotation = 90)
            # plt.yticks(np.arange(0, len(test_label)/2 , 1))
            # plt.xlabel('Predicted Label\naccuracy = %.2f' % acc)
            # plt.ylabel('True Label')
            # plt.show()
            # print("accuracy: ", acc)
            # time.append(t_lda)
    #         acc_list.append(acc)
    #     acc_total.append(acc_list)
    # plt.plot(m_lda_set, np.array(time) * 1000, 'k--')
    # plt.xticks(np.arange(10, 51, 5))
    # plt.xlabel('m_lda')
    # plt.ylabel('Training Time (ms)')
    # plt.show()
    # plt.plot(m_lda_set, acc_total[0], 'r--', label = 'm_pca = %d' % (m_pca_set[0]))
    # plt.plot(m_lda_set, acc_total[1], 'b--', label = 'm_pca = %d' % (m_pca_set[1]))
    # plt.plot(m_lda_set, acc_total[2], 'g--', label = 'm_pca = %d' % (m_pca_set[2]))
    # plt.plot(m_lda_set, acc_total[3], 'y--', label = 'm_pca = %d' % (m_pca_set[3]))
    # # plt.plot(m_pca_set, acc_total[0], 'r--', label = 'm_lda = %d' % (m_lda_set[0]))
    # # plt.plot(m_pca_set, acc_total[1], 'b--', label = 'm_lda = %d' % (m_lda_set[1]))
    # # plt.plot(m_pca_set, acc_total[2], 'g--', label = 'm_lda = %d' % (m_lda_set[2]))
    # # plt.plot(m_pca_set, acc_total[3], 'y--', label = 'm_lda = %d' % (m_lda_set[3]))
    # # plt.plot(m_pca_set, acc_total[4], 'c--', label = 'm_lda = %d' % (m_lda_set[4]))
    # plt.xlabel('m_lda')
    # # plt.xlabel('m_pca')
    # plt.xticks(np.arange(10, 51, 5))
    # plt.ylabel('Face Recognition Accuracy (%)')
    # plt.legend()
    # plt.show()
    # plt.plot(m_lda_set, time, 'ko')
    # plt.xlabel("m_lda ")
    # plt.ylabel('time')
    # plt.show()
    m0_set = [100]#[50, 100, 150, 200]
    m1_set = np.arange(0, 241, 20)
    m_lda = 50
    acc_tot = list()
    avg_tot = list()
    #pca-lda ensemble majority vote
    # print("majority voting")
    for m0 in m0_set:
        acc_majority = list()
        acc_sum = list()
        avg_list = list()
        for m1 in m1_set:
            train_proj_en, pca_proj_coef, lda_proj_coef= train_ensemble(train, train_label, num_models, m0, m1, m_lda, mean)
            label, avg_acc = test_ensemble(test, train_proj_en, pca_proj_coef, lda_proj_coef, train_label, test_label, mean)
            # print("avg_acc: ", avg_acc)
            fin_label = fusion_majority_voting(label)
            acc_maj = accuracy(fin_label, test_label)
            # print("accuracy: ", acc_ensemble)
            # print("confusion matrix: ", confusion_matrix(test_label, fin_label))
            fin_label_rank = test_ensemble_rank(test, train_proj_en, pca_proj_coef, lda_proj_coef, train_label, mean)
            acc_s = accuracy(fin_label_rank, test_label)
            acc_majority.append(acc_maj)
    #         con_mat = confusion_matrix(test_label, fin_label)
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         cax = ax.matshow(con_mat, cmap = plt.get_cmap('Greys'))
    #         fig.colorbar(cax)
    #         plt.xticks(np.arange(0, len(test_label)/2 , 1), rotation = 90)
    #         plt.yticks(np.arange(0, len(test_label)/2 , 1))
    #         plt.xlabel('Predicted Label\naccuracy = %.2f' % acc_maj)
    #         plt.ylabel('True Label')
    #         plt.show()
            acc_sum.append(acc_s)
            avg_list.append(avg_acc)
        # acc_tot.append(acc_majority)
        # acc_tot.append(acc_sum)
        # avg_tot.append(avg_list)
    # plt.plot(m1_set, acc_majority, 'r--', label = 'Committee Machine')#'Majority Voting')
    # plt.plot(m1_set, avg_list, 'g--', label = 'Avg. of Individual Models')#'Sum')
    # # plt.plot(m1_set, acc_tot[2], 'b--', label = 'm_0 = %d' %(m0_set[2]))
    # # plt.plot(m1_set, acc_tot[3], 'y--', label = 'm_0 = %d' %(m0_set[3]))
    # plt.xlabel('m_1')
    # plt.xticks(np.arange(0, 241, 40))
    # plt.ylabel('Face Recognition Accuracy (%)')
    # plt.title('n_model = 8')
    # plt.legend()
    # plt.show()
    
    plt.plot(m1_set, acc_majority, 'r--', label = 'Majority Voting')
    plt.plot(m1_set, acc_sum, 'g--', label = 'Sum')
    plt.xlabel('m_1')
    plt.xticks(np.arange(0, 241, 40))
    plt.ylabel('Face Recognition Accuracy (%)')
    plt.title('n_model = 8')
    plt.legend()
    plt.show()

    # ##pca-lda ensemble sum
    # print("sum")
    # fin_label_rank = test_ensemble_rank(test, train_proj_en, pca_proj_coef, lda_proj_coef, train_label, mean)
    # acc_ensemble_rank = accuracy(fin_label_rank, test_label)
    # print("accuracy: ", acc_ensemble_rank)
    # print("confusion matrix: ", confusion_matrix(test_label, fin_label_rank))



if __name__ == '__main__':
    # m_pca = 30
    # m_lda = 15
    num_models = 8
    m0 = 60
    m1 = 40
    mat_content = io.loadmat('./face.mat')
    data = mat_content['X']
    data = data.T 
    label = mat_content['l'].flatten()
    # print("m_pca, m_lda: ", m_pca, m_lda)
    # print("m0, m1, m_lda: ", m0, m1, m_lda)
    # print('base model: ', num_models)
    main(data, label, num_models, m0, m1)