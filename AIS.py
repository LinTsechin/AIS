# for updating cut of intervals
# input:
# -- cut: origin cut of intervals
# -- m_cut: sum of |h(x)| in each interval
# -- K: parameter for cutting
# output:
# -- new_cut: new cut of intervals
def cut_update(cut, m_cut, K):
    
    # num of intervals
    N = len(cut) - 1
    # new cut
    new_cut = np.zeros(N+1)
    # compute m[i]
    m = m_cut
    for i in range(N):
        m[i] = m_cut[i] * (cut[i+1] - cut[i])
    m = m / sum(m)
    m = (K*m).astype(np.int) + 1
    
    # num of intervals to combine
    comb = (sum(m)/N).astype(np.int) + 1
    # count for num of combined intervals
    count = 1
    # current index in uncombined intervals
    index = 0
    # current position in uncombined intervals
    pos = 0.0
    # combine intervals
    for i in range(N):
        # length of small interval
        incre = (cut[i+1] - cut[i]) / m[i]
        for j in range(m[i]):
            index += 1
            pos += incre
            if(index == comb):
                new_cut[count] = pos
                count += 1
                index = 0
    new_cut[N] = 1
    return new_cut

# Adaptive Importance Sampling
# input:
# -- obj_fun: function to integrate
# -- n_sample: num of random samples in each iteration
# -- n_iter: num of iterations
# -- N: num of intervals to divide
# -- K: parameter for dividing intervals
# output:
# -- I: Estimate of integration
# -- sigma2: sample variance
def AIS_Int(obj_fun, n_sample, n_iter, N, K):

    # initial cut of [0, 1]
    cut = np.linspace(0, N, N+1) / N
    # estimate of each iteration
    I = np.zeros(n_iter)
    # sample variance of each iteration
    sigma2 = np.zeros(n_iter)

    for k in range(n_iter):
        # uniform random number on [0, 1]
        U = np.random.rand(n_sample)
        # decide which interval to locale
        W = np.random.randint(low = 0, high = N, size = n_sample)
        # sample with distribution g(x)
        X = np.zeros(n_sample)
        # num of cut for each interval
        m_cut = np.zeros(N)
        
        for i in range(n_sample):
            # X[i]: sample with distribution g(x)
            X[i] = (cut[W[i]+1] - cut[W[i]]) * U[i] + cut[W[i]]
            # h(X[i])
            h_val = obj_fun(X[i]) * N * (cut[W[i]+1] - cut[W[i]])
            # compute sum of h(x[i])
            I[k] += h_val
            # compute sum of h(x[i])^2
            sigma2[k] += h_val**2
            # compute num of cut for each interval
            m_cut[W[i]] += abs(h_val)
        
        # estimate of iteration k
        I[k] /= n_sample
        # variance of iteration k
        sigma2[k] = (sigma2[k] - n_sample*(I[k]**2)) / ((n_sample - 1)*n_sample)
        # update cut of intervals
        cut = cut_update(cut, m_cut, K)
    
    # final variance
    final_sigma2 = ((1/sigma2).sum())**(-1)
    # final estimate
    final_I = final_sigma2 * (I/sigma2).sum()
    return final_I, final_sigma2
    