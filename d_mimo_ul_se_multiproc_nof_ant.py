#!/usr/bin/python
#coding=utf-8


# file name: d_mimo_ul_se_multiproc.py
# Discription:
#   Compare the thoretical and numerical results of the UL&DL spectral efficiency
#   for large-scale distributed MIMO system.
#
# This is a part of the codings for chapter 3.
# Author: D. Wang@NCRL, SEU, Dec. 31, 2018


# kill all the processes when using multi-process simulation.
# ps -ef | grep d_mimo_ul_se_multiproc | grep -v grep | awk '{print $2}' | xargs kill -9


#import matplotlib.pylab as plt
import numpy as np
from multiprocessing import Pool
import os
from time import ctime




def gen_bs_location(nof_rau, radius):

    '''
    generate BS locations given the number of BS.
    nof_bs: number of BS
    radius:  cell radius
    '''
    
    #generate BS locations
    bs_location = np.zeros((nof_rau, 2))
    bs_r = radius*np.sqrt(np.random.rand(1))
    
    bs_a = 2*np.pi*np.random.rand(1)
    bs_location[0,0] = bs_r * np.cos(bs_a)
    bs_location[0,1] = bs_r * np.sin(bs_a)
    
    n = 1
    while n<nof_rau :
        
        bs_r = radius*np.random.rand(1)
        bs_a = 2*np.pi*np.random.rand(1)
        
        x = bs_r * np.cos(bs_a)
        y = bs_r * np.sin(bs_a)
        
        min_d = np.sqrt( (x-bs_location[0:n, 0])**2 + (y-bs_location[0:n,1])**2 ).min()
        
        if min_d>10:
            bs_location[n,0] = x
            bs_location[n,1] = y
            n += 1
    
    return bs_location




def gen_ue_location(nof_ue, radius):

    '''
    generate UE locations given the number of UE.
    nof_ue: number of UE
    radius:  cell radius
    '''
    
    #generate UE locations
    ue_r = radius*np.random.rand(nof_ue)
    ue_a = 2*np.pi*np.random.rand(nof_ue)

    ue_location = np.zeros((nof_ue, 2))
    ue_location[:, 0] = ue_r * np.cos(ue_a)
    ue_location[:, 1] = ue_r * np.sin(ue_a)
    
    return ue_location


def sys_snr(p_max, d0):
    
    '''
    get the reference SNR (at 10m) of the system
    p_max: power of each UE, in dBm
    d0:    reference point, in meter
    return SNR, in linear scale
    '''
    
    # pathloss model
    # d<10, pl = 20, d = 10, PL = 34.5 + pl*log10(d) = 54.5dB
    # d>=10, 2*PL*(1 + (d/10)^pl_exp)^(-1)
    c0 = 34.5 + 20*np.log10(d0)
    noise_figure = 9
    n0 = -174
    bw = 10e6
    
    snr = p_max - c0 - noise_figure - 10*np.log10(bw) - n0
    
    snr = 10**(snr/10.0)
    return snr


def basic_thread(nof_rau, nof_ue, radius, p_max, pl_exp, nof_sp_h, nof_pilot, ant_rau, rzf_factor, reuse_factor, rand_seed):
    
    '''
    the basic thread for numerical evaluation
    '''
    prng = np.random.seed(rand_seed)
    
    bs_location = gen_bs_location(nof_rau, radius)
    ue_location = gen_ue_location(nof_ue, radius)
    
    # large-scale fading parameters of UPLINK, dim = N*K
    distance_mtx = np.zeros( (nof_rau, nof_ue) )
    for k in range(nof_ue):
        distance_mtx[:,k] = np.sqrt(( ue_location[k,0] - bs_location[:,0])**2\
                                  + ( ue_location[k,1] - bs_location[:,1])**2)
    
    # reference point, 10 meters
    d0 = 10
    snr = sys_snr(p_max, d0)
    
    # power of the large-scale fading
    pow_lsf = 2*snr*( 1/(1 + (distance_mtx/d0)**pl_exp) )
    
    # channel model under pilot contamination
    sigma_p = np.zeros((nof_rau, nof_pilot))
    eq_lsf  = np.zeros((nof_rau, nof_ue))
    for p in range(nof_pilot):
        
        # index
        idx_p = range(p*reuse_factor, (p+1)*reuse_factor)
        
        # p-th pilot, {\Sigma}_p
        x = 1 + pow_lsf[:, idx_p].sum(axis=1)
        sigma_p[:,p] = x              # x is 1-d vector
        
        # equivalent large-scale fading gain, !!! NOT POWER !!!
        eq_lsf[:,idx_p] = pow_lsf[:, idx_p] \
                * np.sqrt(1/x[:,None]).repeat(reuse_factor, axis=1)

    # covaraince of the estimation error for each link, mean-squared-error
    eq_lsf2 = eq_lsf*eq_lsf
    est_mse = pow_lsf - eq_lsf2
    
    # interference power of the users with differet pilot
    intf_lsf = np.dot(eq_lsf2.T, eq_lsf2)
    
    # covariance matrix of the interference plus noise, \tilde{\Sigma}
    tilde_sigma = est_mse.sum(axis=1) + 1
    # temporary
    sqrt_tilde_sigma = np.sqrt(tilde_sigma)
    inv_sqrt_tilde_sigma = np.sqrt(1/tilde_sigma)
    
    xi = np.zeros((reuse_factor, nof_ue))
    xi2 = np.zeros((reuse_factor, nof_ue))
    hat_xi = np.zeros((reuse_factor, nof_ue))
    tilde_xi = np.zeros((reuse_factor, nof_ue))
    eq_lsf_inv_tilde = np.zeros((nof_rau, nof_ue))
    eq_lsf_tilde = np.zeros((nof_rau, nof_ue))
    anly_sinr_mrc = np.zeros((nof_pilot,reuse_factor))
    anly_sinr_zf = np.zeros((nof_pilot,reuse_factor))
    anly_sinr_mmse = np.zeros((nof_pilot,reuse_factor))
    for p in range(nof_pilot):
        
        # index
        idx_p = range(p*reuse_factor, (p+1)*reuse_factor)
        
        # {\Xi}_p
        tmp_cov = np.dot(eq_lsf[:,idx_p].T, eq_lsf[:,idx_p])
        xi[:,idx_p] = tmp_cov
        tmp_cov2 = tmp_cov*tmp_cov
        xi2[:,idx_p] = tmp_cov2

        # \hat{\Xi}_p
        tmp = eq_lsf[:,idx_p] * sqrt_tilde_sigma[:,None].repeat(reuse_factor, axis=1)
        eq_lsf_tilde[:,idx_p] = tmp
        tmp_cov_hat = np.dot(tmp.T, tmp)
        hat_xi[:,idx_p] = tmp_cov_hat
        
        # \tilde{\Xi}_p
        tmp = eq_lsf[:,idx_p] * inv_sqrt_tilde_sigma[:,None].repeat(reuse_factor, axis=1)
        eq_lsf_inv_tilde[:,idx_p] = tmp
        tmp_cov_tilde = np.dot(tmp.T, tmp)
        tilde_xi[:,idx_p] = tmp_cov_tilde
        
        # asympotic results
        # MRC
        intf_p = intf_lsf[idx_p,:].sum(axis=1) - intf_lsf[idx_p][:,idx_p].sum(axis=1)
        anly_sinr_mrc[p,:] = np.diag(tmp_cov2) / ( (tmp_cov2.sum(axis=1) - np.diag(tmp_cov2)) + 1.0/ant_rau * intf_p + 1.0/ant_rau * np.diag(tmp_cov_hat))
        # ZF
        tmp_cov_inv = np.linalg.inv(tmp_cov)
        anly_sinr_zf[p,:] = ant_rau/np.diag(np.dot(np.dot(tmp_cov_inv, tmp_cov_hat), tmp_cov_inv))
        # MMSE
        anly_sinr_mmse[p,:] = 1/np.diag(np.linalg.inv( tmp_cov_tilde * ant_rau + np.eye(reuse_factor) )) - 1
    
    
    # channel estimation error
    hat_sigma2_mrt = np.zeros(nof_ue)
    # numerical results
    sinr_mrc  = np.zeros((nof_sp_h,nof_ue))
    sinr_zf  = np.zeros((nof_sp_h,nof_ue))
    sinr_mmse  = np.zeros((nof_sp_h,nof_ue))
    for n in range(nof_sp_h):
        
        # small-scale fading
        h = (np.random.randn(nof_rau*ant_rau, nof_pilot) \
            + 1j*np.random.randn(nof_rau*ant_rau, nof_pilot))/np.sqrt(2)
        
        # pilot contamination
        h = h.repeat(reuse_factor,axis=1)
        
        hat_g = (eq_lsf.repeat(ant_rau,axis=0)) * h
        hat_g_zf = (eq_lsf_tilde.repeat(ant_rau,axis=0)) * h
        hat_g_mmse = (eq_lsf_inv_tilde.repeat(ant_rau,axis=0)) * h
        
        # pre-computed matrix for ul and dl
        gg = np.dot(hat_g.conj().T, hat_g)
        gg_zf = np.dot(hat_g_zf.conj().T, hat_g_zf)
        gg_mmse = np.dot(hat_g_mmse.conj().T, hat_g_mmse)
        
        # MRC
        gggg = gg * (gg.conj())
        sinr_mrc[n,:] = (np.diag(gggg) / ( (gggg.sum(axis=1) - np.diag(gggg))  + np.diag(gg_zf))).real
        # ZF
        gg_inv = np.linalg.inv(gg)
        sinr_zf[n,:] = 1/(np.diag(np.dot(np.dot(gg_inv, gg_zf), gg_inv))).real
        # MMSE
        sinr_mmse[n,:] = (1/np.diag(np.linalg.inv(np.eye(nof_ue) + gg_mmse)) - 1).real
        
        
    # sum-rate, asymptotic results
    anly_sinr_mrc_sum    = (np.log2(1+anly_sinr_mrc)).sum()
    anly_sinr_zf_sum     = (np.log2(1+anly_sinr_zf)).sum()
    anly_sinr_mmse_sum     = (np.log2(1+anly_sinr_mmse)).sum()
    # sum-rate, numerical results
    sinr_zf_sum   = ((np.log2(1+sinr_zf)).sum())/nof_sp_h
    sinr_mrc_sum  = ((np.log2(1+sinr_mrc)).sum())/nof_sp_h
    sinr_mmse_sum  = ((np.log2(1+sinr_mmse)).sum())/nof_sp_h
    
    print anly_sinr_mrc_sum, sinr_mrc_sum, anly_sinr_zf_sum, sinr_zf_sum, anly_sinr_mmse_sum, sinr_mmse_sum
    print rand_seed/70.0
    return (anly_sinr_mrc_sum, sinr_mrc_sum, anly_sinr_zf_sum, sinr_zf_sum, anly_sinr_mmse_sum, sinr_mmse_sum)
    
    
def main():
    
    # get system parameters
    nof_rau = 512   #nof RAU or cells
    nof_ue  = 32
    p_max     = 17                  # transmit power of each UE
    pl_exp    = 3.7                 # pathloss expoment
    nof_sp_h  = 2                  # nof samples of Rayleigh fading
    nof_pilot = 8
    ant_rau_vec = [i for i in range(2, 6, 2)]
    reuse_factor = int(nof_ue/nof_pilot)
    rzf_factor= 0.1                 # RZF factor
    
    radius = 1000
    
    nof_point = 1
    
    anly_rate_mrc  = np.zeros((len(ant_rau_vec),nof_point))
    simu_rate_mrc  = np.zeros((len(ant_rau_vec),nof_point))
    anly_rate_zf   = np.zeros((len(ant_rau_vec),nof_point))
    simu_rate_zf   = np.zeros((len(ant_rau_vec),nof_point))
    anly_rate_mmse = np.zeros((len(ant_rau_vec),nof_point))
    simu_rate_mmse = np.zeros((len(ant_rau_vec),nof_point))
    
    print ctime()
    
    pool = Pool(processes=4)
    
    # init each thread
    results = []
    for n in range(len(ant_rau_vec)):
        
        results.append([])
        
        ant_rau = ant_rau_vec[n]
        
        for s in range(nof_point):
            
            ret = pool.apply_async(basic_thread, \
                    args=(nof_rau, nof_ue, radius, p_max, pl_exp,\
                          nof_sp_h, nof_pilot, ant_rau, rzf_factor, reuse_factor, n*nof_point+s))
            results[n].append(ret)
                
    pool.close()
    pool.join()
    
    # waiting for the end of each thread
    for n in range(len(ant_rau_vec)):
        for s in range(nof_point):
            
            x = results[n][s].get()
            
            anly_rate_mrc[n][s]  = x[0]
            simu_rate_mrc[n][s]  = x[1]
            anly_rate_zf[n][s]   = x[2]
            simu_rate_zf[n][s]   = x[3]
            anly_rate_mmse[n][s] = x[4]
            simu_rate_mmse[n][s] = x[5]
        
    mean_anly_rate_mrc  = anly_rate_mrc.mean(axis=1)
    mean_simu_rate_mrc  = simu_rate_mrc.mean(axis=1)
    mean_anly_rate_zf   = anly_rate_zf.mean(axis=1)
    mean_simu_rate_zf   = simu_rate_zf.mean(axis=1)
    mean_anly_rate_mmse = anly_rate_mmse.mean(axis=1)
    mean_simu_rate_mmse = simu_rate_mmse.mean(axis=1)
    
    simu_out = np.array([mean_anly_rate_mrc, mean_simu_rate_mrc, mean_anly_rate_zf,\
                         mean_simu_rate_zf, mean_anly_rate_mmse, mean_simu_rate_mmse])
    print simu_out
    
    print ctime()
    
    np.savetxt('d_mimo_ul_se_ant_2_32_4.txt', simu_out)
    

if __name__ == '__main__':
    main()
