import numpy as np
import scipy.io as scio
import math

def Gibbs_analyser(Iterat,x,N,pi_1,sigma2_a,sigma2_alpha,sigma2_gamma,sigma2_gamma2,eta,xi,fshift,onset_list_T,end_list_T,onset_list_P,end_list_P):
	sigma2_gamma3 = 0.01
	K = len(x)
	q = K
	qc = int(np.ceil(N/2))
	G = len(onset_list_P)
	est_f = 1
	est_g3 = 1
	ec = 0

	#Initialize Sampler
	p0factor = np.log((1-pi_1) / 10 * np.sqrt(sigma2_a))
	b_avg_T = np.zeros((K, 1))
	b_avg_P = np.zeros((K, 1))
	b_hat_T = np.zeros((K, 1))
	b_hat_P = np.zeros((K, 1))
	a_hat_T = np.zeros((K, 1))
	a_hat_P = np.zeros((K, 1))
	a_fshift_T = np.zeros((K, 1))
	a_fshift_P = np.zeros((K, 1))
	r_hat_T = np.zeros((K, 1))
	r_hat_P = np.zeros((K, 1))
	s_hat = np.zeros((K, 1))
	f_hat_T = np.zeros((K, 1))
	f_hat_P = np.zeros((K, 1))
	f_fshift_T = np.zeros((K, 1))
	f_fshift_P = np.zeros((K, 1))
	coll_b_T = np.zeros((Iterat, K))
	coll_b_P = np.zeros((Iterat, K))
	coll_a_T = np.zeros((Iterat, K))
	coll_a_P = np.zeros((Iterat, K))
	coll_s = np.zeros((Iterat, 1))
	coll_f_T = np.zeros((Iterat, q))
	coll_f_P = np.zeros((Iterat, q))

	coll_gamma3 = np.zeros((Iterat, G * 6))

	B = scio.loadmat('B.mat')
	B = B['B']
	N = 20
	H_0 = np.squeeze(np.zeros(((q-2*qc)//2,N)))
	H = np.concatenate((H_0,B[(512//2+np.arange(-qc,qc)),0:N],H_0)) / 100


	M_P = np.squeeze(np.zeros((K,G)))
	for i in range(G):
		M_P[int(onset_list_P[i]-1 ):int(end_list_P[i]),i] = 1
		M_P[:,i] = M_P[:,i] / np.linalg.norm(M_P[:,i])
	M_T = np.squeeze(np.zeros((K,G)))
	for i in range(G):
		M_T[int(onset_list_T[i] - 1):int(end_list_T[i]),i] = 1
		M_T[:,i] = M_T[:,i] / np.linalg.norm(M_T[:,i])
	M = M_P + M_T
	non_QRS = np.sum(M.conj().T > 0,axis=0).conj().T.reshape([-1,1])


	M3 = np.squeeze(np.zeros((K,G*6)))
	index = 0
	for i in range(0,G*6,6):
		interval_len = end_list_P[index] - onset_list_T[index] + 1
		M3[onset_list_T[index] - 1: end_list_P[index], i] = 1
		M3[onset_list_T[index] - 1: end_list_P[index], i + 1] = np.linspace(1, interval_len, interval_len)
		M3[onset_list_T[index] - 1: end_list_P[index], i + 2] = np.power(np.linspace(1, interval_len, interval_len),2)
		M3[onset_list_T[index] - 1: end_list_P[index], i + 3] = np.power(np.linspace(1, interval_len, interval_len), 3)
		M3[onset_list_T[index] - 1: end_list_P[index], i + 4] = np.power(np.linspace(1, interval_len, interval_len), 4)
		M3[onset_list_T[index] - 1: end_list_P[index], i + 5] = np.power(np.linspace(1, interval_len, interval_len), 5)
		M3[:,i] = M3[:,i]/np.linalg.norm(M3[:,i])
		M3[:, i+1] = M3[:, i+1] / np.linalg.norm(M3[:, i+1])
		M3[:, i+2] = M3[:, i+2] / np.linalg.norm(M3[:, i+2])
		M3[:, i+3] = M3[:, i+3] / np.linalg.norm(M3[:, i+3])
		M3[:, i+4] = M3[:, i+3] / np.linalg.norm(M3[:, i+4])
		M3[:, i+5] = M3[:, i+3] / np.linalg.norm(M3[:, i+5])
		index = index + 1
	gamma3 = np.squeeze(np.zeros((G*6,1)))
	RH = np.squeeze(np.zeros((q,N)))
	RH2 = np.squeeze(np.zeros((q,N)))
	b_est_T = np.zeros((K, 1))
	b_est_P = np.zeros((K, 1))
	a_est_T = np.sqrt(sigma2_a / 2) * (np.random.randn(K, 1))
	a_est_P = np.sqrt(sigma2_a / 2) * (np.random.randn(K, 1))
	r_est_T = a_est_T * b_est_T
	r_est_P = a_est_P * b_est_P

	gamma3_est = np.sqrt(sigma2_gamma3/2) * np.random.randn(G*6,1)
	sigma2_n_est = np.random.gamma(xi,1/eta,[1,1])
	sigma2_n_est = 1 / sigma2_n_est
	f_est_P = np.squeeze(np.zeros((q,1)))
	f_est_P[q // 2 - 1] = 1
	f_est_T = np.squeeze(np.zeros((q,1)))
	f_est_T[q // 2 - 1] = 1
	x_est_P = np.zeros((K, 1))
	x_est_T = np.zeros((K, 1))

	# begin pcg sampler
	for i in range(Iterat):
		#begin {sample alpha_T}
		for rh_ind in range(N):
			H_row = H[(q-2*qc)//2 + np.arange(0,2*qc)]
			H_col = H_row[:,rh_ind]
			rh = np.convolve(np.squeeze(r_est_T),H_col)
			RH[0:q,rh_ind] = rh[K-1-(q//2-1)-(q-2*qc)//2 + np.arange(0,q)]
		RH = np.dot(np.diag(np.squeeze(non_QRS)),RH)
		RHHR = np.real(np.dot(RH.conj().T,RH))
		sg2_inv = (1 / sigma2_n_est * RHHR + np.diag(1 / sigma2_alpha * np.squeeze(np.ones((1, N)))))
		sqrt_sg2_inv = np.linalg.cholesky(sg2_inv).T
		if sqrt_sg2_inv.shape == (N,N):
			my_f = np.dot(RH.conj().T,(x.reshape([-1,1]) - x_est_P - np.dot(M3,gamma3_est))) / sigma2_n_est
			alpha_est_T = np.linalg.solve(np.linalg.solve(sqrt_sg2_inv,sqrt_sg2_inv.conj().T),my_f) + np.linalg.solve(sqrt_sg2_inv,np.random.randn(N,1))
			#begin {time-shift compensation} to make sure b_T located at its peak
			thrld = 1/2
			if np.random.rand() < 2 * thrld:
				H_alpha = np.abs(np.dot(H,alpha_est_T))
				val_alpha = np.max(H_alpha)
				pos_alpha = np.where(H_alpha == val_alpha)
				uset = pos_alpha[0][0]- q//2
				b_est2 = np.roll(b_est_T,uset)
				b_est2[np.mod(np.arange(0,uset,dtype=np.int),K)] = 0
				a_est2 = np.roll(a_est_T, uset)
				r_est2 = b_est2.conj().T * (a_est2.T)
				for rh_ind in range(N):
					H_row = H[(q-2*qc)// 2 + np.arange(0, 2 * qc)]
					H_col = H_row[:, rh_ind]
					rh2 = np.convolve(np.squeeze(r_est2),H_col)
					RH2[0:q, rh_ind] = rh2[K-1-(q//2-1)-(q-2*qc)//2 + np.arange(0,q)]
				RHHR2 = np.real(np.dot(RH2.conj().T, RH2))
				sg2_inv2 = (1 / sigma2_n_est * RHHR2 + np.diag(1 / sigma2_alpha * np.squeeze(np.ones((1, N)))))
				sqrt_sg2_inv2 = np.linalg.cholesky(sg2_inv2).T
				if 1:
					b_est_T = b_est2
					a_est_T = a_est2
					RH = RH2
					sqrt_sg2_inv = sqrt_sg2_inv2
			my_f = np.dot(RH.conj().T, (x.reshape([-1, 1]) - x_est_P - np.dot(M3, gamma3_est))) / sigma2_n_est
			alpha_est_T = np.linalg.solve(np.linalg.solve(sqrt_sg2_inv, sqrt_sg2_inv.conj().T), my_f) + np.linalg.solve(
				sqrt_sg2_inv, np.random.randn(N, 1))
			f_est_T = np.dot(H,alpha_est_T)
			v = np.max(np.abs(f_est_T))
			p = np.where(np.abs(f_est_T) == v)
			a_est_T = a_est_T * f_est_T[p[0][0]]
			f_est_T = f_est_T / f_est_T[p[0][0]]
		else:
			ec = ec+1
		if est_f == 0:
			#f_est_T = f_T
			pass
		#end {sample alpha_T}
		conv_f_est_T = np.convolve(np.squeeze(f_est_T[q//2 + np.arange(-qc,qc)]),np.squeeze(r_est_T))
		x_est_T = np.concatenate((np.zeros((q//2-qc,1)),conv_f_est_T.reshape([-1,1]),np.zeros((q//2-qc,1))))
		x_est_T = x_est_T[q//2+np.arange(0,K)] * non_QRS

		# begin {sample alpha_P}
		for rh_ind in range(N):
			H_row = H[(q-2*qc)//2 + np.arange(0,2*qc)]
			H_col = H_row[:, rh_ind]
			rh = np.convolve(np.squeeze(r_est_P), H_col)
			RH[0:q, rh_ind] = rh[K-1-(q//2-1)-(q-2*qc)//2 + np.arange(0,q)]
		RH = np.dot(np.diag(np.squeeze(non_QRS)),RH)
		RHHR = np.real(np.dot(RH.conj().T,RH))
		sg2_inv = (1 / sigma2_n_est * RHHR + np.diag(1 / sigma2_alpha * np.squeeze(np.ones((1, N)))))
		sqrt_sg2_inv = np.linalg.cholesky(sg2_inv).T
		if sqrt_sg2_inv.shape == (N,N):
			my_f = np.dot(RH.conj().T,(x.reshape([-1,1]) - x_est_T - np.dot(M3,gamma3_est))) / sigma2_n_est
			alpha_est_P = np.linalg.solve(np.linalg.solve(sqrt_sg2_inv,sqrt_sg2_inv.conj().T),my_f) + np.linalg.solve(sqrt_sg2_inv,np.random.randn(N,1))
			#begin {time-shift compensation} to make sure b_T located at its peak
			thrld = 1/2
			if np.random.rand() < 2 * thrld:
				H_alpha = np.abs(np.dot(H,alpha_est_P))
				val_alpha = np.max(H_alpha)
				pos_alpha = np.where(H_alpha == val_alpha)
				uset = pos_alpha[0][0]- q//2
				b_est2 = np.roll(b_est_P,uset)
				b_est2[np.mod(np.arange(0,uset,dtype=np.int),K)] = 0
				a_est2 = np.roll(a_est_P, uset)
				r_est2 = b_est2.conj().T * (a_est2.T)
				for rh_ind in range(N):
					H_row = H[(q-2*qc)// 2 + np.arange(0, 2 * qc)]
					H_col = H_row[:, rh_ind]
					rh2 = np.convolve(np.squeeze(r_est2), H_col)
					RH2[0:q, rh_ind] = rh2[K-1-(q//2-1)-(q-2*qc)//2 + np.arange(0,q)]
				RHHR2 = np.real(np.dot(RH2.conj().T, RH2))
				sg2_inv2 = 1 / sigma2_n_est * RHHR2 + np.diag(1 / sigma2_alpha * np.squeeze(np.ones((1, N))))
				sqrt_sg2_inv2 = np.linalg.cholesky(sg2_inv2).T
				if 1:
					b_est_P = b_est2
					a_est_P = a_est2
					RH = RH2
					sqrt_sg2_inv = sqrt_sg2_inv2
			#end {time-shift compensation}
			my_f = np.dot(RH.conj().T, (x.reshape([-1, 1]) - x_est_T - np.dot(M3, gamma3_est))) / sigma2_n_est
			alpha_est_P = np.linalg.solve(np.linalg.solve(sqrt_sg2_inv, sqrt_sg2_inv.conj().T), my_f) + np.linalg.solve(
				sqrt_sg2_inv, np.random.randn(N, 1))
			f_est_P = np.abs(np.dot(H,alpha_est_P))
			v = np.max(f_est_P)
			p = np.where(f_est_P == v)
			a_est_P = a_est_P * f_est_P[p[0][0]]
			f_est_P = f_est_P / f_est_P[p[0][0]]
		else:
			ec = ec+1
		if est_f == 0:
			#f_est_P = f_P
			pass
		# end {sample alpha_P}
		conv_f_est_P = np.convolve(np.squeeze(f_est_P[q//2 + np.arange(-qc,qc)]),np.squeeze(r_est_P))
		x_est_P = np.concatenate((np.zeros((q//2-qc,1)),conv_f_est_P.reshape([-1,1]),np.zeros((q//2-qc,1))))
		x_est_P = x_est_P[q//2+np.arange(0,K)] * non_QRS

		eng1 = np.cumsum(np.abs(np.power(f_est_T[::-1],2))).reshape([-1,1])
		eng_T = np.concatenate((eng1[(q+2)//2 - 1:q],eng1[q-1] * np.ones((K-(q-2)//2,1))))

		#begin {sample b_T,a_T}
		for k_ind in range(len(onset_list_T)):
			k = int(onset_list_T[k_ind])
			k_max = int(end_list_T[k_ind])
			b_est_T[k-1:k_max] = 0
			b1_est = b_est_T
			sg = 1 / np.sqrt(eng_T[np.arange(k-1,k_max)-k+1] / sigma2_n_est + 1/sigma2_a)
			r1_est = (a_est_T.reshape([-1,1])) * b1_est
			conv_f_est_T = np.convolve(np.squeeze(f_est_T[q//2 + np.arange(-qc, qc)]), np.squeeze(r1_est))
			x_est = np.concatenate((np.zeros((q//2-qc,1)),conv_f_est_T.reshape([-1,1]),np.zeros((q//2-qc,1))))
			eps = x.reshape([-1,1]) - x_est_P - np.dot(M3,gamma3_est)-x_est[q//2 + np.arange(0,K)] * non_QRS
			conv_f_est_T = np.convolve(np.squeeze(np.conj(f_est_T[q//2+1-1+np.arange(qc-1,-qc-1,-1)])),np.squeeze(eps))
			my =np.concatenate((np.zeros((q//2-qc,1)),conv_f_est_T.reshape([-1,1]),np.zeros((q//2-qc,1)))) / sigma2_n_est
			my = np.power(sg,2) * my[q//2 + np.arange(k-1,k_max)]
			temp = np.power(np.abs(my),2) / np.power(sg,2) / 2
			mmt = np.max(np.max(temp))
			p_est = np.power(sg,2) * np.power(math.e,(temp-mmt)) * pi_1
			p_est = np.append(np.power(math.e,p0factor - mmt),p_est) / (np.sum(p_est) + np.power(math.e,p0factor-mmt))
			u = np.random.rand(1)
			k_rel = np.sum(u > np.cumsum(p_est)) - 1
			if k_rel >= 0:
				b_est_T[k+k_rel-1] = 1
				if np.mean(a_est_T) >= 0:
					my[k_rel] = np.abs(my[k_rel])
				a_est_T[k+k_rel-1] = my[k_rel] + sg[k_rel] * np.random.randn(1)
		#end {sample b_T,a_T}
		r_est_T = b_est_T * (a_est_T.reshape([-1,1]))
		conv_f_est_T = np.convolve(np.squeeze(f_est_T[q // 2 + np.arange(-qc-1, qc-1)]), np.squeeze(r_est_T))
		x_est_T = np.concatenate((np.zeros((q // 2 - qc, 1)),conv_f_est_T.reshape([-1,1]),np.zeros((q // 2 - qc, 1))))
		x_est_T= x_est_T[q // 2 + np.arange(0, K)] * non_QRS

		eng1 = np.cumsum(np.abs(np.power(f_est_P,2)))
		eng_P = np.concatenate((eng1[q-1] * np.ones((K-(q-2)//2,1)),eng1[np.arange(q,(q+2)//2-1,-1)-1].reshape([-1,1])))

		#begin {sample b_P,a_P}
		for k_ind in range(len(onset_list_P)):
			k = int(onset_list_P[k_ind] + 10)
			k_max = int(end_list_P[k_ind])
			b_est_P[k-1:k_max] = 0
			b1_est = b_est_P
			sg = 1 / np.sqrt(eng_P[np.arange(k-1,k_max)-k_max+K]/sigma2_n_est + 1/sigma2_a)
			r1_est = a_est_P.reshape([-1,1]) * b1_est
			conv_f_est_P = np.convolve(np.squeeze(f_est_P[q//2 + np.arange(-qc,qc)]), np.squeeze(r1_est))
			x_est = np.concatenate((np.zeros((q // 2 - qc, 1)), conv_f_est_P.reshape([-1,1]),np.zeros((q // 2 - qc, 1))))
			eps = x.reshape([-1, 1]) - x_est_T - np.dot(M3, gamma3_est) - x_est[q // 2 + np.arange(0,K)] * non_QRS
			conv_f_est_P= np.convolve(np.squeeze(np.conj(f_est_P[q//2+1-1+np.arange(qc-1,-qc-1,-1)])),np.squeeze(eps))
			my = np.concatenate((np.zeros((q//2-qc,1)),conv_f_est_P.reshape([-1,1]),np.zeros((q//2-qc,1)))) / sigma2_n_est
			my = np.power(sg, 2) * my[q//2 + np.arange(k-1,k_max)]
			temp = np.power(np.abs(my), 2) / np.power(sg, 2) / 2
			mmt = np.max(np.max(temp))
			p_est = np.power(sg, 2) * np.power(math.e, (temp - mmt)) * pi_1
			p_est = np.append(np.power(math.e,p0factor - mmt),p_est) / (np.sum(p_est) + np.power(math.e,p0factor-mmt))
			u = np.random.rand(1)
			k_rel = np.sum(u > np.cumsum(p_est)) - 1
			if k_rel >= 0:
				b_est_P[k + k_rel - 1] = 1
				if np.mean(a_est_P) >= 0:
					my[k_rel] = np.abs(my[k_rel])
				a_est_P[k + k_rel - 1] = my[k_rel] + sg[k_rel] *  np.random.randn(1)
		#end {sample b_P,a_P}
		r_est_P = b_est_P * (a_est_P.reshape([-1,1]))
		conv_f_est_P = np.convolve(np.squeeze(f_est_P[q // 2 + np.arange(-qc-1, qc-1)]), np.squeeze(r_est_P))
		x_est_P = np.concatenate((np.zeros((q // 2 - qc, 1)),conv_f_est_P.reshape([-1,1]),np.zeros((q // 2 - qc, 1))))
		x_est_P= x_est_P[q // 2 + np.arange(0, K)] * non_QRS

		#begin {sample gamma}
		M3M3 = np.real(np.dot(M3.conj().T,M3))
		sg2_g3_inv = 1 / sigma2_n_est * M3M3 + np.diag(1 / sigma2_gamma3 * np.squeeze(np.ones((1, G*6))))
		sqrt_sg2_g3_inv = np.linalg.cholesky(sg2_g3_inv).T
		if sqrt_sg2_g3_inv.shape == (G*6,G*6):
			my_g3 = np.dot(M3.conj().T,x.reshape([-1,1])-x_est_T-x_est_P) / sigma2_n_est
			gamma3_est =  np.linalg.solve(sg2_g3_inv,my_g3) + np.linalg.solve(sqrt_sg2_g3_inv,np.random.randn(G*6,1))
		else:
			ec = ec + 1
		if est_g3 == 0:
			gamma3_est = gamma3
		#end {sample gamma3}

		#begin {sample sigma2_n}
		sigma2_n_est = np.random.gamma(xi+K/2, 1/(eta + np.sum(np.power(np.abs(x.reshape([-1,1]) - np.dot(M3,gamma3_est) -x_est_T - x_est_P ),2),axis=0) / 2), [1,1])
		sigma2_n_est = 1 / sigma2_n_est

		coll_b_T[i,:] = b_est_T.conj().T
		coll_b_P[i,:] = b_est_P.conj().T
		coll_a_T[i,:] = r_est_T.T
		coll_a_P[i,:] = r_est_P.T
		coll_s[i,:] = sigma2_n_est
		coll_f_T[i,:] = f_est_T.T
		coll_f_P[i,:] = f_est_P.T
		coll_gamma3[i,:] = gamma3_est.T

	burnin = int(np.round(Iterat * 0.6))
	iii = 0
	gamma3_hat = np.zeros((G * 6, Iterat))

	i = Iterat
	f_fshift_T[:, iii] = np.mean(coll_f_T[burnin-1:i,:], axis=0).T
	f_hat_T[:, iii] = f_fshift_T[:, iii]
	b_avg_T[0: K, iii] = np.mean(coll_b_T[burnin-1:i,:],axis=0).conj().T

	b_hat_T[0:K,iii] = 0
	onset_list_T = onset_list_T.astype(np.int)
	end_list_T = end_list_T.astype(np.int)

	for ib in range(len(onset_list_T)):
		temp_b_avg_T = b_avg_T[onset_list_T[ib]-1:end_list_T[ib],iii]
		v = np.max(temp_b_avg_T)
		p = np.where(temp_b_avg_T == v)
		if v > 1 - np.sum(temp_b_avg_T):
			b_hat_T[p[0][0]+onset_list_T[ib],iii] = 1
		else:
			print('the probability of haveing a T-wave in this interval is smaller than the threshold.')
			np.sum(temp_b_avg_T,axis=0)
	a_sum = np.sum(coll_a_T[burnin-1:i,:] != 0, axis=0)
	a_fshift_T[0:K,iii] = (np.sum(coll_a_T[burnin-1:i,:],axis=0)/(a_sum + (a_sum==0))).T
	a_hat_T[0: K, iii] = a_fshift_T[0: K, iii]
	r_hat_T[0: K, iii] = b_hat_T[:, iii] * a_hat_T[:, iii]
	s_hat[0: K, iii] = np.mean(coll_s[burnin-1: i], axis=0).conj().T
	gamma3_hat[0: G * 6, iii] = np.mean(coll_gamma3[burnin-1: i,:], axis=0).conj().T

	iii = 0
	i = Iterat
	f_fshift_P[:, iii] = np.mean(coll_f_P[burnin-1:i,:], axis=0).T
	f_hat_P[:, iii] = f_fshift_P[:, iii]
	b_avg_P[0: K, iii] = np.mean(coll_b_P[burnin-1: i,:],axis=0).conj().T
	b_hat_P[0:K,iii] = 0

	onset_list_P = onset_list_P.astype(np.int)
	end_list_P = end_list_P.astype(np.int)
	for ib in range(len(onset_list_P)):
		temp_b_avg_P = b_avg_P[onset_list_P[ib]-1:end_list_P[ib],iii]
		v = np.max(temp_b_avg_P)
		p = np.where(temp_b_avg_P == v)
		if v > 1 - np.sum(temp_b_avg_P):
			b_hat_P[p[0][0]+onset_list_P[ib],iii] = 1
		else:
			print('the probability of haveing a P-wave in this interval is smaller than the threshold.')
			np.sum(temp_b_avg_P)
	a_sum = np.sum(coll_a_P[burnin-1:i, :] != 0, axis=0)
	a_fshift_P[0:K, iii] = (np.sum(coll_a_P[burnin-1:i, :], axis=0) / (a_sum + (a_sum == 0))).T
	a_hat_P[0: K, iii] = a_fshift_P[0: K, iii]
	r_hat_P[0: K, iii] = b_hat_P[:, iii] * a_hat_P[:, iii]

	conv_f_est_T = np.convolve(np.squeeze(f_est_T[q // 2 + np.arange(-qc - 1, qc-1)]), np.squeeze(r_hat_T))
	x_hat_T = np.concatenate((np.zeros((q // 2 - qc, 1)),conv_f_est_T.reshape([-1,1]),np.zeros((q // 2 - qc, 1))))
	x_hat_T = x_hat_T[q//2 + np.arange(0,K)] * non_QRS
	conv_f_est_P = np.convolve(np.squeeze(f_est_P[q // 2 + np.arange(-qc - 1, qc-1)]), np.squeeze(r_hat_P))
	x_hat_P = np.concatenate((np.zeros((q // 2 - qc, 1)),conv_f_est_P.reshape([-1,1]),np.zeros((q // 2 - qc, 1))))
	x_hat_P = x_hat_P[q//2 + np.arange(0,K)] * non_QRS
	baseline_total = np.dot(M3 , gamma3_hat)

	return b_hat_T,r_hat_T,f_hat_T,b_hat_P,r_hat_P,f_hat_P,baseline_total,x_hat_T,x_hat_P




