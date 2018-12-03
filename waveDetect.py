import math
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import lfilter,firls,firwin,firwin2
import pywt
import ctypes as ct
import WDen
from Gibbs_analyser import Gibbs_analyser
import pandas as pd
import sys

def BPFilter(x,lf,uf):
	N = x.shape[1]
	S = np.fft.fft(x,N,2)
	k = np.arange(1,math.ceil(lf*N))
	if k is not None:
		S[:,k] = 0
		S[:,N-k+2] = 0
	k = np.arange(math.floor(uf*N),math.ceil(N/2)+1)
	S[:, k] = 0
	S[:, N - k + 2] = 0
	y = np.real(np.fft.ifft(S,N,2))
	return y

def BaseLine(x,L,approach):
	N = x.shape[1]
	b = np.zeros((x.shape))
	flen = int(np.floor(L/2))

	if approach == 'mn':
		for i in range(N):
			index = np.arange(max(i-flen,0),min(i+flen,N-1)+1)
			b[:,i] = np.mean(x[:,index],1)
	elif approach == 'md':
		for i in range(N):
			index = np.arange(max(i-flen,0),min(i+flen,N-1)+1)
			a = x[:,index]
			c = a.shape
			b[:, i] = np.median(x[:, index],1)
	return b

def BaseLineTOS(ecgin,iso_t,fs,cfs):
	pass

def thresholding(qmean,nmean,TH):
	dmed = qmean - nmean
	temp = TH * dmed
	dmed = temp
	thresh = nmean + dmed

	return  thresh

def R_wave_loc(*args):
	# R wave location after QRS detection
	# Once a QRS has been detected using the Output signal from the pre-processing
	# we have to carefully locate it. Indeed the QRS detection is based on the maxima location of a transformed signal, according to the
	nargin = len(args)
	if nargin == 0 or nargin == 1:
		print('pas d''entree correcte pour la location de l''onde R')
		return
	elif nargin == 2:
		signal = args[0]
		ind_recal = args[1]
		#default sampling frequency
		fs = 500
		#Default : the search window is firstly 200ms around the proposed QRS location
		WIN_SEARCH = np.fix(0.2 * fs + 0.5)
	elif nargin == 3:
		signal = args[0]
		ind_recal = args[1]
		fs = args[2]
		#Default : the search window is firstly 200ms around the proposed QRS location
		WIN_SEARCH = np.fix(0.2 * fs + 0.5)
	elif nargin == 4:
		signal = args[0]
		ind_recal = args[1]
		fs = args[2]
		WIN_SEARCH = args[3]

	interv_search = 2 * np.fix(WIN_SEARCH / 2)
	s1, s2, s3, s4 = Beat_analysis_preproc(signal, [])
	signal_f = s3

	b1 = np.ones((1,6))
	b1 = np.squeeze(b1)
	a1 = 1
	signal_f = lfilter(b1,a1,signal_f)
	delay = np.fix((len(b1) - 1) / 2)

	signal_pb = signal_f
	delay_pb = delay

	b_der = np.array([1,0,-1])
	b_der = b_der / 4
	a_der = 1
	derivative = lfilter(b_der,a_der,signal_pb)
	deriv_delay = np.fix((len(b_der)-1)/2) + delay_pb
	derivation_analysed = derivative[int(len(signal)/2+deriv_delay-interv_search/2) - 1:int(len(signal)/2+deriv_delay+interv_search/2)]

	val_max = np.max(derivation_analysed)
	loc_max = int(np.where(derivation_analysed == val_max)[0])
	val_min = np.min(derivation_analysed)
	loc_min = int(np.where(derivation_analysed == val_min)[0])

	WIN_MAX = np.fix(0.15 * fs + 0.5)

	if (abs(val_min / 2) > abs(val_max) or abs(val_max / 2) > abs(val_min)):
		interv_search = 2 * np.fix(1.5 * WIN_SEARCH / 2)
		derivation_analysed = derivative[len(signal) / 2 + deriv_delay - interv_search / 2:len(
			signal) / 2 + deriv_delay + interv_search / 2]
		val_max, loc_max = np.max(derivation_analysed)
		val_min, loc_min = np.min(derivation_analysed)
		if abs(loc_max - loc_min) > WIN_MAX:
			print('max correspondant trop ecartes... peut pas un QRS')
	if loc_max >= loc_min:
		temp_derivation_analysed = abs(derivation_analysed[loc_max:loc_min+1])
		ind_loc = int(np.where(temp_derivation_analysed == np.min(temp_derivation_analysed))[0])
		ind_loc = ind_loc + loc_min + ind_recal + len(signal) / 2 - interv_search / 2

		QRS_neg = np.zeros((0, 0))
		QRS_pos = ind_loc
		QRS_loc = ind_loc
	else:
		temp_derivation_analysed = abs(derivation_analysed[loc_max:loc_min+1])
		ind_loc = int(np.where(temp_derivation_analysed == np.min(temp_derivation_analysed))[0])
		ind_loc = ind_loc + loc_max + len(signal) / 2 + ind_recal - interv_search / 2
		QRS_neg = ind_loc
		QRS_pos = np.zeros((0, 0))
		QRS_loc = ind_loc
	return QRS_loc,QRS_pos,QRS_neg

def nextpow2(x):
	class FloatBits(ct.Structure):
		_fields_ = [
			('M', ct.c_uint, 23),
			('E', ct.c_uint, 8),
			('S', ct.c_uint, 1)
		]
	class Float(ct.Union):
		_anonymous_ = ('bits',)
		_fields_ = [
			('value', ct.c_float),
			('bits', FloatBits)
		]
	if x < 0:
		x = -x
	elif x == 0:
		return 0
	d = Float()
	d.value = x
	if d.M == 0:
		return d.E - 127
	return d.E - 127 + 1

def Beat_analysis_preproc(*args):
	nargin = len(args)
	if nargin == 0:
		print('pas d''entree correcte pour la location de l''onde R')
		return
	elif nargin == 1:
		signal = args[0]
		level_max_BLD = 8
		wavelet_BLD = 'sym4'
		build_level_BLD = 8
		level_max_DEN = 3
		wavelet_DEN = 'sym4'
	elif nargin == 2:
		signal = args[0]
		level_max_BLD = args[1]
		wavelet_BLD = 'sym4'
		build_level_BLD = args[1]
		level_max_DEN = 3
		wavelet_DEN = 'sym4'
	elif nargin == 3:
		signal = args[0]
		level_max_BLD = args[1]
		wavelet_BLD = args[2]
		build_level_BLD = args[1]
		level_max_DEN = 3
		wavelet_DEN = 'sym4'
	elif nargin == 4:
		signal = args[0]
		level_max_BLD = args[1]
		wavelet_BLD = args[2]
		build_level_BLD = args[3]
		level_max_DEN = 3
		wavelet_DEN = 'sym4'
	elif nargin == 5:
		signal = args[0]
		level_max_BLD = args[1]
		wavelet_BLD = args[2]
		build_level_BLD = args[3]
		level_max_DEN = args[4]
		wavelet_DEN = 'sym4'
	else:
		signal = args[0]
		level_max_BLD = args[1]
		wavelet_BLD = args[2]
		build_level_BLD = args[3]
		level_max_DEN = args[4]
		wavelet_DEN = args[5]
	varargout = list()
	if level_max_BLD:
		coeffs = pywt.wavedec(signal, wavelet_BLD,'sym',level_max_BLD )
		baseline = pywt.upcoef('a',coeffs[0],'sym4',8,take=len(signal)+6)[:int(len(signal))]
		signal_basrem = signal - baseline

		varargout.append(signal_basrem)
		varargout.append(baseline)
	else:
		signal_basrem = signal
		varargout.append(signal_basrem)
		varargout.append(np.zeros((len(signal))))

	if level_max_DEN:
		deb = signal[0]
		sig_basrem_d1 = WDen.wden(signal_basrem - deb, 'sqtwolog', 'soft', 'sln', level_max_DEN, wavelet_DEN) + deb
		sig_basrem_d2 = WDen.wden(signal_basrem - deb, 'minimaxi', 'soft', 'sln', level_max_DEN, wavelet_DEN) + deb
		varargout.append(sig_basrem_d1)
		varargout.append(sig_basrem_d2)
	else:
		varargout.append(signal_basrem)
		varargout.append(signal_basrem)

	return varargout[0],varargout[1],varargout[2],varargout[3]

def QRSestim(QRS_loc,necg,Fs):
	QRS_width_buff = np.zeros((0, 0))
	QRS_buff_on = np.zeros((0, 0))
	QRS_buff_off = np.zeros((0, 0))

	MS_PER_SAMPLE = np.fix(1000 / Fs + 0.5)
	MS200 = np.fix(200 / MS_PER_SAMPLE + 0.5)
	MS400 = np.fix(400 / MS_PER_SAMPLE + 0.5)

	half_beat_length = np.power(2,nextpow2(np.around(1.2 * MS400)) + 1)

	for i in range(len(QRS_loc)):
		if QRS_loc[i] - half_beat_length + 1 > 0 and QRS_loc[i] + half_beat_length < len(necg):
			interv = np.arange(QRS_loc[i] - half_beat_length + 1,QRS_loc[i] + half_beat_length + 1).astype(np.int)
			beat_segment = necg[interv]

			s1, s2, s3, s4 = Beat_analysis_preproc(beat_segment)

			interv_recherche = np.arange(-1 * MS200 -1,MS200+1,1,dtype=np.int)
			signal_qrsw = s3[len(beat_segment) // 2 + interv_recherche - 1]
			QRS_width,QRS_on,QRS_off = QRS_width_func(signal_qrsw,'no')
		elif QRS_loc[-1] - MS200 + 1 > 0 and QRS_loc[-1] + MS200 < len(necg):
			interv_recherche =np.arange(-1 *MS200+1,MS200+1,1)
			signal_qrsw = necg[int(QRS_loc[-1]) + interv_recherche.astype(np.int)]
			QRS_width, QRS_on, QRS_off = QRS_width_func(signal_qrsw, 'ok')
		else:
			print('no way to segment the beat')
			QRS_width = 0
			QRS_on = 0
			QRS_off = 0
		QRS_width_buff = np.append(QRS_width_buff,QRS_width)
		QRS_buff_on = np.append(QRS_buff_on,QRS_on+QRS_loc[i] - MS200)
		QRS_buff_off = np.append(QRS_buff_off,QRS_off + QRS_loc[i] - MS200)
	return QRS_width_buff,QRS_buff_on,QRS_buff_off

def QRS_width_func(*args):
	nargin = len(args)
	if nargin == 0:
		print('pas d''entree correcte pour la location de l''onde R')
		return
	elif nargin == 1:
		signal = args[0]
		pre_proc = 'no'
		FC = 12
		Npoint = 8
		fs = 500
	elif nargin == 2:
		signal = args[0]
		pre_proc = args[1]
		FC = 12
		Npoint = 8
		fs = 500
	elif nargin == 3:
		signal = args[0]
		pre_proc = args[1]
		FC = args[2]
		Npoint = 8
		fs = 500
	elif nargin == 4:
		signal = args[0]
		pre_proc = args[1]
		FC = args[2]
		Npoint = args[3]
		fs = 500
	elif nargin == 5:
		signal = args[0]
		pre_proc = args[1]
		FC = args[2]
		Npoint = args[3]
		fs = args[4]
	fact = 10
	pre_proc = 'ok'
	if pre_proc == 'ok':
		s1, s2, s3, s4 = Beat_analysis_preproc(signal, [])
		signal_f = s3
		b1 = np.ones((1,10))
		b1 = np.squeeze(b1)
		a1 = 1
		signal_pb = lfilter(b1,a1,signal_f)
		delay_bp = np.fix((len(b1) - 1) / 2)
	else:
		signal_pb = signal
		delay_bp = 0

	b_der = np.array([1,0,-1])
	b_der = b_der / 6
	a_der = 1
	derivative = lfilter(b_der, a_der, signal_pb)
	deriv_delay = np.fix((len(b_der) - 1) / 2) + delay_bp
	derivation_analysed = derivative

	portion1 = derivation_analysed[int(len(signal) / 2 + deriv_delay):]
	portion2 = derivation_analysed[int(len(signal) / 2 + deriv_delay - 2)::-1]
	if np.prod(portion1[0:3] - derivation_analysed[int(len(signal)/2+deriv_delay - 1)]) > 0 or np.prod(portion2[0:3] - derivation_analysed[int(len(signal)/2+deriv_delay - 1)]) < 0:
		i = 1
		while i < len(portion1) - 4 and (portion1[i] < portion1[i-1] or portion1[i] < portion1[i+1] or portion1[i] < portion1[i+2] or portion1[i] < portion1[i+3]):
			i = i + 1
		loc_max = len(signal) / 2 + deriv_delay + i + 2
		val_max = derivation_analysed[loc_max]

		i = 1
		while i < len(portion2) - 4 and (portion2[i] > portion2[i-1] or portion2[i] > portion2[i+1] or portion2[i] > portion2[i+2] or portion2[i] > portion2[i+3]):
			i = i + 1
		loc_min = len(signal) / 2 + deriv_delay - i - 2
		val_min = derivation_analysed[loc_min]
	elif np.prod(portion1[0:3]-derivation_analysed[int(len(signal)/2+deriv_delay-1)]) < 0 or np.prod(portion2[0:3]-derivation_analysed[int(len(signal)/2+deriv_delay-1)]) > 0:
		i = 1
		while i < len(portion2) - 4 and (portion2[i] < portion2[i-1] or portion2[i] < portion2[i+1] or portion2[i] < portion2[i+2] or portion2[i] < portion2[i+3]):
			i = i + 1
		loc_max = int(len(signal) / 2 + deriv_delay - i - 2)
		val_max = derivation_analysed[loc_max]

		i = 1
		while i < len(portion2) - 4 and (portion2[i] > portion2[i-1] or portion2[i] > portion2[i+1] or portion2[i] > portion2[i+2] or portion2[i] > portion2[i+3]):
			i = i + 1
		loc_min = int(len(signal) / 2 + deriv_delay + i + 2)
		val_min = derivation_analysed[loc_min]
	else:
		print('attention, du bruit doit perturber l''utilisation de la derivee pour la mesure du QRS width')
		val_min = 0
		val_max = 0
		loc_min = 0
		loc_max = 0

	val_max2 = np.max(derivation_analysed)
	loc_max2 = int(np.where(derivation_analysed == val_max2)[0])
	val_min2 = np.min(derivation_analysed)
	loc_min2 = int(np.where(derivation_analysed == val_min2)[0])

	if val_max2 != val_max and (np.sign(loc_max-loc_min) * loc_max+ np.sign(loc_min-loc_max) * loc_max2) >= 0 and (np.sign(loc_min-loc_max) * loc_min+np.sign(loc_max-loc_min) * loc_max2) >= 0:
		val_max = val_max2
		loc_max = loc_max2
	if val_min2 != val_min and (np.sign(loc_min-loc_max) * loc_min+np.sign(loc_max-loc_min) * loc_min2) >= 0 and (np.sign(loc_max-loc_min) * loc_max + np.sign(loc_min-loc_max) * loc_max2) >= 0:
		val_min = val_min2
		loc_min = loc_min2

	slope_max = val_max
	slope_min = val_min

	if loc_max >= loc_min:
		i = loc_min
		while i > 2 and 'QRS_on' not in locals():
			i = i - 1
			if abs(np.prod(derivation_analysed[i-2:i+1])) < abs(np.power(slope_min/fact,3)):
				QRS_on = i

		i = loc_max
		while i < (len(derivation_analysed) - 1) and 'QRS_off' not in locals():
			i = i + 1
			if abs(np.prod(derivation_analysed[i:i+3])) < abs(np.power(slope_max/fact,3)):
				QRS_off = i
	else:
		i = loc_max
		while i > 2 and 'QRS_on' not in locals():
			i = i - 1
			if abs(np.prod(derivation_analysed[i - 2:i+1])) < abs(np.power(slope_max / fact, 3)):
				QRS_on = i

		i = loc_min
		while i < (len(derivation_analysed) - 1) and 'QRS_off' not in locals():
			i = i + 1
			if abs(np.prod(derivation_analysed[i:i + 3])) < abs(np.power(slope_min / fact, 3)):
				QRS_off = i

	if ('QRS_on' or 'QRS_off') not in locals() or QRS_off == 2 or QRS_on == 2 or QRS_off == len(derivation_analysed) - 2 or QRS_on == len(derivation_analysed) - 2:
		while ('QRS_on' or 'QRS_off') not in locals() and fact > 1e-2:
			fact = fact / 2
			if loc_max >= loc_min:
				i = loc_min
				while i > 2 and 'QRS_on' not in locals():
					i = i - 1
					if abs(np.prod(derivation_analysed[i-2:i+1])) < abs(np.power(slope_min/fact,3)):
						QRS_on = i
				i = loc_max
				while i < len(derivation_analysed) - 2 and 'QRS_off' not in locals():
					i = i + 1
					if abs(np.prod(derivation_analysed[i:i+3])) < abs(np.power(slope_max/fact,3)):
						QRS_off = i
			else:
				i = loc_max
				while i > 2 and 'QRS_on' not in locals():
					i = i - 1
					if abs(np.prod(derivation_analysed[i-2:i+1])) < abs(np.power(slope_max/fact,3)):
						QRS_on = i
				i = loc_min
				while i < len(derivation_analysed) - 2 and 'QRS_off' not in locals():
					i = i + 1
					if abs(np.prod(derivation_analysed[i:i+3])) < abs(np.power(slope_min/fact,3)):
						QRS_off = i
		if 'QRS_on' not in locals():
			QRS_on = 0
		if 'QRS_off' not in locals():
			QRS_off = len(derivation_analysed) - 1

	QRS_width = QRS_off - QRS_on
	QRS_on = QRS_on - deriv_delay + 1
	QRS_off = QRS_off - deriv_delay + 1

	return QRS_width,QRS_on,QRS_off

def Rwavexact(b1,b2,b3,b4,b5,loc_peak,recal,necg,Fs):
	FILTER_DELAY = np.fix((len(b1) - 1) / 2) + np.fix((len(b2) - 1) / 2) + np.fix((len(b3) - 1) / 2) + np.fix(
		(len(b4) - 1) / 2) + np.fix((len(b5) - 1) / 2)
	ind_recal = loc_peak + recal - FILTER_DELAY

	MS_PER_SAMPLE = np.fix(1000 / Fs + 0.5)
	MS200 = np.fix(200 / MS_PER_SAMPLE + 0.5)
	MS400 = np.fix(400 / MS_PER_SAMPLE + 0.5)
	QRS_loc = np.zeros((0, 0))

	for i in range(len(loc_peak)):
		if ind_recal[i] > MS400 and ind_recal[i] + MS400 < len(necg):
			WIN_SEARCH = MS200
			# this signal can be saved ain a circular buffer of length=2*MS400; then an indice should be given as an input
			S_Rw_loc = necg[int(ind_recal[i] - MS400 + 1):int(ind_recal[i] + MS400)+1]
			R_loc, R_pos, R_neg = R_wave_loc(S_Rw_loc, ind_recal[i], Fs, WIN_SEARCH)
			QRS_loc = np.append(QRS_loc,R_loc - MS400)
		elif ind_recal[i] > MS200 and ind_recal[i] + MS200 < len(necg):
			WIN_SEARCH = MS200
			# this signal can be saved ain a circular buffer of length=2*MS400; then an indice should be given as an input
			S_Rw_loc = necg[int(ind_recal[i] - WIN_SEARCH + 1):int(ind_recal[i] + WIN_SEARCH)+1]
			R_loc, R_pos, R_neg = R_wave_loc(S_Rw_loc, ind_recal(i), Fs, WIN_SEARCH)
			QRS_loc = np.append(QRS_loc,R_loc - WIN_SEARCH)
		else:
			print('precise R wave location is not possible!')
			QRS_loc = np.append(QRS_loc,ind_recal[i])
	return QRS_loc

def marque(b1,b2,b3,op,Fs):
	#法语：On laissera tomber les Ndeb premiers points pour la recherche des ECG
	Ndeb = len(b1) + len(b2) + len(b3)
	signal_recherche = op[Ndeb:]
	index_sr = len(signal_recherche) - 1
	MS200 = np.around(0.2 * Fs)
	MS300 = np.around(0.3 * Fs)
	fact = 0.3
	loc_peak = np.zeros((0,0))
	peak = signal_recherche[0]
	timesincemax = 0
	TH_init = np.max(signal_recherche[0:300])
	buff_QRS = [TH_init,TH_init,TH_init,TH_init]
	buff_bruit = [0,0,0,0]
	qmean = np.mean(buff_QRS)
	nmean = np.mean(buff_bruit)
	TH = thresholding(qmean,nmean,fact)
	i = 0
	loc_peak_temp = i

	while i < index_sr:
		i = i + 1
		timesincemax = timesincemax + 1

		if signal_recherche[i] > peak and timesincemax < MS300:
			peak = signal_recherche[i]
			loc_peak_temp = i
			timesincemax = 0
		elif signal_recherche[i] < peak and timesincemax >= MS300:
			#法语：test pour savoir si le peak est du bruit ou bien un QRS
			if peak > TH:
				buff_QRS = np.append(buff_QRS[1:],peak)
				qmean = np.mean(buff_QRS)
				TH = thresholding(qmean,nmean,fact)

				loc_peak = np.append(loc_peak,loc_peak_temp)

				if loc_peak[-1] + MS200 < index_sr:
					i = int(loc_peak[-1] + MS200)
				peak = signal_recherche[i]
				timesincemax = 0
			else:
				buff_bruit = np.append(buff_bruit[1:],peak)
				nmean = np.mean(buff_bruit)
				TH = thresholding(qmean,nmean,fact)
				if loc_peak_temp + MS200 < index_sr:
					i = int(loc_peak_temp + MS200)
				peak = signal_recherche[i]
				loc_peak_temp = i
				timesincemax = 0

	return signal_recherche,loc_peak,Ndeb

def peak_features(preproc_sig,b1,b2,b3,b4,b5,ecg,Fs):
	bpb = np.concatenate((b1,b2,b3))
	signal_recherche,loc_peak,recal = marque(bpb,b4,b5,preproc_sig,Fs)
	QRS_loc = Rwavexact(b1,b2,b3,b4,b5,loc_peak,recal,ecg,Fs)
	QRS_width_buff, QRS_buff_on, QRS_buff_off = QRSestim(QRS_loc, ecg, Fs)
	return QRS_loc,QRS_width_buff,QRS_buff_on,QRS_buff_off

def QRSdetection(necg,Fs):
	#Loading the ECG file
	Ns = len(necg)

	#lowpass filtering
	b1 = np.array([1,0,0,0,0,0,-2,0,0,0,0,0,1]) / 32
	a1 = [1,-2,1]
	lpecg = lfilter(b1,a1,necg)
	lp_ecg = lpecg

	#Low pass
	b2 = np.zeros((1,33))
	b2 = np.squeeze(b2)
	b2[0] = 1
	b2[32] = -1
	a2 = [1,-1]
	h2 = lfilter(b2,a2,lpecg)

	#All pass
	b3 = np.zeros((1,17))
	b3 = np.squeeze(b3)
	b3[16] = 1
	a3 = 1
	h3 = lfilter(b3,a3,lpecg)

	#highpass = allpass - lowpass
	p2 = h3 - h2 / 32

	#signal derivation
	b4 = [1,2,0,-2,-1]
	a4 = 1
	h4 = lfilter(b4,a4,p2)

	#Squaring
	h4 = np.power(h4,2)

	#Integration
	b5 = np.ones((1,30))
	b5 = np.squeeze(b5) / 30
	a5 = 1
	op = lfilter(b5,a5,h4)
	op = np.squeeze(op)

	R_loc,QRS_width,Q_loc,S_loc = peak_features(op,b1,b2,b3,b4,b5,necg,Fs)

	return R_loc,QRS_width,Q_loc,S_loc,lp_ecg

if __name__ == '__main__':
	# data = scio.loadmat(sys.argv[1])
	# data = scio.loadmat('sel48.mat')
	# data = data['ECG_1']
	data = pd.read_csv("./batch4allAbnormal_for_testing.csv")
	data = np.array(data.iloc[0,1:7501])
	# data = np.squeeze(np.reshape(data,[-1,len(data)]))
	# data = np.reshape(data,[-1,len(data)])
	data = np.reshape(data.conj().T,[1,-1])
	Fs = 500
	# _, ax = plt.subplots()
	# ax.plot(np.arange(1/Fs,(len(data) + 1)/Fs,1/Fs), data, 'b', label='ECG')
	# plt.show()
	D = 5
	b2 = BaseLine(data,Fs*.3,'md')
	target_signal = data - b2
	target_signal_show = data - b2
	target_signal = target_signal/np.max(np.abs(target_signal))
	target_signal = np.squeeze(target_signal)

	R_loc_total,QRS_width,Q_loc_total,S_loc_total,target_signal_filtered = QRSdetection(target_signal,Fs)

	R_loc_total = R_loc_total.astype(np.int)
	QRS_width = QRS_width.astype(np.int)
	Q_loc_total = Q_loc_total.astype(np.int)
	S_loc_total = S_loc_total.astype(np.int)

	Ndeb = 5
	target_signal_total = target_signal_filtered[Ndeb:]
	total_beat = len(R_loc_total)
	total_process = int(np.floor(total_beat/(D-1)))
	Q_loc_1_total = Q_loc_total[1:]
	S_loc_2_total = S_loc_total[0:-1]
	QS_interval = Q_loc_1_total - S_loc_2_total
	moy_QS_interval_N = int(np.round(np.mean(QS_interval)/3.5))
	if np.mod(moy_QS_interval_N,2):
		moy_QS_interval_N = moy_QS_interval_N+1
	print('QRS detection done!')

	#creat buffers to save overall delineation results
	#=============
	T_hat_total = np.squeeze(np.zeros((total_process,moy_QS_interval_N)))
	P_hat_total = np.squeeze(np.zeros((total_process,moy_QS_interval_N)))
	T_peak_total = np.array([])
	P_peak_total = np.array([])
	T_onset_total = np.array([])
	P_onset_total = np.array([])
	T_end_total = np.array([])
	P_end_total = np.array([])

	for processing_ind in range(total_process):
		offset = (D-1) * (processing_ind - 0)
		if offset+D <= len(Q_loc_total) - 1:
			target_signal = target_signal_total[Q_loc_total[offset]:Q_loc_total[offset+D-1] + 1]
		else:
			print('Warning: the ending beats are not processed.')
			break
		K = len(target_signal)
		if np.mod(K,2):
			K = K+1
		R_loc = R_loc_total[offset:offset+D] - Q_loc_total[offset]
		S_loc = S_loc_total[offset:offset+D]-Q_loc_total[offset]
		Q_loc = Q_loc_total[offset :offset+D]-Q_loc_total[offset] + 1
		Q_loc_1 = Q_loc[1:]
		S_loc_2 = S_loc[0:-1]
		QS_interval = Q_loc_1 - S_loc_2
		moy_QS_interval = np.ceil(QS_interval / 2)

		# generate pure P and T-wave signal
		ST = 15
		T_onset_list = np.zeros((0,0),dtype=np.int)
		T_end_list = np.zeros((0,0),dtype=np.int)
		for i in range(len(S_loc_2)):
			T_onset_list = np.append(T_onset_list,S_loc[i] + ST)
			T_end_list = np.append(T_end_list,S_loc[i]+moy_QS_interval[i]+ST-1)
		PQ = 10
		P_onset_list = np.zeros((0,0),dtype=np.int)
		P_end_list = np.zeros((0,0),dtype=np.int)
		P_onset_list = T_end_list + 1
		for i in range(len(Q_loc_1)):
			P_end_list = np.append(P_end_list,Q_loc_1[i]-PQ)
		pure_signal = np.squeeze(np.zeros((K,1)))
		for i in range(len(P_onset_list)):
			pure_signal[T_onset_list[i]-1:P_end_list[i]] = target_signal[T_onset_list[i]-1:P_end_list[i]]
		amp_factor = 5
		pure_signal = pure_signal * amp_factor
		print('P and T research region extraction done!')

		#T-wave and P-wave analysis using MCMC
		N_T_wave = moy_QS_interval_N
		N_P_wave = moy_QS_interval_N
		pi_1 = 0.005
		sigma2_a = 0.1
		sigma2_alpha = 0.001
		sigma2_gamma = 0.001
		sigma2_gamma2 = 0.001
		eta = 0.5
		xi = 3
		fshift = 0
		Iterat = 80
		print('Wave detection...')
		b_hat_T, r_hat_T, f_hat_T, b_hat_P, r_hat_P, f_hat_P, baseline_total, x_hat_T, x_hat_P = Gibbs_analyser(Iterat,pure_signal,N_T_wave,pi_1,sigma2_a,sigma2_alpha,sigma2_gamma,sigma2_gamma2,eta,xi,fshift,T_onset_list,T_end_list,P_onset_list,P_end_list)
		print('Wave detection done!')
		inst_T_hat = b_hat_T
		inst_P_hat = b_hat_P
		T_hat = f_hat_T
		P_hat = f_hat_P
		T_loc = np.array(np.nonzero(inst_T_hat)).conj().T[:,0]
		P_loc = np.array(np.nonzero(inst_P_hat)).conj().T[:,0]

		T_hat = T_hat[K // 2 - N_T_wave // 2 - 1:K // 2 + N_T_wave // 2]
		P_hat = P_hat[K // 2 - N_P_wave // 2 - 1:K // 2 + N_P_wave // 2]

		#T wave delineation
		T_hat_proc = T_hat[0:]
		T_peak = np.ceil(len(T_hat_proc)/2)
		deno = np.power(np.sqrt(1+np.power(np.diff(T_hat_proc,axis=0),2)),3)
		curv = np.abs(np.diff(T_hat_proc,2,axis=0)) / deno[0:-1]
		curv = np.squeeze(curv.conj().T)
		curv[0] = curv[1]
		coef_b = firwin(11,0.3)
		curv_s = np.convolve(coef_b,curv)
		curv_s = curv_s[5:]
		local_maxima_list = np.nonzero((curv_s >= np.append(curv_s[1:],math.inf)).astype(np.int) & (curv_s > np.append(math.inf,curv_s[0:- 1])).astype(np.int))[0]
		forbidden_zone = np.ceil(len(T_hat_proc) / 8)
		local_maxima_list_pos_left = np.nonzero((local_maxima_list <= T_peak - forbidden_zone).astype(np.int))[0]
		if len(local_maxima_list_pos_left) == 0:
			T_limit_left_dis = np.ceil(len(T_hat_proc) / 3)
		else:
			temp_curv_s = curv_s[local_maxima_list[local_maxima_list_pos_left]]
			v = np.max(temp_curv_s)
			p = np.where(temp_curv_s == v)
			T_limit_left_curv = local_maxima_list[local_maxima_list_pos_left[p[0][0]]]
			T_limit_left_dis = np.abs(T_peak - T_limit_left_curv)

		local_maxima_list_pos_right = np.nonzero((local_maxima_list > T_peak + forbidden_zone).astype(np.int))[0]
		if len(local_maxima_list_pos_right) == 0:
			T_limit_right_dis = np.ceil(len(T_hat_proc) / 3)
		else:
			temp_curv_s = curv_s[local_maxima_list[local_maxima_list_pos_right]]
			v = np.max(temp_curv_s)
			p = np.where(temp_curv_s == v)
			T_limit_right_curv = local_maxima_list[local_maxima_list_pos_right[p[0][0]]]
			T_limit_right_dis = np.abs(T_limit_right_curv - T_peak)

		# P wave delineation
		P_hat_proc = P_hat[0:]
		P_peak = np.ceil(len(P_hat_proc) / 2)
		deno = np.power(np.sqrt(1 + np.power(np.diff(P_hat_proc, axis=0), 2)), 3)
		curv = np.abs(np.diff(P_hat_proc, 2, axis=0)) / deno[0:-1]
		curv = np.squeeze(curv.conj().T)
		curv[0] = curv[1]
		coef_b = firwin(11, 0.3)
		curv_s = np.convolve(coef_b, curv)
		curv_s = curv_s[5:]
		local_maxima_list = np.nonzero((curv_s >= np.append(curv_s[1:], math.inf)).astype(np.int) & (curv_s > np.append(math.inf, curv_s[0:- 1])).astype(np.int))[0]
		forbidden_zone = np.ceil(len(P_hat_proc) / 8)
		local_maxima_list_pos_left = np.nonzero(local_maxima_list <= P_peak - forbidden_zone)[0]
		if len(local_maxima_list_pos_left) == 0:
			P_limit_left_dis = np.ceil(len(P_hat_proc) / 3)
		else:
			temp_curv_s = curv_s[local_maxima_list[np.array(local_maxima_list_pos_left)]]
			v = np.max(temp_curv_s)
			p = np.where(temp_curv_s == v)
			P_limit_left_curv = local_maxima_list[local_maxima_list_pos_left[p[0][0]]]
			P_limit_left_dis = np.abs(P_peak - P_limit_left_curv)

		local_maxima_list_pos_right = np.nonzero((local_maxima_list > P_peak + forbidden_zone).astype(np.int))[0]
		if len(local_maxima_list_pos_right) == 0:
			P_limit_right_dis = np.ceil(len(P_hat_proc) / 3)
		else:
			temp_curv_s = curv_s[local_maxima_list[local_maxima_list_pos_right]]
			v = np.max(temp_curv_s)
			p = np.where(temp_curv_s == v)
			P_limit_right_curv = local_maxima_list[local_maxima_list_pos_right[p[0][0]]]
			P_limit_right_dis = np.abs(P_limit_right_curv - T_peak)

		# plt.figure(12)
		# plt.subplot(221)
		# xc = np.arange(0,len(T_hat))
		# yc = np.squeeze(T_hat.conj().T)
		# plt.title('T waveform estimation')
		# plt.plot(xc,yc)
		# pos_left = T_peak - T_limit_left_dis
		# pos_right = T_peak + T_limit_right_dis
		# plt.axvline(pos_left,color='r')
		# plt.text(pos_left,0,'On set',color='r')
		#
		# plt.axvline(pos_right,color='r')
		# plt.text(pos_right,0,'Off set',color='r')
		#
		# plt.subplot(222)
		# xc = np.arange(0, len(P_hat))
		# yc = np.squeeze(P_hat.conj().T)
		# plt.title('P waveform estimation')
		# plt.plot(xc,yc)
		#
		# pos_left = P_peak - P_limit_left_dis
		# pos_right = P_peak + P_limit_right_dis
		# plt.axvline(pos_left,color='r')
		# plt.text(pos_left,0,'On set',color='r')
		#
		# plt.axvline(pos_right,color='r')
		# plt.text(pos_right,0,'Off set',color='r')
		# plt.show()

		T_hat_total[processing_ind, :] = T_hat[0: moy_QS_interval_N].conj().T / np.max(T_hat)
		P_hat_total[processing_ind, :] = P_hat[0: moy_QS_interval_N].conj().T / np.max(P_hat)
		T_peak_total = np.append(T_peak_total, Q_loc_total[offset] + T_loc + 2)
		P_peak_total = np.append(P_peak_total, Q_loc_total[offset] + P_loc + 2)
		T_onset_total = np.append(T_onset_total, Q_loc_total[offset] + T_loc - T_limit_left_dis + 2)
		P_onset_total = np.append(P_onset_total, Q_loc_total[offset] + P_loc - P_limit_left_dis + 2)
		T_end_total = np.append(T_end_total, Q_loc_total[offset] + T_loc + T_limit_right_dis + 2)
		P_end_total = np.append(P_end_total, Q_loc_total[offset] + P_loc + P_limit_right_dis + 2)

	plt.figure()
	plt.plot(np.arange(1 / Fs, (len(target_signal_total)+1)/Fs,1/Fs),target_signal_total,'b',linewidth=2)
	plt.scatter(R_loc_total / Fs, target_signal_total[R_loc_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	# plt.scatter(Q_loc_total / Fs, target_signal_total[Q_loc_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	# plt.scatter(S_loc_total / Fs, target_signal_total[S_loc_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	plt.scatter(T_peak_total / Fs, target_signal_total[T_peak_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	plt.scatter(T_onset_total / Fs, target_signal_total[T_onset_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	plt.scatter(T_end_total / Fs, target_signal_total[T_end_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	plt.scatter(P_peak_total / Fs, target_signal_total[P_peak_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	plt.scatter(P_onset_total / Fs, target_signal_total[P_onset_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	plt.scatter(P_end_total / Fs, target_signal_total[P_end_total.astype(np.int)],c='r',edgecolors='black',linewidths=1)
	plt.show()




