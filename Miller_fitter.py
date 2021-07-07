"""
created 06/01/2020 on rapc@centos06

The purpose of this script is to fit Miller parameters to a given flux surface.

Getting the right combination of coefficients(delta and kappa) for all thetas is
a Minimization problem. One could also add r as a free parameter but it's not 
recommended for a global equilibrium(with iflux neq 0).

Miller parametrization 
R = R_0 + r*cos(th + arcsin(delta)*sin(theta))
Z = kappa*r*sin(th)
The quantitiy we are going to Minimze is 

obj[i] = sqrt(((R0[i]-R1[i])**2 + (Z0[i]-Z1[i])**2))

where (R0,Z0) are points on the target flux surface and R1, Z1 are the points on the 
matching test surface.

Free parameters to be added in the near future: r, dR_0/dr

Multiprocessing coming soon...
"""
from os.path import dirname, abspath
import re
import pdb
import numpy as np
import netCDF4 as nc4
from intersection_lib import intersection as intsec
from lmfit import Minimizer, Parameters, report_fit
from matplotlib import pyplot as plt
import multiprocessing as mp

parnt_dir_nam = dirname(dirname(abspath(__file__)))

rtg = nc4.Dataset('out_170_330_305_Miller_150_100_0_poly_1500_1000_0_0_poly_120_0_0_93_3_50.nc', 'r')

grp1 = rtg.groups['annular_psi_data']
R = grp1.variables['R'][:].data
Z = grp1.variables['Z'][:].data
#rho = grp1.variables['rho'][:].data
#qfac = grp1.variables['q_int'][:].data

len1 = 300
theta =  np.linspace(0, np.pi, len1)
def calc_R1(R_0, r, kappa, delta):
	return np.array([R_0 + r*np.cos(th +np.arcsin(delta)*np.sin(th)) for th in theta])

def calc_Z1(r, kappa):
	return np.array([kappa*r*np.sin(th) for th in theta])

params = Parameters()
params.add('delta', value = 0.9, min=-0.99, max=0.99)
params.add('kappa', value = 1.4, min=0.2, max = 10)


def obj_fun(params, R0, Z0):
	#Objective function to determine the average normal distance between
	#the two curves.

	delta = params['delta']
	kappa = params['kappa']
	R1 = calc_R1(R_0, r, kappa, delta)
	Z1 = calc_Z1(r, kappa)
	diffR0 = np.diff(R0)
	diffZ0 = np.diff(Z0)
	ph = np.arctan2(diffR0, diffZ0)
	lim = 4*r # could be any number as long as its large
	len0 = len(R0)
	dist = np.zeros((len0-1, ))
	for i in range(len0-1):
		R1_int, Z1_int = intsec(np.array([R0[i] - lim*np.cos(ph[i]), R0[i] + lim*np.cos(ph[i])]),np.array([Z0[i] + lim*np.sin(ph[i]), Z0[i] - lim*np.sin(ph[i])]), R1, Z1)
		if len(R1_int) > 1: 	#intersection at two points

			# choose the one that is closer to (R0[i], Z0[i])
			dist_min = min((R1_int-R0[i])**2 +  (Z1_int-Z0[i])**2)
			idx = np.where(dist_min)

			# Choose the closest point from the two intersections
			R1_int = R1_int[idx]
			Z1_int = Z1_int[idx]
			dist[i] += dist_min
		elif len(R1_int) == 0:  # the fitting miller curve is too short(small kappa)
			dist[i] +=  10*(r**2 + z**2) # heavily penalize these profiles
		#print("%d \n"%i)
		#pdb.set_trace()
		else: # intersection at a single point
			dist[i] +=  np.sqrt((R1_int - R0[i])**2 + (Z1_int-Z0[i])**2)

	return dist

processes = []
qout = mp.Queue()
out = []
len0 = np.shape(R)[0]
R_fit = np.zeros((len0, len1))
Z_fit = np.zeros((len0, len1))
out = []
result = []
for i in range(len0):
	R0 = R[i, :][Z[i,:]>0]
	Z0 = Z[i, :][Z[i,:]>0]
	R_0 = (np.min(R0) + np.max(R0))/2
	r =  (np.max(R0) - np.min(R0))/2
	z = np.max(Z0)
	#p0 = mp.process(minner.minimize, )
	#pdb.set_trace()
	minner = Minimizer(obj_fun, params, fcn_args=(R0, Z0))
	result = minner.minimize()
	out.append([result])
	R_fit[i, :] = calc_R1(R_0, r, result.params['kappa'].value, result.params['delta'].value) 
	Z_fit[i, :] = calc_Z1(r, result.params['kappa'].value)
	plt.plot(R0, Z0, '-g', R_fit[i,:], Z_fit[i,:], 'or', ms=2.0); 
	print("done iter %d"%i)
	#plt.legend(['Original', 'fit'])

out = np.reshape(out, (len0, ))
np.save('fit_data_Miller_7_14_99_5_3', out, allow_pickle=True)
plt.show() 
#k_rho = (out[0].params['kappa'] - out[2].params['kappa'])/(rho[0]-rho[2])
#d_rho = (out[0].params['delta'] - out[2].params['delta'])/(rho[0]-rho[2])
#shift = ((R_fit[0][0] + R_fit[0][-1])/2 - (R_fit[2][0] + R_fit[2][-1])/2)/(rho[0]-rho[2])
#s_input = rho[1]/qfac[1]*(qfac[1]-qfac[0])/(rho[1]-rho[0])

pdb.set_trace()
