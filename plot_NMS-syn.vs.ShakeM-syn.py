#!/usr/bin/env python
import argparse
import h5py
import re
import sys,os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from UCBFilter import Filter
DTYPE = np.float32
np.set_printoptions(threshold=np.inf)
import obspy.signal.filter as fltobspy
from obspy.geodetics.base import gps2dist_azimuth
from obspy import read
from glob import glob
import yaml

def scrub_string(s):
    return re.sub(r'\W+', '', s)


def interp(tr):
    ff = InterpolatedUnivariateSpline(tr[:,0], tr[:,1])
    #s = np.array([ff.derivatives(t_i)[1] for t_i in tr[:,0]]) 
    ss = np.array([ff.derivatives(t_i)[1] for t_i in tr[:,0]]) 
    s  =fltobspy.bandpass(ss,freqmin=0.0025,freqmax=0.025,df=1.0,corners=4,zerophase=False) 

    return s

#Set the bandpass here!
passband_spec="400-250-53-40"
passband = [float(T) for T in passband_spec.split('-')]
filt     = Filter([2.0 * np.pi / float(T) for T in passband])

####################################
T1 =  53.0
T2 =  250.0
DT='53-250'

models      = ['NMS', 'ShakeM'] 
###########


S=dict() 

with open("STATIONS","r") as ff:
     for l in ff:
         stn, netw, slat,slon,dpeth,B=l.split() 
         S[stn]=[netw,slat,slon] 

with open("plt.conf","r") as f:
       conf  = yaml.load(f) 
       stns  = conf['receivers'].split() 
       Elat  = float(conf['Elat'])
       Elon  = float(conf['Elon'])
       evname= conf['evname']
       depth = conf['depth']
       #print(conf)

for rcv in stns:
        net   = S[rcv][0] 
        Slat  = float(S[rcv][1]) 
        Slon  = float(S[rcv][2]) 
        net = re.sub('[!@#$] ', '', net)
        #compute the distance between source and event
        D, az, baz = gps2dist_azimuth(Elat,Elon,Slat,Slon)
        D_km  =D/1000.
        D_deg ="%.2f"%(D_km/100.)
        #print(len(net))

        #Create figure with sharing axis
        fig, axs = plt.subplots(3, sharex=True, sharey=True) 
        for compnt in ['N','E','Z']:
                    path  = '%s/%s.%s.LX%s.modes.sac'%('ShakeM-1D',rcv.replace(' ',''),net.replace(' ',''),compnt.replace(' ',''))
                    st    = read(path,debug_headers=True) 
                    data  = st[0].data
                    times = st[0].times()
                    cpnt  = st[0].stats.sac.kcmpnm 
                    netwk = st[0].stats.sac.knetwk 
                    stname= st[0].stats.sac.kstnm
                    dt    = float(st[0].stats.sampling_rate)
                    DELTA = st[0].stats.delta
                    sk    = data
                    Amax  = max(data)
                    #print(Amax)
                    #sk    = fltobspy.bandpass(data,freqmin=1./T2,freqmax=1./T1,df=dt,corners=4,zerophase=False) 
                    sk   = filt.apply(data, 1.0/dt, taper_len=2 * passband[0])
                    path_nms = '%s/U%s_%s'%('NMS',compnt,rcv)
                    data_s= np.loadtxt(path_nms)
                    tnms  = data_s[:,0]
                    s     = data_s[:,1]
                    Amax1=max(data_s[:,1]) 
                    if 'E' in cpnt and compnt=='E':
                        axs[0].plot(times,sk,'r',linewidth=1.0,label='Syn.Shake.%s_%s'%(rcv,compnt))
                        axs[0].plot(tnms, s, 'b',linewidth=1.0,label='Syn.NMS.%s_%s'%(rcv,compnt))
                        axs[0].legend(loc='upper left',prop={'size':5 })

                    if 'N' in cpnt and  compnt=='N':
                        axs[1].plot(times,sk,'r',linewidth=1.0,label='Syn.Shake.%s_%s'%(rcv,compnt))
                        axs[1].plot(tnms, s, 'b',linewidth=1.0,label='Syn.NMS.%s_%s'%(rcv,compnt))
                        axs[1].legend(loc='upper left',prop={'size':5 })

                    if 'Z' in cpnt and compnt=='Z':
                        axs[2].plot(times,sk,'r',linewidth=1.0,label='Syn.Shake.%s_%s'%(rcv,compnt))
                        axs[2].plot(tnms, s, 'b',linewidth=1.0,label='Syn.NMS.%s_%s'%(rcv,compnt))
                        axs[2].legend(loc='upper left',prop={'size':5 })
        axs[0].set_ylabel(r'Disp [m]')
        axs[1].set_ylabel(r'Disp [m]')
        axs[2].set_ylabel(r'Disp [m]')
        ##########################################
        axs[2].set_xlabel(r'time [s]')
        ##########################################
        axs[0].text(750, -Amax/5.0, r'$\Delta$='+str(D_deg)+r'$^{\circ}$',horizontalalignment='center',verticalalignment='center')
        axs[2].text(750, -Amax/5.0, r'$\Delta T$='+str(DT)+r' s',horizontalalignment='center',verticalalignment='center')
        fig.suptitle('%s_%s'%(evname,rcv)) 
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.savefig(evname+'_'+rcv+".png",dpi=400)
