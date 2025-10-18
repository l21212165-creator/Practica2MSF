# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:12:48 2025

@author: OMARS
"""
#!pip install control
#!pip install slycot
import control as ctrl
import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
u= np.array(pd.read_excel('Signal.xlsx',header=None))

x0,t0,tend,dt,w,h = 0,0,10,1E-3,10,5
n = round((tend - t0)/dt) + 1
t = np.linspace(t0,tend,n)

u= np.reshape(signal.resample(u,len(t)),-1)

def cardio(Z,C,R,L):
    num=[L*R,R*Z]
    den=[C*L*R*Z,L*R+L*Z,R*Z]
    sys=ctrl.tf(num,den)
    return sys

#Func de transferencia : NORMOTENSO
Z,C,R,L= 0.033, 1.5, 0.95, 0.01

sysnormo= cardio(Z,C,R,L)
print(f'Funcion de transferencia del normotenso: {sysnormo}')

#Func de transferencia : HIPOTENSO
Z,C,R,L= 0.020, 0.250, 0.600, 0.005
syshipo= cardio(Z,C,R,L)

print(f'Funcion de transferencia del hipotenso: {syshipo}')
#Func de transferencia : HIPERTENSO
Z,C,R,L=0.050, 2.500, 1.400, 0.020
syshiper= cardio(Z,C,R,L)
print(f'Funcion de transferencia del hipertenso: {syshiper}')

print(f'Funcion de transferencia del normotenso: {syshipo}')

#Respuesta en lazo abierto

_,Pp0= ctrl.forced_response(sysnormo,t,u,x0)
_,Pp1= ctrl.forced_response(syshipo,t,u,x0)
_,Pp2= ctrl.forced_response(syshiper,t,u,x0)

fg1= plt.figure()
plt.plot(t,Pp0,'-',linewidth=1,color=[1.000, 0.902, 0.902], label= 'Pp(t):Normotenso')
plt.plot(t,Pp1,'-',linewidth=1,color=[ 0.882, 0.686, 0.820], label= 'Pp(t):Hipotenso')
plt.plot(t,Pp2,'-',linewidth=1,color=[ 0.678, 0.533, 0.776], label= 'Pp(t):Hipertenso')

plt.grid(True)

plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t) [V]')
plt.ylabel('t [s]')

plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)

plt.show()



# sys
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema cardiovascular python.png',dpi=600,bbox_inches='tight')

fg1.savefig('sistema cardiovascular python.pdf')

def controlador(kP, kI, sys):
    Cr = 1e-6
    Re = 1 / (kI * Cr)
    Rr = kP * Re
    
    # Controlador PI: (Re*Cr)*s + 1 / (Re*Cr*s)
    numPI = [Rr * Cr, 1]
    denPI = [Re * Cr, 0]
    PI = ctrl.tf(numPI, denPI)
    
    # Lazo cerrado
    sysPI = ctrl.feedback(PI * sys, 1, sign=-1)
    
    return sysPI

hipoPI=controlador(8.92851562535347e-05,1035.71406250139,syshipo)
hiperPI=controlador(5,13686.5342163355,syshiper)

#Respuesta en lazo cerrado
_,Pp3=ctrl.forced_response(hipoPI,t,Pp0,x0)
_,Pp4=ctrl.forced_response(hiperPI,t,Pp0,x0)



fg2= plt.figure()
#plt.plot(t,Pp0,'-',linewidth=1,color=[1.000, 0.902, 0.902], label= 'Pp(t):Normotenso')
plt.plot(t,Pp1,'-',linewidth=1,color=[ 0.882, 0.686, 0.820], label= 'Pp(t):Hipotenso')
plt.plot(t,Pp0,'-',linewidth=1,color=[1.000, 0.902, 0.902], label= 'Pp(t):Normotenso')
plt.plot(t,Pp3,':',linewidth=3,color=[ 0.678, 0.533, 0.776], label= 'Pp(t):Hipotenso PI')

plt.grid(False)

plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t) [V]')
plt.ylabel('t [s]')

plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)

plt.show()

fg2.set_size_inches(w,h)
fg2.tight_layout()
fg1.savefig('sistema cardiovascular hipoPI python.png',dpi=600,bbox_inches='tight')

fg2.savefig('sistema cardiovascular hipoPI python.pdf')




#---------

fg3= plt.figure()
#plt.plot(t,Pp0,'-',linewidth=1,color=[1.000, 0.902, 0.902], label= 'Pp(t):Normotenso')
plt.plot(t,Pp2,'-',linewidth=1,color=[ 0.882, 0.686, 0.820], label= 'Pp(t):Hipertenso')
plt.plot(t,Pp0,'-',linewidth=1,color=[1.000, 0.902, 0.902], label= 'Pp(t):Normotenso')
plt.plot(t,Pp4,':',linewidth=3,color=[ 0.678, 0.533, 0.776], label= 'Pp(t):HipertensoPI')

plt.grid(False)

plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t) [V]')
plt.ylabel('t [s]')

plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)

plt.show()

fg3.set_size_inches(w,h)
fg3.tight_layout()
fg3.savefig('sistema cardiovascular hiperPI python.png',dpi=600,bbox_inches='tight')

fg3.savefig('sistema cardiovascular hiperPI python.pdf')
