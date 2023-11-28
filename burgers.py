# vector manipulation
import numpy as np
# math functions
import math

# THIS IS FOR PLOTTING
%matplotlib inline
import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")

# Discete Grid
N=10
Nt=10
h=2*np.pi/N
k=1/Nt
r=k/(h*h)
time_steps=10
time=np.arange(0,(time_steps+.5)*k,k)
x=np.arange(0,2*np.pi+h/2,h)

# Create mesh grid in 2D
X, Y = np.meshgrid(x, time)

fig = plt.figure(figsize=(6,3))
plt.subplot(121)
plt.plot(X,Y,'ro');
plt.plot(x,0*x,'bo',label='Initial Condition');
plt.xlim((-h,2*np.pi+h))
plt.ylim((-k,max(time)+k))
plt.xlabel('x')
plt.ylabel('time (ms)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r'Discrete Mesh $\Omega_h$ ',fontsize=24,y=1.08)
plt.show();

# Initial Conditions
w=np.zeros((time_steps+1,N+1))
b=np.zeros(N-1)
# Initial Condition
for j in range (0,N+1):
    w[0,j]=1-np.cos(x[j])


fig = plt.figure(figsize=(6,3))
plt.subplot(122)
plt.plot(x,w[0,:],'o:',label='Initial Condition')
plt.xlim([-0.1,max(x)+h])
plt.title('Initial Condition',fontsize=24)
plt.xlabel('x')
plt.ylabel('w')
plt.legend(loc='best')
plt.show()
ipos = np.zeros(N+1)
ineg = np.zeros(N+1)
for i in range(0,N+1):
   ipos[i] = i+1
   ineg[i] = i-1

ipos[N] = 0
ineg[0] = N

# Lax Friedrich
lamba=k/h
for j in range(0,time_steps):
    for i in range (0,N+1):
        w[j+1,i]=(w[j,int(ipos[i])]+w[j,int(ineg[i])])/2+lamba*w[j,i]/2*(-(w[j,int(ipos[i])]-w[j,int(ineg[i])]))

= plt.figure(figsize=(12,6))

plt.subplot(121)
for j in range (1,time_steps+1):
    plt.plot(x,w[j,:],'o:')
plt.xlabel('x')
plt.ylabel('w')

plt.subplot(122)
X, T = np.meshgrid(x, time)
z_min, z_max = np.abs(w).min(), np.abs(w).max()


plt.pcolormesh( X,T, w, vmin=z_min, vmax=z_max)


#plt.imshow(w, aspect='auto')
plt.xlabel('x')
plt.ylabel('time')
clb=plt.colorbar()
clb.set_label('Temperature (w)')
plt.suptitle('Numerical Solution of the  Burger Equation'%(np.round(r,3)),fontsize=24,y=1.08)
fig.tight_layout()
plt.show()

## Lax-Wendroff
lamba = k / h  # lambda = dt / dx

for j in range(0, time_steps - 1):
    for i in range(1, N):  # Assuming N+1 points, iterate from 1 to N-1
        w[j+1, i] = w[j, i] - 0.5 * lamba * (w[j, i+1] - w[j, i-1]) + 0.5 * lamba**2 * (w[j, i+1] - 2*w[j, i] + w[j, i-1])


fig = plt.figure(figsize=(12,6))

plt.subplot(121)
for j in range (1,time_steps+1):
    plt.plot(x,w[j,:],'o:')
plt.xlabel('x')
plt.ylabel('w')

plt.subplot(122)
X, T = np.meshgrid(x, time)
z_min, z_max = np.abs(w).min(), np.abs(w).max()


plt.pcolormesh( X,T, w, vmin=z_min, vmax=z_max)


#plt.imshow(w, aspect='auto')
plt.xlabel('x')
plt.ylabel('time')
clb=plt.colorbar()
clb.set_label('Temperature (w)')
plt.suptitle('Numerical Solution of the  Burger Equation using Lax-Wendroff'%(np.round(r,3)),fontsize=24,y=1.08)
fig.tight_layout()
plt.show()