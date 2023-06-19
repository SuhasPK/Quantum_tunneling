import numpy as np
import scipy.constants as sc
from bokeh.plotting import figure 
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import Slider, Div, Button, RadioButtonGroup
from bokeh.events import ButtonClick
from bokeh.server.server import Server
from bokeh.models import Label
from bokeh.palettes import Colorblind

# Class for Quantum Tunneling
class Qtunnel:
    
    # Initializations
    def __init__(self, V0, bw, ke, sig):
        self.V0 = V0 * sc.value('electron volt')  # height of potential barrier in Joules
        self.ke = ke * sc.value('electron volt')  # kinetic energy of electron in Joules
        self.k0 = np.sqrt(self.ke * 2 * sc.m_e / (sc.hbar**2))  # wave vector of electron in m^-1
        self.bw = bw * sc.value('Angstrom star')  # potential barrier width in m
        self.sig = sig * sc.value('Angstrom star')  # Initial spread of Gaussian wavefunction
        self.dx = np.minimum((self.bw / 25.0), (self.sig / 25.0))  # grid cell size
        self.dt = 0.9 * sc.hbar / ((sc.hbar**2/(sc.m_e * self.dx**2)) + (self.V0 / 2.0))  # time step size
        length = 40 * np.maximum(self.bw, self.sig)  # length of the simulation domain
        self.ll = int(length / self.dx)  # total number of grid points in the domain
        vel = sc.hbar * self.k0 / sc.m_e
        self.tt = int(0.35 * length / vel / self.dt)  # total number of time steps in the simulation
        self.lx = np.linspace(0.0, length, self.ll)  # 1D position vector along x
        # potential barrier
        self.Vx = np.zeros(self.ll)
        bwgrid = int(self.bw/(2.0 * self.dx))
        bposgrid = int(self.ll/2.0)
        bl = bposgrid - bwgrid
        br = bposgrid + bwgrid
        self.Vx[bl:br] = self.V0
        # wavefunction arrays
        self.psir = np.zeros((self.ll))
        self.psii = np.zeros((self.ll))
        self.psimag = np.zeros(self.ll)
        ac = 1.0 / np.sqrt((np.sqrt(np.pi)) * self.sig)
        x0 = bl * self.dx - 6 * self.sig
        psigauss = ac * np.exp(-(self.lx - x0)**2 / (2.0 * self.sig**2))
        self.psir = psigauss * np.cos(self.k0 * self.lx)
        self.psii = psigauss * np.sin(self.k0 * self.lx)
        self.psimag = self.psir**2 + self.psii**2
        self.psimaginit = self.psimag
        self.psirinit = self.psir
        self.psiiinit = self.psii
        # fdtd update coefficients
        self.c1 = sc.hbar * self.dt / (2.0 * sc.m_e * self.dx**2)
        self.c2 = self.dt / sc.hbar
    
    # FDTD update for solving Schrodinger's equation
    def fdtd_update(self):
        self.psii[1:self.ll - 1] = (self.c1 * (self.psir[2:self.ll] - 2.0 * self.psir[1:self.ll - 1]
                                    + self.psir[0:self.ll - 2]) 
                                    - self.c2 * self.Vx[1:self.ll - 1] * self.psir[1:self.ll - 1]
                                    + self.psii[1:self.ll - 1])
        self.psir[1:self.ll - 1] = (-self.c1 * (self.psii[2:self.ll] - 2.0 * self.psii[1:self.ll - 1]
                                    + self.psii[0:self.ll - 2]) 
                                    + self.c2 * self.Vx[1:self.ll - 1] * self.psii[1:self.ll - 1]
                                    + self.psir[1:self.ll - 1])
        self.psimag = self.psir**2 + self.psii**2

    # Update plots
    def update_plots(self, r12):
        r12.data_source.data['y'] = self.psimag / np.amax(self.psimaginit)    


# Arrays for plotting


# Function to modify webpage (doc)
def modify_doc(doc):

    p1 = figure(plot_width=600, plot_height=500, title='Quantum Tunneling Animation')
    p1.xaxis.axis_label = 'position (Angstrom)'
    p1.yaxis.axis_label = 'Amplitude Squared (normalized)'
    p1.toolbar.logo = None
    p1.toolbar_location = None
    r11 = p1.line([], [], legend='Barrier', color=Colorblind[8][5], line_width=2)
    r12 = p1.line([], [], legend='Wavefunction', color=Colorblind[8][7], line_width=2)

    p2 = figure(plot_width=400, plot_height=250, title='Normalized wavefunctions at start')
    p2.xaxis.axis_label = 'position (Angstrom)'
    p2.yaxis.axis_label = 'Amplitude'
    p2.toolbar.logo = None
    p2.toolbar_location = None
    r21 = p2.line([], [], legend='Barrier', color=Colorblind[8][5])
    r22 = p2.line([], [], legend='Magnitude', color=Colorblind[8][7])
    r23 = p2.line([], [], legend='Real part', color=Colorblind[8][0])
    r24 = p2.line([], [], legend='Imag. part', color=Colorblind[8][6])
 
    p3 = figure(plot_width=400, plot_height=250, title='Normalized wavefunctions at end')
    p3.xaxis.axis_label = 'position (Angstrom)'
    p3.yaxis.axis_label = 'Amplitude'
    p3.toolbar.logo = None
    p3.toolbar_location = None
    r31 = p3.line([], [], color=Colorblind[8][5])
    r32 = p3.line([], [], color=Colorblind[8][7])
    r33 = p3.line([], [], color=Colorblind[8][0])
    r34 = p3.line([], [], color=Colorblind[8][6])

    barrier_height = Slider(title='Barrier Height (eV)', value=600, start=20, end=1000, step=100)
    barrier_width = Slider(title='Barrier Width (Angstrom)', value=0.3, start=0.3, end=1.0, step=0.1)
    electron_energy = Slider(title='Electron Energy (eV)', value=500, start=10, end=900, step=100)
    psi_spread = Slider(title='Wavefunction Spread (Angstrom)', value=0.8, start=0.3, end=1.0, step=0.1)
    startbutton = Button(label='Start', button_type='success')
    textdisp = Div(text='''<b>Note:</b> Wait for simulation  to stop before pressing buttons.''')
    texttitle = Div(text='''<b>QUANTUM TUNNELING</b>''', width=1000)
    textdesc = Div(text='''This application simulates quantum tunneling of an electron across a potential barrier 
                           by solving the Schrodinger's equation using the finite-difference time-domain method.
                           You can change the height and width of the barrier, as well as the energy and spread
                           of the electron to see how it would affect the probability of tunneling.''', width=1000)
    textrel = Div(text='''This for simulation purpose only. Do play with it and get some insights.''', width=1000)

    def run_qt_sim(event):
    
        # Reset plots
        r21.data_source.data['x'] = []
        r22.data_source.data['x'] = []
        r23.data_source.data['x'] = []
        r24.data_source.data['x'] = []
        r21.data_source.data['y'] = []      
        r22.data_source.data['y'] = []
        r23.data_source.data['y'] = []
        r24.data_source.data['y'] = []
        r31.data_source.data['x'] = []
        r32.data_source.data['x'] = []
        r33.data_source.data['x'] = []
        r34.data_source.data['x'] = []
        r31.data_source.data['y'] = []
        r32.data_source.data['y'] = []
        r33.data_source.data['y'] = []
        r34.data_source.data['y'] = []

        # Get widget values
        V0 = barrier_height.value
        bw = barrier_width.value
        ke = electron_energy.value
        sig = psi_spread.value

        # Create Qtunnel object
        qt = Qtunnel(V0, bw, ke, sig)
        
        # Plot initial states
        r21.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r22.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r23.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r24.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r21.data_source.data['y'] = qt.Vx / np.amax(qt.Vx)
        r22.data_source.data['y'] = qt.psimag / np.amax(qt.psimaginit)
        r23.data_source.data['y'] = qt.psir / np.amax(qt.psirinit)
        r24.data_source.data['y'] = qt.psii / np.amax(qt.psiiinit) 
        r11.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r12.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r11.data_source.data['y'] = qt.Vx / np.amax(qt.Vx)

        for n in range(qt.tt):
            qt.fdtd_update()
            qt.update_plots(r12)


        # Plot final states
        r31.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r32.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r33.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r34.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r31.data_source.data['y'] = qt.Vx / np.amax(qt.Vx)
        r32.data_source.data['y'] = qt.psimag / np.amax(qt.psimaginit)
        r33.data_source.data['y'] = qt.psir / np.amax(qt.psirinit)
        r34.data_source.data['y'] = qt.psii / np.amax(qt.psiiinit) 

    # Setup callbacks
    startbutton.on_event(ButtonClick, run_qt_sim)
    doc.add_root(column(texttitle, textdesc, row(barrier_height, barrier_width, textdisp), 
                 row(electron_energy, psi_spread, startbutton), row(p1, column(p2, p3)), textrel)) 

server = Server({'/': modify_doc}, num_procs=1)
server.start()
 
if __name__=='__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()

import matplotlib.pyplot as plt
import numpy as np
from math import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')



a = int(input("render level "))
l = float(input("lenght = "))
r = float(input("size of potential barrer = "))
V_o = float(input("V_o = "))
n = int(input("n = "))
h = 6.62607015
h_bar = h/(2*np.pi)
m = 9.1093837015
E = ((n**2)*(h**2))/(8*m*l**2)
print("the total energy of the particle is ", E)
q = 5.2
alpha = (np.sqrt(2*m*(V_o-E)))/h_bar
print("alpha ", alpha)


def delta(x):
    s = np.exp(-alpha*x)/((2*np.sin(((n*np.pi)/l)*x))**2)
    return(s)

print("q d exp ", np.exp(-alpha*q), "q d sin^2 ", ((2*np.sin(((n*np.pi)/l)*q))**2))
print("q+r d exp ", np.exp(-alpha*(q+r)), "q+r d sin^2 ", ((2*np.sin((n*np.pi/l)*(q+r)))**2))

print("delta q ", delta(q))
print("delta q+r ", delta(q+r))

Constant = np.sqrt((1/((delta(q))*(q - (l/(2*n*np.pi)) * (np.sin(((2*n*np.pi)/l) * q))) - ((1/(2*alpha)) * ((np.exp(-2*alpha*q))) * ((np.exp(-2*alpha*r)) + 1)) + ((delta(q+r))*(l - (q+r) - (l/(2*n*np.pi)) * (np.sin(((2*n*np.pi)/l) * (q+r))) )) ) ))

A = 2*Constant*(delta(q))
B = Constant
C = 2*Constant*(delta(q+r))

c = np.sqrt(1/(delta(q)*(q-(l/(2*n*np.pi)*np.sin((2*n*np.pi/l)*q))) - (1/(2*alpha))*((e**(-alpha*q))*((e**(-alpha*r))+1)) + delta(q+r)*(l-q+r-(l/(2*n*np.pi))*np.sin((2*np.pi/l)*(q+r)))))



xa = np.linspace(0, q, a)
xb = np.linspace(q, q+r, a)
xc = np.linspace(q+r, l, a)

t = np.linspace(0, 40, a)

x_a, t_ = np.meshgrid(xa, t)
x_b, t_ = np.meshgrid(xb, t)
x_c, t_ = np.meshgrid(xc, t)

def aR(x,t):
    s = np.cos(-(E/h_bar)*t)*(A*np.sin((n*np.pi/l)*x))
    return(s)

def aC(x,t):
    s = np.sin(-(E/h_bar)*t)*(A*np.sin((n*np.pi/l)*x))
    return(s)

def aP(x):
    s = (A*np.sin((n*np.pi/l)*x))**2
    return(s)


def bR(x,t):
    s = np.cos(-(E/h_bar)*t)*B*e**(-alpha*x)
    return(s)

def bC(x,t):
    s = np.sin(-(E/h_bar)*t)*B*e**(-alpha*x)
    return(s)

def bP(x):
    s = (B*e**(-alpha*x))**2
    return(s)


def cR(x,t):
    s = np.cos(-(E/h_bar)*t)*(C*np.sin((n*np.pi/l)*x))
    return(s)

def cC(x,t):
    s = np.sin(-(E/h_bar)*t)*(C*np.sin((n*np.pi/l)*x))
    return(s)

def cP(x):
    s = (C*np.sin((n*np.pi/l)*x))**2
    return(s)

ZaR = aR(x_a,t_)
ZaC = aC(x_a,t_)

ZbR = bR(x_b,t_)
ZbC = bC(x_b,t_)

ZcR = cR(x_c,t_)
ZcC = cC(x_c,t_)

ax.plot_surface(x_a, t_, ZaR, cmap = cm.plasma, linewidth=0, antialiased=True, color = 'blue')
ax.plot_surface(x_a, t_, ZaC, cmap = cm.plasma, linewidth=0, antialiased=True, color = 'green')

ax.plot_surface(x_b, t_, ZbR, cmap = cm.plasma, linewidth=0, antialiased=True, color = 'blue')
ax.plot_surface(x_b, t_, ZbC, cmap = cm.plasma, linewidth=0, antialiased=True, color = 'green')

ax.plot_surface(x_c, t_, ZcR, cmap = cm.plasma, linewidth=0, antialiased=True, color = 'blue')
ax.plot_surface(x_c, t_, ZaC, cmap = cm.plasma, linewidth=0, antialiased=True, color = 'green')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('value of the quantum wave')

ax.legend()

plt.show()


plt.plot(xa, aP(xa), color = 'red')
plt.plot(xb, bP(xb), color = 'red')
plt.plot(xc, cP(xc), color = 'red')

plt.show()

# Quantum Tunneling Simulation for ES 170
# This code simulates quantum mechanical tunneling of an electron across a square potential barrier
# The electron is represented by a 1D Gaussian wavefunction
# 1D time-dependent Schrodinger equation is solved using the finite-difference time-domain method

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


# Class for QMT FDTD
class QMfdtd:
    def __init__(self, V0, bw, ke, sig):
        self.V0 = V0 * sc.value('electron volt')  # height of potential barrier in Joules
        self.ke = ke * sc.value('electron volt')  # kinetic energy of electron in Joules
        self.k0 = np.sqrt(self.ke * 2 * sc.m_e / (sc.hbar ** 2))  # wave vector of electron in m^-1
        self.bw = bw * sc.value('Angstrom star')  # potential barrier width in m
        self.sig = sig * sc.value('Angstrom star')  # Initial spread of Gaussian wavefunction
        self.dx = np.minimum((self.bw / 25.0), (self.sig / 25.0))  # grid cell size
        self.dt = 0.9 * sc.hbar / ((sc.hbar ** 2 / (sc.m_e * self.dx ** 2)) + (self.V0 / 2.0))  # time step size
        length = 40 * np.maximum(self.bw, self.sig)  # length of the simulation domain
        self.ll = int(length / self.dx)  # total number of grid points in the domain
        vel = sc.hbar * self.k0 / sc.m_e
        self.tt = int(0.35 * length / vel / self.dt)  # total number of time steps in the simulation
        self.lx = np.linspace(0.0, length, self.ll)  # 1D position vector along x
        # potential barrier
        self.Vx = np.zeros(self.ll)
        bwgrid = int(self.bw / (2.0 * self.dx))
        bposgrid = int(self.ll / 2.0)
        bl = bposgrid - bwgrid
        br = bposgrid + bwgrid
        self.Vx[bl:br] = self.V0
        # wavefunction arrays
        self.psir = np.zeros((self.ll))
        self.psii = np.zeros((self.ll))
        self.psimag = np.zeros(self.ll)
        ac = 1.0 / np.sqrt((np.sqrt(np.pi)) * self.sig)
        x0 = bl * self.dx - 6 * self.sig
        psigauss = ac * np.exp(-(self.lx - x0) ** 2 / (2.0 * self.sig ** 2))
        self.psir = psigauss * np.cos(self.k0 * self.lx)
        self.psii = psigauss * np.sin(self.k0 * self.lx)
        self.psimag = self.psir ** 2 + self.psii ** 2
        # fdtd update coefficients
        self.c1 = sc.hbar * self.dt / (2.0 * sc.m_e * self.dx ** 2)
        self.c2 = self.dt / sc.hbar

    # The main FDTD update function.
    def fdtd_update(self):
        self.psii[1:self.ll - 1] = (self.c1 * (self.psir[2:self.ll] - 2.0 * self.psir[1:self.ll - 1]
                                               + self.psir[0:self.ll - 2])
                                    - self.c2 * self.Vx[1:self.ll - 1] * self.psir[1:self.ll - 1]
                                    + self.psii[1:self.ll - 1])
        self.psir[1:self.ll - 1] = (-self.c1 * (self.psii[2:self.ll] - 2.0 * self.psii[1:self.ll - 1]
                                                + self.psii[0:self.ll - 2])
                                    + self.c2 * self.Vx[1:self.ll - 1] * self.psii[1:self.ll - 1]
                                    + self.psir[1:self.ll - 1])
        self.psimag = self.psir ** 2 + self.psii ** 2


def run_sim(V0_in, bw_in, ke_in, sig_in):
    q1 = QMfdtd(V0_in, bw_in, ke_in, sig_in)
    print('')
    print('Potential barrier =', round(q1.V0 / sc.value('electron volt'), 2), 'eV')
    print('Potential barrier width =', round(q1.bw / sc.value('Angstrom star'), 2), 'A')
    print('(The boundary of the simulation domain is assumed to be an infinite barrier)')
    print('Electron energy =', round(q1.ke / sc.value('electron volt'), 2), 'eV')
    print('Electron spread =', round(q1.sig / sc.value('Angstrom star'), 2), 'A')
    print('')
    print('Grid size =', '%.2e' % (q1.dx / sc.value('Angstrom star')), 'A')
    print('Time step =', "%.2e" % (q1.dt * 1e15), 'fs')
    plt.ion()
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.set_xlabel('position ($\AA$)')
    ax0.set_ylabel('$\Psi$')
    ax0.set_title('Initial wavefunctions (normalized)')
    ax0.plot(q1.lx / sc.value('Angstrom star'), q1.psimag / np.amax(q1.psimag), label='$|\Psi|^2$')
    ax0.plot(q1.lx / sc.value('Angstrom star'), q1.Vx / np.amax(q1.Vx), label='barrier')
    ax0.plot(q1.lx / sc.value('Angstrom star'), q1.psii / np.amax(q1.psii), label='$\Im[\Psi]$', alpha=0.5)
    ax0.plot(q1.lx / sc.value('Angstrom star'), q1.psir / np.amax(q1.psir), label='$\Re[\Psi]$', alpha=0.5)
    ax0.legend()
    fig0.show()
    fig0.tight_layout()
    fig0.canvas.draw()
    input('Press enter to start the simulation...')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('position ($\AA$)')
    ax1.set_ylabel('norm magnitude')
    fig1.show()
    fig1.canvas.draw()
    for nn in range(0, q1.tt):
        q1.fdtd_update()
        if nn % 50 == 0:
            tstr = 'Time = ' + str(round(nn * q1.dt * 1e15, 4)) + ' fs'
            ax1.clear()
            ax1.plot(q1.lx / sc.value('Angstrom star'), q1.psimag / np.amax(q1.psimag), label='$|\Psi|^2$')
            ax1.plot(q1.lx / sc.value('Angstrom star'), q1.Vx / np.amax(q1.Vx), label='barrier')
            ax1.legend()
            ax1.set_title(tstr)
            ax1.set_xlabel('position ($\AA$)')
            ax1.set_ylabel('normalized magnitude')
            fig1.canvas.draw()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('position ($\AA$)')
    ax2.set_ylabel('$\Psi$')
    ax2.set_title('Final wavefunctions (normalized)')
    ax2.plot(q1.lx / sc.value('Angstrom star'), q1.psimag / np.amax(q1.psimag), label='$|\Psi|^2$')
    ax2.plot(q1.lx / sc.value('Angstrom star'), q1.Vx / np.amax(q1.Vx), label='barrier')
    ax2.plot(q1.lx / sc.value('Angstrom star'), q1.psii / np.amax(q1.psii), label='$\Im[\Psi]$', alpha=0.5)
    ax2.plot(q1.lx / sc.value('Angstrom star'), q1.psir / np.amax(q1.psir), label='$\Re[\Psi]$', alpha=0.5)
    ax2.legend()
    fig2.show()
    fig2.tight_layout()
    fig2.canvas.draw()


print('')
V0_in = float(input('Enter the barrier height in eV (try 600): '))
bw_in = float(input('Enter the barrier width in Angstrom (try 0.25): '))
ke_in = float(input('Enter the electron energy in eV (try 500): '))
sig_in = float(input('Enter the initial electron wavefunction spread in Angstrom (try 0.8): '))
run_sim(V0_in, bw_in, ke_in, sig_in)
