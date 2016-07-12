# ipython --pylab

# import some needed functions
from scipy.integrate import odeint
from numpy import *
from matplotlib.pyplot import *

###seterr(all='raise')

import pint
ureg = pint.UnitRegistry()
_0 = ureg['']
_prob = _0 # TODO: custom 'probability' unit?
_m0 = _0/1000
_m = ureg['meter']
_s = ureg['second']
_m_s = _m/_s
_m_s2 = _m_s/_s
_mV = ureg['mV']
_V = ureg['V']
_mM = ureg['millimol/liter']
_microM = ureg['micromol/liter']
_M = ureg['mol/liter']
_ms = _s/1000
_nm = ureg['nanometer']
_nS = ureg['nanomho']
_pF = ureg['picofarad']
_microN = ureg['micronewton']
_pN = ureg['piconewton']
_l = ureg['liter']
_S = ureg['siemens']

# constants

kB = 1.38064852 * (10 ^ -23) * ureg['J/K'] # Boltzmann constant
F = 96485.3329 * ureg['s A / mol']  # faraday constants
R = 8.3144598 * ureg['J / mol / K'] # universal gas constants

# parameters equation (1)

Cm = 10*_pF # cell capacitance

# parameters equation (2) (3) and (4)

Lambda = 2.8*_microN*_s/_m
X0 = 12*_nm # typical for the bullfrog
Z = 0.7*_pN # gating force typical for bullfrog
T = 295.15 * ureg['K'] # temperature
EMET = 0*_mV # reversal potential
### gMET = 0.65*_nS # maximum conductance Met channels according to Holton and Hudspeth
gMET = 0*_nS
K = 1350*_microN/_m # stiffness hair bundle


# parameters equation (8)

EK = -95*_mV # Potassium equilibrium potential

# parameters equation (9a)

gh = 2.2*_nS # maximum conductance
Eh = -45*_mV # reversal potential of h

# parameters equation (10)

PDRK = 2.4*pow(10,-14)*_l/_s  # maximum permeability of DRK
Kin = 112*_mM   #intracellular  K+ concentration
Kex = 2*_mM     #extracellular  K+ concentration

# parameters equation (11a)

gCa = 1.2*_nS # the maximum Ca2+ conductance
ECa = 42.5*_mV # todo

# parameters equation (12)

PBKS = 2*pow(10,-13)*_l/_s # maximum permeability of BKS
PBKT = 14*pow(10,-13)*_l/_s # maximum permeability of BKT

# parameters equation (14)

K1_0 = 6*_microM
K2_0 = 45*_microM
K3_0 = 20*_microM
k_1 = 300/_s
k_2 = 5000/_s
k_3 = 1500/_s
delta1 = 0.2*_0
delta2 = 0*_0
delta3 = 0.2*_0
VA = 33*_mV
z = 2*_0
alphac0 = 450/_s
betac = 2500/_s

# for equation (A29) in [14], replacing the 0.00061 constant

U = 0.005*_0
epsilon = 3.4e-5*_0
Cvol = 1.2*ureg['picoliter']
point00061 = U/(z*F*Cvol*epsilon)

# parameters equation (17)

EL = 0*_mV #todo

# make explicit the unit for each ODE in our ODE system

STATE_UNITS = [_mV, _nm, _0, _0, _0, _0, _0, _prob, _prob, _prob, _prob, _mM, _0]

def state_to_quantities(state):
        return [ mag * unit for (mag, unit) in zip(state, STATE_UNITS) ]

def state_to_numbers(state, unit_factor=ureg['']):
        return [ val.to(unit/unit_factor).magnitude for (val, unit) in zip(state, STATE_UNITS) ]

# define our ODE function

def hair_cell(state, t):
        """
         Purpose: simulate TODO model for the action potential using
         the equations from TODO.
         Input: state (TODO),
                t (time),
                and the params (parameters of neuron; see paper).
         Output: statep (state derivatives).
        """

        (V, X, mK1f, mK1s, mh, mDRK, mCa, C1, C2, O2, O3, Ca, hBKT) = state_to_quantities(state)

        FV = F*V
        RT = R*T


        #computation 8 computes the delayed rectifier current

        IK1 = gK1*(V-EK)*(0.7*mK1f+0.3*mK1s) # equation 8a
        mK1infinite = pow(1+ exp((V+110*_mV)/(11*_mV)),-1) # equation 8b (originally 8c)
        tauK1f = 0.7*_ms*exp(-(V+120*_mV)/(43.8*_mV))+0.04*_ms # equation 8c (originally 8d)
        tauK1s = 14.1*_ms*exp(-(V+120*_mV)/(28*_mV))+0.04*_ms # equation 8d (originally 8e)
        dmK1f_dt = (mK1infinite-mK1f)/tauK1f # equation 8e1 (originally 8b1)
        dmK1s_dt = (mK1infinite-mK1s)/tauK1s # equation 8e2 (originally 8b2)

        #computation 9 computes the Cation h-current

        Ih = gh*(V-Eh)*(3*pow(mh,2)*(1-mh)+pow(mh,3))# equation (9a)
        mhinfinite = pow(1+exp((V+87*_mV)/(16.7*_mV)),-1) # equation 9b (originally 9c)
        tauh = 63.7*_ms+135.7*_ms*exp(-pow((V+91.4*_mV)/(21.2*_mV),2)) # equation 9c (originally 9d)
        dmh_dt = (mhinfinite-mh)/tauh # equation 9d (9b)

        #computation 10 computes The DRK current

        IDRK = PDRK * ((V*pow(F,2))/RT) * ((Kin - Kex * exp(-FV/RT))/(1-exp(-FV/RT))) * pow(mDRK,2) # equation (10a)
        alphaDRK = pow(3.2*_ms*exp(-V/(20.9*_mV))+3*_ms, -1) # equation (10e)
        betaDRK = pow(1467*_ms*exp(V/(5.96*_mV))+9*_ms, -1) # equation (10f)
        tauDRK = pow(alphaDRK+betaDRK,-1) # equation (10d)
        mDRKinfinite = pow(1+exp(-(V+48.3*_mV)/(4.19*_mV)),-0.5) # equation (10c)
        dmDRK_dt = (mDRKinfinite-mDRK)/tauDRK # equation (10b)
# mdrkinf=  0.59296250511768533
# alfdrk=   25.327338763213305
# betdrk=   107.66080256193725
# dmdrk_dt=  -22.357289277952258
# Idrk=   4.0827377021036806E-011
# t:   1.7311500000000002
        #computation 11 computes Voltage-gated Ca2+current

        ICa = gCa*pow(mCa,3)*(V-ECa) # equation (11a)
        mCainfinite = pow(1+exp(-(V+55*_mV)/(12.2*_mV)),-1) # equation (11c)
        tauCa = 0.046*_ms+0.325*_ms*exp(-pow((V+77*_mV)/(51.67*_mV), 2)) # equation (11d)
        dmCa_dt = (mCainfinite-mCa)/tauCa # equation (11b)

        #[14] equation A14 and A15

        k1 = k_1/(K1_0*exp(-delta1*z*FV/RT))
        k2 = k_2/(K2_0*exp(-delta2*z*FV/RT))
        k3 = k_3/(K3_0*exp(-delta3*z*FV/RT))
        alphac = alphac0*exp(V/VA)

        #computation 14 computes The kinetics of Ca-activated BK currents

        C0 = 1*_prob-(C1+C2+O2+O3) # equation (14e)
        dC1_dt = k1*Ca*C0+k_2*C2-(k_1+k2*Ca)*C1 # equation (14a)
        dC2_dt = k2*Ca*C1+(alphac)*(O2)-(k_2+betac)*C2 # equation (14b)
        dO2_dt = betac*C2+k_3*O3-(alphac +k3*Ca)*O2 # equation (14c)
        dO3_dt = k3*Ca*O2-k_3*O3 # equation (14d)

        #computation 15 computes the dynamics of the Ca2+ concentration

        dCa_dt = -point00061*ICa-2800/_s*Ca # equation (15)

        hBKTinfinite = pow(1+exp((V+61.6*_mV)/(3.65*_mV)),-1) # equation (16b)
        tauBKT = 2.1*_ms+9.4*_ms*exp(-pow(((V+66.9*_mV)/(17.7*_mV)),2)) # equation (16c)
        dhBKT_dt = (hBKTinfinite-hBKT)/tauBKT # equation (16a)

        IBKS = b*PBKS * ((V*pow(F,2))/RT) * ((Kin - Kex * exp(-FV/RT))/(1-exp(-FV/RT))) * (O2 + O3) # equation (12)
        IBKT = b*PBKT * ((V*pow(F,2))/RT) * ((Kin - Kex * exp(-FV/RT))/(1-exp(-FV/RT))) * (O2 + O3)*hBKT # equation (13)

        IL = gL*(V-EL) # equation (17)

        dX_dt = (-K*X)/Lambda # equation (4)
        POX = 1/(1+exp(-(Z*(X-X0)/(kB*T)))) # equation (3)
        IMET = gMET*POX*(V-EMET) # equation (2)

        dV_dt = (-IK1-Ih-IDRK-ICa-IBKS-IBKT-IL-IMET) / Cm # equation (1)

        statep = [dV_dt, dX_dt, dmK1f_dt, dmK1s_dt, dmh_dt, dmDRK_dt, dmCa_dt, dC1_dt, dC2_dt, dO2_dt, dO3_dt, dCa_dt, dhBKT_dt]

        return state_to_numbers(statep, unit_factor=_s)

# simulate

# control parameters
gK1 = 29.25*_nS # maximum conductance of the MET channels
b = 0.01*_0 # dimensionless
gL = 0.174*_nS # leak conductance

# set initial states and time vector
state0 = [-70.*_mV, 0*_nm, 0.1*_0, 0.1*_0, 0.1*_0, 0.1*_0, 0.1*_0, 0.1*_prob, 0.1*_prob, 0.1*_prob, 0.1*_prob, 3*_microM, 0.1*_0]
t = arange(0, 5, 0.0001)

#if False:

        #t:   1.7311500000000002
        #input_state=[-5.0864253782598942E-002*_V, 0*_nm,   4.4823959651917305E-003*_prob,   3.9063196933636409E-003*_prob,  0.22635496281413278*_prob,       0.76107741412517049*_prob,    0.59362705410106820*_prob,    0.14128592217145175*_prob,        1.8027339494674426E-002*_prob,  0.50349641569042014*_prob,        6.7372605859965212E-002*_prob,   5.3808312530758806E-006*_M ,  2.0695504490729426E-002*_prob]
        #print("input: ", input_state)
        #new_state = hair_cell(state_to_numbers(input_state), 0)
        #print("output:", state_to_quantities(new_state))
        #expected output:  -1.6258547663311815       0.66535236908952200       0.56641281952265254       -1.8347760854750577       -22.357289277952258       -32.525095972570099       -4.0997254733989053       -2.2264539369557497        6.3129513679814124       -9.7525609444206793       -7.7757874071416287E-004   4.7225883106837401

        #import sys
        #sys.exit()

         #t:   1.7311500000000002
         #input:   -5.0863938611937526E-002   4.4825585706179064E-003   3.9064719696103754E-003  0.22635543716756984       0.76106021973173243       0.59363010900463153       0.14128685371835825        1.8028039962771136E-002  0.50348631821283085        6.7373132758083917E-002   5.3810333925573272E-006   2.0700830182175056E-002
         #output:  -1.6257312871138283       0.66376213616037072       0.56618909544822105       -1.8348145403769360       -22.353109221187687       -32.514442035503720       -4.0978859378079164       -2.2284867480385913        6.3139509674742840       -9.7512998346415856       -7.7797236871247379E-004   4.7211759592762359

# TODO: set some specific params here, perhaps

# run simulation
state = odeint(hair_cell, state_to_numbers(state0), t, args=())

# plot the results

print("Start plotting...")

"figure(figsize=(8,12))"
subplot(5,3,1)
plot(t, state[:,0])
title('V')
subplot(5,3,4)
plot(t, state[:,1])
title('X')
subplot(5,3,2)
plot(t, state[:,2])
title('mK1f')
subplot(5,3,5)
plot(t, state[:,3])
title('mK1s')
subplot(5,3,8)
plot(t, state[:,4])
title('mh')
subplot(5,3,11)
plot(t, state[:,5])
title('mDRK')
subplot(5,3,14)
plot(t, state[:,6])
title('mCa')
subplot(5,3,6)
plot(t, state[:,7])
title('C1')
subplot(5,3,9)
plot(t, state[:,8])
title('C2')
subplot(5,3,12)
plot(t, state[:,9])
title('O2')
subplot(5,3,15)
plot(t, state[:,10])
title('O3')
subplot(5,3,10)
plot(t, state[:,11])
title('[Ca]')
subplot(5,3,13)
plot(t, state[:,12])
title('hBKT')
xlabel('TIME (sec)')
show()