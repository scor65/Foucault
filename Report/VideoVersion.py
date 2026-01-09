import numpy as np
import matplotlib.pyplot as plt
import ODE_RK4

g = 9.81
L = 67
mu = 0.0000181   
R = 0.085
# release from 3 meters
amp_init = 3 

#Name [0], Density (kg/m^3) [1], Price (£/kg) [2], Colour [3]]
materials = [
    ["Wood", 750, 1.00, "brown"],
    ["Lead", 11348, 2, "grey"],
    ["Gold", 19283, 103815, "gold"],
]

# Density information of lead and gold from the respective Wikipedia pages,

# Decay of amplitude over one day:
t = np.linspace(0, 24*3600, 1000)

plt.figure(figsize=(10, 6))

# Calculate volume and air resistance
vol = (4/3) * np.pi*R**3
gamma = 6 * np.pi*mu*R  

for mat in materials: 
    mass = mat[1] * vol
    amp = np.exp(-(gamma / (2*mass)) * t)
    plt.plot(t/3600, amp_init*amp, label=mat[0], color=mat[3])

#Ping pong ball
massPP = 0.0027  
RPP = 0.02

gammaPP = 6 * np.pi * mu * RPP 
ampPP = np.exp(-(gammaPP / (2 * massPP)) * t)

plt.plot(t/3600, amp_init*ampPP, label="Ping Pong Ball", color="blue")

plt.title(f"Amplitude decay over 1 day by material", fontsize=16)
plt.xlabel("Time (h)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("AmpltudeDecay")
plt.show()


# Cost

names = []
costs = []
half_time = []
colors = []

#Name [0], Density (kg/m^3) [1], Price (£/kg) [2], Colour [3]]
for mat in materials:  
    #mass = dens * vol
    mass = mat[1] * vol
    cost = mass * mat[2]
    
    # Time for amplitude to half
    k = gamma / (2 * mass)
    t_half = (np.log(2) / k)

    names.append(mat[0])
    costs.append(cost)
    colors.append(mat[3])
    half_time.append(t_half/3600)
    
    print(f"{mat[0]}: Mass={mass:.2f}kg, Cost=£{cost:,.2f}, Time for Amplitude to Halve={t_half/3600:.1f} h")


#Amplitude decay over time
plt.figure()

bars = plt.bar(names, half_time, color=colors, edgecolor='black')

plt.ylabel('Time taken for amplitude to halve (h)')
plt.title("Time taken for amplitude to halve by material")

# Labels (AI Disclaimer: GPT5 was used to generate these)
for bar, hl in zip(bars, half_time):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{hl:.1f} h',
             ha='center', va='bottom')

plt.savefig("DecayHalf.png")
plt.show()

# Bar chart for cost of each
plt.figure()

bars = plt.bar(names, costs, color=colors, edgecolor='black')

plt.ylabel("Cost (£)")
plt.semilogy()
# Log scale since gold is so expensive
plt.title("Cost of 17cm Diameter Pendulum Bob by Material")

# Labels (AI Disclaimer: GPT5 was used to generate these)
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height * 1.1, 
             f'£{cost:,.2f}',
             ha='center', va='bottom')

plt.ylim(top=2*10**7) 
plt.savefig("PendulumCost.png")
plt.show()

# Using Drag Equation

rho_air = 1.225 #density of air
#https://en.wikipedia.org/wiki/Density_of_air
C_d = 0.47 #Sphere drag coefficient
# https://en.wikipedia.org/wiki/Drag_coefficient
mass = 0
area = 0

def F_dragequ(t, V):

    eq = np.array([0, 0, 0, 0], dtype=float)
    x  = V[0]
    vx = V[1]
    y  = V[2] 
    vy = V[3] 
    
    newgamma = 0.5 * C_d * rho_air * area
    
    eq[0] = vx
    eq[1] = -(g/L)*x - (newgamma/mass) * (vx)**2 * np.sign(vx)
    eq[2] = vy
    eq[3] = -(g/L)*y - (newgamma/mass) * (vy)**2 * np.sign(vy)
    
    return eq


t0 = 0
t_end = 24 * 3600
dt = 0.5
fig_dt = 50

plt.figure(figsize=(13, 6))

for mat in materials:
    name = mat[0]
    density = mat[1]
    
    area = np.pi * R**2
    vol = (4/3) * np.pi * R**3
    mass = density * vol
    
    gamma_stokes = 6 * np.pi * mu * R
    amp_stokes = np.exp(-(gamma_stokes / (2*mass)) * t)
    
    #Stokes' Law
    plt.plot(t/3600, amp_init*amp_stokes, linestyle='--', color=mat[3])

    V0 = [3.0, 0.0, 0.0, 0.0]   
    times, V_list = ODE_RK4.runge_kutta_2nd_order_system(F_dragequ, V0, t0, t_end, dt, fig_dt)
    
    #Get just x
    x_vals = [v[0] for v in V_list]
    time_vals = np.array(times)
    
    min_list, max_list = ODE_RK4.get_min_max(time_vals, x_vals)
    

    t_peaks = [m[0]/3600 for m in max_list] 
    amp_peaks = [m[1] for m in max_list]
        
    t_peaks.insert(0, 0.0)
    amp_peaks.insert(0, amp_init)
        
    plt.plot(t_peaks, amp_peaks, linestyle='-', linewidth=2, color=mat[3])

plt.title("Stokes vs Drag Equation", fontsize=16)
plt.xlabel("Time (h)")
plt.ylabel("Amplitude (m)")
plt.xticks(np.arange(0, 25, 1)) 
plt.savefig("DragComparison.png")
plt.show()
