import numpy as np
import ODE_RK4 
import matplotlib.pyplot as plt

# --- Global Parameters ---
g = 9.81          
L = 67.0          
gamma = 0.00    # friction parameter 
m = 28

# --- Simulation Parameters ---
dt = 0.01
fig_dt = 0.01
t0 = 0
t_end = 3600*3

#####################################################
# The function describing the right and side of the equation
# dV/dt = F(t,V)
# where V is a vector containing the current position (x,y) and corresponding
# speeds (vx,vy)of the pendulum in the order (x, vx, y, vy)
# It returns the right hand side of teh equation
####################################################
def F(t, V):
    eq = np.array([0,0,0,0], dtype=float)
    x = V[0]
    vx = V[1]
    y = V[2]
    vy = V[3]

    Ls = L*L
    A = Ls - x*x - y*y
    term = (Ls*(vx*vx+vy*vy)-(x*vy-y*vx)**2)/A

    eq[0] = vx
    eq[1] = -(x/Ls)*term - (g/Ls)*x*np.sqrt(abs(A)) - (gamma/m)*vx #added abs
    eq[2] = vy
    eq[3] = -(y/Ls)*term - (g/Ls)*y*np.sqrt(abs(A)) - (gamma/m)*vy
    
    return eq

def plot(V0,lab):
    t_list, V_list =  ODE_RK4.runge_kutta_2nd_order_system(F, V0, t0, t_end, dt, fig_dt)
    x_list = [val[0] for val in V_list]
    y_list = [val[2] for val in V_list]
    plt.plot(x_list, y_list, linewidth=1, label=lab)

if __name__ == "__main__":

  # This is the line that performs the integration of the equation
  # F : the right hand side of the equation
  # V0: the intial condition (x0,vx0,y0,vy0)
  # t0 : the initial time, usuaully 0
  # t_end : when to stop the integration
  # dt : the integration time step.
  # fig_dt : the time intervals between data output for figures.
  # The values returned:
  # t_list : the time values at which data for a graph has been saved
  # V_list : the values saved for graphics: as a list off [x, vx, y, vy ]
  
  # Code was run twice, modifying the value 0.1 to 1 on the second run to produce the two plots shown in Figure 1.
  
  plot([1,0,0,0.1],r'$\dot{x}=0, \dot{y}=0.1 $')
  plot([1,0,0,0],r'$\dot{x}=\dot{y}=0$')
  
  plt.xlabel("x", fontsize=20)
  plt.ylabel("y", fontsize=20)
  #size is weird
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=20)
  plt.axis('equal')
  plt.tight_layout(rect=[0.02, 0, 0.98, 1], pad=0.5)
  plt.legend(loc="upper left", fontsize = 20)
  plt.savefig("airyprecession.png")
  plt.show()


