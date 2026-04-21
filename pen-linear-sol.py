import sympy as sp

l1 = sp.symbols("l1")
m, m1 = sp.symbols("m, m1")
J = sp.symbols("J")
t = sp.symbols("t")
g = sp.symbols("g")

p = sp.Function("p")(t)
th1 = sp.Function("th1")(t)

dp   = sp.diff(p, t)
dth1 = sp.diff(th1, t)

x1_x = p
x1_y = 0.0

x2_x = p + l1*sp.sin(th1)
x2_y = l1*sp.cos(th1)

dx1_x = sp.diff(x1_x, t)
dx2_x = sp.diff(x2_x, t)

dx1_y = sp.diff(x1_y, t)
dx2_y = sp.diff(x2_y, t)

T = 1/2*m*dx1_x**2 + 1/2*m1*dx2_x**2 + \
    1/2*m*dx1_y**2 + 1/2*m1*dx2_y**2 + \
    1/2*J*dth1**2

V = m1*g*x2_y 

L = T - V

eq1 = sp.diff( sp.diff(L, dth1), t ) - sp.diff(L, th1)
eq1 = sp.trigsimp(eq1)

print("eq1 - ddth1")
sp.pprint(sp.simplify(eq1))

########################## linear equation solution #############################
ddth1 = sp.diff(th1, t, 2)
ddp   = sp.diff(p, t, 2)

unknowns = [ddth1]

eqs = [eq1]

M, rhs = sp.linear_eq_to_matrix(eqs, unknowns) 
M_simpl = sp.simplify(M)
rhs_simpl = sp.simplify(rhs)

print("---------------------- linear equation -----------------------------")
print("M * ddq = rhs")
sp.pprint(M_simpl)  
sp.pprint(rhs_simpl) 

print("---------------------- ddq equation -----------------------------")

eq = M.inv() @ rhs
eq = sp.simplify(eq)
sp.pprint(eq)  