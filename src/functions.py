import scipy.special
import numpy as np

class Parameters_layers:
    def __init__(self, N, Nlayer,Rs,rhos,dms,nus):
        # Azimuthal index
        self.N = N
        # Number of layers (including background)
        self.Nlayer = Nlayer
        # Radii (from innermost to outermost)
        self.Rs = np.array(Rs)
        # mass density (from innermost to outermost)
        self.rhos = np.array(rhos)
        # Lame's constants mu
        self.dms = np.array(dms)
        # Poisson's ratio
        self.nus = np.array(nus)

        # Lame's constants lambda
        self.dls = 2*self.dms*self.nus / (1-2*self.nus)
        

class Parameters_solidBG:
    def __init__(self, rho_0, dm_0, nu_0):
        # exterior mass density
        self.rho_0 = rho_0
        # exterior Lame's constant mu
        self.dm_0   = dm_0
        # exterior Poisson's ratio
        self.nu_0  = nu_0

        # exterior Lame's constant lambda
        self.dl_0 = 2*self.dm_0*self.nu_0 / (1-2*self.nu_0)

class Parameters_fluidBG:
    def __init__(self, rho_0, kappa_0):
        # exterior mass density
        self.rho_0 = rho_0
        # exterior bulk modulus
        self.kappa_0 = kappa_0

def besj(n,z):
    return scipy.special.jv(n, z)

def besj_d(n,z):
    return scipy.special.jvp(n, z)

def besh(n,z):
    return scipy.special.hankel1(n, z)

def besh_d(n,z):
    return scipy.special.h1vp(n, z)

def calc_U1(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_U1 = kL * besj_d(n,kL*r)

    return func_U1

def calc_U2(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_U2 = 1j*n/r * besj(n,kT*r)

    return func_U2

def calc_V1(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_V1 = 1j*n/r * besj(n,kL*r)

    return func_V1

def calc_V2(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_V2 = -kT * besj_d(n,kT*r)

    return func_V2

def calc_Ut1(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Ut1 = kL * besh_d(n,kL*r)

    return func_Ut1

def calc_Ut2(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Ut2 = 1j*n/r * besh(n,kT*r)

    return func_Ut2

def calc_Vt1(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Vt1 = 1j*n/r * besh(n,kL*r)

    return func_Vt1

def calc_Vt2(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Vt2 = -kT * besh_d(n,kT*r)

    return func_Vt2

def calc_T11(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_T11 = (n**2+n-0.5*(kT*r)**2) * besj(n,kL*r) - kL*r*besj(n-1,kL*r)

    func_T11 = func_T11 * 2*dm/r**2

    return func_T11

def calc_T12(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_T12 = -1j*n*( (n+1)*besj(n,kT*r) - kT*r*besj(n-1,kT*r)  )

    func_T12 = func_T12 * 2*dm/r**2

    return func_T12

def calc_T41(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_T41 = -1j*n*( (n+1)*besj(n,kL*r) - kL*r*besj(n-1,kL*r)  )
    
    func_T41 = func_T41 * 2*dm/r**2

    return func_T41

def calc_T42(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_T42 = -(n**2+n-0.5*(kT*r)**2) * besj(n,kT*r) + kT*r*besj(n-1,kT*r)

    func_T42 = func_T42 * 2*dm/r**2

    return func_T42

def calc_Tt11(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Tt11 = (n**2+n-0.5*(kT*r)**2) * besh(n,kL*r) - kL*r*besh(n-1,kL*r)

    func_Tt11 = func_Tt11 * 2*dm/r**2

    return func_Tt11

def calc_Tt12(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Tt12 = -1j*n*( (n+1)*besh(n,kT*r) - kT*r*besh(n-1,kT*r)  )

    func_Tt12 = func_Tt12 * 2*dm/r**2

    return func_Tt12

def calc_Tt41(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Tt41 = -1j*n*( (n+1)*besh(n,kL*r) - kL*r*besh(n-1,kL*r)  )

    func_Tt41 = func_Tt41 * 2*dm/r**2

    return func_Tt41

def calc_Tt42(n, r, w, rho, dl, dm):
    kL = w * np.sqrt(rho/(dl+2*dm))
    kT = w * np.sqrt(rho/(     dm))

    func_Tt42 = -(n**2+n-0.5*(kT*r)**2) * besh(n,kT*r) + kT*r*besh(n-1,kT*r)

    func_Tt42 = func_Tt42 * 2*dm/r**2

    return func_Tt42


def calc_M(n, r, w, ilayer, parameters_layers):
    rho = parameters_layers.rhos[ilayer]
    dl  = parameters_layers.dls[ilayer]
    dm  = parameters_layers.dms[ilayer]

    X = np.array([
        [calc_U1(n, r, w, rho, dl, dm), calc_U2(n, r, w, rho, dl, dm), calc_Ut1(n, r, w, rho, dl, dm), calc_Ut2(n, r, w, rho, dl, dm)],
        [calc_V1(n, r, w, rho, dl, dm), calc_V2(n, r, w, rho, dl, dm), calc_Vt1(n, r, w, rho, dl, dm), calc_Vt2(n, r, w, rho, dl, dm)],
        [calc_T11(n, r, w, rho, dl, dm), calc_T12(n, r, w, rho, dl, dm), calc_Tt11(n, r, w, rho, dl, dm), calc_Tt12(n, r, w, rho, dl, dm)],
        [calc_T41(n, r, w, rho, dl, dm), calc_T42(n, r, w, rho, dl, dm), calc_Tt41(n, r, w, rho, dl, dm), calc_Tt42(n, r, w, rho, dl, dm)]
    ], dtype=complex)

    return np.matrix(X)

# (A^i,B^i) -> (A^{i+1},B^{i+1})
def calc_S_layer(n, w, ilayer, parameters_layers):
    # X^{ilayer}(R^{ilayer})
    X1 = calc_M(n, parameters_layers.Rs[ilayer], w, ilayer, parameters_layers)
    # X^{ilayer+1}(R^{ilayer})
    X2 = calc_M(n, parameters_layers.Rs[ilayer], w, ilayer+1, parameters_layers)

    return np.linalg.inv(X2) @ X1

# # A^1 -> A^N
# def calc_S(n, w):
#     # 単位ベクトル
#     A1 = np.array([1.0,0.0], dtype=complex)
#     A2 = np.array([0.0,1.0], dtype=complex)

#     # S = np.zeros((2,2),dtype=complex)
#     S = None

#     for A,i in zip([A1,A2],range(2)):
#         vec = np.array([A[0], A[1], 0.0, 0.0], dtype=complex).reshape(4,1)

#         for ilayer in range(Nlayer-1):
#             vec = calc_S_layer(n, w, ilayer) @ vec
            
        

#         #S[:,i] = vec[0:2,0]
#         if S is None:
#             S = vec[0:2,0]
#         else:
#             S = np.hstack([S,vec[0:2,0]])

#     return S

def calc_X(n, w, parameters_layers):
    X = None

    # 各基底
    for basis in [np.array([1,0]), np.array([0,1])]:
        # [AL,AT,BL,BT] = [basis[0],basis[1],0,0]からスタート
        vec = np.array([basis[0], basis[1], 0.0, 0.0], dtype=complex).reshape(4,1)

        for ilayer in range(parameters_layers.Nlayer-2):
            vec = calc_S_layer(n, w, ilayer, parameters_layers) @ vec

        if X is None:
            X = vec[0:4,0]
        else:
            X = np.hstack([X,vec[0:4,0]])

    return X
        

def calc_Y_solid(n, w, parameters_solidBG, R):

    r = R
    rho = parameters_solidBG.rho_0
    dl = parameters_solidBG.dl_0
    dm = parameters_solidBG.dm_0

    X = np.array([
        [calc_Ut1(n, r, w, rho, dl, dm), calc_Ut2(n, r, w, rho, dl, dm)],
        [calc_Vt1(n, r, w, rho, dl, dm), calc_Vt2(n, r, w, rho, dl, dm)],
        [calc_Tt11(n, r, w, rho, dl, dm), calc_Tt12(n, r, w, rho, dl, dm)],
        [calc_Tt41(n, r, w, rho, dl, dm), calc_Tt42(n, r, w, rho, dl, dm)]
    ], dtype=complex)

    return np.matrix(X)

def calc_Z_solid(n, w, parameters_layers):
    r = parameters_layers.Rs[-1]
    rho = parameters_layers.rhos[-1]
    dl = parameters_layers.dls[-1]
    dm = parameters_layers.dms[-1]

    X = np.array([
        [calc_U1(n, r, w, rho, dl, dm), calc_U2(n, r, w, rho, dl, dm), calc_Ut1(n, r, w, rho, dl, dm), calc_Ut2(n, r, w, rho, dl, dm)],
        [calc_V1(n, r, w, rho, dl, dm), calc_V2(n, r, w, rho, dl, dm), calc_Vt1(n, r, w, rho, dl, dm), calc_Vt2(n, r, w, rho, dl, dm)],
        [calc_T11(n, r, w, rho, dl, dm), calc_T12(n, r, w, rho, dl, dm), calc_Tt11(n, r, w, rho, dl, dm), calc_Tt12(n, r, w, rho, dl, dm)],
        [calc_T41(n, r, w, rho, dl, dm), calc_T42(n, r, w, rho, dl, dm), calc_Tt41(n, r, w, rho, dl, dm), calc_Tt42(n, r, w, rho, dl, dm)]
    ], dtype=complex)

    return np.matrix(X)

def calc_Y_fluid(n, w, parameters_fluidBG, R):

    r = R # Rs[Nlayer-2]
    rho = parameters_fluidBG.rho_0
    kappa = parameters_fluidBG.kappa_0
    k = w * np.sqrt(rho/kappa)
    #dl = dl_0
    #dm = dm_0

    X = np.array([
        [-besh(n,k*r)],
        [0.0],
        [k/(rho*w**2) * besh_d(n,k*r)]
    ], dtype=complex)

    return np.matrix(X)

def calc_Z_fluid(n, w, parameters_layers):
    r = parameters_layers.Rs[-1]
    rho = parameters_layers.rhos[-1]
    dl = parameters_layers.dls[-1]
    dm = parameters_layers.dms[-1]

    X = np.array([
        [calc_T11(n, r, w, rho, dl, dm), calc_T12(n, r, w, rho, dl, dm), calc_Tt11(n, r, w, rho, dl, dm), calc_Tt12(n, r, w, rho, dl, dm)],
        [calc_T41(n, r, w, rho, dl, dm), calc_T42(n, r, w, rho, dl, dm), calc_Tt41(n, r, w, rho, dl, dm), calc_Tt42(n, r, w, rho, dl, dm)],
        [calc_U1(n, r, w, rho, dl, dm), calc_U2(n, r, w, rho, dl, dm), calc_Ut1(n, r, w, rho, dl, dm), calc_Ut2(n, r, w, rho, dl, dm)]
    ], dtype=complex)

    return np.matrix(X)