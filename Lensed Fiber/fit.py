# def rabi(p, x):
#     O0 = params.get('Omega', 1)
#     T = params.get('T', np.pi)
#     a = params.get('a', 1)
#     b = params.get('b', 0)
#     x0 = params.get('x0', 0)
#     O = np.sqrt(O0**2 + (2 * np.pi * (x - x0))**2)
#     return a * O0**2 * np.sin(O * T / 2.)**2 / O**2 + b

# p = {
#     'Omega': {'value': 1}, 
#     'T': {'value': np.pi}, 
#     'a': {'value': 1}, 
#     'b': {'value': 0}, 
#     'x0': {'value': 0}
#     }

# params = paramdict_to_lmfitparams(p)
# x = np.linspace(-np.pi, np.pi, 100)
# y = rabi(params, x) + 0.002 * np.random.randn(x.size)

# fig, ax = plt.subplots(1)
# ax.plot(x, y)

# out = fit_ax(ax, rabi, p, x_range=(-1, 1))

# print out.params['a']

import numpy  as np
import lmfit

def residual_from_model(model):
    def residual(params, x, data, eps_data=None):
        if eps_data == None:
            eps_data = np.ones_like(data)
        return (data - model(params, x)) / eps_data
    return residual

def get_add_many_tuple(name, settings):
    value = settings['value']
    vary = settings.get('vary', True)
    min_ = settings.get('min', None)
    max_ = settings.get('max', None)
    expr = settings.get('expr', None)
    brute_step = settings.get('brute_step', None)

    return (name, value, vary, min_, max_, expr, brute_step)

def paramdict_to_lmfitparams(paramdict):
    params = lmfit.Parameters()
    params.add_many(*(get_add_many_tuple(name, settings) for name, settings in paramdict.items()))
    return params

#def fit_ax(ax, model, p, x_range=(None, None), show_guess=0, line_choice=-1, color='r', alpha = 1):

def fit(model, p, x, y, x_range=(None, None),show_guess=0, alpha = 1):
    params = paramdict_to_lmfitparams(p)
    
    # get x and y data from figure
    # create original figure first then generate the fitline
    #print(ax.get_lines)
    #x, y = ax.get_lines()[line_choice].get_xydata().T  

   # decide over what range to fit
    if x_range[0] == None:
        fit_min = x.min()
    else:
        fit_min = x_range[0]
    if x_range[1] == None:
        fit_max = x.max()
    else:
        fit_max = x_range[1]
        
    x_plot = np.linspace(fit_min, fit_max, 1000)
               
    i_fit = np.argwhere((fit_min <= x) & (x <= fit_max))

    residual = residual_from_model(model)
    x_fit = x[i_fit]
    y_fit = y[i_fit]
    
    out = lmfit.minimize(residual, params, args=(x_fit, y_fit, None))
    
    # if mirror == False:
    #     ax.plot(x_plot, model(out.params, x_plot), color, alpha = alpha)
    # else:
    #     ax.plot(model(out.params, x_plot),x_plot,  color, alpha = alpha)

    return x_plot,model,out

def linear2(params, x):
    a = params['a']
    b = params['b']
    return a * x + b

""" fit functions """

def exponential(params, x):
    a = params['a']
    b = params['b']
    tau = params['tau']
    return a * np.exp(-x/tau) + b

def lorentzian(p):
    return lambda f: p['a'] / (1.0 + (2.0 * (f - p['x0']) / p['Gamma'])**2) + p['b']

#def linear(p):
#    return lambda x: p['a'] * x + p['b']

def sine(p):
    a = p.get('a', 1)
    b = p.get('b', 0)
    f = p.get('f', 1)
    phi = p.get('phi', 0)
    return lambda t: a * np.sin(2 * np.pi * f * t + phi) + b

def sine_exp(p):
    a = p.get('a', 1)
    b = p.get('b', 0)
    f = p.get('f', 1)
    phi = p.get('phi', 0)
    tau = p.get('tau', np.inf)
    return lambda t: a * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + phi) + b

def sin2(p):
    return lambda y: p['a'] * np.sin(np.pi * y / p['c'] + p['phi']) + p['b']
    
def rabi(p):
    O0 = p.get('Omega', 1)
    T = p.get('T', 1)
    a = p.get('a', 1)
    b = p.get('b', 0)
    x0 = p.get('x0', 0)
    
    def r(x):
        O = np.sqrt(O0**2 + (2 * np.pi * (x - x0))**2)
        return a * O0**2 * np.sin(O * T / 2.)**2 / O**2 + b
    return r    

def lorentzian(params, x):
    a = params.get('a', 1)
    b = params.get('b', 0)
    x0 = params.get('x0', 0)
    Gamma = params.get('Gamma', 1)
    return a / (1.0 + (2.0 * (x - x0) / Gamma)**2) + b


def gaussian(params, x):
    a = params.get('a', 1)
    b = params.get('b', 0)
    x0 = params.get('x0', 0)
    sigma = params.get('sigma', 1)
    return a * np.exp(-0.5*((x - x0) / sigma)**2) + b

def lorentzian2(params, x):
    a = params.get('a', 1)
    a2 = params.get('a2', 1)
    b = params.get('b', 0)
    x0 = params.get('x0', 0)
    delta = params.get('delta', 0)
    Gamma = params.get('Gamma', 1)
    Gamma2 = params.get('Gamma2', 1)
    return a / (1.0 + (2.0 * (x - x0 +delta/2.) / Gamma)**2)+a2 / (1.0 + (2.0 * (x - x0-delta/2.) / Gamma2)**2) + b

def rabi(params, x):
    O0 = params.get('Omega', 1)
    T = params.get('T', 1)
    a = params.get('a', 1)
    b = params.get('b', 0)
    x0 = params.get('x0', 0)
    O = np.sqrt(O0**2 + (2 * np.pi * (x - x0))**2)
    return a * O0**2 * np.sin(O * T / 2.)**2 / O**2 + b

def rabi_flop(params, x):
    O0 = params.get('Omega', 1)
    a = params.get('a', 1)
    b = params.get('b', 0)
    return a * np.sin(O0 * x)**2 + b

def sin2(params, x):
    O = params.get('Omega', 1)
    a = params.get('a', 1)
    b = params.get('b', 0)
    phi = params.get('phi', 1)
    return a * np.sin(O * x + phi)**2 + b

def linear(params, x):
    a = params.get('a', 1)
    b = params.get('b', 0)
    return a * x + b


def troid_coupling(params, x):
    G0 = params.get('G0', 1)
    Ge = params.get('Ge', 1)
    a = params.get('a', 1)
    b = params.get('b', 0)    
    d = params.get('d', 0)   
    return a*((G0 - Ge)**2+4*(x-d)**2)/((G0 + Ge)**2+4*(x-d)**2)+b

def fano(params, x):
    G0 = params.get('G0', 1)
    q = params.get('q', 1)
    a = params.get('a', 1)
    b = params.get('b', 0)    
    d = params.get('d', 0)   
    return a*(2*(x-d)/(G0)+q)**2/((2*(x-d)/G0)**2+1)+b