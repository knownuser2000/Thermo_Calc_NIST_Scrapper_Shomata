# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:36:17 2025

@author: Bayra
"""
import numpy as np
#from mixture_averaging import convert_to_mole_fractions, average_molar_mass

from unit_conversion import to_SI

def solve_ideal_gas(
    P=None, P_unit=None,
    V=None, V_unit=None,
    T=None, T_unit=None,
    n=None, n_unit=None,
    c=None, c_unit=None,
    m=None, m_unit=None,
    M=None, M_unit=None,
    rho=None, rho_unit=None,
    R=8.3145
):
    inputs = {
        "P": to_SI(P, P_unit) if P is not None and P_unit else P,
        "V": to_SI(V, V_unit) if V is not None and V_unit else V,
        "T": to_SI(T, T_unit) if T is not None and T_unit else T,
        "n": to_SI(n, n_unit) if n is not None and n_unit else n,
        "c": to_SI(c, c_unit) if c is not None and c_unit else c,
        "m": to_SI(m, m_unit) if m is not None and m_unit else m,
        "M": to_SI(M, M_unit) if M is not None and M_unit else M,
        "rho": to_SI(rho, rho_unit) if rho is not None and rho_unit else rho,
    }
    inputs = {k: (np.asarray(v) if v is not None else None) for k, v in inputs.items()}

    # Rules paired with the variable they compute
    rules = [
        ("n", lambda v: v["c"] * v["V"] if v["n"] is None and v["c"] is not None and v["V"] is not None else None),
        ("n", lambda v: (v["rho"] * v["V"]) / v["M"] if v["n"] is None and v["rho"] is not None and v["V"] is not None and v["M"] is not None else None),
        ("n", lambda v: v["m"] / v["M"] if v["n"] is None and v["m"] is not None and v["M"] is not None else None),

        ("m", lambda v: v["n"] * v["M"] if v["m"] is None and v["n"] is not None and v["M"] is not None else None),

        ("rho", lambda v: v["m"] / v["V"] if v["rho"] is None and v["m"] is not None and v["V"] is not None else None),

        ("P", lambda v: v["n"] * R * v["T"] / v["V"] if v["P"] is None and v["n"] is not None and v["T"] is not None and v["V"] is not None else None),
        ("V", lambda v: v["n"] * R * v["T"] / v["P"] if v["V"] is None and v["n"] is not None and v["T"] is not None and v["P"] is not None else None),
        ("T", lambda v: v["P"] * v["V"] / (v["n"] * R) if v["T"] is None and v["P"] is not None and v["V"] is not None and v["n"] is not None else None),

        ("c", lambda v: v["n"] / v["V"] if v["c"] is None and v["n"] is not None and v["V"] is not None else None),
    ]

    max_iter = 10
    for _ in range(max_iter):
        updated = False
        for var, rule in rules:
            if inputs[var] is None:
                val = rule(inputs)
                if val is not None:
                    inputs[var] = val
                    updated = True
        if not updated:
            break

    known = sum(inputs[k] is not None for k in ["P", "V", "T", "n"])
    if known < 3:
        raise ValueError("Insufficient inputs to solve ideal gas equation. Provide at least 3 of P, V, T, n.")

    return {
        "P (Pa)": inputs["P"],
        "V (m³)": inputs["V"],
        "T (K)": inputs["T"],
        "n (mol)": inputs["n"],
        "c (mol/m³)": inputs["c"],
        "m (kg)": inputs["m"],
        "M (kg/mol)": inputs["M"],
        "rho (kg/m³)": inputs["rho"]
    }

def solve_ideal_gas_mixture(
    P=None, P_unit=None,
    V=None, V_unit=None,
    T=None, T_unit=None,
    n_i=None, n_i_unit=None,
    y=None,                        # mole fractions
    w=None,                        # mass fractions
    p_i=None, p_i_unit=None,       # partial pressures
    m_i=None, m_i_unit=None,       # masses
    M_i=None, M_i_unit=None,       # molar masses
    rho_unit=None,
    R=8.3145,
    to_SI=None                     # your unit conversion function
):
    # --- Unit conversion ---
    P = to_SI(P, P_unit) if to_SI else P
    V = to_SI(V, V_unit) if to_SI else V
    T = to_SI(T, T_unit) if to_SI else T
    n_i = to_SI(n_i, n_i_unit) if to_SI else n_i
    p_i = to_SI(p_i, p_i_unit) if to_SI else p_i
    m_i = to_SI(m_i, m_i_unit) if to_SI else m_i
    M_i = to_SI(M_i, M_i_unit) if to_SI else M_i

    # --- Calculate mole fractions using the helper ---
    y = convert_to_mole_fractions(y=y, w=w, p_i=p_i, P=P, M_i=M_i,)

    # --- Determine n_i if not provided ---
    if n_i is None:
        if V is not None and T is not None:
            if p_i is not None:
                n_i = p_i * V / (R * T)
            elif P is not None:
                n_total = P * V / (R * T)
                n_i = y * n_total

    n_total = np.sum(n_i) if n_i is not None else None

    # --- Molar mass ---
    M_mix = average_molar_mass(y, M_i) if M_i is not None else None

    # --- Masses ---
    if m_i is None and M_i is not None and n_i is not None:
        m_i = n_i * M_i
    m_total = np.sum(m_i) if m_i is not None else None

    # --- Density & Concentration ---
    rho = m_total / V if (m_total is not None and V is not None) else None
    c = n_total / V if (n_total is not None and V is not None) else None

    # --- Calculate missing state variables ---
    if P is None and T is not None and V is not None and n_total is not None:
        P = n_total * R * T / V
    if V is None and P is not None and T is not None and n_total is not None:
        V = n_total * R * T / P
    if T is None and P is not None and V is not None and n_total is not None:
        T = P * V / (n_total * R)

    # --- Return results ---
    return {
        "P (Pa)": P, "V (m³)": V, "T (K)": T,
        "n_i (mol)": n_i, "n_total (mol)": n_total, "y (mol/mol)": y,
        "m_i (kg)": m_i, "m_total (kg)": m_total,
        "M_i (kg/mol)": M_i, "M_mix (kg/mol)": M_mix,
        "rho (kg/m³)": rho, "c (mol/m³)": c
    }

def solve_process_eos(
    process_type: str,           # "polytropic", "isentropic_pv", etc.
    P1=None, V1=None, T1=None,
    P2=None, V2=None, T2=None,
    n=None,                      # polytropic exponent
    gamma=None,                  # isentropic exponent
    c_p=None,                    # [J/mol/K]
    c_v=None,                    # [J/mol/K]
    R=8.3145                     # [J/mol/K]
):
    """
    General EOS process solver for ideal gas processes.
    Supports: polytropic, isentropic_pv, isentropic_tv, isentropic_tp,
              isothermal, isobaric, isochoric.
    """

    def compute_gamma_fallback():
        if gamma is not None:
            return gamma
        if c_p and c_v:
            return c_p / c_v
        elif c_p:
            return c_p / (c_p - R)
        elif c_v:
            return 1 / (1 - R / c_v)
        raise ValueError("Gamma or c_p/c_v must be provided for isentropic process.")

    process_type = process_type.lower()

    if process_type == "polytropic":
        if n is None:
            raise ValueError("Polytropic process requires exponent 'n'.")
        return solve_polytropic_eos(P1=P1, V1=V1, P2=P2, V2=V2, n=n)

    elif process_type == "isentropic_pv":
        gamma = compute_gamma_fallback()
        return solve_isentropic_PV(P1=P1, V1=V1, P2=P2, V2=V2, gamma=gamma)

    elif process_type == "isentropic_tv":
        gamma = compute_gamma_fallback()
        return solve_isentropic_TV(T1=T1, V1=V1, T2=T2, V2=V2, gamma=gamma)

    elif process_type == "isentropic_tp":
        gamma = compute_gamma_fallback()
        return solve_isentropic_TP(T1=T1, P1=P1, T2=T2, P2=P2, gamma=gamma)

    elif process_type == "isothermal":
        return solve_isothermal_eos(P1=P1, V1=V1, P2=P2, V2=V2)

    elif process_type == "isobaric":
        return solve_isobaric_eos(V1=V1, T1=T1, V2=V2, T2=T2)

    elif process_type == "isochoric":
        return solve_isochoric_eos(P1=P1, T1=T1, P2=P2, T2=T2)

    else:
        raise ValueError(f"Unsupported process type: '{process_type}'")

def solve_relation(variables, equation_map):
    """
    Generic solver for simple 2-variable equations where one variable is missing.
    """
    missing = [k for k, v in variables.items() if v is None]
    if len(missing) != 1:
        raise ValueError("Exactly one variable must be None.")
    key = missing[0]
    return {key: equation_map[key](variables)}

def solve_isothermal_eos(P1=None, V1=None, P2=None, V2=None):
    variables = {"P1": P1, "V1": V1, "P2": P2, "V2": V2}
    equations = {
        "P1": lambda v: v["P2"] * v["V2"] / v["V1"],
        "V1": lambda v: v["P2"] * v["V2"] / v["P1"],
        "P2": lambda v: v["P1"] * v["V1"] / v["V2"],
        "V2": lambda v: v["P1"] * v["V1"] / v["P2"]
    }
    return solve_relation(variables, equations)

def solve_isochoric_eos(P1=None, T1=None, P2=None, T2=None):
    variables = {"P1": P1, "T1": T1, "P2": P2, "T2": T2}
    equations = {
        "P1": lambda v: v["P2"] * v["T1"] / v["T2"],
        "T1": lambda v: v["T2"] * v["P1"] / v["P2"],
        "P2": lambda v: v["P1"] * v["T2"] / v["T1"],
        "T2": lambda v: v["T1"] * v["P2"] / v["P1"]
    }
    return solve_relation(variables, equations)

def solve_isobaric_eos(V1=None, T1=None, V2=None, T2=None):
    variables = {"V1": V1, "T1": T1, "V2": V2, "T2": T2}
    equations = {
        "V1": lambda v: v["V2"] * v["T1"] / v["T2"],
        "T1": lambda v: v["T2"] * v["V1"] / v["V2"],
        "V2": lambda v: v["V1"] * v["T2"] / v["T1"],
        "T2": lambda v: v["T1"] * v["V2"] / v["V1"]
    }
    return solve_relation(variables, equations)

def solve_polytropic_eos(P1=None, V1=None, P2=None, V2=None, n=None):
    if n is None:
        raise ValueError("Polytropic exponent 'n' must be provided.")
    
    variables = {"P1": P1, "V1": V1, "P2": P2, "V2": V2}
    
    equations = {
        "P1": lambda v: v["P2"] * (v["V2"] / v["V1"]) ** n,
        "V1": lambda v: v["V2"] * (v["P2"] / v["P1"]) ** (1 / n),
        "P2": lambda v: v["P1"] * (v["V1"] / v["V2"]) ** n,
        "V2": lambda v: v["V1"] * (v["P1"] / v["P2"]) ** (1 / n)
    }
    
    return solve_relation(variables, equations)
    
def solve_isentropic_TV(T1=None, V1=None, T2=None, V2=None, gamma=None):
    if gamma is None:
        raise ValueError("Isentropic exponent 'gamma' must be provided.")
    
    variables = {"T1": T1, "V1": V1, "T2": T2, "V2": V2}
    
    equations = {
        "T1": lambda v: v["T2"] * (v["V2"] / v["V1"]) ** (gamma - 1),
        "V1": lambda v: v["V2"] * (v["T2"] / v["T1"]) ** (1 / (gamma - 1)),
        "T2": lambda v: v["T1"] * (v["V1"] / v["V2"]) ** (gamma - 1),
        "V2": lambda v: v["V1"] * (v["T1"] / v["T2"]) ** (1 / (gamma - 1))
    }
    
    return solve_relation(variables, equations)

def solve_isentropic_TP(T1=None, P1=None, T2=None, P2=None, gamma=None):
    if gamma is None:
        raise ValueError("Isentropic exponent 'gamma' must be provided.")
    
    variables = {"T1": T1, "P1": P1, "T2": T2, "P2": P2}
    
    equations = {
        "T1": lambda v: v["T2"] * (v["P1"] / v["P2"]) ** ((1 - gamma) / gamma),
        "P1": lambda v: v["P2"] * (v["T2"] / v["T1"]) ** (gamma / (1 - gamma)),
        "T2": lambda v: v["T1"] * (v["P2"] / v["P1"]) ** ((1 - gamma) / gamma),
        "P2": lambda v: v["P1"] * (v["T1"] / v["T2"]) ** (gamma / (1 - gamma))
    }
    
    return solve_relation(variables, equations)

def solve_isentropic_PV(P1=None, V1=None, P2=None, V2=None, gamma=None):
    if gamma is None:
        raise ValueError("Isentropic exponent 'gamma' must be provided.")
    
    variables = {"P1": P1, "V1": V1, "P2": P2, "V2": V2}
    
    equations = {
        "P1": lambda v: v["P2"] * (v["V2"] / v["V1"]) ** gamma,
        "V1": lambda v: v["V2"] * (v["P2"] / v["P1"]) ** (1 / gamma),
        "P2": lambda v: v["P1"] * (v["V1"] / v["V2"]) ** gamma,
        "V2": lambda v: v["V1"] * (v["P1"] / v["P2"]) ** (1 / gamma)
    }
    
    return solve_relation(variables, equations)


# Example wrapper function to get all properties from any input form:


# 1 mol gas at 25°C, 1 bar, what’s the volume?
result = solve_ideal_gas(P=101325, V=0.01, T=298)
print(result)

# Using density instead of mass
#result = solve_ideal_gas(T=[300, 310], V=[0.01, 0.02], c=[1, 2], c_unit="mol/L")
#print(result)

a= solve_process_eos("polytropic", P1=1e5, V1=1.0, V2=0.5, n=1.3)
b= solve_process_eos("isentropic_pv", P1=1e5, V1=1.0, V2=0.5, c_p=29.1, c_v=20.8)
c= solve_process_eos("isothermal", P1=1e5, V1=1.0, V2=2.0)
d= solve_process_eos("isochoric", P1=1e5, T1=300, T2=600)
