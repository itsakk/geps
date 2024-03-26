"""

Driven-damped pendulum equation:



    d^2(theta)/dt^2 + alpha*d(theta)/dt + w0*sin(theta) = f0*cos(wf*t)


    

we distinguish between the following cases:

    - ideal: alpha = 0, wf = , "f0": 00, f0 = 0

            d^2(theta)/dt^2 + w0*sin(theta) = 0

    - damped: f0 = 0, wf = 0

            d^2(theta)/dt^2 + alpha*d(theta)/dt + w0*sin(theta) = 0

        can showcase 3 different subcases: underdamped, critically damped, overdamped; depending on ratio w0/alpha

    
    - driven: alpha = 0
    
                d^2(theta)/dt^2 + w0*sin(theta) = f0*cos(wf*t)
    
        can showcase 3 different subcases: resonance, subresonance, superresonance; depending on ratio w0/wf
    
    
    - general: all parameters are non-zero

                d^2(theta)/dt^2 + alpha*d(theta)/dt + w0*sin(theta) = f0*cos(wf*t)

        here we can observe different kinds of phenomena:
            - beats
            - chaos (aperiodicity)
            - steady state motion (assymptotically)
            - transient motions
            - multistability (steady state dependent on IC)       

            
"""



params_ideal =     [{"alpha": 0., "w0": 0.10, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.15, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.20, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.25, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.30, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.35, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.40, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.45, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.50, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.55, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.60, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.65, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.70, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.75, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.80, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.85, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.90, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 0.95, "wf":0, "f0": 0},
                    {"alpha": 0., "w0": 1.00, "wf":0, "f0": 0}]


# w0/alpha < 1 ---> underdamped, w0/alpha = 1 ---> critically damped, w0/alpha > 1 ---> overdamped

params_damped =    [{"alpha": 0.05, "w0": 0.10, "wf":0, "f0": 0},
                    {"alpha": 0.10, "w0": 0.10, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.10, "wf":0, "f0": 0},
                    {"alpha": 0.05, "w0": 0.20, "wf":0, "f0": 0},
                    {"alpha": 0.10, "w0": 0.20, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.20, "wf":0, "f0": 0},
                    {"alpha": 0.10, "w0": 0.30, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.30, "wf":0, "f0": 0},
                    {"alpha": 0.40, "w0": 0.30, "wf":0, "f0": 0},
                    {"alpha": 0.10, "w0": 0.40, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.40, "wf":0, "f0": 0},
                    {"alpha": 0.40, "w0": 0.40, "wf":0, "f0": 0},
                    {"alpha": 0.10, "w0": 0.50, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.50, "wf":0, "f0": 0},
                    {"alpha": 0.40, "w0": 0.50, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.60, "wf":0, "f0": 0},
                    {"alpha": 0.30, "w0": 0.60, "wf":0, "f0": 0},
                    {"alpha": 0.40, "w0": 0.60, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.70, "wf":0, "f0": 0},
                    {"alpha": 0.40, "w0": 0.70, "wf":0, "f0": 0},
                    {"alpha": 0.60, "w0": 0.70, "wf":0, "f0": 0},
                    {"alpha": 0.20, "w0": 0.80, "wf":0, "f0": 0},
                    {"alpha": 0.40, "w0": 0.80, "wf":0, "f0": 0},
                    {"alpha": 0.60, "w0": 0.80, "wf":0, "f0": 0},
                    {"alpha": 0.30, "w0": 0.90, "wf":0, "f0": 0},
                    {"alpha": 0.70, "w0": 0.90, "wf":0, "f0": 0},
                    {"alpha": 0.90, "w0": 0.90, "wf":0, "f0": 0},
                    {"alpha": 0.50, "w0": 1.00, "wf":0, "f0": 0},
                    {"alpha": 0.80, "w0": 1.00, "wf":0, "f0": 0},
                    {"alpha": 1.00, "w0": 1.00, "wf":0, "f0": 0}]



# w0/wf < 1 ---> subresonance, w0/wf = 1 ---> resonance, w0/wf > 1 ---> superresonance


params_driven =    [{"alpha": 0., "w0": 0.10, "wf":0.05, "f0": 0.5},
                    {"alpha": 0., "w0": 0.10, "wf":0.10, "f0": 0.5},
                    {"alpha": 0., "w0": 0.10, "wf":0.20, "f0": 0.5},
                    {"alpha": 0., "w0": 0.20, "wf":0.10, "f0": 0.5},
                    {"alpha": 0., "w0": 0.20, "wf":0.20, "f0": 0.5},
                    {"alpha": 0., "w0": 0.20, "wf":0.30, "f0": 0.5},
                    {"alpha": 0., "w0": 0.30, "wf":0.10, "f0": 0.5},
                    {"alpha": 0., "w0": 0.30, "wf":0.30, "f0": 0.5},
                    {"alpha": 0., "w0": 0.30, "wf":0.50, "f0": 0.5},
                    {"alpha": 0., "w0": 0.40, "wf":0.20, "f0": 0.5},
                    {"alpha": 0., "w0": 0.40, "wf":0.40, "f0": 0.5},
                    {"alpha": 0., "w0": 0.40, "wf":0.60, "f0": 0.5},
                    {"alpha": 0., "w0": 0.50, "wf":0.10, "f0": 0.5},
                    {"alpha": 0., "w0": 0.50, "wf":0.30, "f0": 0.5},
                    {"alpha": 0., "w0": 0.50, "wf":0.50, "f0": 0.5},
                    {"alpha": 0., "w0": 0.60, "wf":0.20, "f0": 0.5},
                    {"alpha": 0., "w0": 0.60, "wf":0.55, "f0": 0.5},
                    {"alpha": 0., "w0": 0.60, "wf":1.00, "f0": 0.5},
                    {"alpha": 0., "w0": 0.70, "wf":0.10, "f0": 0.5},
                    {"alpha": 0., "w0": 0.70, "wf":0.69, "f0": 0.5},
                    {"alpha": 0., "w0": 0.70, "wf":0.99, "f0": 0.5},
                    {"alpha": 0., "w0": 0.80, "wf":0.20, "f0": 0.5},
                    {"alpha": 0., "w0": 0.80, "wf":0.60, "f0": 0.5},
                    {"alpha": 0., "w0": 0.80, "wf":0.80, "f0": 0.5},
                    {"alpha": 0., "w0": 0.90, "wf":0.10, "f0": 0.5},
                    {"alpha": 0., "w0": 0.90, "wf":0.50, "f0": 0.5},
                    {"alpha": 0., "w0": 0.90, "wf":0.90, "f0": 0.5},
                    {"alpha": 0., "w0": 1.00, "wf":0.20, "f0": 0.5},
                    {"alpha": 0., "w0": 1.00, "wf":1.00, "f0": 0.5},
                    {"alpha": 0., "w0": 1.00, "wf":2.00, "f0": 0.5}]


#  |w0 - wf| < alpha,  f0 << 1  ---> non-chaotic

params_damped_driven =[{"alpha": 0.2, "w0": 0.10, "wf":0.05, "f0": 0.1},
                    {"alpha": 0.3, "w0": 0.10, "wf":0.10, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.10, "wf":0.20, "f0": 0.1},
                    {"alpha": 0.3, "w0": 0.20, "wf":0.10, "f0": 0.1},
                    {"alpha": 0.4, "w0": 0.20, "wf":0.20, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.20, "wf":0.30, "f0": 0.1},
                    {"alpha": 0.3, "w0": 0.30, "wf":0.10, "f0": 0.1},
                    {"alpha": 0.3, "w0": 0.30, "wf":0.30, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.30, "wf":0.50, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.40, "wf":0.20, "f0": 0.1},
                    {"alpha": 0.1, "w0": 0.40, "wf":0.40, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.40, "wf":0.60, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.50, "wf":0.10, "f0": 0.1},
                    {"alpha": 0.4, "w0": 0.50, "wf":0.30, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.50, "wf":0.50, "f0": 0.1},
                    {"alpha": 0.6, "w0": 0.60, "wf":0.20, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.60, "wf":0.55, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.60, "wf":1.00, "f0": 0.1},
                    {"alpha": 0.6, "w0": 0.70, "wf":0.10, "f0": 0.1},
                    {"alpha": 0.7, "w0": 0.70, "wf":0.69, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.70, "wf":0.99, "f0": 0.1},
                    {"alpha": 0.6, "w0": 0.80, "wf":0.20, "f0": 0.1},
                    {"alpha": 0.7, "w0": 0.80, "wf":0.60, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.80, "wf":0.80, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.90, "wf":0.10, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.90, "wf":0.50, "f0": 0.1},
                    {"alpha": 0.5, "w0": 0.90, "wf":0.90, "f0": 0.1},
                    {"alpha": 0.9, "w0": 1.00, "wf":0.20, "f0": 0.1},
                    {"alpha": 0.5, "w0": 1.00, "wf":1.00, "f0": 0.1},
                    {"alpha": 0.8, "w0": 1.00, "wf":2.00, "f0": 0.1}]




# |w0 - wf| > alpha,  f0 >> 1  ---> chaotic


params_chaotic =   [{"alpha": 0.02, "w0": 0.10, "wf":0.05, "f0": 1.2},
                    {"alpha": 0.03, "w0": 0.10, "wf":0.10, "f0": 1.5},
                    {"alpha": 0.05, "w0": 0.10, "wf":0.20, "f0": 1.1},
                    {"alpha": 0.03, "w0": 0.20, "wf":0.10, "f0": 1.1},
                    {"alpha": 0.04, "w0": 0.20, "wf":0.20, "f0": 1.2},
                    {"alpha": 0.05, "w0": 0.20, "wf":0.30, "f0": 1.5},
                    {"alpha": 0.03, "w0": 0.30, "wf":0.10, "f0": 1.5},
                    {"alpha": 0.03, "w0": 0.30, "wf":0.30, "f0": 1.7},
                    {"alpha": 0.05, "w0": 0.30, "wf":0.50, "f0": 1.1},
                    {"alpha": 0.05, "w0": 0.40, "wf":0.20, "f0": 2.0},
                    {"alpha": 0.01, "w0": 0.40, "wf":0.40, "f0": 1.5},
                    {"alpha": 0.05, "w0": 0.40, "wf":0.60, "f0": 1.2},
                    {"alpha": 0.05, "w0": 0.50, "wf":0.10, "f0": 1.3},
                    {"alpha": 0.04, "w0": 0.50, "wf":0.30, "f0": 1.1},
                    {"alpha": 0.05, "w0": 0.50, "wf":0.50, "f0": 1.3},
                    {"alpha": 0.06, "w0": 0.60, "wf":0.20, "f0": 1.5},
                    {"alpha": 0.05, "w0": 0.60, "wf":0.55, "f0": 1.6},
                    {"alpha": 0.05, "w0": 0.60, "wf":1.00, "f0": 1.5},
                    {"alpha": 0.06, "w0": 0.70, "wf":0.10, "f0": 2.0},
                    {"alpha": 0.07, "w0": 0.70, "wf":0.69, "f0": 1.7},
                    {"alpha": 0.05, "w0": 0.70, "wf":0.99, "f0": 1.8},
                    {"alpha": 0.06, "w0": 0.80, "wf":0.20, "f0": 1.3},
                    {"alpha": 0.07, "w0": 0.80, "wf":0.60, "f0": 1.2},
                    {"alpha": 0.05, "w0": 0.80, "wf":0.80, "f0": 1.1},
                    {"alpha": 0.05, "w0": 0.90, "wf":0.10, "f0": 1.7},
                    {"alpha": 0.05, "w0": 0.90, "wf":0.50, "f0": 1.8},
                    {"alpha": 0.05, "w0": 0.90, "wf":0.90, "f0": 1.3},
                    {"alpha": 0.09, "w0": 1.00, "wf":0.20, "f0": 1.2},
                    {"alpha": 0.05, "w0": 1.00, "wf":1.00, "f0": 1.8},
                    {"alpha": 0.08, "w0": 1.00, "wf":2.00, "f0": 1.5}]



params_train =     [# DAMPED
                    {"alpha": 0.30, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Underdamped
                    {"alpha": 0.50, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Critically damped
                    {"alpha": 1.50, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Overdamped
                    
                    # DRIVEN
                    {"alpha": 0.00, "w0": 0.50, "wf":0.10, "f0": 0.2}, # Subresonance
                    {"alpha": 0.00, "w0": 0.50, "wf":0.50, "f0": 0.2}, # Resonance
                    {"alpha": 0.00, "w0": 0.50, "wf":0.80, "f0": 0.2}, # Beats
                    {"alpha": 0.00, "w0": 0.50, "wf":1.30, "f0": 0.2}] # Superresonance




params_test =       [# DRIVEN-DAMPED: transient + steady state
                    {"alpha": 0.05, "w0": 1.00, "wf":0.30, "f0": 0.1},
                    {"alpha": 0.05, "w0": 1.00, "wf":2.00, "f0": 0.1},
                    # DRIVEN-DAMPED: chaotic
                    {"alpha": 0.10, "w0": 1.00, "wf":1.00, "f0": 1.5},
                    {"alpha": 0.10, "w0": 1.00, "wf":1.23, "f0": 1.5}]