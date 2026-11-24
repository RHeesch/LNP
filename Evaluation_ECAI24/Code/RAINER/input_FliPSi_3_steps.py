import pandas as pd

integrate_ml = True
cc = False
approx = True

# Tabelle der fixen parameter als 0.0 angeben, da ansonsten Probleme mit Z3 Integer
Parameter_fix = ["volume", "density", "specific_heat_capacity"]
#Values_fix = pd.Series([1.5, 7.870, 4.77], index = Parameter_fix) #Stahl
#Values_fix = pd.Series([1.5, 2.700, 8.96], index = Parameter_fix) #Alu
Values_fix = pd.Series([1.5, 8.730, 3.77], index = Parameter_fix) #Messing
para_fix = pd.concat([Values_fix], axis = 1)
Grenzen_fix = ['Values_fix']
para_fix.columns = Grenzen_fix

# Tabelle der variablen parameter
Parameter_var = ["energy_input"]
Top_var = pd.Series([1000000], index = Parameter_var)
Down_var = pd.Series([-1000000], index = Parameter_var)
para_var = pd.concat([Top_var, Down_var], axis = 1)
Grenzen_var = ["Top_var", "Down_var"]
para_var.columns = Grenzen_var

Variables = ["coloured", "temperature", "milled", "drilled", "rounded"]
sym_actions = ["Mill", "Drill", "CNC", "Paint", "Heat"]
learn_actions = ["L_Heat"]

# Tabelle der Effekte der Aktionen
Mill_eff = pd.Series(["0.0", "0.0", "10", "0.0", "0.0"], index=Variables)
Drill_eff = pd.Series(["0", "0", "0", "10", "0"], index=Variables)
CNC_eff = pd.Series(["0", "0", "0", "0", "10"], index=Variables)
Paint_eff = pd.Series(["10", "0", "0", "0", "0"], index=Variables)
Heat_eff = pd.Series(["0", "(energy_input*1000000)*((volume*density*specific_heat_capacity*1000*100)**(-1))", "0", "0", "0"], index=Variables)
effects = pd.concat([Mill_eff, Drill_eff, CNC_eff, Paint_eff, Heat_eff], axis = 1)
effects.columns = sym_actions

if integrate_ml == True:
    
    #Tabelle der Modelle
    learned_models = ["L_Heat"]
    trained_model = pd.Series(["model(state_in)"], index = learned_models)
    ld_models = pd.concat([trained_model], axis = 1)
    ld_models.columns = learned_models

    # Tabelle der Preconditions der gelernten Aktionen
    L_Heat_pre = pd.Series(["<=10", "<=100.0", "", "", ""], index=Variables)
    L_precons = pd.concat([L_Heat_pre], axis = 1)
    L_precons.columns = learn_actions

# Tabelle der Preconditions der Aktionen
Mill_pre = pd.Series(["0.0", "", "0.0", "", ""], index=Variables)
Drill_pre = pd.Series(["", "", "", "0.0", "0.0"], index=Variables)
CNC_pre = pd.Series(["", "", "", "", "0.0"], index=Variables)
Paint_pre = pd.Series(["0.0", ">=20", "", "", ""], index=Variables)
Heat_pre = pd.Series(["<=10", ">=100.0", "", "", ""], index=Variables)
precons = pd.concat([Mill_pre, Drill_pre, CNC_pre, Paint_pre, Heat_pre], axis = 1)
precons.columns = sym_actions

if cc == True:
    # Tabelle der Kostenfunktionen
    Cost_factors = ["Dauer", "Energie"]
    Wasser_kochen_Cost = pd.Series(["zugeführte_Energie_Wasserkocher*((4190*Wassermenge)**(-1))", "0"], index=Cost_factors)
    #Wasser_kochen_Cost = pd.Series(["4190", "0"], index=Cost_factors)
    Kaffee_mahlen_Cost = pd.Series(["25", "Kaffeemenge * Mahlgeschwindigkeit"], index=Cost_factors)
    Kaffee_kochen_Cost = pd.Series(["25", "35"], index=Cost_factors)
    Kaffee_abkuehlen_Cost = pd.Series(["0", "0"], index=Cost_factors)
    cost_function = pd.concat([Wasser_kochen_Cost, Kaffee_mahlen_Cost, Kaffee_kochen_Cost, Kaffee_abkuehlen_Cost], axis = 1)
    cost_function.columns = sym_actions

# Tabelle Initial- & Endzustand
Init = pd.Series(["0.0", "5.0", "0.0", "0.0", "0.0"], index=Variables)
Goal = pd.Series(["10", "20.0", "10", "0", "0"], index=Variables)
start_end = pd.concat([Init, Goal], axis = 1)

def print_all():
    #results = open('Results', 'a')
    #results.write('\r\n')
    #results.write(para_fix)
    print(para_fix)
    #results.write('\r\n')
    #results.write(para_var)
    print(para_var)
    #results.write('\r\n')
    #results.write(precons)
    print(precons)
    #print(ld_models)
    #print(L_precons)
    #results.write('\r\n')
    #results.write(effects)
    print(effects)
    #results.write('\r\n')
    #results.write(start_end)
    print(start_end)
    #results.write('\r\n')
    #results.write(cost_function)
    #print(cost_function)

print_all()

#sep_cost = cost_function.iloc[0, 0]
#print(sep_cost)