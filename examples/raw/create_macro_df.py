'''This file creates the macroeconomic data series from raw files and saves to examples/data/macro.pkl.'''
import numpy as np
import pandas as pd
import os
import pickle
from collections import deque

dtVarDict = pd.read_excel('examples/raw/var_dict.xlsx')
dtVarDict['var'] = ''

dtMacro = None
dtfilegroup = dtVarDict.groupby('file')
for crntFileName, crntFileInfo in dtfilegroup:
    crntDF = pd.read_excel(
        os.path.join('examples/raw', crntFileName)
        , skiprows=range(0,crntFileInfo.iloc[0]['rowstart'])
        , usecols=[crntFileInfo.iloc[0]['datecol']]+list(crntFileInfo['varcol'].unique())
    )        
    varnames = list(crntDF.columns[1:])
    crntDF.columns = ['date'] + varnames
    crntDF = crntDF.dropna()
    crntDF['date'] = crntDF['date'].dt.date

    dtVarDict.loc[crntFileInfo.index, 'var'] = varnames

    if dtMacro is None:
        dtMacro = crntDF
    else:
        dtMacro = pd.merge(dtMacro, crntDF, on='date')

vardict_temp = {'var':deque(), 'group':['stock']*8, 'transformation':['N/A','LND1']*4, 'source': ['']*8}
for exVar, stockVar in zip(['EXCAUS', 'EXJPUS', 'EXUSUK', None], ['TSX','N225','FTSE100', 'SP500']):
    usd_var_name = stockVar + '_DfUSD'
    if exVar is None:
        dtMacro[usd_var_name] = dtMacro[stockVar] / dtMacro['CPI_US_CORE']
    elif exVar == 'EXUSUK':
        dtMacro[usd_var_name] = dtMacro[stockVar] * dtMacro[exVar] / dtMacro['CPI_US_CORE']
    else:
        dtMacro[usd_var_name] = dtMacro[stockVar] / dtMacro[exVar] / dtMacro['CPI_US_CORE']
    vardict_temp['var'].append(usd_var_name)

    lnd_var_name = usd_var_name + '_LND'
    dtMacro[lnd_var_name] = np.log(dtMacro[usd_var_name] / dtMacro[usd_var_name].shift(1))
    vardict_temp['var'].append(lnd_var_name)
dtVarDict = pd.concat([dtVarDict, pd.DataFrame(vardict_temp)], axis=0)
    

dtVarDict.loc[dtVarDict['transformation'].isna(), 'transformation'] = 'N/A'
dtVarDict = dtVarDict[['var', 'group', 'transformation', 'source']].sort_values(['group', 'transformation', 'var']).reset_index(drop=True)

# move columns in the same order as they appear in dtVarDict
dtMacro = dtMacro[['date']+dtVarDict['var'].values.tolist()]

# pfile = pd.HDFStore('examples/data/macro.dt', 'w')
# pfile.put('dtMacro', dtMacro)
# pfile.put('dtVarDict', dtVarDict[['var', 'group', 'transformation', 'source']])
# pfile.close()

with open('examples/data/macro.pkl', 'wb') as hMacroFile:
    pickle.dump([dtMacro, dtVarDict[['var', 'group', 'transformation', 'source']]], file=hMacroFile)

#dtMacro.to_excel('aa.xlsx')