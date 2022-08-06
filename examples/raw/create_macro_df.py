import pandas as pd
import os
import pickle

dtVarDict = pd.read_excel('examples/raw/var_dict.xlsx')
dtVarDict['var'] = ''

dtMacro = None
for iRow, crntRow in dtVarDict.iterrows():
    crntDF = pd.read_excel(
        os.path.join('examples/raw', crntRow['file'])
        , skiprows=range(0,crntRow['rowstart'])
        , usecols=[crntRow['datecol'], crntRow['varcol']]
    )
    varName = crntDF.columns[1]
    crntDF.columns = ['date', varName]
    crntDF = crntDF.dropna()
    crntDF['date'] = crntDF['date'].dt.date

    dtVarDict.loc[iRow, 'var'] = varName

    if dtMacro is None:
        dtMacro = crntDF
    else:
        dtMacro = pd.merge(dtMacro, crntDF, on='date')

with open('examples/data/macro.pkl', 'wb') as hMacroFile:
    pickle.dump([dtMacro, dtVarDict], file=hMacroFile)