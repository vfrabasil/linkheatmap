from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import base64

from time import mktime
from datetime import datetime


#import plotly.express as px

#Modo de Ejecucion:
#streamlit run HeatmapStreamlit.py
#
#From Heroku
#https://link-heatmap.herokuapp.com/

# Global parameters
pd.set_option('display.max_columns', 100)
debug = False
stlit = True



# Setting Cache for dataset:
@st.cache
def load_dataset(pre_file ,post_file):
    #dfOld = pd.read_csv('rcardOld.csv', sep=';')
    #dfNew = pd.read_csv('rcardNew.csv', sep=';')

    dfOld = pd.read_csv(pre_file, sep=';')
    dfNew = pd.read_csv(post_file, sep=';')
    return dfOld, dfNew



def main():
    pre_file = None
    post_file = None
    count = 1

    if stlit == True:
        st.title("Seleccionar Dataframes:")

        with st.beta_expander("Cargar Archivos CSV :", expanded=True):
            pre_file  = st.file_uploader("Seleccion PRE implementacion:", 
                                        accept_multiple_files=False,
                                        type='csv', key = count)
            count += 1                        
            post_file  = st.file_uploader("Seleccion POST implementacion:", 
                                        accept_multiple_files=False,
                                        type='csv', key = count)

            if st.checkbox('Usar CSV de ejemplo'):
                pre_file = 'rcard1104.csv'
                post_file = 'rcard1105.csv'

    else:
        pre_file = 'rcard1104.csv'
        post_file = 'rcard1105.csv'




    if pre_file is not None and post_file is not None:
        dfOld, dfNew = load_dataset(pre_file, post_file )

        if stlit == True:
            page = st.sidebar.selectbox("Seleccionar opcion", ['Analisis', 'Exploracion', 'Prediccion'])

            if page == 'Analisis':
                muestra = 30
                if st.checkbox('Muestra de los archivos cargados:'):
                    #st.header('Muestra de los DataSet:')
                    st.subheader("PRE implementacion:")
                    st.dataframe(dfOld.head(muestra))
                    st.subheader("POS implementacion:")
                    st.dataframe(dfNew.head(muestra))
                st.title('Analisis por mapa de calor')
                runTest(dfNew, dfOld)

            elif page == 'Exploracion':
                st.title('Datos exploratorios del Data-set: {}'.format("Post Implementacion"))
                if st.checkbox('Descripcion:'):
                    st.dataframe(dfNew.describe())
                if st.checkbox("Por Cantidad de transacciones:"):
                    txCant(dfNew)
                if st.checkbox("Por tipo de terminal:"):
                    txTerm(dfNew)
                if st.checkbox("Tipo-tran por transaccion:"):
                    txTipoTran(dfNew)
                if st.checkbox("Tran-cde por FIID:"):
                    txFiid(dfNew)
                if st.checkbox("Transacciones por FIID/Tran-cde:"):
                    txFiidTran(dfNew)
                    
            else:
                st.title('Prediccion')
                runPred(dfNew)

        else:
            runTest(dfNew, dfOld)




# Cantidad de transacciones por Fiid + trancde:
def txFiidTran(dfNew):
    df = dfNew[['card-fiid', 'tran-cde']].copy()
    df['cantTx']=1
    grouping_columns = ['card-fiid', 'tran-cde']
    columns_to_show = ['cantTx']
    df = df.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()
    st.write(df.sort_values(['card-fiid', "cantTx"], ascending=[True,False]))
    



# Transacciones utilizadas por cada FIID"
def txFiid(dfNew):
    df = dfNew.copy()
    df.sort_values(["card-fiid"], inplace=True)
    df.sort_values(["tran-cde"], inplace=True)
    #sns.set_style("darkgrid")
    #sns.set_context("talk", font_scale=0.5)
    #plt.figure(figsize=(10,16))

    #sns.scatterplot(df["card-fiid"], df["tran-cde"], data=df)
    #plt.xlabel("FIID")
    #plt.ylabel("Transaccion")
    #plt.title("Cantidad de tipo de transacciones por FIID")
    #st.pyplot(plt)

    fig3 = plt.figure()
    g = sns.scatterplot(data=df, x=df["card-fiid"], y=df["tran-cde"] )
    plt.xticks(rotation=45)
    st.pyplot(fig3)


# Convierto dos columnas (Codigo de transaccion y tipo-tran) en un diccionario:
# Quiero saber de cada transaccion que tipo-tran posibles tienen
def txTipoTran(dfNew):
    df = dfNew.copy()
    st.write("Tipo tran por transaccion:")
    st.write(df.groupby(by='tran-cde')['tipo-tran'].unique().apply(lambda x:x.tolist()).to_dict() )


# Cantidad de transacciones por tipo de terminal:
def txTerm(dfNew):
    df = dfNew.copy()
    counts = df['term-typ'].value_counts().to_frame()
    counts.rename(columns={'term-typ': 'value_counts'}, inplace=True)
    counts.index.name = 'term-typ'
    st.write("Cantidad de transacciones por tipo de terminal:")
    st.write(counts)


# Tansacciones con mas operaciones
def txCant(dfNew):
    df = dfNew.copy()
    counts = df['tran-cde'].value_counts().to_frame()
    counts.rename(columns={'tran-cde': 'value_counts'}, inplace=True)
    counts.index.name = 'tran-cde'

    st.write("Tansacciones con mas operaciones:")
    st.write(counts.head(10))
    st.write("Tansacciones con menos operaciones:")
    st.write(counts.tail(10))

    counts.head(10).plot(kind = "bar")
    plt.xlabel("Codigos de Respuesta")
    plt.ylabel("Cantidad de transacciones")
    plt.title("Cantidad de transacciones por RESP-CDE (primeras 10)")
    st.pyplot(plt)


    


def genRatio(df, debug = False):
    """ Generar un nuevo DF reducido con la columna de ratios de aprobacion: """

    dfRatio = df[['term-fiid', 'term-typ', 'tran-cde', 'resp-cde']]

    #print("agrego columna con resultado: OK para aprobado, NO para rechazo")
    #dfHeat["resultado"] = dfHeat["resp-cde"].apply(lambda x: 'ok' if (x == '000' or x == '001') else 'no')
    dfRatio["resultado"] = dfRatio["resp-cde"].apply(lambda x: 'ok' if (x == 0 or x == 1) else 'no')

    #Agrego una columna para contabilizar aprobadas vs rechazadas:
    dfRatio['counter'] =1
    grouping_columns = ['term-fiid', 'term-typ', 'tran-cde', 'resultado']
    columns_to_show = ['counter']
    dfRatio = dfRatio.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()

    #Separo aprobadas y rechazadas en dos nuevas columnas y agrupo
    dfRatio['aprobadas'] = np.where(dfRatio['resultado'] == "ok", dfRatio['counter'].astype('int64'), 0)
    dfRatio['rechazadas'] = np.where(dfRatio['resultado'] == "no", dfRatio['counter'].astype('int64'), 0)

    grouping_columns = ['term-fiid', 'term-typ', 'tran-cde']
    columns_to_show = ['aprobadas', 'rechazadas']
    dfRatio = dfRatio.groupby(by=grouping_columns)[columns_to_show].max().reset_index()

    #Creo una unica columna TERMINAL, concatenando el term-fiid y el term-typ de la termina:
    dfRatio['terminal'] = dfRatio["term-fiid"].str.strip() + "-" + dfRatio["term-typ"].str.strip()
    del dfRatio['term-fiid']
    del dfRatio['term-typ']

    # Creo una nueva columna calculando el ratio = aprob/(aprob+rech)
    def calRatio(fila):
        resultado = fila["aprobadas"]/( fila["aprobadas"] + fila["rechazadas"] )
        return resultado
    dfRatio['ratio'] = dfRatio.apply(calRatio, axis = 1)

    return dfRatio



def genRatioMatrix(df, eda):
    """ HEATMAP por transaccion (terminal FIID + Term Typ vs Codigo de transacion) """

    #Pivoteo tabla dejando el ratio como valor en la matriz de terminal vs tran-cde:
    dfRatios = df[['terminal', 'tran-cde', 'ratio']]
    dfRatios = dfRatios.pivot_table(index=['terminal'], columns='tran-cde', values='ratio')
    if (eda == True):
        print(dfRatios.head(10))
    return dfRatios


def generateHeatmap(dfRatios, heatTit):
    """ HEATMAP diario: """

    df = dfRatios.fillna(-1).copy()
    if stlit == False:
        plt.pcolor(df)
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
        g = sns.heatmap(df, linewidths=.3, xticklabels=True, yticklabels=True, cmap='YlGnBu_r',
                            cbar_kws={'label': ' sin datos ( < 0)     |     %Aprob ( >= 0) '} )
        g.set_xticklabels(g.get_xticklabels(), rotation=90, fontsize=3)
        g.set_yticklabels(g.get_yticklabels(), fontsize=3)

        g.set_title(heatTit)
        plt.tight_layout()
        plt.show()
    else:
        #fig1 = plt.figure(figsize=(16,5))
        fig1 = plt.figure()
        ax = plt.axes()
        sns.heatmap(df, linewidths=.3, xticklabels=True, yticklabels=True, cmap='YlGnBu_r',
                    cbar_kws={'label': ' sin datos ( < 0)     |     %Aprob ( >= 0) '}, ax = ax )
        ax.set_title(heatTit)
        st.pyplot(fig1)



def generateDifHeatmap(dfRatios, heatTit):
    """ HEATMAP por diferencia entre dia previo y posterior: """

    dfRatios *=100
    if stlit == False:
        #new_df4 = new_df3.copy()
        #new_df3 *= 100
        plt.pcolor(dfRatios)
        plt.yticks(np.arange(0.5, len(dfRatios.index), 1), dfRatios.index)
        plt.xticks(np.arange(0.5, len(dfRatios.columns), 1), dfRatios.columns)
        g = sns.heatmap(dfRatios, linewidths=.5, xticklabels=True, yticklabels=True, cmap='coolwarm_r',
                        cbar_kws={'label': ' Disminucion X% ratio     |     Aumento X% ratio '} )
        g.set_xticklabels(g.get_xticklabels(), rotation=50, fontsize=7)
        g.set_yticklabels(g.get_yticklabels(), fontsize=7)
        g.set_title(heatTit)
        plt.tight_layout()
        plt.show()
    else:
        fig2 = plt.figure()
        ax = plt.axes()
        g = sns.heatmap(dfRatios, linewidths=.5, xticklabels=True, yticklabels=True, cmap='coolwarm_r',
                        cbar_kws={'label': ' Disminucion X% ratio     |     Aumento X% ratio '}, ax = ax )
        ax.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 
        st.pyplot(fig2)


def generateHeatmapNeg(dfConcatNeg, heatTit):
    """ HEATMAP por variacion negativa: """

    if stlit == False:
        g = sns.heatmap(dfConcatNeg, linewidths=.5, xticklabels=True, yticklabels=True)
        g.set_xticklabels(g.get_xticklabels(), rotation=75, fontsize=7)
        g.set_yticklabels(g.get_yticklabels(), fontsize=7)
        g.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 
        plt.tight_layout()
        plt.show()
    else:
        fig3 = plt.figure()
        ax = plt.axes()
        sns.heatmap(dfConcatNeg, linewidths=.5, xticklabels=True, yticklabels=True, ax = ax)
        ax.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 
        st.pyplot(fig3)
        
        # g = sns.heatmap(dfConcatNeg)
        # g.set_title(heatTit)
        # st.write(g)


def generateHeatmapNegReduc(dfConcatNeg, heatTit):
    """ HEATMAP por variacion negativa reducido: """

    if stlit == False:
        g = sns.heatmap(dfConcatNeg, linewidths=.5, xticklabels=True, yticklabels=True)
        g.set_xticklabels(g.get_xticklabels(), rotation=75, fontsize=7)
        g.set_yticklabels(g.get_yticklabels(), fontsize=7)
        g.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 
        plt.tight_layout()
        plt.show()
    else:
        fig4 = plt.figure()   
        ax = plt.axes()
        sns.heatmap(dfConcatNeg, linewidths=.5, xticklabels=True, yticklabels=True, ax = ax)
        ax.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 

        st.pyplot(fig4)




def runPred(dfNew):

    with st.beta_expander("Seleccionar transaccion :", expanded=False):
        optionTX = st.selectbox("", options=dfNew['tran-cde'].unique() )
        if len(optionTX) > 0:
            dfNew = dfNew.loc[dfNew['tran-cde'] == optionTX ]


    df = dfNew[['tran-cde', 'date', 'time', 'amt']].copy()
    compras = df.loc[df['tran-cde'] == optionTX]
    st.write("Muestra:")
    st.write(compras.head(15))
    st.write(compras.shape)
    

    st.markdown('##')
    st.markdown('____')
    st.write("Agrupado por HH:MM :")
    #compras = compras.groupby('time')['amt'].sum().reset_index()

    radiosel = st.radio("Tipo de agregacion:", ["Minimo", "Maximo", "Promedio", "Suma", "Cantidad"] )
    if radiosel == "Minimo":
        compras = compras.groupby('time')['amt'].min().reset_index()
    if radiosel == "Maximo":
        compras = compras.groupby('time')['amt'].max().reset_index()
    if radiosel == "Promedio":
        compras = compras.groupby('time')['amt'].mean().reset_index()
    if radiosel == "Suma":
        compras = compras.groupby('time')['amt'].sum().reset_index()
    if radiosel == "Cantidad":
        compras = compras.groupby('time')['amt'].count().reset_index()

    #st.write(compras.head(15))
    

    st.write("Conversion:")




    #compras['time']= datetime.strptime(compras['time'], '%H:%M')
    #compras['time'] = (pd.to_datetime(compras['time'].str.strip(), format='%H:%M'))
    #compras['time'] = pd.to_datetime(compras['time'], format='%H:%M')
    #st.write(df.dtypes)

    #compras['time'] = compras['time'].astype(str)
    #compras['time'] = pd.to_datetime(compras['time'].astype(str).apply('{:0>4}'.format)    , format='%H:%M')

    #compras['time'] = datetime.strptime(pd.to_datetime(compras['time'].astype(str).apply('{:0>4}'.format), '%H:%M'))
    compras['time'] = compras['time'].astype(str).apply('{:0>4}'.format)
    f = lambda x:  x[:2] + ':' + x[2:]
    compras['dtime'] = compras['time'].apply(f)  
    #datetime.datetime.strptime(compras['time'], '%H:%M').time()
    #compras['datetime'] = pd.to_timedelta(compras.time)

    
    #compras['dtime'] = pd.to_datetime(compras['dtime'], format='%H:%M').dt.time
    #compras['dtime'] =  pd.to_datetime(compras['dtime'], format='%H:%M:%S', errors='coerce' )

    #compras['dtime'] =  datetime.strptime(compras['dtime'], "%H:%M:%S").time()

    #st.write(compras)
    #st.write(compras.dtypes)

    tsCompras = compras[['dtime','amt']]
    #tsCompras.rename(columns={"dtime": "ds", "amt": "y"}, errors="raise", inplace=True)
    tsCompras = tsCompras.set_index('dtime')
    #tsCompras.index


    st.write(tsCompras)
    st.write(tsCompras.shape)
    

    #tsCompras = tsCompras['dtime'].resample('30T').mean()
    


    tsCompras.plot(kind = "bar")
    st.pyplot(plt)

    st.line_chart(tsCompras['amt'])

    #y = compras['amt'].resample('3T').mean()
    #st.write(print("y"))


def download_csv(name, df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv()
    base = base64.b64encode(csv.encode()).decode()
    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>' % (name))
    return file



def runTest(dfNew, dfOld):


    with st.beta_expander("Reducir Matriz :", expanded=False):
        optionFIID = st.multiselect("Por FIID", dfNew['term-fiid'].unique() )
        optionTERM = st.multiselect("Por TERM-TYPE", options=dfNew['term-typ'].unique() )
        optionTX = st.multiselect("Por TRANSACCION", options=dfNew['tran-cde'].unique() )
        st.write(f"Opcion seleccionada: FIID:{optionFIID}, TERM:{optionTERM}, TRANSACTION:{optionTX}  ")

        if len(optionFIID) > 0:
            dfOld = dfOld.loc[dfOld['term-fiid'].isin(optionFIID) ]
            dfNew = dfNew.loc[dfNew['term-fiid'].isin(optionFIID) ]
        if len(optionTERM) > 0:
            dfOld = dfOld.loc[dfOld['term-typ'].isin(optionTERM) ]
            dfNew = dfNew.loc[dfNew['term-typ'].isin(optionTERM) ]
        if len(optionTX) > 0:
            dfOld = dfOld.loc[dfOld['tran-cde'].isin(optionTX) ]
            dfNew = dfNew.loc[dfNew['tran-cde'].isin(optionTX) ]
    st.markdown('##')
    st.markdown('____')


    dfHeatOld = genRatio(dfOld, False)
    dfHeatOld = genRatioMatrix(dfHeatOld, False)
    generateHeatmap(dfHeatOld, "Ratio aprobacion (aprob/total) PRE-Imple")
    st.markdown(download_csv('Data Frame PREVIO',dfHeatOld),unsafe_allow_html=True)
    st.markdown('##')
    st.markdown('____')

    dfHeatNew = genRatio(dfNew, False)
    dfHeatNew = genRatioMatrix(dfHeatNew, False)
    generateHeatmap(dfHeatNew, "Ratio aprobacion (aprob/total) POS-Imple")
    st.markdown(download_csv('Data Frame POSTERIOR',dfHeatNew),unsafe_allow_html=True)
    st.markdown('##')
    st.markdown('____')


    dfConcat = dfHeatNew.append((dfHeatOld*(-1)), sort=False)
    dfConcat = dfConcat.groupby(dfConcat.index).sum()
    generateDifHeatmap(dfConcat, "Variacion en ratios de aprobacion (TermTyp vs TranCde)")
    st.markdown(download_csv('Data Frame VARIACION',dfConcat),unsafe_allow_html=True)
    st.markdown('##')
    st.markdown('____')

    # Para este analisis elimino todos los valores positivos
    # ya que solo me interesa ver que transacciones disminuyeron en aprobaciones:
    f = lambda x:  float("NaN") if x > 0.0 else x
    dfConcatNeg = dfConcat.copy()
    dfConcatNeg = dfConcatNeg.applymap(f)
    generateHeatmapNeg(dfConcatNeg, "Variacion NEGATIVA en ratios de aprobacion")
    st.markdown(download_csv('Data Frame VARIACION NEG',dfConcatNeg),unsafe_allow_html=True)
    st.markdown('##')
    st.markdown('____')

    # Reduzco la matriz eliminando valores muy bajos de variacion
    # Hay una variacion de +/-X% en los ratios de aprobacion que es aceptable y no interesa analizar 
    # Por ejemplo si el dia anterior hubo 91% de aprobadas y el siguiente bajo a 89% (-2%), no lo analizo
    # A esa variable la denomino threshold:
    f = lambda x:  float("NaN") if x > threshold else x

    threshold = -15
    if stlit == False:
        threshold = -15
    else:
        #st.text('Reduzco la matriz eliminando valores muy bajos de variacion:')
        optionals = st.beta_expander("Reduccion de matriz:", True)
        threshold = optionals.slider( "Umbral (threshold)", float(-100), float(0), float(threshold) )

        dfConcatNeg = dfConcatNeg.applymap(f)
        dfConcatNeg = dfConcatNeg.dropna(how='all')
        dfConcatNeg = dfConcatNeg.dropna(axis=1, how='all')
        st.write(f"Tamaño de la matriz con umbral de  {threshold}: {dfConcatNeg.shape}")



    print("Tamaño original: {}".format(dfConcatNeg.shape ) )
    dfConcatNeg = dfConcatNeg.applymap(f)
    dfConcatNeg = dfConcatNeg.dropna(how='all')
    dfConcatNeg = dfConcatNeg.dropna(axis=1, how='all')
    print("Tamaño despues de aplicar el umbral: {}".format(dfConcatNeg.shape ) )

    generateHeatmapNegReduc(dfConcatNeg, 'Variacion ( superior a {}% ) en ratios de aprobacion'.format(threshold) )



if __name__ == '__main__':
	main()