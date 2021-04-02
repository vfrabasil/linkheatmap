from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import base64
import math

from time import mktime
from datetime import datetime


#import plotly.express as px

#Modo de Ejecucion:
# streamlit run HeatmapStreamlit.py --server.maxUploadSize=1028
#
#From Heroku:
# https://link-heatmap.herokuapp.com/

# Global parameters
pd.set_option('display.max_columns', 100)
debug = False
stlit = True



# Setting Cache for dataset:
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_dataset(pre_file ,post_file):
    #dfOld = pd.read_csv('rcardOld.csv', sep=';')
    #dfNew = pd.read_csv('rcardNew.csv', sep=';')
    dfOld = pd.read_csv(pre_file, sep=';', dtype={"term-fiid": str, "term-typ": str, "card-fiid": str, "resp-cde": str}, low_memory=False)
    dfNew = pd.read_csv(post_file, sep=';', dtype={"term-fiid": str, "term-typ": str, "card-fiid": str, "resp-cde": str}, low_memory=False)
    return dfOld, dfNew



def main():
    pre_file = None
    post_file = None
    count = 1

    if stlit == True:
        st.title("1 - Seleccionar Dataframes:")

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

        #Simplificar el dataset
        #dfOld.drop(dfOld.columns.difference(['term-fiid', 'term-typ', 'card-fiid', 'tran-cde', 'resp-cde']), 1, inplace=True)
        #dfNew.drop(dfNew.columns.difference(['term-fiid', 'term-typ', 'card-fiid', 'tran-cde', 'resp-cde']), 1, inplace=True)
 
        if stlit == True:
            page = st.sidebar.selectbox("Seleccionar opcion", ['Exploracion', 'Analisis', 'Serie Temporal'])

            st.markdown('##')
            if page == 'Analisis':
                st.title('2 - Analisis por mapa de calor')
                runTest(dfNew, dfOld)

            elif page == 'Exploracion':
                st.title('2 - Datos exploratorios:')

                radioselx = st.radio("Seleccion Tabla:", ["PRE-implementacion", "POS-implementacion"] , key = 20)
                if radioselx == "PRE-implementacion":
                    dfSel = dfOld
                if radioselx == "POS-implementacion":
                    dfSel = dfNew
                
                st.markdown('##')
                st.subheader(f"{radioselx}:")

                muestra = 30
                if st.checkbox('Muestra Aleatoria'):
                    st.dataframe(dfSel.sample(muestra))
                    st.markdown('##')
                if st.checkbox('Descripcion:'):
                    st.dataframe(dfSel.describe())
                    st.dataframe(dfSel.describe(include=[object]))
                    st.markdown('##')
                if st.checkbox("Cantidad de operaciones por Transaccion (tran-cde):"):
                    txCant(dfSel)
                    st.markdown('##')
                if st.checkbox("Cantidad de operaciones por Terminal (term-cde):"):
                    txTerm(dfSel)
                    st.markdown('##')
                if st.checkbox("Tipo-tran por Transaccion:"):
                    txTipoTran(dfSel)
                    st.markdown('##')
                if st.checkbox("Transacciones (tran-cde) por FIID:"):
                    txFiid(dfSel)
                    st.markdown('##')
                if st.checkbox("Volumen de transacciones por FIID:"):
                    txVolFiid(dfSel)
                    st.markdown('##')

                    
            else:
                st.title('2 - Serie Temporal:')

                radiosels = st.radio("Seleccion Tabla:", ["PRE-implementacion", "POS-implementacion"] , key = 30)
                if radiosels == "PRE-implementacion":
                    dfSer = dfOld
                if radiosels == "POS-implementacion":
                    dfSer = dfNew
                
                st.markdown('##')
                st.subheader(f"{radiosels}:")

                runPred(dfSer)

        else:
            runTest(dfNew, dfOld)




def txVolCardFiid(dfNew):

    df = dfNew[['card-fiid', 'tran-cde']].copy()
    df['cantTx']=1
    grouping_columns = ['card-fiid', 'tran-cde']
    columns_to_show = ['cantTx']
    df = df.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()

    # Para mostrar en listado, ordeno por cantidad de transacciones:
    st.write(df.sort_values(['card-fiid', "cantTx"], ascending=[True,False]))
    
    # Para mostrar grafico, ordeno por tran-cde:
    df.sort_values(["card-fiid"], inplace=True)
    df.sort_values(["tran-cde"], inplace=True)

    # Dimensionamiento automatico de la matriz
    xx = df['card-fiid'].nunique()
    yy = df['tran-cde'].nunique()
    fig_dims = (math.ceil(xx/3), math.ceil(yy/5))

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xticks(rotation=90)
    sns.scatterplot(data=df, x="card-fiid", y="tran-cde", ax=ax, size="cantTx", sizes=(5, 600), alpha=.8, palette="muted")
    st.write(fig)
    st.write(f'Tamaño matriz X:{xx} Y:{yy}')

    # OK: otra opcion:
    #fig3 = plt.figure()
    #g = sns.scatterplot(data=df, x="card-fiid", y="tran-cde", size="cantTx", sizes=(5, 300), alpha=.5, palette="muted")
    #plt.xticks(rotation=45)
    #st.pyplot(fig3)


def txVolTermFiid(dfNew):

    df = dfNew[['term-fiid', 'tran-cde']].copy()
    df['cantTx']=1
    grouping_columns = ['term-fiid', 'tran-cde']
    columns_to_show = ['cantTx']
    df = df.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()

    # Para mostrar en listado, ordeno por cantidad de transacciones:
    st.write(df.sort_values(['term-fiid', "cantTx"], ascending=[True,False]))
    
    # Para mostrar grafico, ordeno por tran-cde:
    df.sort_values(["term-fiid"], inplace=True)
    df.sort_values(["tran-cde"], inplace=True)

    xx = df['term-fiid'].nunique()
    yy = df['tran-cde'].nunique()
    fig_dims = (math.ceil(xx/3), math.ceil(yy/5))
    fig, ax = plt.subplots(figsize=fig_dims)
    plt.xticks(rotation=90)

    #sns.set_palette('rainbow')
    sns.scatterplot(data=df, x="term-fiid", y="tran-cde", ax=ax, size="cantTx", sizes=(10, 500), legend="brief", alpha=.8)
    st.write(fig)
    st.write(f'Tamaño matriz X:{xx} Y:{yy}')

    # OK: otra opcion:
    #fig3 = plt.figure()
    #g = sns.scatterplot(data=df, x="term-fiid", y="tran-cde", size="cantTx", sizes=(5, 300), alpha=.5, palette="muted")
    #plt.xticks(rotation=45)
    #st.pyplot(fig3)


# Cantidad de transacciones por Fiid + trancde:
def txVolFiid(dfNew):

    radioselv = st.radio("Seleccion FIID:", ["term-fiid", "card-fiid"] , key = 10)
    if radioselv == "term-fiid":
        txVolTermFiid(dfNew)
    if radioselv == "card-fiid":
        txVolCardFiid(dfNew)   


def txTermFiid(dfNew):
    df = dfNew[["term-fiid", "tran-cde"]].copy()
    df.sort_values(["term-fiid"], inplace=True)
    df.sort_values(["tran-cde"], inplace=True)

    st.write(df['term-fiid'].value_counts().to_frame())
    xx = df['term-fiid'].nunique()
    yy = df['tran-cde'].nunique()
    fig_dims = (math.ceil(xx/3), math.ceil(yy/5))
    fig, ax = plt.subplots(figsize=fig_dims)
    ax.grid(linestyle=':')
    plt.xticks(rotation=45)
    sns.scatterplot(data=df, x="term-fiid", y="tran-cde", ax=ax)
    st.write(fig)
    st.write(f'Tamaño matriz X:{xx} Y:{yy}')

def txCardFiid(dfNew):
    df = dfNew[["card-fiid", "tran-cde"]].copy()
    df.sort_values(["card-fiid"], inplace=True)
    df.sort_values(["tran-cde"], inplace=True)

    st.write(df['card-fiid'].value_counts().to_frame())
    xx = df['card-fiid'].nunique()
    yy = df['tran-cde'].nunique()
    fig_dims = (math.ceil(xx/3), math.ceil(yy/5))

    fig, ax = plt.subplots(figsize=fig_dims)
    ax.grid(linestyle=':')
    plt.xticks(rotation=45)
    sns.scatterplot(data=df, x="card-fiid", y="tran-cde", ax=ax)
    st.write(fig)
    st.write(f'Tamaño matriz X:{xx} Y:{yy}')


# Transacciones utilizadas por cada FIID"
def txFiid(dfNew):

    radiosel = st.radio("Seleccion FIID:", ["term-fiid", "card-fiid"], key = 11 )
    if radiosel == "term-fiid":
        txTermFiid(dfNew)
    if radiosel == "card-fiid":
        txCardFiid(dfNew)


# Convierto dos columnas (Codigo de transaccion y tipo-tran) en un diccionario:
# Quiero saber de cada transaccion que tipo-tran posibles tienen
def txTipoTran(dfNew):
    df = dfNew.copy()
    st.write("Tipo tran por transaccion:")
    dic = df.groupby(by='tran-cde')['tipo-tran'].unique().apply(lambda x:x.tolist()).to_dict() 


    for tc, tt in dic.items():
        st.markdown(f'__{tc}__: {tt}')


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
    #counts = counts.reset_index()
    counts.rename(columns={'tran-cde': 'value_counts'}, inplace=True)
    #counts.index.name = 'tran-cde'


    st.write("Tansacciones con mas operaciones:")
    st.write(counts.head(15))
    st.write("Tansacciones con menos operaciones:")
    st.write(counts.tail(15))

    counts.head(15).plot(kind = "bar")
    plt.xlabel("tran-cde")
    plt.ylabel("Cantidad de operaciones")
    plt.title("Cantidad de operaciones por transaccion (primeras 10)")
    st.pyplot(plt)

    #fig1 = plt.figure()
    #ax = sns.countplot(data=counts , x = 'value_counts' )
    #plt.xticks(rotation=45)
    #st.pyplot(fig1)
    


def genRatio(dfRatio, radioselr):
    """ Generar un nuevo DF reducido con la columna de ratios de aprobacion: """

    #dfRatio = df[['term-fiid', 'term-typ', 'tran-cde', 'resp-cde']]

    #print("agrego columna con resultado: OK para aprobado, NO para rechazo")
    #dfRatio["resultado"] = dfRatio["resp-cde"].apply(lambda x: 'ok' if (x == 0 or x == 1) else 'no')
    dfRatio["resultado"] = dfRatio["resp-cde"].apply(lambda x: 'ok' if (x == '000' or x == '001') else 'no')

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

    # Creo una nueva columna calculando el ratio = aprob
    def calRatioV(fila):
        resultado = fila["aprobadas"]
        return resultado

    def calRatioLN(fila):
        resultado = np.log(fila["aprobadas"]+1)
        return resultado


    if radioselr == "Porcentual":
        dfRatio['ratio'] = dfRatio.apply(calRatio, axis = 1)
    if radioselr == "Volumen":
        dfRatio['ratio'] = dfRatio.apply(calRatioV, axis = 1)
    if radioselr == "VolumenLN":
        dfRatio['ratio'] = dfRatio.apply(calRatioLN, axis = 1)

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

        xx = len(df.columns)
        yy = len(df)
        fig_dims = (math.ceil(xx/3), math.ceil(yy/5))
        fig1, ax = plt.subplots(figsize=fig_dims)

        sns.heatmap(df, linewidths=.3, xticklabels=True, yticklabels=True, cmap='YlGnBu_r',
                    cbar_kws={'label': ' sin datos ( < 0)     |     %Aprob ( >= 0) '}, ax = ax )
        ax.set_title(heatTit)
        st.pyplot(fig1)
        st.write(f'Tamaño matriz X:{xx} Y:{yy}')




@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calRatios(dfOld, dfNew, radioselr):

    dfHeatOld = genRatio(dfOld, radioselr)
    dfHeatOld = genRatioMatrix(dfHeatOld, False)
    dfHeatOld = dfHeatOld.T
    #generateHeatmap(dfHeatOld, "Ratio PRE-Imple")
    #st.markdown(download_csv('Data Frame PREVIO',dfHeatOld),unsafe_allow_html=True)
    #st.markdown('##')
    #st.markdown('____')

    dfHeatNew = genRatio(dfNew, radioselr)
    dfHeatNew = genRatioMatrix(dfHeatNew, False)
    dfHeatNew = dfHeatNew.T
    #generateHeatmap(dfHeatNew, "Ratio POS-Imple")
    #st.markdown(download_csv('Data Frame POSTERIOR',dfHeatNew),unsafe_allow_html=True)
    #st.markdown('##')
    #st.markdown('____')

    return dfHeatOld, dfHeatNew



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calDifHeatmap(dfHeatNew, dfHeatOld, radioselr):
    dfConcat = dfHeatNew.append((dfHeatOld*(-1)), sort=False).copy()
    dfConcat = dfConcat.groupby(dfConcat.index).sum()
    if radioselr == "Porcentual":
        dfConcat *= 100
    return dfConcat


def generateDifHeatmap(dfRatios, heatTit):
    """ HEATMAP por diferencia entre dia previo y posterior: """

    #dfRatios *=100
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

        xx = len(dfRatios.columns)
        yy = len(dfRatios)
        fig_dims = (math.ceil(xx/3), math.ceil(yy/5))
        fig2, ax = plt.subplots(figsize=fig_dims)

        g = sns.heatmap(dfRatios, linewidths=.5, xticklabels=True, yticklabels=True, cmap='coolwarm_r',
                        cbar_kws={'label': ' Disminucion X% ratio     |     Aumento X% ratio '}, ax = ax )
        ax.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 
        st.pyplot(fig2)
        st.write(f'Tamaño matriz X:{xx} Y:{yy}')
        st.markdown(download_csv('Data Frame VARIACION',dfRatios),unsafe_allow_html=True)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calHeatmapNeg(dfConcat):
    """ HEATMAP por variacion negativa: """

    f = lambda x:  float("NaN") if x > 0.0 else x
    dfConcat = dfConcat.copy()
    dfConcat = dfConcat.applymap(f)



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


        xx = len(dfConcatNeg.columns)
        yy = len(dfConcatNeg)
        fig_dims = (math.ceil(xx/3), math.ceil(yy/5))
        fig3, ax = plt.subplots(figsize=fig_dims)

        sns.heatmap(dfConcatNeg, linewidths=.5, xticklabels=True, yticklabels=True, ax = ax)
        ax.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 
        st.pyplot(fig3)
        st.write(f'Tamaño matriz X:{xx} Y:{yy}')

        st.markdown(download_csv('Data Frame VARIACION NEG',dfConcatNeg),unsafe_allow_html=True)
        # g = sns.heatmap(dfConcatNeg)
        # g.set_title(heatTit)
        # st.write(g)



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calHeatmapNegReduc(dfConcatNeg, threshold):
    """ HEATMAP por variacion negativa reducido: """
    f = lambda x:  float("NaN") if x > threshold else x

    #print("Tamaño original: {}".format(dfConcatNeg.shape ) )
    dfConcatNeg = dfConcatNeg.applymap(f)
    dfConcatNeg = dfConcatNeg.dropna(how='all')
    dfConcatNeg = dfConcatNeg.dropna(axis=1, how='all')
    return dfConcatNeg



def generateHeatmapNegReduc(dfConcatNeg, heatTit):
    """ HEATMAP por variacion negativa reducido: """
    #f = lambda x:  float("NaN") if x > threshold else x
    ##print("Tamaño original: {}".format(dfConcatNeg.shape ) )
    #dfConcatNeg = dfConcatNeg.applymap(f)
    #dfConcatNeg = dfConcatNeg.dropna(how='all')
    #dfConcatNeg = dfConcatNeg.dropna(axis=1, how='all')
    ##print("Tamaño despues de aplicar el umbral: {}".format(dfConcatNeg.shape ) )

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

        xx = len(dfConcatNeg.columns)
        yy = len(dfConcatNeg)
        fig_dims = (math.ceil(xx/3), math.ceil(yy/5))
        fig4, ax = plt.subplots(figsize=fig_dims)

        sns.heatmap(dfConcatNeg, linewidths=.5, xticklabels=True, yticklabels=True, ax = ax)
        ax.set_title(heatTit)
        plt.xlabel("TranCode") 
        plt.ylabel("TermTyp por FIID") 

        st.pyplot(fig4)
        #st.write(f'Tamaño matriz X:{xx} Y:{yy}')




def runPred(dfNew):

    with st.beta_expander("Seleccionar transaccion :", expanded=False):
        optionTX = st.selectbox("", options= sorted(dfNew['tran-cde'].unique()) )
        if len(optionTX) > 0:
            dfNew = dfNew.loc[dfNew['tran-cde'] == optionTX ]


    df = dfNew[['tran-cde', 'date', 'time', 'amt']].copy()
    compras = df.loc[df['tran-cde'] == optionTX]

    compras['datetime'] = pd.to_datetime(2020*100000000 + df['date']* 10000  + df['time'] , format='%Y%m%d%H%M' )

    st.markdown('##')
    st.markdown('##')
    st.write("Muestra: Transaccion {}".format(optionTX))
    compras.set_index(['datetime'],inplace=True)

    #st.write(compras.head(20))
    #st.write(compras.tail(15))
    #st.write(compras.shape)
    #st.write(compras.dtypes)
    

    st.markdown('##')
    st.markdown('____')
    st.write("Modificar rango de la serie :")
    


    compras = compras[['amt']]
    column_3, column_4 = st.beta_columns(2)
    with column_3:
        radioselag = st.radio("Rango de agrupacion:", ["1h", "30m", "15m", "5m", "1m"], key="40")
        if radioselag == "1h":
            rango = "H"
        if radioselag == "30m":
            rango = "30min"
        if radioselag == "15m":
            rango = "15min"
        if radioselag == "5m":
            rango = "5min"
        if radioselag == "1m":
            rango = "1min"
    with column_4:
        radiosel = st.radio("Tipo de agregacion:", ["Minimo", "Maximo", "Promedio", "Suma", "Cantidad"], key="50" )
        if radiosel == "Minimo":
            compras = compras.amt.resample(rango).min()
        if radiosel == "Maximo":
            compras = compras.amt.resample(rango).max()
        if radiosel == "Promedio":
            compras = compras.amt.resample(rango).mean()
        if radiosel == "Suma":
            compras = compras.amt.resample(rango).sum()
        if radiosel == "Cantidad":
            compras = compras.amt.resample(rango).count()
    st.markdown('____')

    #if radiosel == "Minimo":
    #    compras = compras.groupby('time')['amt'].min().reset_index()
    #if radiosel == "Maximo":
    #    compras = compras.groupby('time')['amt'].max().reset_index()
    #if radiosel == "Promedio":
    #    compras = compras.groupby('time')['amt'].mean().reset_index()
    #if radiosel == "Suma":
    #    compras = compras.groupby('time')['amt'].sum().reset_index()
    #if radiosel == "Cantidad":
    #    compras = compras.groupby('time')['amt'].count().reset_index()

    #compras = compras[['amt']]

    st.markdown('##')
    st.write("Conversion:")

    #compras['time'] = compras['time'].astype(str).apply('{:0>4}'.format)
    #f = lambda x:  x[:2] + ':' + x[2:]
    #compras['dtime'] = compras['time'].apply(f)  
    #tsCompras = compras[['dtime','amt']]
    #tsCompras = tsCompras.set_index('dtime')
    
    #tsCompras = compras[['datetime','amt']]
    #tsCompras = tsCompras.set_index('datetime')

    column_1, column_2 = st.beta_columns(2)
    with column_1:
        st.write(compras)
    with column_2:
        compras.plot()
        st.pyplot(plt)

    st.markdown('##')
    st.line_chart(compras)


def download_csv(name, df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv(sep=";")
    base = base64.b64encode(csv.encode()).decode()
    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>' % (name))
    return file



def runTest(dfNew, dfOld):

    f = lambda x:  float("NaN") if x > threshold else x


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


    # 0- Formula del ratio en LATEX:
    st.write("Formula del Calculo del ratio:")
    radioselr = st.radio("Tipo de ratio:", ["Porcentual", "Volumen", "VolumenLN"], key="40" )
    if radioselr == "Porcentual":
        st.latex(r'''
        \rho = x / (x+y)   :   \begin{cases}
        x: & \sum_ \text{ transacciones aprobadas} \    \\
        y: & \sum_ \text{ transacciones rechazadas} \   
        \end{cases}
        ''')
    if radioselr == "Volumen":
        st.latex(r''' \rho = \sum_{}^{} \text{ transacciones aprobadas} ''')
    if radioselr == "VolumenLN":
        st.latex(r''' \rho = \ln \lparen \sum_{}^{} \text{ transacciones aprobadas} + 1 \rparen''')
    st.markdown('##')
    st.markdown('____')

    # 1- Calculo del ratio:
    dfHeatOld, dfHeatNew = calRatios(dfOld, dfNew, radioselr )
    st.subheader('Ratios PRE implementacion:')
    generateHeatmap(dfHeatOld, "Ratio PRE-Imple")
    st.markdown(download_csv('Data Frame PREVIO',dfHeatOld),unsafe_allow_html=True)
    st.markdown('##')
    st.markdown('____')
    st.subheader('Ratios POS implementacion:')
    generateHeatmap(dfHeatNew, "Ratio POS-Imple")
    st.markdown(download_csv('Data Frame POSTERIOR',dfHeatNew),unsafe_allow_html=True)
    st.markdown('##')
    st.markdown('____')
    

    # 2- Calculo matriz de diferencias:
    st.subheader('Variacion de ratios:')
    dfConcat = calDifHeatmap(dfHeatNew, dfHeatOld, radioselr )
    generateDifHeatmap(dfConcat, "Variacion en ratios de aprobacion (TermTyp vs TranCde)")
    st.markdown('##')
    st.markdown('____')


    # 3- Calculo diferencias negativas:
    # Para este analisis elimino todos los valores positivos
    # ya que solo me interesa ver que transacciones disminuyeron en aprobaciones:
    st.subheader('Variacion negativa de ratios:')
    calHeatmapNeg(dfConcat)
    generateHeatmapNeg(dfConcat, "Variacion NEGATIVA en ratios de aprobacion")
    st.markdown('##')
    st.markdown('____')


    # 4- Calculo matriz negativa reducida:
    # Reduzco la matriz eliminando valores muy bajos de variacion
    # Hay una variacion de +/-X% en los ratios de aprobacion que es aceptable y no interesa analizar 
    # Por ejemplo si el dia anterior hubo 91% de aprobadas y el siguiente bajo a 89% (-2%), no lo analizo
    # A esa variable la denomino threshold:
    threshold = min(dfConcat.min()) * 0.1
    if stlit == False:
        threshold = -10
    else:
        optionals = st.beta_expander("Reducir por nivel de tolerancia:", False)
        threshold = optionals.slider( "Umbral (threshold)", float(min(dfConcat.min())), float(0), float(threshold) )
        #dfConcatNeg = dfConcat.applymap(f)
        #dfConcatNeg = dfConcatNeg.dropna(how='all')
        #dfConcatNeg = dfConcatNeg.dropna(axis=1, how='all')


    st.markdown('##')
    st.markdown('##')
    st.subheader('Variacion negativa con tolerancia:')
    dfConcatNeg = calHeatmapNegReduc(dfConcat, threshold )
    generateHeatmapNegReduc(dfConcatNeg, 'Variacion NEGATIVA con nivel de tolerancia' )
    st.write(f"Tamaño matriz: {dfConcatNeg.shape}  Tolerancia: {threshold}")



# -----------------------------------------------------------------------------
if __name__ == '__main__':
	main()
# -----------------------------------------------------------------------------