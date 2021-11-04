  
"""
*** Streamlit HeapMap Data Explorer ***
App to analyze the transaction data obtained on two different days and evaluate the variation in the values.

*** Modo de Ejecucion ***
streamlit run HeatmapStreamlit.py --server.maxUploadSize=1024

*** From Heroku ***
https://link-heatmap.herokuapp.com/

v21.08 August 2022
Author: @VictorFrabasil
"""

from os import write
from altair.vegalite.v4.schema.channels import Tooltip
from numpy.core.numeric import NaN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import base64
import math
import time
import threading
import queue

import altair as alt
from altair import Row, Column, Chart, Text, Scale, Color


from streamlit.report_thread import add_report_ctx
import sweetviz as sv



#import plotly.express as px
#import numba as nb
#from typing import Optional
#from datetime import datetime
#from numpy.core.numeric import False_



# Global parameters
global ejex

hide_menu_style = """ 
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: visible;}
                footer:after {content: '▪️ vfrabasil@gmail.com';
                display:block;
                position:relative;
                color:lightgray;
                padding:5px;
                top:-10px;}
                </style>
                """
#MainMenu {visibility: hidden;}
#color:tomato;

st.markdown(hide_menu_style, unsafe_allow_html=True)



pd.set_option('display.max_columns', 100)
debug = False
stlit = True
debugSteamlit = False
debugInfo = False
label1 = 'DIA 1'
label2 = 'DIA 2'

# HEROKU:
#mylogo = 'logoHeat.png'
samplef1 = 'f210708.csv'
samplef2 = 'f210715.csv'
inputcsv = 'img1.png'
outputcsv = 'img2.png'

# LINK:
mylogo = 'logohd.png'
#samplef1 = '.\Sample\\f210708.csv'
#samplef2 = '.\Sample\\f210715.csv'
#inputcsv = '.\images\img1.png'
#outputcsv = '.\images\img2.png'


useAltair = True
dfAltairOld = pd.DataFrame()
dfAltairNew = pd.DataFrame()



# Setting Cache for dataset:
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_dataset(pre_file ,post_file):
    #dfOld = pd.read_csv('rcardOld.csv', sep=';')
    #dfNew = pd.read_csv('rcardNew.csv', sep=';')
    dfOld = pd.read_csv(pre_file, sep=';', dtype={"pan": str,"term-fiid": str, "term-typ": str, "card-fiid": str, "resp-cde": str, "terminal": str}, low_memory=False)
    dfNew = pd.read_csv(post_file, sep=';', dtype={"pan": str,"term-fiid": str, "term-typ": str, "card-fiid": str, "resp-cde": str, "terminal": str}, low_memory=False)
    return dfOld, dfNew

def sep():
    st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)

def seps():
    #st.sidebar.markdown('##')
    #st.sidebar.markdown('___')
    #st.sidebar.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:gray">', unsafe_allow_html=True)
    st.sidebar.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)

def main():
    global ejex
    pre_file = None
    post_file = None
    count = 1

    def getCount():
        count=+1
        return count

    def getFiles(num, f1, f2):
        pre_file = None
        post_file = None
        #st.title("Seleccionar Archivos:")

        global label1
        global label2

        with st.sidebar.expander("💾 Cargar Archivos CSV :", expanded=True):
            pre_file  = st.file_uploader(label1, 
                                        accept_multiple_files=False,
                                        type='csv', key = "prefile") # getCount())

            post_file  = st.file_uploader(label2, 
                                        accept_multiple_files=False,
                                        type='csv', key = "posfile") #  getCount())

        seps()
        global debugInfo
        if st.sidebar.checkbox('Prueba', value=False, key= "SampleFile"):
            pre_file  = samplef1
            post_file = samplef2


        global debugInfo
        if st.sidebar.checkbox('Debug', value=False, key= "debugmode"):
            debugInfo  = True
        else:
            debugInfo  = False

        with st.sidebar.expander("File labels :", expanded=False):
            id1 = st.text_input('Identificador:', label1)
            id2 = st.text_input('Identificador:', label2)
            label1 = id1
            label2 = id2


        return pre_file, post_file


    def countByRech(dfIni):

        df = dfIni.copy()
        ejex = df.columns[0]

        df["resultado"] = df["resp-cde"].astype(str).apply(lambda x: 'ok' if (x == '000' or x == '001'
            or x == '00' or x == '01' or x == '0' or x == '1' or x == ' 1' or x == ' 0') else 'no')

        #df['counter'] =1
        grouping_columns = [ejex, 'tran-cde', 'resultado']
        columns_to_show = ['count']
        df = df.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()

        #Separo aprobadas y rechazadas en dos nuevas columnas y agrupo
        df['aprobadas'] = np.where(df['resultado'] == "ok", df['count'].astype('int64'), 0)
        df['rechazadas'] = np.where(df['resultado'] == "no", df['count'].astype('int64'), 0)

        grouping_columns = [ejex, 'tran-cde']
        columns_to_show = ['aprobadas', 'rechazadas']
        df = df.groupby(by=grouping_columns)[columns_to_show].max().reset_index()

        return df

        
    def sweetEDA(dfpre, dfpos, title=None):

        st.title(title)


        radioseda = st.radio("Origen de los datos :", [label1, label2], key="swEDA" )
        if radioseda == label1:
            df = dfpre
        if radioseda == label2:
            df = dfpos

        st.write(df.head())

        #skip_columns_years = [str(y) for y in range(2016, 2026)]
        #skip_columns_time_series = ['Date of last update', 'Databank code', 'Scenario', 'Location code', 'Indicator code'] + skip_columns_years

        # Use the analysis function from sweetviz module to create a 'DataframeReport' object.
        analysis = sv.analyze([df,'EDA'], feat_cfg=sv.FeatureConfig(
            #skip=skip_columns_time_series,
            force_text=[]), target_feat=None)

        # Render the output on a web page.
        #analysis.show_html(filepath='./frontend/public/EDA.html', open_browser=False, layout='vertical', scale=1.0)
        #components.iframe(src='http://localhost:3001/EDA.html', width=1100, height=1200, scrolling=True)
        analysis.show_html(filepath='./EDA.html', open_browser=False, layout='vertical', scale=1.0)

        HtmlFile = open("./EDA.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, width=1100, height=1200, scrolling=True)

        #components.iframe(source_code, width=1100, height=1200, scrolling=True)

        st.subheader("Comparacion entre PRE y POS")
        df1 = sv.compare(dfpre, dfpos)
        df1.show_html(filepath='./EDAc.html', open_browser=False, layout='vertical', scale=1.0)
        HtmlFile = open("./EDAc.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, width=1100, height=1200, scrolling=True)



    #st.title('DATA EXPLORER', )
    st.markdown("<h1 style='text-align: center; color: green;'>DATA EXPLORER</h1>", unsafe_allow_html=True)


    #st.subheader('<-- Seleccionar tipo de analisis')


    if stlit == True:

        st.sidebar.image(mylogo, width=256) #, use_column_width=True)
        #page = st.sidebar.selectbox("Seleccionar opcion:", [' ', 'Mapa de transacciones', 'Estadistico', 'Serie', 'EDA'])
        page = st.sidebar.selectbox("Seleccionar opcion:", [' ', 'Generar archivos', 'Mapa de transacciones'])
        seps()

        global useAltair
        useAltair = True
        #radioselm = st.sidebar.radio("Libreria Grafica:", ["Altair", "Seaborn"], key="41" )
        #global useAltair
        #if radioselm == "Altair":
        #    useAltair = True
        #if radioselm == "Seaborn":
        #    useAltair = False
        #seps()



        if page == 'EDA':

            pre_file, post_file = getFiles('4', 'prod_pr_PRE.csv', 'prod_pr_POS.csv' )
            if pre_file is not None and post_file is not None:
                
                dfpre, dfpos = load_dataset(pre_file, post_file )
                sweetEDA(dfpre, dfpos, title='Analisis Exploratorio de Datos')

        if page == ' ':
            st.write("##")
            st.write("""
            ### Analisis de variacion en transacciones:
            """) 
            st.write("""
            Applicacion para analizar los codigos de respuesta de transacciones obtenidos en dos días distintos 
            """) 
            st.write("""
            y evalua la diferencia entre ambos valores.
            """)
            st.write("""
            Utiliza dos archivos .csv para crear una matriz de diferencias de ratios
            """)
            st.write("""
            Version Beta [online](https://link-heatmap.herokuapp.com/).
            """)

            sep()
            #st.markdown("<h2 style='text-align: center; color: green;'>Seleccionar Opcion </h2>", unsafe_allow_html=True)
            pre_file = ''
            post_file = ''
            #st.markdown("<h1><hr></h1>", unsafe_allow_html=True)




        #Deprecated:
        #if page == 'Mapeo':
        #    st.success("1 - ANALISIS POR MAPA DE CALOR:")
        #    pre_file, post_file = getFiles('1', 'desa_rcard_PRE.csv', 'desa_rcard_POS.csv' )
        #    #dfOld = pd.DataFrame()
        #    #dfNew = pd.DataFrame()
        #    #if dfOld.empty == False and  dfNew.empty == False:
        #    #    dfOld.drop(dfOld.index, inplace=True)
        #    #    dfNew.drop(dfNew.index, inplace=True)
        #    if pre_file is not None and post_file is not None:
        #        dfOld, dfNew = load_dataset(pre_file, post_file )
        #        runTest(dfNew, dfOld, 0, '',dfNew, dfOld)




        elif page == 'Estadistico':
            #st.header('2 - DATOS EXPLORATORIOS:')
            st.info("2 - ANALISIS ESTADISTICOS:")
            pre_file, post_file = getFiles('2', 'desa_rcard_PRE.csv', 'desa_rcard_POS.csv' )

            #dfOld2 = pd.DataFrame()
            #dfNew2 = pd.DataFrame()
            #if dfOld2.empty == False and  dfNew2.empty == False:
            #    dfOld2.drop(dfOld2.index, inplace=True)
            #    dfNew2.drop(dfNew2.index, inplace=True)

            if pre_file is not None and post_file is not None:
                
                dfpre, dfpos = load_dataset(pre_file, post_file )

                st.markdown('##')
                radioselx = st.radio("Seleccion Tabla:", [label1, label2] , key = "estadistico")
                if radioselx == label1:
                    dfSel = dfpre
                if radioselx == label2:
                    dfSel = dfpos
            
                st.markdown('##')
                st.subheader(f"{radioselx}:")

                muestra = 30
                if st.checkbox('Muestra Aleatoria'):
                    st.dataframe(dfSel.sample(muestra))
                    st.markdown('##')
                if st.checkbox('Descripcion:'):
                    st.dataframe(dfSel.describe())
                    #st.dataframe(dfSel.describe(include=[object]))
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
                #if st.checkbox("Volumen de transacciones por FIID (HD):"):
                #    txVolFiidAlt(dfSel)
                #    st.markdown('##')

        elif page == 'Generar archivos':
            #sep()
            #st.write("""    
            #**Generar los archivos de input para el mapa de transacciones:**
            #""") 
            st.write("##")
            st.write("""
            ### Generacion de archivos:
            """) 


            with st.expander("formato de los archivos:", expanded=False):
                st.write("""
                A partir de un archivo .CSV (obtenido por RCARD o QUERY)
                """)

                st.image(inputcsv, use_column_width=True)
                st.write("""
                *term-fiid ; term-typ ; card-fiid ; pan ; tran-cde ; tipo-tran ; responder ; resp-cde ; reversal ; date ; time ; amt* 
                """)
                st.write('##')

                st.write("""
                Genera otro archivo .CSV (contabilizando resp-cde) con el siguiente formato:
                """)

                st.image(outputcsv, width= 512)
                st.write("""
                *terminal (fiid-term + term-typ) ; tran-cde ; resp-cde ; count ; card-fiid*
                """)

            sep()                
            user_input = st.text_input("Abrir Archivo", 'full path file', help= 'Shift + Boton Derecho (sobre el archivo) y <Copy as path>')
            if st.button('Generar'): #validar ubicacion del archivo
                user_input = user_input.strip('"')
                user_output = runChunk(user_input)
                st.write('Archivo Generado:')
                st.caption(user_output)


        elif page == 'Mapa de transacciones':
            #st.header('4 - ARCHIVO PREPROCESADO:')
            #st.success("4 - ANALISIS CON ARCHIVO PREPROCESADO:") 
            #with st.expander("formato de los archivos:", expanded=False): 
            st.write("#")
            st.write("""
            ### Mapa de transacciones:
            """) 
            sep()

            pre_file, post_file = getFiles('4', 'prod_pr_PRE.csv', 'prod_pr_POS.csv' )
            if pre_file is not None and post_file is not None:
                
                dfpre, dfpos = load_dataset(pre_file, post_file )
                st.success("CONFIGURACION:") 
                global dfpreg
                global dfposg
                dfpreg = dfpre
                dfposg = dfpos

                #st.markdown('##')
                with st.expander(" 🔗 Metodo de agrupacion:", expanded=False):
                    radioEjex = st.radio("Agrupar por:", ["FIID Terminal", "FIID Tarjeta"] , key = 'radioEjex', 
                                help='FIID (Tarjeta o Terminal) vs Codigo de Transaccion' )
                    if radioEjex == "FIID Tarjeta":
                        dfpregx = dfpreg[dfpreg['card-fiid'].notna()]
                        dfposgx = dfposg[dfposg['card-fiid'].notna()]
                        dfpregx = dfpregx.drop('terminal', axis=1)
                        dfposgx = dfposgx.drop('terminal', axis=1)
                        dfpregx = dfpregx[['card-fiid', 'tran-cde', 'resp-cde', 'count']]
                        dfposgx = dfposgx[['card-fiid', 'tran-cde', 'resp-cde', 'count']]
                    if radioEjex == "FIID Terminal":
                        dfpregx = dfpreg[dfpreg['terminal'].notna()]
                        dfposgx = dfposg[dfposg['terminal'].notna()]
                        dfpregx = dfpregx.drop('card-fiid', axis=1)
                        dfposgx = dfposgx.drop('card-fiid', axis=1)
                        dfpregx = dfpregx[['terminal', 'tran-cde', 'resp-cde', 'count']]
                        dfposgx = dfposgx[['terminal', 'tran-cde', 'resp-cde', 'count']]


                dfOldp = countByRech(dfpregx)
                dfNewp = countByRech(dfposgx)

                #dfOldp = dfpre
                #dfNewp = dfpos

                #v1
                #dfOldp.rename(columns={'apr': 'aprobadas', 'rec': 'rechazadas'}, inplace=True)
                #dfNewp.rename(columns={'apr': 'aprobadas', 'rec': 'rechazadas'}, inplace=True)
                #st.write(dfOldp.head(10))

                dfOld4 = dfOldp
                dfNew4 = dfNewp
                #-----------------------------------------
                #st.markdown('##')
                #with st.expander("Reducir Matriz :", expanded=False):
                #    radioselag = st.radio("Agrupar por:", ["ambos", "fiid", "terminal"] , key = 23)
                #    if radioselag == "ambos":
                #        dfOld4 = dfOldp
                #        dfNew4 = dfNewp
                #    if radioselag == "fiid":
                #        dfOld4['terminal'] = dfOldp['terminal'].str.slice(0,4)
                #        dfNew4['terminal'] = dfNewp['terminal'].str.slice(0,4)
                #    if radioselag == "terminal":
                #        dfOld4['terminal'] = dfOldp['terminal'].str.slice(5,7)
                #        dfNew4['terminal'] = dfNewp['terminal'].str.slice(5,7)

                ## movi al final


                #-----------------------------------------
                #optiontx = st.selectbox("Minimo de Transacciones", [ 0, 1, 10, 100, 1000] )
                #optionApr = st.selectbox("Minimo de Aprobadas", [ 0, 1, 10, 100, 1000] )
                #optionRec = st.selectbox("Minimo de Rechazadas", [ 0, 1, 10, 100, 1000] )
                #st.write(f"Opcion seleccionada: Transacciones>{optiontx}, Aprobadas>{optionApr}, Rechazadas>{optionRec}")
                #if optiontx >= 0:
                #    dfOld4 = dfOldp[(dfOldp['aprobadas'] + dfOldp['rechazadas'])  >= optiontx]
                #    dfNew4 = dfNewp[(dfNewp['aprobadas'] + dfNewp['rechazadas'])  >= optiontx]
                #if optionApr >= 0:
                #    dfOld4 = dfOldp[dfOldp['aprobadas'] >= optionApr]
                #    dfNew4 = dfNewp[dfNewp['aprobadas'] >= optionApr]
                #if optionRec >= 0:
                #    dfOld4 = dfOldp[dfOldp['rechazadas'] >= optionRec]
                #    dfNew4 = dfNewp[dfNewp['rechazadas'] >= optionRec]
                #if optiontx >= 0:
                #    dfOld4 = dfOldp[(dfOldp['aprobadas'] + dfOldp['rechazadas'])  >= optiontx]
                #    dfNew4 = dfNewp[(dfNewp['aprobadas'] + dfNewp['rechazadas'])  >= optiontx]
                #if optionApr >= 0:
                #    dfOld4 = dfOldp[dfOldp['aprobadas'] >= optionApr]
                #    dfNew4 = dfNewp[dfNewp['aprobadas'] >= optionApr]
                #if optionRec >= 0:
                #    dfOld4 = dfOldp[dfOldp['rechazadas'] >= optionRec]
                #    dfNew4 = dfNewp[dfNewp['rechazadas'] >= optionRec]
                #-----------------------------------------

                # 0- Formula del ratio en LATEX:
                preselect =  selectRatioType()
                st.markdown('##')

                # 1- Genero el ratio que corresponda
                dfOld4 = genRatioBySelect(dfOld4, preselect)
                dfNew4 = genRatioBySelect(dfNew4, preselect)

                #st.write(dfOld4.sort_values(by=['aprobadas'], ascending=True).head(15))
                #1 dfOld4.drop(dfOld4.columns.difference(['terminal', 'tran-cde', 'ratio']), 1, inplace=True)
                #1 dfNew4.drop(dfNew4.columns.difference(['terminal', 'tran-cde', 'ratio']), 1, inplace=True)


                dfOldM, dfNewM = runPreprocesado(dfOld4, dfNew4)
                runTest(dfNewM, dfOldM, 1, preselect, dfNew4, dfOld4)


        elif page == 'Serie':
            #st.header('3 - SERIE TEMPORAL:')
            st.warning("3 - ANALISIS POR SERIE TEMPORAL:")
            pre_file, post_file = getFiles('3', 'desa_rcard_PRE.csv', 'desa_rcard_POS.csv' )
            if pre_file is not None and post_file is not None:
                dfOld3, dfNew3 = load_dataset(pre_file, post_file )

                st.markdown('##')
                radiosels = st.radio("Seleccion Tabla:", [label1, label2] , key = 'serie')
                if radiosels == label1:
                    dfSer = dfOld3
                if radiosels == label2:
                    dfSer = dfNew3
                
                st.markdown('##')
                st.subheader(f"{radiosels}:")

                runPred(dfSer)


    else:
        print("No hay archivos seleccionados")




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
    st.write(f'Tamaño X:{xx} Y:{yy}')

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
    st.write(f'Tamaño X:{xx} Y:{yy}')

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



def txVolFiidAlt(dfNew):

    df = dfNew[['term-fiid', 'tran-cde']].copy()
    df['cantTx']=1
    grouping_columns = ['term-fiid', 'tran-cde']
    columns_to_show = ['cantTx']
    df = df.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()

    c = alt.Chart(df).mark_circle().encode(x="term-fiid", y="tran-cde", size='cantTx')
    st.write(c)


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
    st.write(f'Tamaño X:{xx} Y:{yy}')

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
    st.write(f'Tamaño X:{xx} Y:{yy}')


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

    df2 = dfNew[['term-fiid', 'term-typ']].copy()
    df2 = df2.groupby(['term-fiid','term-typ']).agg(count = pd.NamedAgg('term-typ', 'count')).reset_index()
    chart = alt.Chart(df2).mark_point().encode(
        x='count',
        y='term-typ',
        tooltip=['term-typ','term-fiid','count']
    )
    st.altair_chart(chart)



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
    

def genRatioBySelect(dfRatio, radioselr):
    max = 0;
    def calRatio(fila):
        resultado = (fila["aprobadas"]/( fila["aprobadas"] + fila["rechazadas"] ))*100
        return resultado

    def calRatioV(fila):
        resultado = fila["aprobadas"]
        return resultado

    def calRatioLN(fila):
        resultado = np.log(fila["aprobadas"]+1)
        return resultado

    def calRatioMAX(fila):
        resultado = fila["aprobadas"]/( max )
        return resultado

    def calRatioVAR(fila):
        rho = fila["aprobadas"]/( fila["aprobadas"] + fila["rechazadas"] )
        resultado = rho*(1-rho)*100
        return resultado

    if radioselr == "Porcentual":
        dfRatio.loc[:, 'ratio'] = dfRatio.apply(calRatio, axis = 1)
    if radioselr == "Volumen":
        dfRatio['ratio'] = dfRatio.apply(calRatioV, axis = 1)
    if radioselr == "VolumenLN":
        dfRatio['ratio'] = dfRatio.apply(calRatioLN, axis = 1)
    if radioselr == "VolumenMAX":
        max = dfRatio["aprobadas"].max()
        dfRatio['ratio'] = dfRatio.apply(calRatioMAX, axis = 1)
    if radioselr == "Varianza":
        dfRatio['ratio'] = dfRatio.apply(calRatioVAR, axis = 1)

    return dfRatio.copy()



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def genRatio(dfRatio, radioselr ):
    """ Generar un nuevo DF reducido con la columna de ratios de aprobacion: """

    #dfRatio = df[['term-fiid', 'term-typ', 'tran-cde', 'resp-cde']]

    #print("agrego columna con resultado: OK para aprobado, NO para rechazo")
    #dfRatio["resultado"] = dfRatio["resp-cde"].apply(lambda x: 'ok' if (x == 0 or x == 1) else 'no')
    dfRatio["resultado"] = dfRatio["resp-cde"].astype(str).apply(lambda x: 'ok' if (x == '000' or x == '001'
        or x == '00' or x == '01' or x == '0' or x == '1' or x == ' 1' or x == ' 0') else 'no')

    #vif debugInfo:
    #v    st.write("genRatio 1 ---->")
    #v    st.write(dfRatio.loc[ dfRatio['term-typ'].isin(['82']) & dfRatio['tran-cde'].isin(['301100'])])
    #v    #print(dfRatio.loc[ dfRatio['term-typ'].isin(['82']) & dfRatio['tran-cde'].isin(['301100'])]['tran-cde'].unique())
    #v    st.write('---> genRatio 1 b')
    #v    st.write(dfRatio[dfRatio['term-typ'] == '82'])

    #Agrego una columna para contabilizar aprobadas vs rechazadas:
    #dfRatio.loc['counter'] =1
    dfRatio['counter'] =1
    grouping_columns = ['term-fiid', 'term-typ', 'tran-cde', 'resultado']
    columns_to_show = ['counter']
    dfRatio = dfRatio.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()


    #vif debugInfo:
    #v    st.write("genRatio 2 ---->")
    #v    dfRatio = dfRatio.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()
    #v    st.write(dfRatio.loc[ dfRatio['term-typ'].isin(['82']) & dfRatio['tran-cde'].isin(['301100'])])
    #v    st.write(dfRatio.sort_values(['tran-cde', 'counter'], ascending=[True, True]))
    #v    #duplicate = dfRatio[dfRatio.duplicated(['term-fiid', 'term-typ', 'tran-cde'])]
    #v    #print("Duplicate Rows: ")
    #v    #print(dfRatio[dfRatio['tran-cde'] == '301100'])


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

    dfRatio = genRatioBySelect(dfRatio, radioselr)

    return dfRatio.copy()
    #q.put(dfRatio)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def genRatioMatrix(df, eda):
    """ HEATMAP por transaccion (terminal FIID + Term Typ vs Codigo de transacion) """

    t0= time.time()

    #Pivoteo tabla dejando el ratio como valor en la matriz de terminal vs tran-cde:
    dfRatios = df[['terminal', 'tran-cde', 'ratio']]

    #opcion 1:
    dfRatios = dfRatios.pivot_table(index=['terminal'], columns='tran-cde', values='ratio')
    
    #opcion 2:
    #grouping_columns = ['terminal', 'tran-cde']
    #columns_to_show = ['ratio']
    #dfRatios = dfRatios.groupby(by=grouping_columns)[columns_to_show].max().unstack()


    t1 = time.time()
    if debugSteamlit == True:
        st.write(f'process time genRatioMatrix: {t1 - t0}')


    dfRatios = dfRatios.T
    return dfRatios.copy()
    #q.put(dfRatios)


def generateHeatmap(dfRatios, heatTit):
    """ HEATMAP diario: """

    t0 = time.time()
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

        t1 = time.time()
        st.write(f'Tamaño X:{xx} Y:{yy} - Tiempo: {t1 - t0:.5f}s')

        # Construct 2D histogram from data using the 'plasma' colormap
        #plt.hist2d(x, y,  cmap='plasma')
        #cb = plt.colorbar()
        #cb.set_label('Number of entries')
        #plt.title('Heatmap of 2D normally distributed data points')
        #plt.xlabel('x axis')
        #plt.ylabel('y axis')
        #plt.show()



def displayAltairHeatmapOrig(df, myscheme):

    global debugInfo
    ta0 = time.time()
    #2dfAltair = df[['terminal', 'tran-cde', 'ratio']].copy()
    dfAltair = df.copy()
    #print('----------------------------')
    #print(dfAltair.head(5))

    global ejex
    if (dfAltair.shape[0] == 0):
        st.warning("Tabla Vacia")
        return

    xx = dfAltair[ejex].nunique()
    yy = dfAltair['tran-cde'].nunique()

    #if debugInfo:
    #    print(dfAltair.shape)

    if debugInfo:
        st.write(dfAltair[[ejex, 'tran-cde', 'ratio']].sample(min( min(xx, yy) , 5)))
    
    if xx < 1 or yy < 1:
        st.write("No se puede armar matriz")
        return



    if myscheme == 'greenblue':
        myfill='#ebf7f1'
        myTooltip=[ejex, 'tran-cde', 'ratio:Q']
    else:
        myfill='#fcfcf2'
        #print('dfAltair-------------------------')
        #print(dfAltair.sample())
        myTooltip=[ejex, 'tran-cde', 
            'ratio:Q', 'aprobadas_pre:Q', 'rechazadas_pre:Q', 'aprobadas_pos:Q', 'rechazadas_pos:Q']



    if xx < 20:
        factorX = 40
    elif 20 <= xx and xx < 40:
        factorX = 30
    elif 40 <= xx and xx < 80:
        factorX = 20
    elif 80 <= xx and xx < 120:
        factorX = 12
    else:
        factorX = 8

    if yy < 20:
        factorY = 40
    elif 20 <= yy and yy < 40:
        factorY = 30
    elif 40 <= yy and yy < 80:
        factorY = 20
    elif 80 <= yy and yy < 120:
        factorY = 12
    else:
        factorY = 8

    myEjeX = ejex + ':O'
    factor = min(factorX, factorY)
    chart = alt.Chart(dfAltair).mark_rect(stroke='black', strokeWidth=0.5
    ).encode(
        x=myEjeX,
        y='tran-cde:O',
        color=alt.Color("ratio:Q", scale=alt.Scale(scheme=myscheme )), #, reverse=True) ),  #'ratio:Q'
        tooltip=myTooltip
        #stroke='black', strokeWidth=2
        #strokeWidth=alt.StrokeWidthValue(0, condition=alt.StrokeWidthValue(3, selection=highlight.name))
    ).configure_scale(
        bandPaddingInner=0.01
    ).configure_legend(
        gradientLength= max(200, yy*6),
        gradientThickness=15
    ).configure_view(
        fill=myfill,
    ).properties(
        #width='container',
        #height='container'
        width = max( 400, xx*factor),
        height = max( 400, yy*factor),
        autosize=alt.AutoSizeParams(
            type='fit',
            contains='padding'
        )
    ).configure_axis(
        labelFontSize=(factor+2)/2,
        titleFontSize=15
    )
    

    #nulls = chart.transform_filter(
    #"!isValid(datum.ratio)"
    #).mark_rect(opacity=0.5).encode(
    #alt.Color('ratio:N', scale=alt.Scale(scheme='greys'))
    #)
    #st.altair_chart(chart+nulls) #, use_container_width=True)
    st.altair_chart(chart)
    ta1 = time.time()
    if debugInfo:
        st.write(f'Tamaño X:*{xx}*, Y:*{yy}* - Tiempo: *{ta1 - ta0:.5f}s*')
    st.markdown(download_csv('Data Frame',df), unsafe_allow_html=True)

    sep()

    #chart.configure_view(
    #    continuousHeight=200,
    #    continuousWidth=200,
    #    strokeWidth=4,
    #    fill='#FFEEDD',
    #    stroke='blue'
    #)

    #nulls = c.transform_filter(
    #    "!isValid(datum.rate)"
    #    ).mark_rect(opacity=0.5).encode(
    #    alt.Color('rate:N', scale=alt.Scale(scheme='greys'))
    #)

    #st.altair_chart(c)



##DEPRECATE
#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
#def calRatios(dfOld, dfNew, radioselr):
#
#    dfOld = genRatio(dfOld, radioselr )
#    dfNew = genRatio(dfNew, radioselr )
#
#    dfOld = genRatioMatrix(dfOld, False )
#    dfNew = genRatioMatrix(dfNew, False )
#
#    return dfOld.copy(), dfNew.copy()





def displayAltairHeatmap(df, myscheme):

    global debugInfo
    ta0 = time.time()
    dfAltair = df.copy()

    global ejex
    if (dfAltair.shape[0] == 0):
        st.warning("Tabla Vacia")
        return


    xx = dfAltair[ejex].nunique()
    yy = dfAltair['tran-cde'].nunique()

    if debugInfo:
        st.write(dfAltair[[ejex, 'tran-cde', 'ratio']].sample(min( min(xx, yy) , 5)))
    
    if xx < 1 or yy < 1:
        st.write("No se puede armar matriz")
        return

    if myscheme == 'greenblue':
        myfill='#ebf7f1'
        myTooltip=[ejex, 'tran-cde', 'ratio:Q']
    else:
        myfill='#fcfcf2'
        myTooltip=[ejex, 'tran-cde', 
            alt.Tooltip('ratio:Q', format='.2f'),
            alt.Tooltip('aprobadas_pre:Q',  title = f'Apr {label1.lower()}'),
            alt.Tooltip('rechazadas_pre:Q', title = f'Rec {label1.lower()}'), 
            alt.Tooltip('aprobadas_pos:Q',  title = f'Apr {label2.lower()}'), 
            alt.Tooltip('rechazadas_pos:Q', title = f'Rec {label2.lower()}')  ]


    if xx < 20:
        factorX = 40
    elif 20 <= xx and xx < 40:
        factorX = 30
    elif 40 <= xx and xx < 80:
        factorX = 20
    elif 80 <= xx and xx < 120:
        factorX = 12
    else:
        factorX = 8

    if yy < 20:
        factorY = 40
    elif 20 <= yy and yy < 40:
        factorY = 30
    elif 40 <= yy and yy < 80:
        factorY = 20
    elif 80 <= yy and yy < 120:
        factorY = 12
    else:
        factorY = 8

    myEjeX = ejex + ':O'
    factor = min(factorX, factorY)

    rMin = dfAltair['ratio'].min()
    rMax = dfAltair['ratio'].max()

    #selector = alt.selection_single()
    scale2 = alt.Scale(
        #domain=[-100,  -75, -15, 0, 15,  75, 100],
        domain=[ rMin,  rMin*(0.75), rMin*(0.15), 0, rMax*(0.15),  rMax*(0.75), rMax],
        range=['darkred', 'red', 'yellow', 'lightyellow', 'greenyellow', 'green', 'darkgreen'],
        type='linear'
    )

    scalewb = alt.Scale(
        domain=[-100, 0, 100],
        range=['black', 'white', 'black'],
        type='linear'
    )

    selector = alt.selection_single(fields = [ejex, "tran-cde"] )#     , empty='none')
    scale = alt.Scale(scheme='brownbluegreen') #myscheme)
    color = alt.Color("ratio:Q", scale=scale2)
    colorwb = alt.Color("ratio:Q", scale=scalewb)


#    chart = alt.Chart(dfAltair).mark_rect(stroke='black', strokeWidth=0.5
#    ).encode(
#        x=myEjeX,
#        y='tran-cde:O',
#        color=alt.condition(selector, color, alt.value('lightgray')), #, reverse=True) ),  #'ratio:Q'
#        tooltip=myTooltip
#        #stroke='black', strokeWidth=2
#        #strokeWidth=alt.StrokeWidthValue(0, condition=alt.StrokeWidthValue(3, selection=highlight.name))
##    ).configure_scale(
##        bandPaddingInner=0.01
##    ).configure_legend(
##        gradientLength= max(200, yy*6),
##        gradientThickness=15
##    ).configure_view(
##        fill=myfill,
#    ).properties(
#        #opc1:
#        #width='container',
#        #height='container'
#        #opc2:
#        width = max( 400, xx*factor),
#        height = max( 400, yy*factor)
#        #opc3:
#        #autosize=alt.AutoSizeParams(
#        #    type='fit',
#        #    contains='padding'
#        #)
##    ).configure_axis(
##        labelFontSize=(factor+2)/2,
##        titleFontSize=15
#    ).add_selection(
#        selector
#    )



    base0 = alt.Chart(dfAltair).encode(
        x=myEjeX,
        y='tran-cde:O',
        color=alt.Color("ratio:Q", scale=color),
        tooltip=myTooltip
        #stroke='black', strokeWidth=2
        #strokeWidth=alt.StrokeWidthValue(0, condition=alt.StrokeWidthValue(3, selection=highlight.name))
#    ).configure_scale(
#        bandPaddingInner=0.01
#    ).configure_legend(
#        gradientLength= max(200, yy*6),
#        gradientThickness=15
#    ).configure_view(
#        fill=myfill,
    ).properties(
        #opc1:
        #width='container',
        #height='container'
        #opc2:
        width = max( 500, xx*factor),
        height = max( 500, yy*factor)
        #opc3:
        #autosize=alt.AutoSizeParams(
        #    type='fit',
        #    contains='padding'
        #)


#    ).configure_axis(
#        labelFontSize=(factor+2)/2,
#        titleFontSize=15
    )





    heat = base0.mark_rect(stroke='black', strokeWidth=0.5).encode(
        color=alt.condition(~selector, color, alt.value('white')),
        #strokeWidth=alt.StrokeWidthValue(0.5, condition=alt.StrokeWidthValue(1.5, selection=selector.name))
        strokeWidth= alt.condition(~selector, alt.value(0.5), alt.value(2)  ) 
    ).add_selection(
        selector
    )


    #box = base0.mark_rule( stroke='orange', strokeWidth=2)



    global dfpreg
    global dfposg



    base1 = alt.Chart(dfpreg).encode(
        y= alt.Y('resp-cde:N', sort='-x'),
        x='count:Q',
        tooltip=['resp-cde', 'count']
    ).transform_filter(
        selector
    ).properties(
        title=f"Codigos de respuesta {label1}:",
        height = {"step":15}
    ).transform_window(
        rank='rank(resp-cde)'
    ).transform_filter(
        alt.datum.rank <= 15
    )
    

    base2 = alt.Chart(dfposg).encode(
        y= alt.Y('resp-cde:N', sort='-x'),
        x='count:Q',
        tooltip=['resp-cde', 'count']
    ).transform_filter(
        selector
#    ).transform_filter(
#        alt.FieldValidPredicate(field='tran-cde', valid=True)
#    ).transform_filter(
#        'datum["tran-cde""] != null'
    ).properties(
        title=f"Codigos de respuesta {label2}:",
        height = {"step":15}
    ).transform_window(
        rank='rank(resp-cde)'
    ).transform_filter(
        alt.datum.rank <= 15
    )


    bars1 = base1.mark_bar().encode(
        color='resp-cde:N',
    )

    bars2 = base2.mark_bar().encode(
        color='resp-cde:N',
    )

    text1 = base1.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text='count:Q'
    )

    text2 = base2.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text='count:Q'
    )

    prebar = bars1 + text1
    posbar = bars2 + text2
    finalbar = prebar | posbar

    finalChart = (  (heat) & finalbar )
    
    #alt.layer(heat, cu(prebar | posbar )).resolve_scale(y='independent')

    #finalChart.configure_axisY(
    #    titleAngle=0, 
    #    titleY=-10,
    #    titleX=-60,
    #    labelPadding=160, 
    #    labelAlign='left'
    #)

    #finalChart.properties(
    #width=600 ,
    #height=1000,
    ##autosize=alt.AutoSizeParams(
    ##    type='fit',
    ##    contains='padding')
    #)
    

    #finalChart.resolve_legend(
    #    color="independent",
    #    size="independent"
    #)

    #finalChart.resolve_scale(x='shared', y='shared').resolve_axis(x='shared', y='shared' )

    #finalChart.resolve_scale(size='independent')

    st.altair_chart( finalChart )


    ta1 = time.time()
    if debugInfo:
        st.write(f'Tamaño X:*{xx}*, Y:*{yy}* - Tiempo: *{ta1 - ta0:.5f}s*')
    st.markdown(download_csv('Data Frame',df), unsafe_allow_html=True)
    sep()




@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calDifHeatmap(dfHeatNew, dfHeatOld, radioselr):
    dfConcat = dfHeatNew.append((dfHeatOld*(-1)), sort=False).copy()
    dfConcat = dfConcat.groupby(dfConcat.index).sum()
    if radioselr == "Porcentual":
        dfConcat *= 100
    return dfConcat.copy()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calDifHeatmapRaw(dfHeatNew, dfHeatOld, radioselr):
    #dfConcat = dfHeatNew.append((dfHeatOld*(-1)), sort=False).copy()
    #dfConcat = dfHeatNew.copy()
    #dfConcat.add(dfHeatOld*(-1), fill_value=0)
    #dfConcat = dfConcat.groupby(['terminal','tran-cde']).sum().reset_index()
    #dfHeatOld = dfHeatOld['ratio']*(-1)
    


    #dfConcat = pd.merge(dfHeatNew, dfHeatOld, on=['terminal', 'tran-cde']).set_index(['terminal', 'tran-cde']).sum(axis=1).reset_index()
    #dfConcat.columns = ['terminal', 'tran-cde', 'ratio']

    dfHeatOld['ratio'] = dfHeatOld['ratio'].apply(lambda x: x*(-1) ).copy()

    #if debugInfo:
    #    st.markdown('##')
    #    st.write(dfHeatOld[dfHeatOld.terminal == 'LINK-82'])
    #    st.write(dfHeatNew[dfHeatNew.terminal == 'LINK-82'])

    #dfConcat = pd.concat([dfHeatNew, dfHeatOld]).groupby(['terminal', 'tran-cde']).sum().reset_index()

    #dfHeatNew = dfHeatNew.set_index(['terminal', 'tran-cde'])
    #dfHeatOld = dfHeatOld.set_index(['terminal', 'tran-cde'])
    #dfConcat = dfHeatNew + dfHeatOld

    #dfConcat = dfHeatNew.copy() # 
    #st.write('Debug')
    #st.write(dfHeatNew.head(8))
    #st.write(dfHeatOld.head(8))
    #dfConcat = pd.concat([dfHeatNew, dfHeatOld])#, sort=False)


    #vif debugInfo:
    #v    st.write("calDifHeatmapRaw 1 ---->")
    #v    st.write(dfHeatOld[dfHeatOld.terminal == 'LINK-82'])
    #v    st.write(dfHeatNew[dfHeatNew.terminal == 'LINK-82'])

    dfConcat = pd.DataFrame()
    #dfConcat = pd.concat([dfHeatNew, dfHeatOld])

    global ejex
    dfConcat = pd.merge(dfHeatNew, dfHeatOld, how="outer", on=[ejex, "tran-cde"], suffixes=("_pos", "_pre"))
    dfConcat['ratio'] = dfConcat['ratio_pre'].fillna(0) + dfConcat['ratio_pos'].fillna(0)


    #4
    #if debugInfo:
    #    st.write(dfConcat.sample(200))

    #print(list(dfConcat.columns))
    grouping = [ejex, 'tran-cde', 'aprobadas_pos', 'rechazadas_pos', 'ratio_pos', 'aprobadas_pre', 'rechazadas_pre', 'ratio_pre']

    dfConcat = dfConcat.fillna(0).groupby(by=grouping).agg(ratio = pd.NamedAgg('ratio', 'sum')).reset_index()
    #dfConcat = dfConcat.groupby(['terminal','tran-cde']).agg(ratio = pd.NamedAgg('ratio', 'sum'), 
    #                                                         aprobadas = pd.NamedAgg('aprobadas', 'sum'),
    #                                                         rechazadas = pd.NamedAgg('rechazadas', 'sum')).reset_index()

    #vif debugInfo:
    #v    st.write("calDifHeatmapRaw 2 concat---->")
    #v    st.write(dfConcat[dfConcat.terminal == 'LINK-82'])



    #df.groupby('class')[['alcohol','hue']].agg(['sum','count'])
    #st.write(dfConcat.head(8))

    #dfConcat['ratio'] = dfConcat['ratio'].apply(lambda x: x*100)
    #if radioselr == "Porcentual":
    #    dfConcat *= 100

    return dfConcat.copy()



def generateDifHeatmap(dfRatios, heatTit):
    """ HEATMAP por diferencia entre dia previo y posterior: """

    #fig = go.Figure(data=go.Heatmap(
    #        z=dfRatios,
    #        x=dfRatios.index,
    #        y=dfRatios.columns,
    #        colorscale='Viridis'))
    #fig.update_layout(
    #    title='GitHub commits per day',
    #    xaxis_nticks=36)
    ##fig.show()
    #st.pyplot(fig)

    #altair

    #st.write(dfRatios.head(4))
    #df = dfRatios.melt('terminal', var_name='tran-cde', value_name='ratio').copy()
    #df = dfRatios.set_index('terminal').stack().reset_index(name='tran-cde').rename(columns={'level_1':'ratio'})
    #df = pd.DataFrame(dfRatios.to_records())
    #df = df.T
    #st.write(df.head(4))

    #dfRatios *=100
    t0 = time.time()
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
        t1 = time.time()
        st.write(f'Tamaño X:{xx} Y:{yy} - Tiempo: {t1 - t0:.5f}s')

        st.markdown(download_csv('Data Frame VARIACION',dfRatios),unsafe_allow_html=True)



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calHeatmapNegRaw(dfConcat):
    """ HEATMAP por variacion negativa: """
    dfNeg = dfConcat.copy()
    #dfNeg[dfNeg['ratio'] > 0.0] = 0.0
    dfNeg.loc[dfNeg['ratio'] > 0, 'ratio'] = 0
    return dfNeg.copy()




@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calHeatmapNeg(dfConcat):
    """ HEATMAP por variacion negativa: """

    #f = lambda x:  float("NaN") if x > 0.0 else x
    #dfConcat = dfConcat.copy()
    #dfConcat = dfConcat.applymap(f)
    dfNeg = dfConcat.copy()
    dfNeg[dfNeg > 0.0] = 0.0
    return dfNeg



def generateHeatmapNeg(dfConcatNeg, heatTit):
    """ HEATMAP por variacion negativa: """

    t0 = time.time()
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
        t1 = time.time()
        st.write(f'Tamaño X:{xx} Y:{yy} - Tiempo: {t1 - t0:.5f}s')

        st.markdown(download_csv('Data Frame VARIACION NEG',dfConcatNeg),unsafe_allow_html=True)
        # g = sns.heatmap(dfConcatNeg)
        # g.set_title(heatTit)
        # st.write(g)



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calHeatmapNegReducRaw(df, ratioMin, ratioMax):
    """ HEATMAP por variacion negativa reducido: """

    dfConcatNegRaw = df.copy()
    #dfConcatNegRaw[dfConcatNegRaw['ratio'] >= ratioMin and dfConcatNegRaw['ratio'] <= ratioMax ] = float("NaN")
    #dfConcatNegRaw['ratio'] = dfConcatNegRaw['ratio'].apply(lambda x: NaN if (x >= ratioMin and x <= ratioMax) else dfConcatNegRaw['ratio'] )
    dfConcatNegRaw = dfConcatNegRaw.loc[(dfConcatNegRaw['ratio'] >= ratioMin) & (dfConcatNegRaw['ratio']  <= ratioMax)] 

    dfConcatNegRaw = dfConcatNegRaw.dropna(how='all')
    dfConcatNegRaw = dfConcatNegRaw.dropna(axis=1, how='all')
    return dfConcatNegRaw


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calHeatmapNegReduc(dfConcatNeg, threshold):
    """ HEATMAP por variacion negativa reducido: """
    f = lambda x:  float("NaN") if x > threshold else x

    #print("Tamaño original: {}".format(dfConcatNeg.shape ) )
    dfConcatNeg = dfConcatNeg.applymap(f)
    dfConcatNeg = dfConcatNeg.dropna(how='all')
    dfConcatNeg = dfConcatNeg.dropna(axis=1, how='all')
    return dfConcatNeg


#@cuda.jit
#@jit
#@nb.njit(fastmath=True,error_model="numpy")
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

    with st.expander("Seleccionar transaccion :", expanded=False):
        optionTX = st.selectbox("", options= sorted(dfNew['tran-cde'].unique()) )
        if len(optionTX) > 0:
            dfNew = dfNew.loc[dfNew['tran-cde'] == optionTX ]


    df = dfNew[['tran-cde', 'date', 'time', 'amt']].copy()
    compras = df.loc[df['tran-cde'] == optionTX]

    compras['datetime'] = pd.to_datetime(2020*100000000 + df['date']* 10000  + df['time'] , format='%Y%m%d%H%M' )

    st.markdown('##')
    #st.markdown('##')
    #st.write("Muestra: Transaccion {}".format(optionTX))
    compras.set_index(['datetime'],inplace=True)

    #st.write(compras.head(20))
    #st.write(compras.tail(15))
    #st.write(compras.shape)
    #st.write(compras.dtypes)
    

    st.markdown('##')
    st.markdown('____')
    st.write("Modificar rango de la serie :")
    


    compras = compras[['amt']]
    column_3, column_4 = st.columns(2)
    with column_3:
        radioselag = st.radio("Rango de agrupacion:", ["1h", "30m", "15m", "5m", "1m"], key="agrupacion") # "40")
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
        radiosel = st.radio("Tipo de agregacion:", ["Minimo", "Maximo", "Promedio", "Suma", "Cantidad"], key="agregacion") # "50" )
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

    column_1, column_2 = st.columns(2)
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
    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv"> 📁 Download file</a>' % (name))
    return file




def runPreprocesado(dfOld, dfNew):
    # Creo una nueva columna calculando el ratio = aprob/(aprob+rech)
    #def calRatio(fila):
    #    resultado = fila['aprobadas']/( fila['aprobadas'] + fila['rechazadas'] )
    #    return resultado
#
    #radioselr = "Porcentual"
#
    ## 1- Calculo del ratio:
    #if radioselr == "Porcentual":
    #    dfOld['ratio'] = dfOld.apply(calRatio, axis = 1)
    #    dfNew['ratio'] = dfNew.apply(calRatio, axis = 1)


    #genRatioMatrix
    #dfRatios = df[['terminal', 'tran-cde', 'ratio']] vic
    #dfOld.drop(dfOld.columns.difference(['terminal', 'tran-cde', 'ratio']), 1, inplace=True)

    # ejex: Nombre de la columa principal que se transformara en el eje X de la matriz
    #       Puede ser por fiid-card, fiid-term, o terminal(fiid-term + tipo-term)
    #       lo obtengo de cualquiera de los dos dataframes, ya que deben ser el mismo sino da error
    global ejex
    ejex = dfOld.columns[0]

    dfOld = dfOld.pivot_table(index=[ejex], columns='tran-cde', values='ratio')
    dfOld = dfOld.T


    #dfNew.drop(dfNew.columns.difference(['terminal', 'tran-cde', 'ratio']), 1, inplace=True)
    dfNew = dfNew.pivot_table(index=[ejex], columns='tran-cde', values='ratio')
    dfNew = dfNew.T

    return dfOld.copy(), dfNew.copy()


def selectRatioType():
    with st.expander("⚗️ Formula del Calculo del ratio:", expanded=False):
        #st.write("Formula del Calculo del ratio:")
        radioselr = st.radio("Tipo de ratio:", ["Porcentual", "Volumen", "VolumenLN", "VolumenMAX", "Varianza"], key="tiporatio",help='Seleccionar la formula a aplicar para generar el valor (ratio) de cada elemento de la matriz ' )
        if radioselr == "Porcentual":
            st.latex(r''' \boxed{
            \rho = x / (x+y)   :   \begin{cases}
            x: & \sum_ \text{ transacciones aprobadas} \    \\
            y: & \sum_ \text{ transacciones rechazadas} \   
            \end{cases}
            }''')
        if radioselr == "Volumen":
            st.latex(r''' \boxed{ \rho = \sum_{}^{} \text{ transacciones aprobadas} }''')
        if radioselr == "VolumenLN":
            st.latex(r''' \boxed{ \rho = \ln \lparen \sum_{}^{} \text{ transacciones aprobadas} + 1 \rparen }''')
        if radioselr == "VolumenMAX":
            st.latex(r''' \boxed{
            \rho = x / (max(x))  
            }''')
        if radioselr == "Varianza":
            st.latex(r''' \boxed{ \sigma = \lparen \rho * (1 - \rho) \rparen }''')
        #st.markdown('____')

    return radioselr



def residualPlot(dfraw):
    # residual plot with seaborn library

    df = dfraw.copy()

    global ejex

    #df.dropna(subset = [ejex], inplace=True)

    #df = df.loc[df[ejex].isin(['0011', '0014']) ]

    #df['height']= df[ejex].str.strip() + "-" + df['tran-cde'].str.strip()
    #st.write(df)
    
    #myTooltip=[ejex, 'tran-cde', 'ratio:Q']
    st.write("##")
    myTooltip=[ejex, 'tran-cde', 
        alt.Tooltip('ratio:Q', format='.2f'),
        alt.Tooltip('aprobadas_pre:Q', title = 'Apr PRE'),
        alt.Tooltip('rechazadas_pre:Q', title = 'Rec PRE'), 
        alt.Tooltip('aprobadas_pos:Q', title = 'Apr POS'), 
        alt.Tooltip('rechazadas_pos:Q', title = 'Rec POS') ]


    scale = alt.Scale(
        domain=[-100, -50, 0, 50, 100],
        range=['darkred', 'red', 'yellow', 'green', 'darkgreen'],
        type='linear'
    )

    rMin = df['ratio'].min()
    rMax = df['ratio'].max()

    #selector = alt.selection_single()
    scale2 = alt.Scale(
        #domain=[-100,  -75, -15, 0, 15,  75, 100],
        domain=[ rMin,  rMin*(0.75), rMin*(0.15), 0, rMax*(0.15),  rMax*(0.75), rMax],
        range=['darkred', 'red', 'yellow', 'lightyellow', 'greenyellow', 'green', 'darkgreen'],
        type='linear'
    )




    selector = alt.selection_single(fields = [ejex, "tran-cde"])
    brush = alt.selection_interval()
    color = alt.Color('ratio:Q', scale=scale2)

    chart2 = alt.Chart(df).mark_point().encode(
        x=alt.X('tran-cde'),
        y=alt.Y('ratio:Q'),
        #color='ratio',
        #color=alt.Color('ratio', scale=scale),
        color=alt.condition(brush, color, alt.value('lightgray')),
        tooltip=myTooltip
        #).properties(
        #    width=1600,
        #    height=500
        ).properties(
            width = {"step":10}

        ).add_selection(
            brush
        )


    barsx = alt.Chart(df).mark_bar().encode(
        y=alt.X('tran-cde:N', sort='-x'),
        x=alt.Y('count(ejex):Q'),
        tooltip=[ejex, 'tran-cde', 'ratio']
    ).transform_filter(
        brush
    ).transform_window(
        rank='rank(tran-cde)'
    ).transform_filter(
        alt.datum.rank <= 15
    )

    barsy = alt.Chart(df).mark_bar().encode(
        x=alt.X('ratio:Q', bin=alt.Bin(maxbins=40)),
        y='count()',
        color=ejex,
        tooltip=[ejex, 'tran-cde', 'ratio']
    ).transform_filter(
            brush
        ).properties(
            width=600,
            height=400
        )

    st.altair_chart(chart2 & barsy)


def runTest(dfNew, dfOld, preprocesado, preselect, dfNewR, dfOldR ):

    if (preprocesado == 0):
        f = lambda x:  float("NaN") if x > threshold else x
        st.markdown('##')
        with st.expander("Reducir Matriz :", expanded=False):
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

        #if st.checkbox('Grafico en ALTAIR/SEABORN', value=False):
        #    useAltair = True
        #else:
        #    useAltair = False
        #st.markdown('##')

        #radioselr = st.radio("Motor Grafico:", ["Altair", "Seaborn"], key="41" )
        #if radioselr == "Altair":
        #    useAltair = True
        #if radioselr == "Seaborn":
        #    useAltair = False
        #st.markdown('##')

        # 0- Formula del ratio en LATEX:
        radioselr = selectRatioType()


        # 1- Calculo del ratio:
        dfOld = genRatio(dfOld, radioselr )
        dfNew = genRatio(dfNew, radioselr )

        #vif debugInfo:
        #v    st.write('---> runtest 1')
        #v    st.write(dfOld[dfOld.terminal == 'LINK-82'])
        #v    st.write(dfNew[dfNew.terminal == 'LINK-82'])

        dfOldRaw = dfOld[['terminal', 'tran-cde', 'ratio']].copy()
        dfNewRaw = dfNew[['terminal', 'tran-cde', 'ratio']].copy()
        dfHeatOld = genRatioMatrix(dfOld, False )
        dfHeatNew = genRatioMatrix(dfNew, False )

        #vif debugInfo:
        #v    st.write('---> runtest 2')
        #v    st.write(dfHeatOld['LINK-82'])
        #v    st.write(dfHeatNew['LINK-82'])

        # 2 - a partir de aca:
        # dfOld/dfNew son los df RAW            -> para graficar ALTAIR
        # dfHeatOld/dfHeatNew son los df MATRIX -> para graficar SEABORN
    else:
        # Utilizo el DF de preprocesado
        dfOldRaw = dfOldR 
        dfNewRaw = dfNewR 
        radioselr = preselect
        dfHeatNew = dfNew
        dfHeatOld = dfOld


    if debugSteamlit == True:
        st.subheader(f'Ratios {label1}:')
        st.write(dfHeatOld.head(10))


    dfConcatRaw = calDifHeatmapRaw(dfNewRaw, dfOldRaw, radioselr )
    if debugInfo == True:
        # 2- Calculo matriz de diferencias:
        #st.subheader('Variacion de ratios:')
        st.success("MATRIZ POR DIA Y DIFERENCIA TOTAL DE RATIOS:")

        if st.checkbox(f'Grafico {label1}', value=False, help=f'Matriz de ratios {label1}'):
            if useAltair == True:
                displayAltairHeatmapOrig(dfOldRaw, 'greenblue')
            else:
                generateHeatmap(dfHeatOld, f"Ratio {label1}")
                st.markdown(download_csv('Data Frame PREVIO',dfHeatOld),unsafe_allow_html=True)

        if st.checkbox(f'Grafico {label2}', value=False, help=f'Matriz de ratios {label2}' ):
            if useAltair == True:
                displayAltairHeatmapOrig(dfNewRaw, 'greenblue')
            else:
                generateHeatmap(dfHeatNew, f"Ratio {label2}")
                st.markdown(download_csv('Data Frame POSTERIOR',dfHeatNew),unsafe_allow_html=True)

        if useAltair == True:
            #st.write(dfNewRaw.head())
            #
            #st.write(dfConcatRaw.head())
            if st.checkbox('Grafico variacion Ratios', value=False, help='Diferencia de ratios entre ambos dias' ):
                #displayAltairHeatmap(dfConcatRaw, 'inferno')
                displayAltairHeatmapOrig(dfConcatRaw, 'redyellowgreen')
                
        else:
            #st.write(dfHeatNew.head())
            dfConcat = calDifHeatmap(dfHeatNew, dfHeatOld, radioselr )
            #st.write(dfConcat.head())
            generateDifHeatmap(dfConcat, "Variacion en ratios de aprobacion (TermTyp vs TranCde)")
        #st.markdown('##')
        #st.markdown('____')


    # 3- Calculo diferencias negativas:
    # Para este analisis elimino todos los valores positivos
    # ya que solo me interesa ver que transacciones disminuyeron en aprobaciones:
    #st.subheader('Variacion negativa de ratios:')
    #st.markdown('##')
    #st.success("VARIACION <NEGATIVA> DE RATIOS:")

    #simplifico y no uso el grafico para solo negativos:
    if useAltair == True:
        dfNegRaw = dfConcatRaw 
        #dfNegRaw = calHeatmapNegRaw(dfConcatRaw )
        #if st.checkbox('Grafico variacion negativa Ratios', value=False, help='Diferencia de ratios solo valores negativos' ):
        #    displayAltairHeatmap(dfNegRaw, 'inferno')
    else:
        dfNeg = dfConcat
        #dfNeg = calHeatmapNeg(dfConcat)
        #generateHeatmapNeg(dfNeg, "Variacion NEGATIVA en ratios de aprobacion")
    #st.markdown('##')
    #st.markdown('____')

    #  relplot:
    #  xx = dfNegRaw['terminal'].nunique()
    #  yy = dfNegRaw['tran-cde'].nunique()
    #  #fig_dims = (math.ceil(xx/3), math.ceil(yy/5))
    #  #fig, ax = plt.subplots(figsize=fig_dims)
    #  plt.xticks(rotation=90)
    #  #sns.scatterplot(data=df, x="card-fiid", y="tran-cde", ax=ax, size="cantTx", sizes=(5, 600), alpha=.8, palette="muted")
    #  fig, ax = plt.subplots()
    #  sns.relplot(
    #      data=dfNegRaw,
    #      x='terminal', y='tran-cde', hue="ratio")
    #  st.pyplot()
    #  st.write(f'Tamaño X:{xx} Y:{yy}')





    # 4- Calculo matriz negativa reducida:
    # Reduzco la matriz eliminando valores muy bajos de variacion
    # Hay una variacion de +/-X% en los ratios de aprobacion que es aceptable y no interesa analizar 
    # Por ejemplo si el dia anterior hubo 91% de aprobadas y el siguiente bajo a 89% (-2%), no lo analizo
    # A esa variable la denomino threshold:

    #st.subheader('Variacion negativa con tolerancia:')
    #st.markdown('##')
    st.success("MAPA DE VARIACION DE TRANSACCIONES:")

    if useAltair == True:
        ratioMin = dfNegRaw['ratio'].min()
        ratioMax = dfNegRaw['ratio'].max()
    else:
        ratioMin = min(dfNeg.min())
        ratioMax = max(dfNeg.max())


    selMin = float(ratioMin)
    selMax = float(ratioMin * 0.1)

    #st.write(f'Minimo:  {threshold}')
    dfNegRawMin = dfNegRaw
    dfNegRawReduc = dfNegRaw
    if st.checkbox('Reducir matriz', value=False):
        if stlit == False:
            threshold = -10
        else:
            with st.form(key='my_form1'):
                optionals = st.expander("📏 Por valores minimos y maximos:", False)
                #threshold = optionals.slider( "Umbral (threshold)", float(threshold), float(0), float(threshold * 0.1) )


                selMin, selMax = optionals.slider( "Seleccionar topes de variacion minima/maxima:", float(ratioMin), float(ratioMax), (selMin, selMax) )
                threshold = selMax


                with st.expander("📋 Por codigo de transaccion o terminal (Fila/Columna)  :", expanded=False):

                    optionTX = st.multiselect("Por TRANSACCION", options=dfNegRawMin['tran-cde'].unique() )
                    myquery = st.text_input('por FIID/TERMINAL:')

                    if myquery:
                        dfNegRawMin = dfNegRawMin[dfNegRawMin[ejex].str.contains(myquery, case=False)]
                    if len(optionTX) > 0:
                        dfNegRawMin = dfNegRawMin.loc[dfNegRawMin['tran-cde'].isin(optionTX) ]


                with st.expander("💳 Por cantidad de transacciones :", expanded=False):
                    col3, col4 = st.columns(2)
                    col3.write(f'**{label1}:**')
                    optiontx23 = col3.selectbox("Min de Transacciones (en total)", [ 0, 1, 10, 100, 1000] )
                    optionApr23 = col3.selectbox("Min de Aprobadas", [ 0, 1, 10, 100, 1000] )
                    optionRec23 = col3.selectbox("Min de Rechazadas", [ 0, 1, 10, 100, 1000] )

                    col4.write(f'**{label2}**')
                    optiontx24 = col4.selectbox("Min de Transacciones  (en total)", [ 0, 1, 10, 100, 1000] )
                    optionApr24 = col4.selectbox("Min de Aprobadas ", [ 0, 1, 10, 100, 1000] )
                    optionRec24 = col4.selectbox("Min de Rechazadas ", [ 0, 1, 10, 100, 1000] )

                    if optiontx23 >= 0:
                        dfNegRawMin = dfNegRawMin[((dfNegRawMin['aprobadas_pre'] + dfNegRawMin['rechazadas_pre'])  >= optiontx23)]
                    if optionApr23 >= 0:
                        dfNegRawMin = dfNegRawMin[(dfNegRawMin['aprobadas_pre'] >= optionApr23)]
                    if optionRec23 >= 0:
                        dfNegRawMin = dfNegRawMin[(dfNegRawMin['rechazadas_pre'] >= optionRec23)]
                    if optiontx24 >= 0:
                        dfNegRawMin = dfNegRawMin[((dfNegRawMin['aprobadas_pos'] + dfNegRawMin['rechazadas_pos'])  >= optiontx24)]
                    if optionApr24 >= 0:
                        dfNegRawMin = dfNegRawMin[(dfNegRawMin['aprobadas_pos'] >= optionApr24)]
                    if optionRec24 >= 0:
                        dfNegRawMin = dfNegRawMin[(dfNegRawMin['rechazadas_pos'] >= optionRec24)]





                st.form_submit_button(label='Aplicar')


            #with st.expander("💳 Reducir por cantidad de transacciones :", expanded=False):
            #    optiontx2 = st.selectbox("Min de Transacciones", [ 0, 1, 10, 100, 1000] )
            #    optionApr2 = st.selectbox("Min de Aprobadas", [ 0, 1, 10, 100, 1000] )
            #    optionRec2 = st.selectbox("Min de Rechazadas", [ 0, 1, 10, 100, 1000] )
            #    st.write(f"Opcion seleccionada: Transacciones>{optiontx2}, Aprobadas>{optionApr2}, Rechazadas>{optionRec2}")
            #    if optiontx2 >= 0:
            #        dfNegRawMin = dfNegRaw[(((dfNegRaw['aprobadas_pre'] + dfNegRaw['rechazadas_pre'])  >= optiontx2) & \
            #                                ((dfNegRaw['aprobadas_pos'] + dfNegRaw['rechazadas_pos'])  >= optiontx2))]
            #    if optionApr2 >= 0:
            #        dfNegRawMin = dfNegRawMin[(dfNegRawMin['aprobadas_pre'] >= optionApr2) & (dfNegRawMin['aprobadas_pos'] >= optionApr2)]
            #    if optionRec2 >= 0:
            #        dfNegRawMin = dfNegRawMin[(dfNegRawMin['rechazadas_pre'] >= optionRec2) & (dfNegRawMin['rechazadas_pos'] >= optionRec2)]




        st.markdown('##')
    st.markdown('##')
    if useAltair == True:
        dfNegRawReduc = calHeatmapNegReducRaw(dfNegRawMin, selMin, selMax)
        displayAltairHeatmap(dfNegRawReduc, 'inferno')
    else:
        dfConcatNeg = calHeatmapNegReduc(dfNeg, threshold )
        t0 = time.time()
        generateHeatmapNegReduc(dfConcatNeg, 'Variacion NEGATIVA con nivel de tolerancia' )
        t1 = time.time()
        #st.write(f"Tamaño matriz: {dfConcatNeg.shape} - Tolerancia: {selMin:.3f}~{selMax:.3f} - Tiempo: {t1 - t0:.5f}s")



    #interval = alt.selection_interval()
    #base = alt.Chart(dfNegRaw).mark_point().encode(
    #    y='term-typ',
    #    x='ratio',
    #    color=alt.condition(interval, 'term-fiid', alt.value('lightgray')),
    #    tooltip='term-fiid'
    #).properties(
    #    selection=interval
    #)
    #st.altair_chart(base) #, use_container_width=True)

    if (debug == True):
        dfNegRaw['term-fiid'] = dfNegRaw['terminal'].str.slice(0,4)
        dfNegRaw['term-typ'] = dfNegRaw['terminal'].str.slice(5,7)
        st.write(dfNegRaw.head())
        interval = alt.selection_interval()
        base = alt.Chart(dfNegRaw).mark_point().encode(
            x='term-typ:O',
            y='term-fiid:O',
            #color='ratio',
            color=alt.condition(interval, 'ratio', alt.value('lightgray')),
            tooltip=['tran-cde:O', 'term-fiid:O', 'ratio']
        ).properties(
            selection=interval
        ).interactive()
        st.altair_chart(base, use_container_width=True)

    st.markdown('##')
    st.success("REPORTE:")
    #if st.checkbox('Activar Reporte:', value=False):
    st.write(f'Reporte del analisis **{label1}** vs **{label2}**')

    #dfRepo = dfConcatRaw
    dfRepo = dfNegRawReduc
    
    cm = sns.light_palette("seagreen", reverse=True,  as_cmap=True)
    cm2 = sns.light_palette("seagreen", reverse=True, as_cmap=True)
    #s = df.style.background_gradient(cmap=cm)


    dfRepo = dfRepo.rename(columns={'aprobadas_pre': 'apr PRE', 'rechazadas_pre': 'rec PRE', 'aprobadas_pos': 'apr POS', 'rechazadas_pos': 'rec POS'})
    dfRepo['apr PRE'] = dfRepo['apr PRE'].astype(int)
    dfRepo['rec PRE'] = dfRepo['rec PRE'].astype(int)
    dfRepo['apr POS'] = dfRepo['apr POS'].astype(int)
    dfRepo['rec POS'] = dfRepo['rec POS'].astype(int)

    #col1, col2 = st.columns(2)
    #col1.write('Transacciones con *menor* ratio:')
    ##col1.write(dfRepo.sort_values(['ratio', "terminal"], ascending=[True,True])[['terminal', 'tran-cde','ratio']].head(20).style.background_gradient(cmap=cm) )
    #col1.write(dfRepo.sort_values(['ratio', ejex], ascending=[True,True])[[ejex, 'tran-cde', 'PRE apr', 'PRE rec', 'POS apr', 'POS rec', 'ratio']].head(20).style.background_gradient(cmap=cm, subset=['ratio']).format({"ratio": lambda x: "{:.2f}".format(x)}) ) #.hide_index()  )
    #col2.write('Transacciones con *mayor* ratio:')
    ##col2.write(dfRepo.sort_values(['ratio', "terminal"], ascending=[False,True])[['terminal', 'tran-cde','ratio']].head(20).style.background_gradient(cmap=cm2) )
    #col2.write(dfRepo.sort_values(['ratio', ejex], ascending=[False,True])[[ejex, 'tran-cde', 'PRE apr', 'PRE rec', 'POS apr', 'POS rec', 'ratio']].head(20).style.background_gradient(cmap=cm2, subset=['ratio']).format({"ratio": lambda x: "{:.2f}".format(x)}) ) #.hide_index() )
    ##col2.write(dfConcatRaw.sort_values(['ratio', "terminal"], ascending=[False,True]).head(15).style.highlight_max(axis=0))


    #col1, col2 = st.columns(2)
    #col1.metric("SPDR S&P 500", "$437.8", "-$1.25")
    #col2.metric("FTEC", "$121.10", "0.46%")



    st.write('Transacciones con *menor* ratio:')
    st.write(dfRepo.sort_values(['ratio', 'apr PRE'], ascending=[True,False])[[ejex, 'tran-cde', 'apr PRE', 'rec PRE', 'apr POS', 'rec POS', 'ratio']].head(20).style.background_gradient(cmap=cm, subset=['ratio']).format({"ratio": lambda x: "{:.2f}".format(x)}) ) #.hide_index()  )
    st.write('##')
    st.write('Transacciones con *mayor* ratio:')
    st.write(dfRepo.sort_values(['ratio', 'apr POS'], ascending=[False,False])[[ejex, 'tran-cde', 'apr PRE', 'rec PRE', 'apr POS', 'rec POS', 'ratio']].head(20).style.background_gradient(cmap=cm2, subset=['ratio']).format({"ratio": lambda x: "{:.2f}".format(x)}) ) #.hide_index() )
    st.write('##')
    
    #Debug:
    #residualPlot(dfNegRawReduc)

        #seleccionar transaccion:
        #with st.expander("📋 Codigos de respuesta por fiid / tran-cde  :", expanded=False):
        #    global dfpreg
        #    global dfposg
        #    #rechByFiid = st.selectbox("Por FIID:", options=sorted(dfNegRawReduc[ejex].str.slice(0,4).unique()) )
        #    rechByFiid = st.selectbox("Por FIID:", options=sorted(dfNegRawReduc[ejex].unique()) )
        #    rechByTran = st.selectbox("Por TRANSACCION:", options=sorted(dfNegRawReduc['tran-cde'].unique()) )
        #    dfRechPre = dfpreg[dfpreg[ejex].str.contains(rechByFiid, case=False)]
        #    dfRechPre = dfRechPre[dfRechPre['tran-cde'].str.contains(rechByTran, case=False)]
        #    dfRechPos = dfposg[dfposg[ejex].str.contains(rechByFiid, case=False)]
        #    dfRechPos = dfRechPos[dfRechPos['tran-cde'].str.contains(rechByTran, case=False)]
        #    #st.write(dfRechPre)
        #    #st.write(dfRechPos)
        #    col3, col4 = st.columns(2)
        #    col3.write('Codigos de respuesta (preimplementacion):')
        #    col3.write(dfRechPre.style.background_gradient(cmap=cm))
        #    #col3.write(dfRepo.sort_values(['ratio', ejex], ascending=[True,True])[[ejex, 'tran-cde','ratio']].head(20).style.background_gradient(cmap=cm).format({"ratio": lambda x: "{:.2f}".format(x)})  )
        #    col4.write('Codigos de respuesta (posimplementacion):')
        #    col4.write(dfRechPos.style.background_gradient(cmap=cm))
        #    #col4.write(dfRepo.sort_values(['ratio', ejex], ascending=[False,True])[[ejex, 'tran-cde','ratio']].head(20).style.background_gradient(cmap=cm2).format({"ratio": lambda x: "{:.2f}".format(x)}) )


            #if rechByFiid and rechByTran:
            #    dfNegRawMin = dfNegRawMin[dfNegRawMin[ejex].str.contains(myquery, case=False)]
            #if len(optionTX) > 0:
            #    dfRech = .loc[dfNegRawMin['tran-cde'].isin(optionTX) ]


        #listado de codigos de rechazos ordenados por rechazos:

        #residualPlot(dfNegRaw)


def runChunk(myfile):
    pd.options.mode.chained_assignment = None  # default='warn'

    cont = 0    
    chunkSize = 200000
    chunk_list3 = []
    chunk_list4 = []
    chunk_list5 = []
    grouping_columns3 = ['card-fiid', 'tran-cde', 'resp-cde']
    grouping_columns4 = ['terminal', 'tran-cde', 'resp-cde']

    input = myfile
    #input = 'produccion5\prod_PRE.csv'
    #outputTerm = f"{mys[:10]}_{mys[10:]}"

    index = input.find('.csv')
    outputRespTC = input[:index] + '_PREP' + input[index:]

    t0= time.time()
    filas = 0
    #for chunk in pd.read_csv('Caso trx71_pocho_210531.csv', sep='\t', low_memory=False, dtype={'TERM_FIID': str, 'PAN': str, 'TRAN_DAT': str, 'TRAN_TIM': str, 'TERM_TYPE': str, 'RESP_CDE': str}, chunksize=chunkSize):

    # Add a placeholder (PROGRESS)
    ntot = sum(1 for row in open(input, 'r'))
    st.write(f'Total registros a procesar: **{ntot}**' )
    latest_iteration = st.empty()
    pbar = st.progress(0)
    i = 0


    # 1 -- Por SQL:
    #for chunk in pd.read_csv(input, sep='\t', low_memory=False, dtype={'FIID_TERM': str, 'PAN': str, 'TRAN_DAT': str, 'TRAN_TIM': str, 'TERM_TYPE': str, 'RESP_CDE': str}, chunksize=chunkSize):
    # 2 -- Por RCARD:
    for chunk in pd.read_csv(input, sep=';', low_memory=False, dtype={'term-fiid': str, 'pan': str, 'term-typ': str, 'resp-cde': str, 'card-fiid': str,}, chunksize=chunkSize):

        # Update the progress bar with each iteration (PROGRESS)
        nciclo = i / ntot
        latest_iteration.text(f' % {round(nciclo*100, 2)}')
        pbar.progress( nciclo )


        filas = filas + chunk.shape[0]
        #Tomo solo las columnas que necesito, dropeo el resto y cambio el nombre:

        # 1 -- Por SQL:
        #chunk = chunk.filter(['FIID_TERM', 'FIID_CARD', 'TERM_TYPE', 'TRAN_CDE', 'RESP_CDE'])
        #chunk.columns = ['term-fiid', 'card-fiid','term-typ', 'tran-cde', 'resp-cde']

        # 2 -- Por RCARD:
        # term-fiid;term-typ;card-fiid;pan;tran-cde;tipo-tran;responder;resp-cde;reversal;date;time;amt
        chunk = chunk.filter(['term-fiid', 'card-fiid', 'term-typ', 'tran-cde', 'resp-cde'])
        chunk.columns = ['term-fiid', 'card-fiid','term-typ', 'tran-cde', 'resp-cde']

        #filtro 1: por fiid terminal
        chunk_filter = chunk[['term-fiid','term-typ', 'tran-cde', 'resp-cde']]
        #filtro 2: por fiid tarjeta
        chunk_filter2 = chunk[['card-fiid', 'tran-cde', 'resp-cde']]
        #filtro 3: resp-cde por tran-cde (card-fiid)
        chunk_filter3 = chunk[['card-fiid', 'tran-cde', 'resp-cde']]
        #filtro 4: resp-cde por tran-cde (term-fiid)
        chunk_filter4 = chunk[['term-fiid', 'tran-cde', 'resp-cde']]
        #filtro 5: resp-cde por tran-cde (term-fiid + card-fiid)
        chunk_filter5 = chunk[['term-fiid', 'card-fiid', 'tran-cde', 'resp-cde']]


        chunk_filter['terminal'] = chunk['term-fiid'].str.strip() + "-" + chunk['term-typ'].str.strip() 
        chunk_filter['aprobadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x == '000' or x == '001') else 0)
        chunk_filter['rechazadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x != '000' and x != '001') else 0)
        chunk_filter.drop(['term-fiid', 'term-typ', 'resp-cde'], axis=1, inplace=True)
        
        chunk_filter2['card-fiid'] = chunk['card-fiid']
        chunk_filter2['aprobadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x == '000' or x == '001') else 0)
        chunk_filter2['rechazadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x != '000' and x != '001') else 0)
        chunk_filter2.drop(['resp-cde'], axis=1, inplace=True)


        #chunk_filter3['tran-cde'] = chunk['tran-cde']
        #dic = df.groupby(by='tran-cde')['tipo-tran'].unique().apply(lambda x:x.tolist()).to_dict()

        #chunk_filter3['Subtotal'] = chunk_filter3.groupby('Client')['USD_Balance'].transform('sum')
        chunk_filter3['card-fiid'] = chunk['card-fiid']
        chunk_filter3['tran-cde'] = chunk['tran-cde']
        chunk_filter3['resp-cde'] = chunk['resp-cde']
        chunk_filter3['count']  = 1

        chunk_filter4['terminal'] = chunk['term-fiid'].str.strip() + "-" + chunk['term-typ'].str.strip() 
        #chunk_filter4['term-fiid'] = chunk['term-fiid']
        chunk_filter4['tran-cde'] = chunk['tran-cde']
        chunk_filter4['resp-cde'] = chunk['resp-cde']
        chunk_filter4['count']  = 1

        # Once the data filtering is done, append the chunk to list ok
        chunk_list3.append(chunk_filter3)
        df_concatRespC = pd.concat(chunk_list3)

        chunk_list4.append(chunk_filter4)
        df_concatRespT = pd.concat(chunk_list4)

        chunk_list5.append(chunk_filter3)
        df_concatRespTC = pd.concat(chunk_list3)
        chunk_list5.append(chunk_filter4)
        df_concatRespTC = pd.concat(chunk_list4)

        #print(f'chunk: {cont}')
        cont = cont + 1
        t1 = time.time()
        #print(f'process time: {t1 - t0}')

        # Update the progress bar with each iteration (PROGRESS)
        i = i + chunkSize        
        
       
    # concat the list into dataframe ok
    df_concatRespC  = df_concatRespC.groupby(by=grouping_columns3)['count'].sum().reset_index()
    df_concatRespT  = df_concatRespT.groupby(by=grouping_columns4)['count'].sum().reset_index()
    df_concatRespTC = df_concatRespT.append(df_concatRespC)
    #print('\n \n')
    #print("Resultado:")

    #print("**********")
    #print(df_concatRespTC)
    df_concatRespTC.to_csv(outputRespTC, sep=';', index=False)
    #print("**********")

    t2 = time.time()

    nciclo = 100
    latest_iteration.text(f' % {nciclo}')
    pbar.progress( nciclo )

    return outputRespTC;

    #print(f'total time: {t2 - t0}')
    #print(f'total registros: {filas}')
    #print('\n \n')


# -----------------------------------------------------------------------------
if __name__ == '__main__':
	main()
# -----------------------------------------------------------------------------