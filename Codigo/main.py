import streamlit as st
from PIL import Image
import funciones as ft
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)

plt.style.use("fivethirtyeight")

ft.config_page()

menu = st.sidebar.selectbox('Elige una sección:', ('Introducción', 'Juegos mas visualizados','Tipo de juego','Tendencia visualizaciones', 'Visualizaciones por mes', 'Comparativa','ML: Prediccion TOP 200 Rank Enero/2024. 1/2 - PRESENCIA','ML: Prediccion TOP 200 Enero/2024. 2/2 - VISUALIZACIONES','ML: Conclusiones predicción','Conclusiones'))

if menu == 'Introducción':
    ft.home()

elif menu == 'Juegos mas visualizados':
    fig_vistos_2016 = ft.juegos_vistos_2016()
    fig_vistos_2017 = ft.juegos_vistos_2017()
    fig_vistos_2018 = ft.juegos_vistos_2018()
    fig_vistos_2019 = ft.juegos_vistos_2019()
    fig_vistos_2020 = ft.juegos_vistos_2020()
    fig_vistos_2021 = ft.juegos_vistos_2021()
    fig_vistos_2022 = ft.juegos_vistos_2022()
    fig_vistos_total = ft.juegos_vistos_total()
    with st.expander("JUEGOS MAS VISUALIZADOS CADA AÑO:2016 :bow_and_arrow:"): 
        st.plotly_chart(fig_vistos_2016, use_container_width=True)
    with st.expander("JUEGOS MAS VISUALIZADOS CADA AÑO:2017 :bow_and_arrow:"):
        st.plotly_chart(fig_vistos_2017, use_container_width=True)
    with st.expander("JUEGOS MAS VISUALIZADOS CADA AÑO:2018 :bow_and_arrow:"):
        st.plotly_chart(fig_vistos_2018, use_container_width=True)
    with st.expander("JUEGOS MAS VISUALIZADOS CADA AÑO:2019 :bow_and_arrow:"):
        st.plotly_chart(fig_vistos_2019, use_container_width=True)
    with st.expander("JUEGOS MAS VISUALIZADOS CADA AÑO:2020 :bow_and_arrow:"):
        st.plotly_chart(fig_vistos_2020, use_container_width=True)
    with st.expander("JUEGOS MAS VISUALIZADOS CADA AÑO:2021 :bow_and_arrow:"):
        st.plotly_chart(fig_vistos_2021, use_container_width=True)
    with st.expander("JUEGOS MAS VISUALIZADOS CADA AÑO:2022 :bow_and_arrow:"):
        st.plotly_chart(fig_vistos_2022, use_container_width=True)
    with st.expander("JUEGOS MAS VISUALIZADOS EN 7 AÑOS:trophy:"):
        st.plotly_chart(fig_vistos_total, use_container_width=True)
    img =Image.open("EDA/imagen/minecraft.jpg")
    st.image(img,use_column_width="always")
elif menu == 'Tipo de juego':
    fig_tipo_juegos = ft.tipo_juegos()
    with st.expander("TIPO DE JUEGOS MAS VISUALIZADOS EN 7 AÑOS:european_castle:"):
        st.plotly_chart(fig_tipo_juegos, use_container_width=True)
    img =Image.open("ML/imagen/apex.png")
    st.image(img,use_column_width="always")
elif menu == 'Tendencia visualizaciones':
    fig_viewers = ft.viewers()
    with st.expander("HISTORIAL TOTAL VISUALIZACIONES CADA AÑO EN 7 AÑOS:dragon:"):
        st.plotly_chart(fig_viewers, use_container_width=True)
        
    img =Image.open("ML/imagen/amongus.jpg")
    st.image(img,use_column_width="always")
elif menu == 'Visualizaciones por mes':
    fig_viewers_mes = ft.viewer_mes()
    with st.expander("VISUALIZACIONES POR CADA MES:space_invader:"):
        st.plotly_chart(fig_viewers_mes, use_container_width=True) 
    img =Image.open("ML/imagen/wow.jpg")
    st.image(img,use_column_width="always")  
 
elif menu == 'Comparativa' :
    fig_pico_año = ft.pico_viewer()
    fig_comparativa = ft.comparativa()
    with st.expander("PICOS DE MAXIMAS VISUALIZACIONES AL AÑO (2016-2022):dagger_knife:"):
        st.plotly_chart(fig_pico_año, use_container_width=True)
    with st.expander("COMPARATIVA CON ALGUNOS FINALES DE EVENTOS DE FUTBOL:dagger_knife:"):
        st.markdown("""
                Vamos a comprar con los siguientes 3 eventos futbolísticos:\n
                * Máxima audiencia en la Final del mundial 2010 Sudafrica (España Holanda) en ESPAÑA - (Telecinco + Canal+)\n
                * Máxima audiencia en la Final del mundial 2022 Catar (Argentina-Francia) en ESPAÑA - (TVE +  Gol Mundial)\n
                * Máxima audiencia en la Final del mundial 2022 Catar (Argentina-Francia) en FRANCIA - (TF1 -Television francesa) \n
                """)
        st.plotly_chart(fig_comparativa, use_container_width=True)  
    img =Image.open("ML/imagen/lol.jpg")
    st.image(img,use_column_width="always")
elif menu == 'ML: Prediccion TOP 200 Rank Enero/2024. 1/2 - PRESENCIA':
    fig_binaria_inicial = ft.binaria_inicial()
    fig_feature_importance_clas = ft.feature_importance_clas()
    fig_score_clasificacion_accuracy = ft.score_clasificacion_accuracy()
    fig_score_clasificacion_recall = ft.score_clasificacion_recall() 
    fig_score_clasificacion_precision  = ft.score_clasificacion_precision()

    with st.expander("Creacion columna Clasificacion Binaria. Recuento: 1 presencia/ 0 Ausencia:dagger_knife:"):
        st.plotly_chart(fig_binaria_inicial, use_container_width=True) 
    with st.expander("Feature importance dataset clasificacion:dagger_knife:"):
        st.plotly_chart(fig_feature_importance_clas, use_container_width=True) 
    with st.expander("Modelos y sus hiperparámetros:dagger_knife:"):
        st.markdown("""
                Shuffle al dataset + Train_Test_Split + Estandarizar:\n
                * Regresión Logistica\n
                * SVC\n
                * Voting Hard y Soft a través de (Reg. Log + SVC + Rand. Forest)\n
                * Random Forest\n
                * Bagging  \n
                * AdaBoost\n
                * Gradient Boost Classifier\n
                * XGBoost Clasifier\n
                """)  
    with st.expander("Modelos + Score: Accuracy:dagger_knife:"):
        st.plotly_chart(fig_score_clasificacion_accuracy, use_container_width=True)
    with st.expander("Modelos + Score: Recall:dagger_knife:"):
        st.plotly_chart(fig_score_clasificacion_recall, use_container_width=True)
    with st.expander("Modelos + Score: Precision:dagger_knife:"):
        st.plotly_chart(fig_score_clasificacion_precision, use_container_width=True)
    with st.expander("Mapa de calor de la Matriz: Modelo XGBoost Classifier:dagger_knife:"):
        img =Image.open("ML/imagen/heatmap.png")
        st.image(img,use_column_width=False)
     
elif menu == 'ML: Prediccion TOP 200 Enero/2024. 2/2 - VISUALIZACIONES':
    
    fig_feature_importance_reg = ft.feature_importance_reg()
    fig_score_modelos_reg = ft.score_modelos_reg()

    with st.expander("Matriz correlación lineal columnas:dagger_knife:"):
        img =Image.open("ML/imagen/matriz_corr_original.png")
        st.image(img,use_column_width=False)
        img =Image.open("ML/imagen/matriz_corr_final.png")
        st.image(img,use_column_width=False)
    with st.expander("Feature importance:dagger_knife:"):
        st.plotly_chart(fig_feature_importance_reg, use_container_width=True)
    with st.expander("Modelos Regresion:Score R2:dagger_knife:"):
        st.plotly_chart(fig_score_modelos_reg, use_container_width=True)
        
elif menu == 'ML: Conclusiones predicción': 
    fig_juegos_enero_2024 = ft.juegos_enero_2024()
    fig_juegos_mas_crecimiento = ft.juegos_mas_crecimiento()

    with st.expander("Variacion Hours watched Enero 2023 a enero 2024:dagger_knife:"):
        st.plotly_chart(fig_juegos_enero_2024, use_container_width=True)
    with st.expander("10 juegos con mayor crecimiento Enero2023-Enero2024:dagger_knife:"):
        st.plotly_chart(fig_juegos_mas_crecimiento, use_container_width=True)
else:
    st.header('Conclusiones Visualizaciones 2016-2022 en Twitch')
    ft.conclusiones()
   
