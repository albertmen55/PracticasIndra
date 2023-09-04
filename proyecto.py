import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load():
    if selected_year != '2010':
        year_anterior = str(int(selected_year) - 1)
        # Filtrar los datos para la empresa seleccionada
        selected_row = dat[dat['company'] == selected_company].iloc[0]
        dividends = selected_row['2010':year_anterior].values
        real_dividend = selected_row[selected_year]

        dividends_series = pd.Series(dividends)
        dividends_series = pd.to_numeric(dividends_series, errors='coerce').fillna(0)

        model_arima = ARIMA(dividends_series, order=(1, 1, 1))
        results_arima = model_arima.fit()
        forecast_arima = results_arima.forecast(steps=1)

        # Mostrar resultados en Streamlit
        st.header(f"Predicción de Dividendos para {selected_company} en {selected_year} (ARIMA):")
        forecast = results_arima.get_forecast(steps=1)

        # Agregar la predicción del año 2020 a la matriz de correlación
        correlation_matrix = dat.loc[:, '2010':selected_year].corr()

        # Dividir la pantalla en dos columnas
        col1, col2 = st.columns(2)

        # Renderizar el gráfico de avance de dividendos en la primera columna
        with col1:
            years = dat.columns[dat.columns.get_loc('2010'):dat.columns.get_loc(year_anterior) + 1]
            dividend_values = selected_row[years].values
            predicted_dividend = forecast.predicted_mean.values[0]
            years = [int(year) for year in years]

            years.append(selected_year)
            dividend_values = list(dividend_values)
            dividend_values.append(predicted_dividend)

            plt.figure(figsize=(10, 6))
            plt.plot(years, dividend_values, marker='o')
            plt.title(f"Avance de Dividendos para {selected_company}")
            plt.xlabel("Año")
            plt.ylabel("Dividendos")
            plt.grid(True)
            st.pyplot(plt)

        # Renderizar el mapa de calor en la segunda columna
        with col2:
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f"Matriz de Correlación de Dividendos para {selected_company}")
            st.pyplot(plt)

        # Dividir la pantalla en dos columnas
        co1, co2, co3 = st.columns(3)

        with co1:

            st.subheader(f"Predicción para {selected_year}:")
            st.markdown(predicted_dividend)

        with co2:

            st.subheader(f"Valor Real de {selected_year}:")
            st.markdown(real_dividend)

        with co3:

            st.subheader("Error cometido:")
            st.markdown(abs(real_dividend - predicted_dividend))
    else:
        st.subheader("Introduzca un año mayor que 2010")


def main():
    st.title("Predicción de dividendos de empresas")
    st.divider()
    # Cargar los datos
    data = pd.read_csv('./div.csv')
    columns_to_drop = ['quartile', 'country', 'economic_sector', 'sub_industry', 'pollution', 'metric','wa_10','wa_11', 'wa_12', 'wa_13', 'wa_14', 'wa_15', 'wa_16', 'wa_17', 'wa_18', 'wa_19', 'wa_20']
    data_filtered = data.drop(columns=columns_to_drop)
    dat = data_filtered.fillna(0)
    # Crear una lista de empresas para la selección
    companies = dat['company'].unique().tolist()
    selected_company = st.selectbox('Selecciona una empresa:', companies)
    selected_year = st.selectbox('Selecciona un año para la predicción:', dat.columns[1:])
    st.divider()
    load()

if __name__ == "__main__":
    main()