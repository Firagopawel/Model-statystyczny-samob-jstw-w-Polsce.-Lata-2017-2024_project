### Biblioteki 

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
#from matplotlib import pyplot as plt


#Wczytaj dane z Excela
df = pd.read_excel(r"C:/Users/pawel/OneDrive/Desktop/TwojKatalog/Proby_samobojcze/Stan zdrowia.xlsx")
spol = pd.read_excel(r"C:\Users\pawel\OneDrive\Desktop\TwojKatalog\Proby_samobojcze\Status społeczny.xlsx")
wyk = pd.read_excel(r"C:\Users\pawel\OneDrive\Desktop\TwojKatalog\Proby_samobojcze\Wykształcenie.xlsx")
wiek = pd.read_excel(r"C:\Users\pawel\OneDrive\Desktop\TwojKatalog\Proby_samobojcze\Gruoa wiekowa.xlsx")
mapa = pd.read_excel(r"C:\Users\pawel\OneDrive\Desktop\TwojKatalog\Proby_samobojcze\Mapa_samobójstw_Polska2024.xlsx")
mapka = pd.read_excel(r"C:\Users\pawel\OneDrive\Desktop\TwojKatalog\Proby_samobojcze\Mapa_samobójstw_Polska2024 — kopia.xlsx")


### Wybierz kolumny numeryczne oprócz 'Rok'
numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'Rok']




### Wykresy
fig0 = px.line(spol, x="Rok", y="Rolnik", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig2 = px.bar(spol, x="Rok", y="Bezrobotny", color='KWP', title="Liczba samobójstw w latach 2017-2024")
fig3 = px.bar(spol, x="Rok", y="Uczeń", color='KWP', title="Liczba samobójstw w latach 2017-2024")
fig4 = px.bar(spol, x="Rok", y="Mężczyźni", color='KWP', title="Liczba samobójstw w latach 2017-2024")
fig5 = px.bar(spol, x="Rok", y="Kobiety",color='KWP', title="Liczba samobójstw w latach 2017-2024")
fig6 = px.funnel(spol, x="Rok", y="Liczba całkowita samobójstw", color='KWP', title="Liczba samobójstw w latach 2017-2024")

fig7 = px.area(wyk, x="Rok", y="Podstawowe", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig8 = px.area(wyk, x="Rok", y="Gimnazjalne", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig9 = px.area(wyk, x="Rok", y="Zasadnicze zawodowe", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig10 = px.area(wyk, x="Rok", y="Średnie", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig11= px.area(wyk, x="Rok", y="Wyższe", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")

fig14 = px.line(wiek, x="Rok", y="Grupa wiekowa - '19-24'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig15 = px.area(wiek, x="Rok", y="Grupa wiekowa - '25-29'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig16 = px.area(wiek, x="Rok", y="Grupa wiekowa - '25-29'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig17 = px.area(wiek, x="Rok", y="Grupa wiekowa - '30-34'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig18 = px.area(wiek, x="Rok", y="Grupa wiekowa - '35-39'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig19 = px.area(wiek, x="Rok", y="Grupa wiekowa - '40-44'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig20 = px.area(wiek, x="Rok", y="Grupa wiekowa - '45-49'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig21 = px.area(wiek, x="Rok", y="Grupa wiekowa - '50-54'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig22 = px.area(wiek, x="Rok", y="Grupa wiekowa - '55-59'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig23 = px.area(wiek, x="Rok", y="Grupa wiekowa - '60-64'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig24 = px.area(wiek, x="Rok", y="Grupa wiekowa - '65-69'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig25 = px.area(wiek, x="Rok", y="Grupa wiekowa - '70-74'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")
fig26 = px.area(wiek, x="Rok", y="Grupa wiekowa - '70-74'", color='KWP', markers=True, title="Liczba samobójstw w latach 2017-2024")

fig27 = px.scatter_map(mapa, lat="lat", lon="long",color = "Liczba samobójstw w 2024", size="Liczba samobójstw w 2024", zoom=10)
fig27.update_traces(cluster=dict(enabled=True))

fig28 = px.scatter_map(mapa, lat="lat", lon="long", size="Wskaźnik samobójstw", color ="Wskaźnik samobójstw", zoom=5)
fig28.update_traces(cluster=dict(enabled=True))

fig29 = px.scatter(mapa,  x="Rok", y="Wskaźnik samobójstw", color = "Wskaźnik samobójstw",size='Wskaźnik samobójstw', title="Polska.Wskaźnik samobójstw w roku 2024")
fig30 = px.scatter(mapa,  x="Rok", y="Liczba samobójstw w 2024", color = "Liczba samobójstw w 2024",size='Liczba samobójstw w 2024', title="Polska.Liczba samobójstw w roku 2024")

fig31 = px.scatter_map(mapa, lat="lat", lon="long",color = "Problemy alkoholowe", size="Problemy alkoholowe", zoom=8)
fig31.update_traces(cluster=dict(enabled=True))

fig32 = px.scatter_map(mapa, lat="lat", lon="long",color = "Zaburzenia psychiatryczne", size="Zaburzenia psychiatryczne", zoom=8)
fig32.update_traces(cluster=dict(enabled=True))

fig33 = px.scatter_map(mapa, lat="lat", lon="long",color = "Bezrobocie", size="Bezrobocie", zoom=8)
fig33.update_traces(cluster=dict(enabled=True))

fig34 = px.scatter(mapa, x= "Zaburzenia psychiatryczne", y=  "Bezrobocie", color="Wskaźnik samobójstw")
fig35 = px.scatter(mapa, x= "Zaburzenia psychiatryczne", y=  "Bezrobocie", color="KWP")
fig36 = px.scatter(mapa, x= "Problemy alkoholowe", y=  "Bezrobocie",color="Wskaźnik samobójstw")
fig38 = px.scatter(mapa,  "Problemy alkoholowe", y=  "Bezrobocie",color="KWP")


numeric_mapka = mapka.select_dtypes(include=['number'])
corr_matrix = numeric_mapka.corr()

fig37 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r")



### Inicjalizacja aplikacji
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

models = {'Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}



navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src="assets/sadnes.jpg",width= "200" ,height="220px")),
                dbc.Col(dbc.NavbarBrand("Model statystyczny samobójstw w Polsce. Lata 2017-2024", className="ms-2",style={"fontSize": "40px"})),
            ], className="g-0"),
            href="#",
            #style={"textDecoration": "none"},
        )
    ]),
    color="dark",
    dark=True
)


#Zawartość zakładki

tab1_content = dbc.Card([
    dbc.CardBody([
        html.P("Samobójstwa w kategorii stanu zdrowia", className="card-text"),
        
        
                html.H5("Wykres rozrzutu", className="card-title"),
                html.H6("Wybierz kolumnę do wykresu: "),
                dcc.Dropdown(
                     id='column-dropdown',
                     options=[{'label': col, 'value': col} for col in numeric_columns],
                     value=numeric_columns[0]),
                dcc.Graph(id='line-chart'),         
        
    ])
])

tab2_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                   
                    dcc.Graph(figure=fig0),
                    dcc.Graph(figure=fig2),
                    dcc.Graph(figure=fig3),
                    #dcc.Graph(figure=fig4),
                    #dcc.Graph(figure=fig5),
                    dcc.Graph(figure=fig6)
                ]
            ),
            
        ),
        width=6  # tutaj ustawiasz szerokość kolumny
    )
])


tab3_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                   
                    dcc.Graph(figure=fig7),
                    dcc.Graph(figure=fig8),
                    dcc.Graph(figure=fig9),
                    dcc.Graph(figure=fig10),
                    dcc.Graph(figure=fig11),

                ]
            ),
            
        ),
        width=6  # tutaj ustawiasz szerokość kolumny
    )

])


tab4_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                   
                    dcc.Graph(figure=fig14),
                    dcc.Graph(figure=fig15),
                    dcc.Graph(figure=fig16),
                    dcc.Graph(figure=fig17),
                    dcc.Graph(figure=fig18),
                    dcc.Graph(figure=fig19),
                    dcc.Graph(figure=fig20),
                    dcc.Graph(figure=fig21),
                    dcc.Graph(figure=fig22),
                    dcc.Graph(figure=fig23),
                    dcc.Graph(figure=fig24),
                    dcc.Graph(figure=fig25),
                    dcc.Graph(figure=fig26),
                ]
            ),
            
        ),
        width=6  # tutaj ustawiasz szerokość kolumny
    )
])

tab5_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(figure=fig27),
                    dcc.Graph(figure=fig30),
        
                ]
            ),
            
        ),
       
    )
])

tab6_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(figure=fig28),
                    dcc.Graph(figure=fig29),
        
                ]
            ),
            
        ),
       
    )
])
 

tab7_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(figure=fig31),
                    
        
                ]
            ),
            
        ),
       
    )
])
 
tab8_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(figure=fig32),
                    
        
                ]
            ),
            
        ),
       
    )
])

tab9_content = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(figure=fig33),
                    
                    
        
                ]
            ),
            
        ),
       
    )
])

tab10_content = dbc.Card([
    dbc.CardBody([
        html.P("Uczenie maszynowe w analizie danych", className="card-text"),
        
        
                html.H5("Model predykcyjny"),
                html.P("Wybierz model:"),
                dcc.Dropdown(
                    id='ml-regression-x-dropdown',
                    options=["Regression", "Decision Tree", "k-NN"],
                    value='Decision Tree',
                    clearable=False
                ),
                dcc.Graph(id="ml-regression-x-graph"),
    ])
])

tab11_content = dbc.Card([
    dbc.CardBody([
        html.P("Analiza danych -prognozowanie", className="card-text"),
        
        
                html.H5("Model predykcyjny"),

                dcc.Graph(figure=fig34),
                dcc.Graph(figure=fig35),
                dcc.Graph(figure=fig36),
                dcc.Graph(figure=fig38),
                dcc.Graph(figure= fig37),
                          
                              
            
                
    ])
])

#Layout z zakładkami
app.layout = html.Div([
    navbar,
    
    dbc.Tabs([
        dbc.Tab(tab1_content, label="Wykres samobójstw według stanu zdrowia"),
        dbc.Tab(tab2_content,label="Wykres samobójstw według statusu społecznego"),
        dbc.Tab(tab3_content,label="Wykres samobójstw według wykształcenia"),
        dbc.Tab(tab4_content,label="Wykres samobójstw według grupy wiekowej"),
        dbc.Tab(tab5_content,label="Mapa samobójstw w Polsce - 2024 rok"),
        dbc.Tab(tab6_content,label="Mapa wskaźników samobójstw w Polsce - 2024 rok"),
        dbc.Tab(tab7_content,label="Mapa samobójstw w Polsce -ALKOHOLIZM -2024 rok"),
        dbc.Tab(tab8_content,label="Mapa samobójstw w Polsce -ZABURZENIA PSYCHIATRYCZNE -2024 rok"),
        dbc.Tab(tab9_content,label="Mapa samobójstw w Polsce -BEZROBOCIE -2024 rok"),
        dbc.Tab(tab10_content,label="Uczenie maszynowe w analizie danych"),
        dbc.Tab(tab11_content,label="Prognozowanie w analizie danych"),


        
    ]),
   
])


#Callback do aktualizacji wykresu
@app.callback(
    Output('line-chart', 'figure'),
    Input('column-dropdown', 'value')
)
def update_chart(selected_column):
    fig1= px.scatter(df, x="Rok", y=selected_column, color='KWP',
                  title=f'{selected_column} wg wojewódzkich komend Policji [KWP]')
    return fig1




@app.callback(
    Output("ml-regression-x-graph", "figure"),
    Input("ml-regression-x-dropdown", "value")
)
def train_and_display(name):
    # Wczytanie danych
    #mapa = pd.read_excel( r"C:\Users\pawel\OneDrive\Desktop\TwojKatalog\Proby_samobojcze\Mapa_samobójstw_Polska2024.xlsx")

    # Wybierz odpowiednie kolumny (przykład – dostosuj do twojego pliku)
    X = mapa["Zaburzenia psychiatryczne"].values.reshape(-1, 1)
    y = mapa["Problemy alkoholowe"]

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Wybór i trening modelu
    model = models[name]()
    model.fit(X_train, y_train)

    # Zakres predykcji
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    # Rysowanie wykresu

    fig34 = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='Train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='Test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name='Prediction', mode='lines')
    ])

    fig34.update_layout(title="Regresja liczby samobójstw w kategorii leczenia psychiatrycznegp i zaburzeń alkoholowych w 2024 roku")

    return fig34




#Uruchomienie aplikacji
if __name__ == '__main__':
    app.run(debug=True, port=8051)
