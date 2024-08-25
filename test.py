import plotly.express as px
df = px.data.gapminder().query("year == 2007")
df.to_csv('gapminder.csv')