
# Greydon O'Keefe, gokeefe@usc.edu
# ITP 216, Fall 2024
# Section: 32081
# Final Project
# Description:
    # Flask sqlLite db powered cryptocurrency data displayer, with capability to project future prices
    # through sklearn's polynomial fitting pipeline

import datetime
import io
import os
import sqlite3 as sl

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for, send_file
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
db = "coin_data.db"

@app.route('/')
def home():
    """
    allows users to select what type of data they want to see for which coin
    calls db_get_tickers to get unique ticker symbols from coin_data.db
    :return: returns home template
    """
    options = {
        "open_price": "Open Prices",
        "price_pct_change": "Percent Change"
    }
    return render_template(
        "home.html",
        tickers=db_get_tickers(),
        message="Choose a Cryptocurrency",
        options=options
    )

@app.route('/submit_ticker', methods=['POST'])
def submit_ticker():
    """
    endpoint accessed when user presses submit on home template
    :return: redirects user to template for the ticker & data selected from the home template form
    """
    session['ticker'] = request.form['ticker']
    if 'ticker' not in session or session['ticker'] == "":
        return redirect(url_for("home"))
    if "data_request" not in request.form:
        return redirect(url_for("home"))
    session["data_request"] = request.form["data_request"]
    return redirect(
        url_for(
            "ticker_current",
            data_request=session["data_request"],
            ticker=session["ticker"]
        )
    )

@app.route('/api/crypto/<data_request>/<ticker>')
def ticker_current(data_request, ticker):
    """
    displays a graph of the users desired data
    :param data_request: user selects to view price or percent change data
    :param ticker: the users selected cryptocurrency ticker
    :return: renders a graph of the historic data the user desires to see
    """
    return render_template(
        "ticker.html",
        data_request=data_request,
        ticker=ticker,
        project=False
    )

@app.route("/submit_projection", methods=['POST'])
def submit_projection():
    """
    handles the use case where a user inputs a desired date they want a projection for
    :return: redirects the user to the ticket template with functionality for projection of future data
    """
    if 'ticker' not in session:
        return redirect(url_for("home"))
    session["date"] = request.form["date"]

    return redirect(
        url_for(
            "ticker_projection",
            data_request=session["data_request"],
            ticker=session["ticker"]
        )
    )

@app.route("/api/crypto/<data_request>/projection/<ticker>")
def ticker_projection(data_request, ticker):
    """
    renders the ticker template, but with the functionality to project future data to the users desired date
    :param data_request: user selects to view prices or percent change data
    :param ticker: users selected cryptocurrency ticker
    :return: returns a graph with historic data and future projections
    """
    return render_template(
        "ticker.html",
        data_request=data_request,
        ticker=ticker,
        project=True,
        date=session["date"]
    )

@app.route("/fig/<data_request>/<ticker>")
def fig(data_request, ticker):
    """
    allows functionality to display the user's desired graph
    :param data_request: user selects to view prices or percent change data
    :param ticker: user's selected cryptocurrency ticker
    :return: a png file capable of displaying the matplotlib graph
    """
    fig = create_figure(data_request, ticker)

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

def create_figure(data_request, ticker):
    """
    conditionally creates the graph of the users desired data, includes strictly historical data or a combination of historical and projection
    if user desires projection, uses sklearn's pipeline to create a polynomial fit of the user's selected data and project to their desired date
    :param data_request: user selects to view prices or percent change data
    :param ticker: user's selected cryptocurrency ticker
    :return: a matplotlib figure containing the user's desired graph
    """
    df = db_create_dataframe(data_request, ticker)

    df['datemod'] = df['Date'].map(datetime.datetime.toordinal)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    if not 'date' in session:
        fig.suptitle(f"{data_request} for {ticker}")
        ax.plot(df['Date'], df['Values'], label=data_request, color='blue')
        ax.set(xlabel='Date', ylabel=data_request)
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        fig.subplots_adjust(bottom=0.2)
    else:
        # filter data for the last year
        two_years_ago = df['Date'].max() - pd.Timedelta(days=730)
        recent_df = df[df['Date'] >= two_years_ago]

        # extract recent data for ml
        X = recent_df['datemod'].values.reshape(-1, 1)
        y = recent_df['Values'].values

        # user-selected prediction date
        dt = datetime.datetime.strptime(session['date'], "%m/%d/%y")
        draw = datetime.datetime.toordinal(dt)

        # polynomial regression
        degree = 3
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(X, y)

        # generate a range of dates for the entire curve (historical + prediction range)
        last_date = df['datemod'].iloc[-1]
        date_range = np.linspace(last_date, draw, 200).reshape(-1, 1)

        smooth_predictions = poly_model.predict(date_range)
        date_range_dates = [datetime.datetime.fromordinal(int(d)) for d in date_range.flatten()]

        # plot historical data
        ax.plot(
            df["Date"],
            df["Values"],
            color="blue",
            label="Historical Data"
        )

        # plot prediction curve
        ax.plot(
            date_range_dates,
            smooth_predictions,
            color="orange",
            label="Prediction (Curve)",
        )

        # plot labels and legend
        fig.suptitle(f"Prediction for {data_request} on {session['date']} ({ticker})")
        ax.set(xlabel="Date", ylabel=data_request)
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        fig.subplots_adjust(bottom=0.2)

    return fig


def db_create_dataframe(data_request, ticker):
    """
    :param data_request: user selects to view prices or percent change data
    :param ticker: user's selected cryptocurrency ticker
    :return: returns a pandas df based upon the user's selected data, which is then used to create graph and other functionality
    """
    conn = sl.connect(db)

    stmt = "SELECT date, " + data_request + " FROM coin_data WHERE ticker = ? ORDER BY date;"
    df = pd.read_sql_query(
        stmt,
        conn,
        params=(ticker,)
    )

    df['date'] = pd.to_datetime(df['date'])
    df.columns = ['Date', 'Values']

    conn.close()

    return df

def db_get_tickers():
    """
    :return: list of all unique tickers within coin_data.db, obtained through SQL query
    """
    conn = sl.connect(db)
    curs = conn.cursor()

    table = "coin_data"
    stmt = "SELECT DISTINCT ticker from " + table
    data = curs.execute(stmt)
    # sort a set comprehension for unique values
    tickers = sorted({result[0] for result in data})
    conn.close()
    return tickers


@app.route('/<path:path>')
def catch_all():
    """
    :return: if user enters a miscellaneous url, route them to home
    """
    return redirect(url_for("home"))


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)