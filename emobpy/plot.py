"""
This module contains a class that can be used to visualise the data. There are different visualisation functions.
"""

import pandas as pd
import os

try:
    import plotly.graph_objects as go
    import plotly.colors as pc
    from plotly.subplots import make_subplots
    from IPython.display import display, HTML
except ImportError:
    raise Exception("This plotly code only works within a jupyter notebook")

from .functions import balance
from .constants import CWD
from .tools import display_all
from .logger import get_logger

logger = get_logger(__name__)

class NBplot:
    
    """
    Work in Jupyter notebooks only.
    Set of plots for a single time series and groups.
    Three kind of plots:

    - self.sgplot_dp(tscode) for driving profiles
    - self.sgplot_ga(tscode) for grid availability profiles
    - self.sgplot_ged(tscode) for grid electricity demand profiles
    - tscode: time series code (string of profile name)

    self.__init__(db)
        db is an instance of a DataBase class that contains the time series.

    """

    def __init__(self, db):
        self.db = db
        try:
            display_all()
        except:
            pass

    def sgplot_dp(self, tscode, rng=None, to_html=False, path=None):
        """
        Plot of a single driving profile.

        Args:
            tscode (str): Time series code. Eg. 'ev_user_W3_85e59_avai_65t2p'
            rng (tuple): (a,b) index if only part of timeseries should be copied. Defaults to None.
            to_html (bool): Save as a html file. Defaults to False.
            path (str): Path if plot should be saved to file. Defaults to None.

        Returns:
            plotly.plot: Plot object.
        """
        self.db.update()
        if rng is None:
            df = self.db.db[tscode]["timeseries"].copy()
        else:
            df = self.db.db[tscode]["timeseries"].iloc[rng[0]: rng[1]].copy()
        if self.db.db[tscode]["kind"] != "driving":
            raise Exception(
                "code '{}' does not correspond to a driving profile".format(tscode)
            )

        cnt = df.groupby([df.index, "state"])["state"].count()
        cn = (
            pd.DataFrame(cnt)
                .rename(columns={"state": "count"})
                .unstack(level=-1)
                .fillna(0)
        )
        cn.columns = cn.columns.droplevel()
        rr = (cn.T / cn.T.sum(axis=0)).T.astype(float)
        # Plotly-native stacked area + line (avoids legacy third-party dataframe plotting)
        palette = pc.qualitative.Plotly
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.32, 0.68],
        )
        for i, col in enumerate(rr.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=rr.index,
                    y=rr[col],
                    name=str(col),
                    mode="lines",
                    stackgroup="dp_states",
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["distance"].astype("float64"),
                name="distance",
                mode="lines",
                line=dict(color="#636EFA"),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Location",
            rangemode="tozero",
            tickformat=".1%",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Distance (km)",
            rangemode="tozero",
            row=2,
            col=1,
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10, pad=0),
        )
        if to_html:
            if path is None:
                raise Exception(
                    """when to_html is True then path must be given with .html extension"""
                )
            else:
                fig.write_html(file=path)
        return fig

    def sgplot_ga(self, tscode, rng=None, to_html=False, path=None):
        """
        Plot of a single grid availability profile.

        Args:
            tscode (str): Time series code. Eg. 'ev_user_W3_85e59_avai_65t2p'
            rng (tuple): (a,b) index if only part of timeseries should be copied. Defaults to None.
            to_html (bool): Save as a html file. Defaults to False.
            path (str): Path if plot should be saved to file. Defaults to None.

        Returns:
            plotly.plot: Plot object.
        """
        self.db.update()
        if rng is None:
            df = self.db.db[tscode]["timeseries"].copy()
        else:
            df = self.db.db[tscode]["timeseries"].iloc[rng[0]: rng[1]].copy()
        if self.db.db[tscode]["kind"] != "availability":
            raise Exception(
                "code '{}' does not correspond to a grid availability profile".format(
                    tscode
                )
            )

        dt = df[["state", "consumption", "charging_point", "charging_cap", "soc"]]
        cnt = dt.groupby([dt.index, "state"])["state"].count()
        cn = (
            pd.DataFrame(cnt)
                .rename(columns={"state": "count"})
                .unstack(level=-1)
                .fillna(0)
        )
        cn.columns = cn.columns.droplevel()
        rr = (cn.T / cn.T.sum(axis=0)).T.astype(float)
        dk = dt[["consumption", "charging_cap"]]
        dd = pd.to_numeric(dt["soc"], errors="coerce").astype("float64")
        palette = pc.qualitative.Plotly
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.28, 0.36, 0.36],
        )
        for i, col in enumerate(rr.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=rr.index,
                    y=rr[col],
                    name=str(col),
                    mode="lines",
                    stackgroup="ga_states",
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=dk.index,
                y=pd.to_numeric(dk["consumption"], errors="coerce").astype("float64"),
                name="consumption",
                mode="lines",
                line=dict(color="#636EFA"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dk.index,
                y=pd.to_numeric(dk["charging_cap"], errors="coerce").astype("float64"),
                name="charging_cap",
                mode="lines",
                line=dict(color="#EF553B"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                name="soc",
                mode="lines",
                line=dict(color="#00CC96"),
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(
            tickfont=dict(family="Arial, sans-serif", size=13, color="black"),
            row=3,
            col=1,
        )
        fig.update_yaxes(
            title=dict(text="Location", font=dict(size=12)),
            showgrid=False,
            showline=True,
            rangemode="tozero",
            zeroline=True,
            tickformat=".1%",
            tickfont=dict(family="Arial, sans-serif", size=12, color="black"),
            linewidth=2,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title=dict(text="Grid Availability (kW)", font=dict(size=12)),
            showgrid=True,
            showline=True,
            rangemode="tozero",
            tickfont=dict(family="Arial, sans-serif", size=12, color="black"),
            linewidth=2,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title=dict(text="SOC", font=dict(size=12)),
            showgrid=True,
            showline=True,
            rangemode="tozero",
            tickformat=".1%",
            tickfont=dict(family="Arial, sans-serif", size=12, color="black"),
            linewidth=2,
            row=3,
            col=1,
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10, pad=0),
        )
        if to_html:
            if path is None:
                raise Exception(
                    """when to_html is True then path must be given with .html extension"""
                )
            else:
                fig.write_html(file=path)
        return fig

    def sgplot_ged(self, tscode, rng=None, to_html=False, path=None):
        """
        Plot of grid electricity demand profiles associated with the same grid availability profile.

        Args:
            tscode (str): Time series code. Eg. 'ev_user_W3_85e59_avai_65t2p'
            rng (tuple): (a,b) index if only part of timeseries should be copied. Defaults to None.
            to_html (bool): Save as a html file. Defaults to False.
            path (str): Path if plot should be saved to file. Defaults to None.

        Returns:
            plotly.plot: Plot object.
        """
        self.db.update()
        if self.db.db[tscode]["kind"] != "charging":
            raise Exception(
                "code '{}' does not correspond to a grid electricity demand profile".format(
                    tscode
                )
            )
        df = pd.DataFrame()
        availcode = self.db.db[tscode]["input"]
        for k in self.db.db.keys():
            if self.db.db[k]["kind"] == "charging":
                if self.db.db[k]["input"] == availcode:
                    tmp = self.db.db[k]["timeseries"].copy()
                    tmp.loc[:, "option"] = self.db.db[k]["option"]
                    df = pd.concat([df,tmp])

        if rng is None:
            pass
        else:
            df = df.iloc[rng[0] : rng[1]].copy()

        dt = df[["state", "actual_soc", "charge_grid", "option"]]
        dt = dt.astype(dtype={'actual_soc':float, 'charge_grid':float})
        cnt = dt.groupby([dt.index, "state"])["state"].count()
        cn = (
            pd.DataFrame(cnt)
            .rename(columns={"state": "count"})
            .unstack(level=-1)
            .fillna(0)
        )
        cn.columns = cn.columns.droplevel()
        rr = (cn.T / cn.T.sum(axis=0)).T.astype(float)
        dff = dt.pivot_table(
            index=dt.index, columns="option", values="actual_soc", aggfunc="sum"
        )
        dg = dt.pivot_table(
            index=dt.index, columns="option", values="charge_grid", aggfunc="sum"
        )
        palette = pc.qualitative.Plotly
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.32, 0.44, 0.24],
        )
        for i, col in enumerate(dff.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=dff.index,
                    y=pd.to_numeric(dff[col], errors="coerce").astype("float64"),
                    name="{} (SOC)".format(col),
                    mode="lines",
                    line=dict(color=color),
                ),
                row=1,
                col=1,
            )
        for i, col in enumerate(dg.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=dg.index,
                    y=pd.to_numeric(dg[col], errors="coerce").astype("float64"),
                    name="{} (kW)".format(col),
                    mode="lines",
                    line=dict(color=color),
                ),
                row=2,
                col=1,
            )
        for i, col in enumerate(rr.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=rr.index,
                    y=rr[col],
                    name=str(col),
                    mode="lines",
                    stackgroup="ged_loc",
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
                ),
                row=3,
                col=1,
            )
        fig.update_xaxes(
            tickfont=dict(family="Arial, sans-serif", size=14, color="black"),
            row=3,
            col=1,
        )
        fig.update_yaxes(
            title=dict(text="SOC", font=dict(size=14)),
            showgrid=False,
            showline=True,
            rangemode="tozero",
            zeroline=True,
            tickformat=".1%",
            tickfont=dict(family="Arial, sans-serif", size=14, color="black"),
            linewidth=2,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title=dict(text="Actual charge (kW)", font=dict(size=14)),
            showgrid=True,
            showline=True,
            rangemode="tozero",
            tickfont=dict(family="Arial, sans-serif", size=12, color="black"),
            linewidth=2,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title=dict(text="Location", font=dict(size=14)),
            showgrid=True,
            showline=True,
            rangemode="tozero",
            tickformat=".1%",
            tickfont=dict(family="Arial, sans-serif", size=12, color="black"),
            linewidth=2,
            row=3,
            col=1,
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10, pad=0),
            showlegend=True,
        )
        if to_html:
            if path is None:
                raise Exception(
                    """when to_html is True then path must be given with .html extension"""
                )
            else:
                fig.write_html(file=path)
        return fig


    def sankey(self, tscode, include=None, to_html=False, path=None):
        """
        Plot of sankey diagram which shows the energy consumption flows.

        Args:
            tscode (str): Time series code. Eg. 'ev_user_W3_85e59_avai_65t2p'
            include (int): Index which part to include. Defaults to None.
            to_html (bool): Save as a html file. Defaults to False.
            path (str): Path if plot should be saved to file. Defaults to None.

        Returns:
            plotly.plot: Plot object.
        """
        self.db.update()
        distance, consumption, rate, label, source, target, value = balance(
            self.db, tscode, include=include
        )

        link = dict(source=source, target=target, value=value)
        node = dict(label=label, pad=50, thickness=10)
        data = go.Sankey(link=link, node=node)
        fig = go.Figure(data)
        if to_html:
            if path is None:
                raise Exception(
                    """when to_html is True then path must be given with .html extension"""
                )
            else:
                fig.write_html(file=path)
        logger.info(f"Consumption [kWh]: {round(consumption,3)}")
        logger.info(f"Distance [km]: {round(distance,3)}")
        logger.info(f"Specific consumption [kWh/100 km]: {round(rate,3)}")
        return fig

    def overview(self, tscode, date_range=None, to_html=False, path=None, share_x=True):
        """
        Plot overview of all time series of a vehicle. Give one charging name profile and it collects upstream related profile names.
        
        args:
        
            tscode(string): Time series code. Eg. 'ev_user_W3_85e59_avai_65t2p'
            date_range(list): List of two datetime objects. Defaults to None. E.g. [datetime.datetime(2019, 1, 1, 0, 0), datetime.datetime(2019, 1, 31, 23, 59)]
            to_html(bool): Save as a html file. Defaults to False.
            path(string): Path if plot should be saved to file. Defaults to None.
            share_x(bool): Share x axis. Defaults to True.
            
        returns:
            plotly.plot: Plot object.

        """
        self.db.update()
        gavailability_name = self.db.db[tscode]['input']
        consumption_name = self.db.db[gavailability_name]['input']

        ts = self.db.db[gavailability_name]['timeseries'].copy().reset_index(drop=False).rename(
            columns={'date': 'datetime', 'hh': 'hr'})

        logger.info(f"Actual time-series date range = [{ts.datetime.min()},{ts.datetime.max()}]")
        if date_range is None:
            start = ts.datetime.min()
            end = ts.datetime.max()
        else:
            start = date_range[0]
            end = date_range[1]

        cons = self.db.db[consumption_name]

        from .consumption import include_weather, Weather
        wt = Weather()
        D = wt.humidair_density
        temp_arr = wt.temp(cons['weather_country'], cons['weather_year'])
        pres_arr = wt.pressure(cons['weather_country'], cons['weather_year'])
        dp_arr = wt.dewpoint(cons['weather_country'], cons['weather_year'])
        hum_arr = wt.calc_rel_humidity(dp_arr, temp_arr)
        r_ha = wt.humidair_density(temp_arr, pres_arr, h=hum_arr)
        dfs = include_weather(ts, cons['refdate'], temp_arr, pres_arr, dp_arr, hum_arr, r_ha)

        cdf = self.db.db[consumption_name]['profile'].copy()
        _mg = pd.merge_asof(
            dfs,
            cdf[["datetime", "speed km/h"]],
            on="datetime",
            tolerance=pd.Timedelta("900s"),
            direction="nearest",
        )
        for _col in _mg.columns:
            if _col == "datetime":
                continue
            _ser = _mg[_col]
            if pd.api.types.is_numeric_dtype(_ser):
                _mg[_col] = _ser.fillna(0.0)
            else:
                _mg[_col] = _ser.where(pd.notna(_ser), 0.0)
        dfg = _mg.set_index("datetime")  
        df = pd.DataFrame()
        availcode = self.db.db[tscode]["input"]
        for k in self.db.db.keys():
            if self.db.db[k]["kind"] == "charging":
                if self.db.db[k]["input"] == availcode:
                    tmp = self.db.db[k]["timeseries"].copy()
                    tmp.loc[:, "option"] = self.db.db[k]["option"]
                    df = pd.concat([df,tmp])

        dt = df[["state", "actual_soc", "charge_grid", "option"]]
        dt = dt.astype(dtype={'actual_soc':float, 'charge_grid':float})
        dff = dt.pivot_table(index=dt.index, columns="option", values="actual_soc", aggfunc="sum")
        dg = dt.pivot_table(index=dt.index, columns="option", values="charge_grid", aggfunc="sum")

        dg.loc[:, 'Grid Availability'] = dfg["charging_cap"]
        cnt = dfg.groupby([dfg.index, "state"])["state"].count()
        cn = (
            pd.DataFrame(cnt)
                .rename(columns={"state": "count"})
                .unstack(level=-1)
                .fillna(0)
        )
        cn.columns = cn.columns.droplevel()
        rr = (cn.T / cn.T.sum(axis=0)).T.astype(float)

        # imput is the name of grid demand time series (charging class) and database 'db'
        # rr, dfg, dff, dg are dataframes resulting from a preprocessing step

        rr_s = rr.loc[start:end]
        dfg_s = dfg.loc[start:end]
        dg_s = dg.loc[start:end]
        dff_s = dff.loc[start:end]

        palette = pc.qualitative.Plotly
        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True if share_x else False,
            vertical_spacing=0.04,
            specs=[[{"secondary_y": True}] for _ in range(5)],
        )

        for i, col in enumerate(rr_s.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=rr_s.index,
                    y=rr_s[col],
                    name=str(col),
                    mode="lines",
                    stackgroup="ov_row1",
                    line=dict(width=0.6, color=color),
                    fillcolor=color,
                    hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(
                x=dfg_s.index,
                y=pd.to_numeric(dfg_s["distance"], errors="coerce").astype("float64"),
                name="Distance",
                mode="lines",
                line=dict(color="green", width=2.2),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=dfg_s.index,
                y=pd.to_numeric(dfg_s["consumption"], errors="coerce").astype("float64"),
                name="Consumption",
                mode="lines",
                line=dict(color="pink", width=1.2),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=dfg_s.index,
                y=pd.to_numeric(dfg_s["temp_degC"], errors="coerce").astype("float64"),
                name="Temperature",
                mode="lines",
                line=dict(color="purple", width=2),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=dfg_s.index,
                y=pd.to_numeric(dfg_s["speed km/h"], errors="coerce").astype("float64"),
                name="Average speed",
                mode="lines",
                line=dict(color="#9c8830", width=2),
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

        for i, col in enumerate(dg_s.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=dg_s.index,
                    y=pd.to_numeric(dg_s[col], errors="coerce").astype("float64"),
                    name=str(col),
                    mode="lines",
                    line=dict(width=0.6, color=color),
                ),
                row=4,
                col=1,
                secondary_y=False,
            )

        for i, col in enumerate(dff_s.columns):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=dff_s.index,
                    y=pd.to_numeric(dff_s[col], errors="coerce").astype("float64"),
                    name=str(col),
                    mode="lines",
                    line=dict(width=0.6, color=color),
                    showlegend=True,
                ),
                row=5,
                col=1,
                secondary_y=False,
            )

        fig.update_yaxes(
            title=dict(text="Location", font=dict(color="black")),
            showgrid=False,
            zeroline=True,
            linecolor="black",
            gridcolor="#bdbdbd",
            tickformat=".1%",
            rangemode="tozero",
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title=dict(text="Distance (km)", font=dict(color="green")),
            showgrid=True,
            zeroline=True,
            linecolor="black",
            gridcolor="#bdbdbd",
            rangemode="tozero",
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title=dict(text="Consumption (kWh)", font=dict(color="pink")),
            showgrid=False,
            zeroline=True,
            linecolor="black",
            gridcolor="#bdbdbd",
            rangemode="tozero",
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            title=dict(text="Temp (C)", font=dict(color="purple")),
            showgrid=True,
            zeroline=True,
            linecolor="black",
            gridcolor="#bdbdbd",
            zerolinecolor="black",
            rangemode="tozero",
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title=dict(text="Speed (km/h)", font=dict(color="#9c8830")),
            showgrid=False,
            zeroline=True,
            linecolor="black",
            gridcolor="#bdbdbd",
            rangemode="tozero",
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            title=dict(text="Power rating (kW)", font=dict(color="black")),
            showgrid=True,
            zeroline=True,
            linecolor="black",
            gridcolor="#bdbdbd",
            rangemode="tozero",
            row=4,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title=dict(text="SOC", font=dict(color="black")),
            showgrid=True,
            zeroline=True,
            linecolor="black",
            gridcolor="#bdbdbd",
            tickformat=".1%",
            rangemode="tozero",
            row=5,
            col=1,
            secondary_y=False,
        )

        fig.update_xaxes(showgrid=True, zeroline=True, linecolor='black', gridcolor='#bdbdbd')
        fig.update_yaxes(rangemode='tozero')

        fig["layout"].update(
            {
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "margin": dict(l=10, r=10, t=20, b=10, pad=0),
                # 'width': 1300,
                'height': 1000, 
                'showlegend': True
            }
        )

        if to_html:
            if path is None:
                raise Exception("""when to_html is True then path must be given with .html extension""")
            else:
                fig.write_html(file=path)

        return fig

