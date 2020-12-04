import random, os
import numpy as np
from scipy import interp  # interperlation 
from scipy.stats import sem  # compute standard error
import collections

import matplotlib

# Generate images without having a window appear
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.

from matplotlib import pyplot
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray()


def plot_horizontal_bar(df, **kargs): 
    import plotly.plotly as py
    import plotly.graph_objs as go

    # [params]
    plot_description = kargs.get('description', 'Horizontal Histogram')
    
    # if want to configure a set of 'traces'
    trace_params = kargs.get('trace_params', None)
    if trace_params is not None: assert isinstance(trace_params, dict) and len(trace_params) > 0

    # xcol: column in the input dataframe that serves as x (values in the x axis)
    col_y, col_x = kargs.get('xcol', 'counts'), kargs.get('ycol', 'ngrams')

    y_vals = df[col_y].values 
    x_vals = df[col_x].values
    
    # [params]
    save_fig = kargs.get('save_', True)

    ### plotly code ### 
    trace1 = go.Bar(
        y=y_vals,   
        x=x_vals,   
        name=plot_description,
        orientation = 'h',
        # font=dict(size=10),
        marker = dict(
            color = 'rgba(246, 78, 139, 0.6)',
            line = dict(
                color = 'rgba(246, 78, 139, 1.0)',
                width = 3)
        )
    )

    ### if cascading multiple histograms

    # [params]
    # label2 = "Cluster count"
    # col_y, col_x = "ngram", "count"

    # y_vals = df[col_y].values 
    # x_vals = df[col_x].values

    # trace2 = go.Bar(
    #     y=y_vals,
    #     x=x_vals,
    #     name=label2,
    #     orientation = 'h',
    #     # font=dict(size=10),
    #     marker = dict(
    #         color = 'rgba(58, 71, 80, 0.6)',
    #         line = dict(
    #             color = 'rgba(58, 71, 80, 1.0)',
    #             width = 3)
    #     )
    # )

    # data = []
    # for tracei in trace_set: 
    #     data.append(tracei)

    data = [trace1, ]

    layout = go.Layout(
        barmode='stack'
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='marker-h-bar')

    if save_fig: 
        ext_fig = 'pdf'  # Plotly-Python package currently only supports png, svg, jpeg, and pdf

        outputdir, fname = kargs.get('outputdir', None), kargs.get('fname', None)
        if outputdir is None: 
            outputdir = os.getcwd() 
            # fname <- None
        else: 
            outputdir = outputdir.dirname(outputdir)
            if fname is None: 
                fname = outputdir.basename(outputdir)
                
                # not included in outputdir 
                # fname <- None

        assert os.path.exists(outputdir), "invalid directory: %s" % outputdir
        # if not os.path.exists(outputdir): 
        #     print('io> creating new graphic directory %s' % outputdir)
        #     os.makedirs(outputdir) # test directory

        if not fname: # None or '' 
            identifier = kargs.get('identifier', None) 
            if identifier is None: identifier = 'output'
            fname = "%s.%s" % (identifier, ext_fig)
 
        fpath = os.path.join(outputdir, fname)
        print('io> saving figure to %s' % fpath)
        # sns.savefig(fpath)
        # plt.savefig(fpath, dpi=300)

        # (@) Send to Plotly and show in notebook
        # py.iplot(fig, filename=fname)
        # (@) Send to broswer 
        plot_url = py.plot(fig, filename=fname)
        py.image.save_as({'data': data}, fpath)

    plt.close() 

    return 


def t_plotly(params): 
    """


    Memo
    ----
    1. color codes:  

       #2296E4, #2F22E4: dark blue

       #E3BA22, #E6842A: orange yellow, orange

       #B0E422, #70E422: green 



    """

    # (*) To communicate with Plotly's server, sign in with credentials file
    import plotly.plotly as py

    # (*) Useful Python/Plotly tools
    import plotly.tools as tls

    import plotly.graph_objs as go
    # (*) Graph objects to piece together plots
    # from plotly.graph_objs import *

    print('plotly> setting color scheme ...')
    color_marker = params.get('color_marker', None)
    # if color_marker is None: color_marker = '#E3BA22'
    color_err = params.get('color_err', None)
    # if color_err is None: color_err = '#E6842A' 

    print('plotly> selecting plot type ...')
    plotFunc = params.get('plot_type', 'bar')
    plot_type = 'bar'
    if plotFunc.startswith('b'): 
        plotFunc = go.Bar
    elif plotFunc.startswith('sc'):
        print('plotly> selected scatter plot.')
        plotFunc = go.Scatter 
        plot_type = 'scatter'

    trace_params = params.get('trace_params', None)
    if trace_params is None: 
        trace_params = params

    # Make a Bar trace object
    if color_marker and color_err: 
        # text, opacity
        trace1 = plotFunc(
            x=trace_params['x'],  # a list of string as x-coords
            y=trace_params['y'],   # 1d array of numbers as y-coords
            marker=go.Marker(color=color_marker),  # set bar color (hex color model)
            error_y=go.ErrorY(
                type='data',     # or 'percent', 'sqrt', 'constant'
                symmetric=False,
                array=trace_params.get('array', None),
                arrayminus=trace_params.get('arrayminus', None), 
                color=color_err,  # set error bar color
                thickness=0.6
           )
        )
    else: # default color (blue bars and black error bars)

        trace1 = plotFunc(
            x=trace_params['x'],  # a list of string as x-coords
            y=trace_params['y'],   # 1d array of numbers as y-coords
            error_y=go.ErrorY(
                type='data',     # or 'percent', 'sqrt', 'constant'
                symmetric=False,
                array=trace_params.get('array', None),
                arrayminus=trace_params.get('arrayminus', None), 
            )
        )

    # Make Data object
    data = [trace1, ] # go.Data([trace1])

    titleX = params.get('title_x', None)
    if titleX is None: 
        model = params.get('model', None)
        if model is None: 
            model = 'Combined'
        else: 
            model = model.title()
        titleX = "AUCs of the %s Model" % model # plot's title

    titleY = params.get('title_y', None)
    if titleY is None: 
        titleY = 'Area under an ROC Curve (AUC)'

    axis_range = params.get('range', [0.0, 1.0])

    # Make Layout object
    if plot_type.startswith('b'): 
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend

            xaxis = go.XAxis(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),
           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       )
    else: # automatic range assignment
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend

            xaxis = dict(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),

           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       ) 


    # Make Figure object
    fig = go.Figure(data=data, layout=layout)

    # save file
    fpath = params.get('opath', params.get('path', 'roc_bar'))
    base, fname = os.path.dirname(fpath), os.path.basename(fpath)
    assert os.path.exists(base)

    # (@) Send to Plotly and show in notebook
    # py.iplot(fig, filename=fname)
    # (@) Send to broswer 
    plot_url = py.plot(fig, filename=fname)
    
    py.image.save_as({'data': data}, fpath)

    return (fig, data) 


def t_plotly2(params): # performance evaluation 
    # (*) To communicate with Plotly's server, sign in with credentials file
    import plotly.plotly as py

    # (*) Useful Python/Plotly tools
    import plotly.tools as tls

    import plotly.graph_objs as go
    # (*) Graph objects to piece together plots
    # from plotly.graph_objs import *

    print('plotly> setting color scheme ...')
    color_marker = params.get('color_marker', None)
    # if color_marker is None: color_marker = '#E3BA22'
    color_err = params.get('color_err', None)
    # if color_err is None: color_err = '#E6842A' 

    print('plotly> selecting plot type ...')
    plotFunc = params.get('plot_type', 'bar')
    plot_type = 'bar'
    if plotFunc.startswith('b'): 
        plotFunc = go.Bar
    elif plotFunc.startswith('sc'):
        print('plotly> selected scatter plot.')
        plotFunc = go.Scatter 
        plot_type = 'scatter'

    # Make a Bar trace object
    traces = params.get('traces', None)
    data = []
    for trace_params in traces: 
        color_marker_eff = trace_params.get('color_marker', color_marker)
        if color_marker_eff and color_err: 
            # text, opacity
            trace = plotFunc(
                x=trace_params['x'],  # a list of string as x-coords
                y=trace_params['y'],   # 1d array of numbers as y-coords
                marker=go.Marker(color=color_marker_eff),  # set bar color (hex color model)
                error_y=go.ErrorY(
                    type='data',     # or 'percent', 'sqrt', 'constant'
                    symmetric=False,
                    array=trace_params.get('array', None),
                    arrayminus=trace_params.get('arrayminus', None), 
                    color=color_err,  # set error bar color
                    thickness=0.6
               )
            )
        else: # default color (blue bars and black error bars)
            trace = plotFunc(
                x=trace_params['x'],  # a list of string as x-coords
                y=trace_params['y'],   # 1d array of numbers as y-coords
                marker=go.Marker(color=color_marker_eff), 
                error_y=go.ErrorY(
                    type='data',     # or 'percent', 'sqrt', 'constant'
                    symmetric=False,
                    array=trace_params.get('array', None),
                    arrayminus=trace_params.get('arrayminus', None), 
                )
            )

        # Make Data object
        data.append(trace)
        # data = [trace1, ] # go.Data([trace1])

    titleX = params.get('title_x', None)
    if titleX is None: 
        model = params.get('model', None)
        if model is None: 
            model = 'Combined'
        else: 
            model = model.title()
        titleX = "AUCs of the %s Model" % model # plot's title

    titleY = params.get('title_y', None)
    if titleY is None: 
        titleY = 'Area under the Curve'

    axis_range = params.get('range', [0.0, 1.0])

    # Make Layout object
    if plot_type.startswith('b'): 
        print('info> configuring layout for bar plot ...')
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend

            xaxis = go.XAxis(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),
           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       )
    else: # automatic range assignment
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend

            xaxis = dict(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),

           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       ) 


    # Make Figure object
    fig = go.Figure(data=data, layout=layout)

    # save file
    fpath = params.get('opath', params.get('path', 'roc_bar'))
    base, fname = os.path.dirname(fpath), os.path.basename(fpath)
    assert os.path.exists(base)

    # (@) Send to Plotly and show in notebook
    # py.iplot(fig, filename=fname)
    # (@) Send to broswer 
    plot_url = py.plot(fig, filename=fname)
    
    py.image.save_as({'data': data}, fpath)

    return (fig, data) 
