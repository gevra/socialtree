import plotly
import plotly.graph_objs as go


def plotly_create_fig(traces, layout, image_height=740, image_width=980, margin_dict=None,
                      shapes_list=None, legend_dict=None, annotations_list=None, save_image_path=None):
    if shapes_list is not None:
        layout['shapes'] = shapes_list
    if legend_dict is not None:
        layout['legend'] = legend_dict
    if annotations_list is not None:
        layout['annotations'] = [go.layout.Annotation(a) for a in annotations_list]
    if margin_dict is not None:
        layout['margin'] = margin_dict
        # margin = dict(
        #     l=80,
        #     r=80,
        #     b=80,
        #     t=100,
        #     pad=0
        # )

    fig = go.Figure(
        data=traces,
        layout=layout
    )
    div = plotly.offline.plot(
        fig, auto_open=False, output_type='div', image_height=image_height, image_width=image_width
    )
    if save_image_path is not None:
        plotly.offline.plot(
            fig,
            filename=save_image_path,
            image='svg',
            auto_open=False,
            image_height=image_height, image_width=image_width
        )
    return div


def plotly_scatter_div(list_of_x_y_name_hovertext_tuples, list_of_x_y_name_hovertext_tuples_y2=None,
                       mode='lines+markers', title=None, image_height=740, image_width=980, margin_dict=None,
                       x_title=None, y_title=None, showlegend=False, autorange=None, y2_title=None,
                       shapes_list=None, legend_dict=None, annotations_list=None, save_image_path=None):
    traces = []
    for x, y, name, hovertext in list_of_x_y_name_hovertext_tuples:
        trace = go.Scatter(
            x=x,
            y=y,
            mode=mode,
            hoverinfo='text',
            name=name
        )
        if hovertext is not None:
            trace['hoverinfo'] = 'text'
            trace['text'] = hovertext
        traces.append(trace)
    layout = go.Layout(
        title=title,
        hovermode='closest',
        showlegend=showlegend,
        xaxis=go.layout.XAxis(showgrid=False, title=x_title, autorange=autorange, ticklen=5, rangemode='tozero'),
        yaxis=go.layout.YAxis(title=y_title, rangemode='tozero'),
    )

    # add the second axis
    if list_of_x_y_name_hovertext_tuples_y2 is not None:
        traces_y2 = []
        for x, y, name, hovertext in list_of_x_y_name_hovertext_tuples_y2:
            trace = go.Scatter(
                x=x,
                y=y,
                mode=mode,
                hoverinfo='text',
                name=name,
                yaxis='y2',
                marker=go.scatter.Marker(symbol="x")
            )
            if hovertext is not None:
                trace['hoverinfo'] = 'text'
                trace['text'] = hovertext
            traces_y2.append(trace)
        traces += traces_y2
        layout['yaxis2'] = dict(
            title=y2_title,
            titlefont=dict(
                color='rgb(148, 103, 189)'
            ),
            tickfont=dict(
                color='rgb(148, 103, 189)'
            ),
            overlaying='y', side='right', showgrid=False, zeroline=False,
            rangemode='tozero'  # align both the y axes at 0
        )

    div = plotly_create_fig(
        traces, layout, image_height=image_height, image_width=image_width, margin_dict=margin_dict,
        shapes_list=shapes_list, legend_dict=legend_dict, annotations_list=annotations_list,
        save_image_path=save_image_path
    )
    return div


def plotly_hist_div(list_of_x_name_xbins_opacity_tuples,
                    barmode=None, title=None, image_height=740, image_width=980, margin_dict=None,
                    x_title=None, y_title=None, showlegend=False, autorange=None,
                    shapes_list=None, legend_dict=None, annotations_list=None, save_image_path=None):
    traces = []
    for x, name, xbins, opacity in list_of_x_name_xbins_opacity_tuples:
        trace = go.Histogram(
            x=x,
            name=name,
            opacity=opacity,
            xbins=xbins,
        )
        traces.append(trace)
    layout = go.Layout(
        title=title,
        barmode=barmode,
        hovermode='closest',
        showlegend=showlegend,
        xaxis=go.layout.XAxis(showgrid=False, title=x_title, autorange=autorange, ticklen=3),
        yaxis=go.layout.YAxis(title=y_title),
    )
    div = plotly_create_fig(
        traces, layout, image_height=image_height, image_width=image_width, margin_dict=margin_dict,
        shapes_list=shapes_list, legend_dict=legend_dict, annotations_list=annotations_list,
        save_image_path=save_image_path
    )
    return div
