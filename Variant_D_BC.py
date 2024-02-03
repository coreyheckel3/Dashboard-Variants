import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
# import dash_katex
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
# import json
# import io
# import csv
# import flask
import math
import dash_table

global projectile_mass, release_angle, arm_length, spring_constant
# import pandas as pd
global n_clicks
# import dash
# import dash_table
import random, string
import urllib

# import datetime

columns = ['Mass of Projectile(kg)', 'Release Angle(deg)', 'Length of Arm(m)', 'Spring Constant', 'Height(m)',
           'Distance(m)', 'Cost(USD $)', 'Catapult Weight(kg)', 'Time', 'E']
table = dash_table.DataTable(columns=[{"name": column, "id": column} for column in columns], data=[], id="table")
pre_nclicks = 0


def timer(i):
    minute, second = divmod(i, 60)
    stopwatch = '%02i:%02i' % (minute, second)
    return stopwatch


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H3('Catapult Design Dashboard', style={
        'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'system-ui'}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Instructions', children=[
            html.Div([

                html.Div(children='_', style={'color': 'white'}),
                dcc.Markdown('### Task'),
                # html.Div(children ='Task: You are tasked with creating a catapult design that reaches at least a maximum height of 20 meters and travels a distance of 50 meters while keeping cost below $120k dollars and weight above 530k kg. Increase the height and distance to as much as possible while meeting the weight and cost requirements.',
                #         style = {'textAlign':'vertical'}
                #     ),
                dcc.Markdown('''
                    >
                    > ##### You are designing a catapult that launches rocks. There are some requirements that have been passed down to you:
                    > ###### &#8226 **Cost:** Your budget is capped at $14,000 \n
                    > ###### &#8226 **Weight:** The entire system cannot weigh more than 40,000 kg (for mobility purposes) \n
                    > ###### &#8226 **Height:** The rocks/projectiles must be able to clear a 200-meter high barrier \n
                    > ###### &#8226 **Distance:** The rocks/projectiles must be able to hit a target at least 800 meters away, but preferably much farther \n
                    > ##### Other than those requirements, your goal is to **maximize the distance** that you can reach. Please read the following steps before moving on.
                    ''', style={'text-align': 'justify'}),
                html.Div(children='_', style={'color': 'white'}),
                dcc.Markdown('### Steps'),
                dcc.Markdown('''
                    >
                    > ###### **1.** In the next tab (labeled "Catapult Design Creator'' toward the top of the page), you will be able to explore how your design decisions affect the system performance. Experiment with the “Design Decision” input sliders on the left until you are satisfied with your design, as displayed through the “System Outputs” on the right. This should take a few minutes. If you are still working on this after 10 minutes, you should just stop, pick a decent design, and move on to the next step. \n
                    > ###### **2.** Once you have found a satisfactory design, click the “SUBMIT” button. An ID code will then be displayed, which you should write down or copy to your clipboard. \n
                    > ###### **3.** Return to the Qualtrics survey where you started, paste/enter the ID code, and continue on to answer the remaining questions.\n
                    >
                    ''', style={'text-align': 'justify'}),
            ], className='ten columns offset-by-one')]),

        dcc.Tab(label='Catapult Design Creator', children=[

            html.Div(children='_', style={'color': 'white'}),
            html.Div(children='_', style={'color': 'white'}),

            html.Div([  # row
                html.Div([  # slider columns

                    dcc.Markdown('##### **Design Decisions**'),
                    html.Div(children='_', style={'color': 'white'}),

                    html.Div(children=[html.Div(id='slider_output1'), ], style={'margin-top': '-1vw'}),
                    dcc.Slider(
                        id='projectile_mass',
                        marks=None,
                        value=20,
                        step=0.5,
                        min=1,
                        max=100
                    ),
                    # html.Div(children=[html.Div(id='slider_output1'),],style={'margin-top': '-1vw'}),
                    # html.Div(children='_',style={'color':'white'}),
                    html.Div(children='_', style={'color': 'white'}),

                    html.Div(children=[html.Div(id='slider_output2'), ], style={'margin-top': '-1vw'}),
                    # html.Label('Release Angle'),
                    dcc.Slider(
                        id='release_angle',
                        marks=None,
                        value=20,
                        step=0.5,
                        min=10,
                        max=80,
                    ),
                    # html.Div(children=[html.Div(id='slider_output2'),],style={'margin-top': '-1vw'}),
                    # html.Div(children='_',style={'color':'white'}),
                    html.Div(children='_', style={'color': 'white'}),

                    # html.Label('Length of Arm'),
                    html.Div(children=[html.Div(id='slider_output3'), ], style={'margin-top': '-1vw'}),
                    dcc.Slider(
                        id='arm_length',
                        marks=None,
                        value=23,
                        step=0.5,
                        min=9,
                        max=35,
                    ),
                    # html.Div(children=[html.Div(id='slider_output3'),],style={'margin-top': '-1vw'}),
                    # html.Div(children='_',style={'color':'white'}),
                    html.Div(children='_', style={'color': 'white'}),

                    html.Div(children=[html.Div(id='slider_output4'), ], style={'margin-top': '-1vw'}),
                    # html.Label('Spring Constant'),
                    dcc.Slider(
                        id='spring_constant',
                        marks=None,
                        value=2000,
                        step=5,
                        min=1,
                        max=20000,
                    ),
                    # html.Div(children=[html.Div(id='slider_output4'),],style={'margin-top': '-1vw'}),
                    # html.Div(children='_',style={'color':'white'}),
                    html.Div(children='_', style={'color': 'white'}),
                ], className='three columns offset-by-one'),

                html.Div([  # row
                    # html.Div(children='_',style={'color':'white'}),

                    html.Div(children=[
                        dcc.Markdown('##### **Catapult Drawing**'),
                        dcc.Graph(id='catapult_model', config={'displayModeBar': False},
                                  style={'height': '25vh', 'width': '1hw'}),

                    ], className='four columns'),
                    # style={'width': '375px', 'height': '290px', 'display': 'inline-block', 'vertical-align': 'top','margin-left': '9vw'}),
                    html.Div(children=[
                        dcc.Markdown('##### **System Outputs**'),

                        dcc.Graph(id='bar_outcomes', config={'displayModeBar': False})
                    ], className='four columns'),
                    # style={'width': '500px', 'display': 'inline-block', 'vertical-align': 'top'}

                    html.Div([  # column
                        # dcc.Markdown('##### **System Outputs**'),
                        dbc.Table(
                            [html.Thead(html.Tr([html.Th(""), html.Th("")]))] +  # head
                            [html.Tbody([
                                html.Tr([html.Td(['Maximum Height in Air (m)     ']), html.Td(id='MH')]),
                                html.Tr([html.Td(['Maximum Distance Traveled (m)    ']), html.Td(id='MD')]),
                                html.Tr([html.Td(["Cost of Catapult System (10", html.Sup(3), " USD)    "]),
                                         html.Td(id='TC')]),
                                html.Tr([html.Td(["Catapult Weight (10", html.Sup(3), " kg)    "]), html.Td(id='WC')])
                            ], style={'display': 'none'})],
                            bordered=True, hover=True, responsive=True, striped=True, className="p-5"),
                    ],
                        className='three columns'),

                    html.Div(id='stop_watch',
                             style={'display': 'none'}),
                    dcc.Interval(
                        id='time-interval',
                        interval=1000,  # 1000 milliseconds
                        n_intervals=0),
                    dcc.Store(id="cache", data=[]),
                    html.Div(table, style={'display': 'none'}),
                ], className='row'),

                html.Div([html.Div(id='container-button-basic'),
                          html.Button(id='loading-icon', n_clicks=0,
                                      children=[html.Div(
                                          html.A('Submit', id='download-link', download='data.csv', href="",
                                                 target="_blank"))],
                                      style={'font-family': 'Times New Roman', 'font-size': '20px'}),
                          html.Div(["Input: ",
                                    dcc.Input(id='my-input', value='', type='text')], style={'display': 'none'}),
                          html.Div(id='my-output'),
                          ], className='two columns offset-by-one'),
            ])
        ])
    ])
])


@app.callback(
    [dash.dependencies.Output('slider_output1', 'children'),
     dash.dependencies.Output('slider_output2', 'children'),
     dash.dependencies.Output('slider_output3', 'children'),
     dash.dependencies.Output('slider_output4', 'children')],
    [dash.dependencies.Input('projectile_mass', 'value'),
     dash.dependencies.Input('release_angle', 'value'),
     dash.dependencies.Input('arm_length', 'value'),
     dash.dependencies.Input('spring_constant', 'value')])
def update_output(projectile_mass, release_angle, arm_length, spring_constant):
    h = 0  # mangonel base height
    sm = 5

    al = arm_length  # arm length (m)
    am = 1  # arm mass (kg)
    k = spring_constant  # spring constant (N*m/rad)
    ai = np.deg2rad(0)  # intial angle (deg)
    a0 = np.deg2rad(release_angle)
    a1 = release_angle  # release angle (deg)
    g = 9.81  # gravitational acceleration (m/s^2)
    h0 = np.sin(a0) * al

    grav = 9.8
    airDen = 1.225  # kg/m^3
    areaProj = 3.14  # m^2

    dragCoef = 0.5  # Coefficient of sphere

    mass = projectile_mass  # kg
    step = 0.02

    # interior ballistics
    atot = np.deg2rad(90) - a0  # total angle of travel (rad)
    Itot = ((mass ** 2) * al ** 2) + (am * al ** 2) / 3  # total moment of inertia for arm and shot (kg*m^2)
    w = np.sqrt((k * atot ** 2) / Itot)  # angular velocity at release (rad/s)
    v0 = w * al ** 2  # initi
    # #exterior ballistics
    t_flight = (v0 * np.sin(a0)) / g  # time of flight (s)
    h = (v0 ** 2 * np.sin(a0) ** 2) / (2 * g)  # max height (m)
    r = (v0 ** 2 * np.sin(2 * a0)) / g  # range (m)

    t = [0]
    vx = 15 * [(v0 * np.cos(release_angle * np.pi / 180))]
    vy = 15 * [(v0 * np.sin(release_angle * np.pi / 180))]

    x = [0]
    y = [0]

    dragForce = 0.5 * dragCoef * areaProj * (v0 ** 2) * airDen
    ax = 15 * [(-(dragForce * np.cos(release_angle / 180 * np.pi)) / mass)]
    ay = 15 * [(-grav - (dragForce * np.sin(release_angle / 180 * np.pi) / mass))]

    ####Preliminary Equations and functions for Weight Estimation ################

    na = 15  # number of coils

    MG = 2000000000  # modulus of elasticity; 280 Gpa
    wire_diameter = 0.40  # diamter of wire
    LL1 = 5  # leg length
    LL2 = 5  # leg length
    outer_diameter = np.power(((MG * np.power(wire_diameter, 4)) / (na * 8 * k)), (-1 / 3))
    mean_diameter = outer_diameter - wire_diameter
    length_of_wire = ((np.pi) * mean_diameter * na) + LL1 + LL2
    volume_wire = ((np.pi) / 4) * np.power(wire_diameter, 2) * length_of_wire
    density_steel = 8050  # density of steel, kg/m3
    spring_weight = volume_wire * density_steel

    wood_density = 0.74
    width_of_arm = 6
    height_of_arm = 2.5
    arm_mass = al * width_of_arm * height_of_arm * wood_density
    global total_weight
    total_weight = round(arm_mass * 100 + spring_weight + projectile_mass * 100)
    #####################################################

    ####Preliminary Equations and functions for Cost Estimation ################
    global total_cost
    arm_projectile_cost = 0.24 * (arm_mass * 125 + projectile_mass * 225)  # wood with an average rate of $0.24 per kg
    steel_cost = 0.21 * (spring_weight)
    total_cost = round(
        arm_projectile_cost + steel_cost)  # steel with rate of $0.21, current price for hot rolled carbon steel
    #####################################################

    counter = 0

    while (y[counter] >= 0):
        t.append(t[counter] + step)

        vx.append(vx[counter] + step * ax[counter])
        vy.append(vy[counter] + step * ay[counter])

        x.append(x[counter] + step * vx[counter])
        y.append(y[counter] + step * vy[counter])

        global max_height, max_distance
        max_height = round(max(y), 1)
        max_distance = round(max(x), 1)
        vel = np.sqrt(vx[counter + 1] ** 2 + vy[counter + 1] ** 2)
        dragForce = 0.5 * dragCoef * areaProj * (vel ** 2) * airDen

        angle = np.arctan(vy[counter] / vx[counter]) * (180 / np.pi)

        ax.append(-(dragForce * np.cos(release_angle / 180 * np.pi)) / mass)
        ay.append(-grav - (dragForce * np.sin(release_angle / 180 * np.pi) / mass))

        counter = counter + 1
    return 'Mass of Projectile: {} kg'.format(round(projectile_mass / 10, 2)), ' Release Angle: {} degrees'.format(
        release_angle), 'Length of Arm: {} m'.format(arm_length), 'Spring Constant: {}'.format(spring_constant)


@app.callback(
    [Output('MH', 'children'),
     Output('MD', 'children'),
     Output('TC', 'children'),
     Output('WC', 'children')],
    [dash.dependencies.Input('slider_output1', 'children'),
     dash.dependencies.Input('slider_output2', 'children'),
     dash.dependencies.Input('slider_output3', 'children'),
     dash.dependencies.Input('slider_output4', 'children')])
def callback_a(slider_output1, slider_output2, slider_output3, slider_output4):
    output_value = [max_height, max_distance, total_cost, total_weight]
    return output_value


@app.callback(
    dash.dependencies.Output('catapult_model', 'figure'),
    [dash.dependencies.Input('projectile_mass', 'value'),
     dash.dependencies.Input('release_angle', 'value'),
     dash.dependencies.Input('arm_length', 'value'),
     dash.dependencies.Input('spring_constant', 'value')])
def update_cannon_model(projectile_mass, release_angle, arm_length, spring_constant):
    def draw_cannon(projectile_mass, release_angle, arm_length, spring_constant):
        arm_angle = np.deg2rad(180) - release_angle * (math.pi / 180)
        arm_width = arm_length / 10

        box = 0  # barrel origin, in bottom-left corner
        boy = 11
        bsx2 = box + (arm_length / 3) * 2 * math.cos(arm_angle)
        bsy2 = boy + (arm_length / 3) * 2 * math.sin(arm_angle)
        bsx4 = box - arm_width * math.sin(arm_angle)
        bsy4 = boy + arm_width * math.cos(arm_angle)
        bsx3 = bsx4 + (arm_length / 3) * 2 * math.cos(arm_angle)
        bsy3 = bsy4 + (arm_length / 3) * 2 * math.sin(arm_angle)
        bcx0 = (box + bsx4) / 2  # barrel centerline, for munitions
        bcx1 = (bsx2 + bsx3) / 2
        bcy0 = (boy + bsy4) / 2
        bcy1 = (bsy2 + bsy3) / 2

        arm_string = 'M {} {} L {} {} L {} {} L {} {} Z'.format(box, boy, bsx2, bsy2, bsx3, bsy3, bsx4, bsy4)
        global arm_centerline
        arm_centerline = [bcx0, bcx1, bcy0, bcy1]
        return [arm_string, arm_centerline]

    arm_string = draw_cannon(projectile_mass, release_angle, arm_length, spring_constant)[0]
    wdia = 2
    wdia_1 = 2.15 + (projectile_mass / 100)
    wdia_2 = spring_constant / 13000
    neg_angle = -0.5 + (-1 * (release_angle)) / 7
    angle = -3.5 + (-1 * (release_angle)) / 7
    neg_angle2 = 11 - (0.001 * (release_angle / 7))
    neg_angle3 = -2 - (0.001 * (release_angle / 7))
    lengths = arm_length - (spring_constant / 4000)
    return {'data': [],
            'layout': {
                'margin': dict(t=20, b=20, l=20, r=20),
                'xaxis': {
                    'range': [-30, 20],
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False,
                },
                'yaxis': {
                    'range': [0, 39],
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False,
                },
                'shapes': [
                    {'type': 'rect', 'x0': -18, 'y0': 4, 'x1': 6, 'y1': 12, 'fillcolor': 'rgb(164,116,73)',
                     'line': dict(color='none', width=0)},
                    {'type': 'path', 'path': arm_string, 'fillcolor': 'rgb(164,116,73)',
                     'line': dict(color='none', width=0)},
                    {'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': arm_centerline[1] - wdia_1,
                     'y0': arm_centerline[3] - wdia_1, 'x1': arm_centerline[1] + wdia_1,
                     'y1': arm_centerline[3] + wdia_1, 'fillcolor': 'blue', 'line': dict(color='none', width=0)},
                    {'type': 'rect', 'x0': -18, 'y0': 1, 'x1': -12, 'y1': 5, 'fillcolor': 'rgb(164,116,73)',
                     'line': dict(color='none', width=0)},
                    {'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': 0 - wdia - wdia_2, 'y0': 3 - wdia - wdia_2,
                     'x1': 0 + wdia + wdia_2, 'y1': 5 + wdia + wdia_2, 'fillcolor': 'black',
                     'line': dict(color='none', width=0)},
                ]}}


@app.callback(
    [Output('bar_outcomes', 'figure'),
     Output('table', 'data')],
    [Input('MH', 'children'),
     Input('MD', 'children'),
     Input('TC', 'children'),
     Input('WC', 'children')],
    [State('table', 'data'),
     State('projectile_mass', 'value'),
     State('release_angle', 'value'),
     State('arm_length', 'value'),
     State('spring_constant', 'value'),
     State('time-interval', 'n_intervals')]
)
def bar_chart_figure(MH, MD, TC, WC, data, mass, a1, al, k, n_intervals):
    global barcolors
    barcolors = ['red', 'red', 'green', 'green']
    mh = MH * (1 / 15)
    md = MD * 4 * (1 / 15)
    tc = TC / 1000
    wc = WC / 1000

    data.append({'Mass of Projectile(kg)': round((mass / 10), 2), 'Release Angle(deg)': a1, 'Length of Arm(m)': al,
                 'Spring Constant': k, 'Height(m)': MH, 'Distance(m)': MD, 'Cost(USD $)': TC, 'Catapult Weight(kg)': WC,
                 'Time': stopwatch})

    return [
        {'data': [go.Bar(
            x=[wc, tc, mh, md],
            y=['Weight (max 40,000 kg)', 'Costs (max $14,000)', 'Height (min 200 m)', 'Distance (min 800 m)'],

            hoverinfo="none",
            text=['{:,.0f} kg'.format(WC),
                  '${:,.0f}'.format(TC),
                  '{} m'.format(MH),
                  '{} m'.format(MD)],

            textposition='auto',
            textfont=dict(
                size=16,
                color='black',

            ),
            marker_color=barcolors,
            marker_line_color='black',
            marker_line_width=0.5,
            orientation='h')],

            'layout': {
                'xaxis': dict(
                    type='log',
                    range=[0, 6],
                    showticklabels=False,

                ),
                'yaxis': dict(
                    zeroline=True
                ),
                'height': 350,
                'margin': {
                    'l': 150, 'b': 100, 't': 0, 'r': 0
                }
            }},
        data
    ]


@app.callback(Output('stop_watch', 'children'),
              [Input('time-interval', 'n_intervals')])
def update_layout(n):
    global stopwatch
    stopwatch = timer(n)
    return stopwatch


@app.callback(
    [Output('download-link', 'href'),
     Output(component_id='my-output', component_property='children')],
    [Input("table", "data"),
     Input(component_id='my-input', component_property='value'),
     Input('loading-icon', 'n_clicks')])
def update_download_link(data, input_value, nclicks):
    global pre_nclicks
    variant = 'D'
    df = pd.DataFrame.from_dict(data)
    uniqueID = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    df['uniqueID'] = uniqueID
    df['variant'] = variant
    if nclicks > pre_nclicks:
        csv_string = df.to_csv(uniqueID + '.csv', index=False, encoding='utf-8')
        # csv_string = "data:text/csv;charset=utf-8,"+ uniqueID + "%2C" + variant + "%0D" + urllib.parse.quote(csv_string)
        pre_nclicks = nclicks
        return csv_string, 'Code To Enter: {}'.format(uniqueID)
    else:
        csv_string_empty = df.to_csv('overwrite.csv', index=False, encoding='utf-8')
        return csv_string_empty, 'Press submit for your ID code'


if __name__ == '__main__':
    app.run_server(debug=True)