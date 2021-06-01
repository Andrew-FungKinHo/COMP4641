#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('load_ext', 'jupyternotify')


# In[2]:


df = pd.read_csv('AMiner-Coauthor.txt', delimiter = "\t",header=None,
                 names=['First author', 'Second author', 'Number of collaborations'])

df['First author'] = df['First author'].str[1:]
df = df.sort_values(by='Number of collaborations', ascending=False)
df = df.reset_index(drop=True)
df = df.apply(pd.to_numeric)

s = pd.Series(df['Number of collaborations'].tolist())
ax = s.plot.kde()

pd.set_option('float_format', '{:f}'.format)
df.describe()


# In[3]:


df.head()


# In[4]:


author_df = pd.read_csv('AMiner-Author.txt', delimiter = "\t",header=None)
author_df[0][:10]
author_df[-10:]


# In[5]:


rows = []
for i in range(1712433):
     rows.append(
                 [
                  int(author_df[0][9 * i][7:]),
                  author_df[0][9 * i+1][3:],
                  author_df[0][9 * i+2][3:],
                  int(author_df[0][9 * i+3][4:]),
                  int(author_df[0][9 * i+4][4:]),
                  int(author_df[0][9 * i+5][4:]),
                  float(author_df[0][9 * i+6][4:]),
                  float(author_df[0][9 * i+7][5:]),
                  author_df[0][9 * i+8][3:],
                 ]
                )

author_data = pd.DataFrame(rows, columns=[
                                 'index id',
                                 'name (separated by semicolons)',
                                 'affiliations',
                                 'published papers',
                                 'total number of citations',
                                 'H-index',
                                 'P-index with equal A-index',
                                 'P-index with unequal A-index',
                                 'research interests'
                                ]
                           )


# In[6]:


author_data.describe()


# In[7]:


author_data[author_data['name (separated by semicolons)'] == 'Kuo-Chen Chou']


# In[8]:


plt.xlim([-100, 100])
# plt.ylim([0, 1.5])
ax = author_data['P-index with equal A-index'].plot.kde()


# In[9]:


def find_author_name_by_index(index_concerned):
    return author_data.loc[author_data['index id'] == f'#index {index_concerned}']['name (separated by semicolons)'].item()[3:8]
find_author_name_by_index(5)


# In[10]:


author_data[author_data['P-index with equal A-index'] == author_data['P-index with equal A-index'].max()]


# In[11]:


row = []
rows = []
buffer = ""
with open("test.txt", "r") as file:
    for line in file:
        
        if line == "":
            row.append(buffer)
            continue
            
        if not line.startswith("#"):
            buffer += line
            
        elif line.startswith("#index"):
            int(line[7:])
            row.append(int(line[7:]))
            
        elif line.startswith("#n"):
            row.append(str(line[3:-1]).rstrip("\n"))
            
        elif line.startswith("#a"):
            row.append(str(line[3:]).rstrip("\n"))
            
        elif line.startswith("#pc"):
            row.append(int(line[4:]))
        
        elif line.startswith("#cn"):
            row.append(int(line[4:]))
            
        elif line.startswith("#hi"):
            row.append(int(line[4:]))
            
        elif line.startswith("#pi"):
            row.append(float(line[4:]))
            
        elif line.startswith("#upi"):
            row.append(float(line[5:]))
            
        elif line.startswith("#t"):
            buffer = line[3:].rstrip("\n")
            row.append(buffer)
#         rows.append(row)
print(row)
count = 0
result = []
bigger_list = []
for i in row:
    if count % 8 == 0:
        bigger_list.append(result)
        result = []
    else:
        result.append(i)
    count += 1
print(bigger_list)


# In[12]:


author_data['index id'].tolist()


# In[13]:


pattern = '#index'
index_list = author_data['index id'].tolist()

for i in range(len(index_list)):
    res = index_list[i].startswith(pattern)
    if res == False:
        print(i)
        print(index_list[i])
        break
    else:
        continue


# In[ ]:





# In[14]:


import networkx as nx
import matplotlib.pyplot as plt

G = nx.from_pandas_edgelist(df,source='First author',target='Second author')


# In[15]:


nx.draw(G,with_labels=True)


# In[16]:


G = nx.MultiGraph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(range(100, 110))
H = nx.path_graph(10)
G.add_nodes_from(H)
G.add_node(H)
keys = G.add_edges_from([(1, 2), (1, 3)])
nx.draw(G,with_labels=True)


# In[ ]:





# In[ ]:





# In[17]:


df


# In[18]:


G = nx.Graph()

for i in range(100):
    G.add_edge(str(df['First author'][i]), str(df['Second author'][i]), weight=int(df['Number of collaborations'][i]))

# G.add_edge("a", "b", weight=0.6)


# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

enormal = [(u, v) for (u, v, d) in G.edges(data=True)]
pos = nx.spring_layout(G, k=0.15, iterations=20)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos)

# edges
# nx.draw_networkx_edges(G, pos, edgelist=enormal,width=6,edge_color="b",style="dashed")
nx.draw_networkx_edges(
    G, pos, edgelist=G.edges(data=True), width=6, alpha=0.5, edge_color="b"
)
# nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6,)
# nx.draw_networkx_edges(
#     G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
# )

edge_labels = nx.get_edge_attributes(G,'weight')

# labels
# nx.draw_networkx_labels(G,pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis("off")
plt.show()


# In[ ]:


get_ipython().run_cell_magic('notify', '', "from matplotlib import pyplot as plt\n\ndef find_author_name_by_index(index_concerned):\n    return author_data.loc[author_data['index id'] == index_concerned]['name (separated by semicolons)'].item()\n\n# find_author_name_by_index(746237)\nG = nx.Graph()\n\nfor i in range(len(df)):\n    G.add_edge(find_author_name_by_index(df['First author'][i]), find_author_name_by_index(df['Second author'][i]), weight=int(df['Number of collaborations'][i]))\n    print(i)\n\n# # G = nx.fast_gnp_random_graph(100, .05)\n# plt.figure(figsize=(20,12))\n# pos = nx.spring_layout(G, k=0.8)\n# nx.draw(G, pos , with_labels = True, width=0.4, \n#         node_color='lightblue', node_size=400)\n# edge_labels = nx.get_edge_attributes(G,'weight')\n\n# # labels\n# # nx.draw_networkx_labels(G,pos)\n# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)\n# plt.show()")


# In[ ]:


author_data


# In[ ]:


from pyvis import network as net
g = net.Network(notebook=True)
nxg = nx.complete_graph(5)
g.from_nx(G)

g.show("example.html")
# net.show_buttons(filter_=['physics'])


# In[ ]:


def draw_graph3(networkx_graph,notebook=True,output_filename='graph.html',show_buttons=False,only_physics_buttons=False):
        from pyvis import network as net

        # make a pyvis network
        pyvis_graph = net.Network(notebook=notebook)
        pyvis_graph.width = '1000px'
        # for each node and its attributes in the networkx graph
        for node,node_attrs in networkx_graph.nodes(data=True):
            pyvis_graph.add_node(node,**node_attrs)
    #         print(node,node_attrs)

        # for each edge and its attributes in the networkx graph
        for source,target,edge_attrs in networkx_graph.edges(data=True):
            # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
            if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                # place at key 'value' the weight of the edge
                edge_attrs['value']=edge_attrs['weight']
            # add the edge
            pyvis_graph.add_edge(source,target,**edge_attrs)

        # turn buttons on
        if show_buttons:
            if only_physics_buttons:
                pyvis_graph.show_buttons(filter_=['physics'])
            else:
                pyvis_graph.show_buttons()

        # return and also save
        return pyvis_graph.show(output_filename)

draw_graph3(G,output_filename='graph_output.html', notebook=True)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pyvis import network as net

df = pd.read_csv('AMiner-Coauthor.txt', delimiter = "\t",header=None,
                names=['First author', 'Second author', 'Number of collaborations'])

df['First author'] = df['First author'].str[1:]
df = df.sort_values(by='Number of collaborations', ascending=False)
df = df.reset_index(drop=True)
df = df.apply(pd.to_numeric)

author_df = pd.read_csv('AMiner-Author.txt', delimiter = "\t",header=None)

rows = []
for i in range(1712433):
     rows.append(
                 [author_df[0][9 * i],
                  author_df[0][9 * i+1],
                  author_df[0][9 * i+2],
                  author_df[0][9 * i+3],
                  author_df[0][9 * i+4],
                  author_df[0][9 * i+5],
                  author_df[0][9 * i+6],
                  author_df[0][9 * i+7],
                  author_df[0][9 * i+8],
                 ]
                )

author_data = pd.DataFrame(rows, columns=[
                                 'index id',
                                 'name (separated by semicolons)',
                                 'affiliations',
                                 'published papers',
                                 'total number of citations',
                                 'H-index',
                                 'P-index with equal A-index',
                                 'P-index with unequal A-index',
                                 'research interests'
                                ]
                           )


def find_author_name_by_index(index_concerned):
    return author_data.loc[author_data['index id'] == f'#index {index_concerned}']['name (separated by semicolons)'].item()[3:8]


G = nx.Graph()

for i in range(1000):
    G.add_edge(find_author_name_by_index(df['First author'][i]), find_author_name_by_index(df['Second author'][i]), weight=int(df['Number of collaborations'][i]))


def map_data(letter_data, ep_color="#03DAC6", ms_color="#da03b3", edge_color="#018786", ep_shape="triangle", ms_shape="box", alg="barnes", buttons=False, recipient_color="#FFA300", recipient_shape="ellipse", cited_color="#DFEE9A", cited_shape="square"):
    g = G
    # g = Network(height="1500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    # g.add_node("Alcuin 1", color="#EB9090")
    # if buttons==True:
    #     g.width = "75%"
    #     g.show_buttons(filter_=["edges"])
    # for letter in letter_data[0:10]:
    #     ep = (letter["ep_num"])[0]
    #     mss = (letter["mss"])
    #     recipients = (letter["recipients"])
    #     people_cited = (letter["people_cited"])
    #     g.add_node(ep, color=ep_color, shape=ep_shape)
    #     g.add_edge("Alcuin 1", ep, color=edge_color)
    #     for ms in mss:
    #         g.add_node(ms, color=ms_color, shape=ms_shape)
    #         g.add_edge(ep, ms, color=edge_color)
    #     for recipient in recipients:
    #         g.add_node(recipient, color=recipient_color, shape=recipient_shape)
    #         g.add_edge(ep, recipient, color=recipient_color)
    #         g.add_edge("Alcuin 1", recipient, color="#EB9090")
    #     for cited in people_cited:
    #         if "Alcuin 1" not in cited:
    #             g.add_node(cited, color=recipient_color, shape=recipient_shape)
    #             g.add_edge(ep, cited, color=cited_color)

    map_algs(g, alg=alg)
    g.set_edge_smooth("dynamic")
    g.show("yayeet.html")


# In[ ]:


import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pyvis import network as net

# G = nx.random_geometric_graph(10, 0.125)
G = nx.complete_graph(100)
# G = nx.Graph()

# for i in range(10):
#     G.add_edge(find_author_name_by_index(df['First author'][i]), find_author_name_by_index(df['Second author'][i]), weight=int(df['Number of collaborations'][i]))

# nodes = list(G.nodes())
# edges = list(G.edges())


edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

# print(edge_x)
# print()
# print(edge_y)

edge_trace = go.Scatter(
    x=edge_x,y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x,y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlOrRd',
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: ",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()


# In[ ]:


import chart_studio.plotly as py
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# return the researchers' name by index id
def find_author_name_by_index(index_concerned):
    return author_data.loc[author_data['index id'] == f'#index {index_concerned}']['name (separated by semicolons)'].item()[3:]

nnodes = 2100
pairs = list(df.loc[:, df.columns != 'Number of collaborations'][0:nnodes].itertuples(index=False,name=None))

# plotting the top 2100 collaborator with the most collaborations betweeen each other
G = nx.Graph()
for i in range(nnodes):
    G.add_edge((df['First author'][i]), (df['Second author'][i]), weight=int(df['Number of collaborations'][i]), length = 1)


    
nodes_list = set([(pair[0]) for pair in pairs] + [(pair[1]) for pair in pairs])

positions = np.random.rand(len(nodes_list),2)
pos = dict(zip(nodes_list,positions))

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)


edge_trace = go.Scatter(x=edge_x,y=edge_y,line=dict(width=0.5, color='#888'),hoverinfo='text',mode='lines',name="Collaboration Edge")

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(x=node_x,y=node_y,mode='markers',hoverinfo='text',
#                         texttemplate = "%{label}: fuck %{value:$,s} <br>(%{percent})",
                marker=dict(showscale=True, colorscale='YlOrRd',color=[],size=10,
                    colorbar=dict(thickness=15,
                                title='Researchers Connections',
                                xanchor='left',
                                titleside='right'
                             ),
                            
                line_width=2)
                )

node_adjacencies = []
node_text = []
edge_text = []

for item in G.edges():
    weight = G.get_edge_data(item[0],item[1])['weight']
    edge_text.append(f'Count: {weight}')

for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f'Researcher: {str(find_author_name_by_index(adjacencies[0]))} # of connections: {str(len(adjacencies[1]))}')
    
node_trace.marker.color = node_adjacencies
node_trace.text = node_text
edge_trace.text = edge_text

print('started plotting')
# fig.show()
data = [edge_trace, node_trace]

py.plot(data, layout=go.Layout(
                title='Research Network graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: ",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)), 
                filename = 'researcher-network', auto_open=False)


# In[ ]:


import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt

nnodes = 30

pairs = list(df.loc[:, df.columns != 'Number of collaborations'][0:nnodes].itertuples(index=False,name=None))

G = nx.Graph()
for i in range(nnodes):
    G.add_edge(find_author_name_by_index(df['First author'][i]), find_author_name_by_index(df['Second author'][i]), weight=int(df['Number of collaborations'][i]), length = 1)
nodes_list = set([find_author_name_by_index(pair[0]) for pair in pairs] + [find_author_name_by_index(pair[1]) for pair in pairs])

positions = np.random.rand(len(nodes_list),2)
pos = dict(zip(nodes_list,positions))

nx.draw(G,pos,with_labels=True, node_color='lightblue')
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'go.Scatter')


# In[ ]:


df.loc[:, df.columns != 'Number of collaborations']


# In[ ]:


citation_net = nx.DiGraph()


# In[ ]:


pd.set_option('display.max_colwidth', None)
def print_basic_information(network):
    print(f'Number of nodes: {nx.number_of_nodes(coauthor_net)}')
    print(f'Number of edges: {nx.number_of_edges(coauthor_net)}')
    print(f'Number of self loops: {nx.number_of_selfloops(coauthor_net)}')
    print(f'denisty: {(nx.density(coauthor_net))}')
    
print_basic_information(G)
# print(coauthor_net.degree(weight='wieght'))


# In[ ]:


# entire coauthor network
coauthor_net=nx.from_pandas_edgelist(df, 'First author', 'Second author', ['Number of collaborations'])


# In[ ]:


# print(f'Number of nodes: {}')
for keys, values in nx.neighbors(coauthor_net,746236):
    print(keys,values)
# print(degree_centrality(coauthor_net))
# print(degree_centrality(coauthor_net))
# print(degree_centrality(coauthor_net))


# In[ ]:


row = []
buffer = ""
with open("test_paper.txt", "r") as file:
    for line in file:
        if line == "":
            row.append(buffer)
            print('yeet')
            continue
            
        elif line.startswith("#index"):
            row.append(int(line[7:]))
            
        elif line.startswith("#*"):
            row.append(line[3:])
            
        elif line.startswith("#@"):
            row.append(line[3:])
            
        elif line.startswith("#t"):
            row.append(line[3:])
            
        elif line.startswith("#c"):
            row.append(line[3:])
            
        elif line.startswith("#%"):
            row.append(int(line[3:]))

print(row)


# In[ ]:


# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# A custom function to calculate
# probability distribution function
def pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    return y_out

# To generate an array of x-values
x = author_data['P-index with equal A-index'].tolist()

# To generate an array of
# y-values using corresponding x-values
y = pdf(x)

# To fill in values under the bell-curve
x_fill = np.arange(-2, 2, 0.1)
y_fill = pdf(x_fill)

# Plotting the bell-shaped curve
plt.style.use('seaborn')
plt.figure(figsize = (6, 6))
plt.plot(x, y, color = 'black',linestyle = 'dashed')

plt.scatter(x, y, marker = 'o',s = 25, color = 'red')

plt.fill_between(x_fill, y_fill, 0, alpha = 0.2, color = 'blue')
plt.show()

