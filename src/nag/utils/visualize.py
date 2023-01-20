# -*- coding: utf-8 -*-
import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_attn_heatmap(
        input_words: List[str],
        output_words: List[str],
        attentions: torch.Tensor,  # len_src x len_tgt
        name='attn',
        show_label=True,
        input_label='src',
        output_label='tgt',
        cmap='bone'):
    assert cmap in ['hot', 'bone', 'cool', 'gray', 'spring', 'summer', 'autumn', 'winter'],\
        "param: \'cmap\' should in [hot, bone, cool, gray, spring, summer, autumn, winter]"
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap=cmap)
    fig.colorbar(cax)
    ax.set_xlabel(xlabel=output_label)
    ax.set_ylabel(ylabel=input_label)
    if show_label:
        # Set up axes
        ax.set_xticklabels([''] + output_words, rotation=45)
        ax.set_yticklabels([''] + input_words, rotation=45)
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(name + "_heatmap.pdf", format="pdf")


def draw_attn_bipartite(
        input_words: List[str],
        output_words: List[str],
        attentions: torch.Tensor,  # len_src x len_tgt
        name='attn',
        show_label=True):
    input_words = [word + '   ' for word in input_words]
    output_words = ['   ' + word for word in output_words]
    attn = attentions.cpu().numpy()
    left, right, bottom, top = .4, .6, .1, .9
    mid = (top + bottom)/2.
    layer_sizes = [len(input_words), len(output_words)]
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)
    src_layer_top = v_spacing*(layer_sizes[0]-1)/2. + mid
    tgt_layer_top = v_spacing*(layer_sizes[1]-1)/2. + mid
    src_layer_left = left
    tgt_layer_left = left + h_spacing
    # add nodes and edges
    G = nx.Graph()
    for i in range(layer_sizes[0]):
        G.add_node(input_words[i], pos=(src_layer_left, src_layer_top - i*v_spacing))
        for j in range(layer_sizes[1]):
            G.add_node(output_words[j], pos=(tgt_layer_left, tgt_layer_top - j*v_spacing))
            G.add_edge(input_words[i], output_words[j], weight=attn[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = [edge[-1]['weight'] for edge in G.edges(data=True)]
    # draw graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = plt.cm.Purples
    nx.draw_networkx_nodes(
        G, pos, node_shape='s', alpha=0)
    edges = nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=0.8, edge_cmap=color_map)
    if show_label:
        nx.draw_networkx_labels(G, pos)

    edges.cmap = color_map
    plt.colorbar(edges, ax=ax)
    plt.savefig(name + "_bipartite.pdf", format="pdf")


def draw_attn_heatmap_subplot(
        fig,
        base_shape: int,
        head_idx: int,
        input_words: List[str],
        output_words: List[str],
        attentions,  # len_src x len_tgt
        show_label=True,
        input_label='src',
        output_label='tgt',
        cmap='bone'):
    ax = fig.add_subplot(base_shape[0], base_shape[1], head_idx)
    cax = ax.matshow(attentions, cmap=cmap)
    if show_label and head_idx == 1:
        ax.set_xlabel(xlabel=output_label, fontsize=5)
        ax.set_ylabel(ylabel=input_label, fontsize=5)
        # Set up axes
        ax.set_xticklabels([''] + output_words, rotation=45, fontsize=5)
        ax.set_yticklabels([''] + input_words, rotation=45, fontsize=5)
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    else:
        ax.set_xticklabels(['']*(len(output_words)+1), rotation=45, fontsize=5)
        ax.set_yticklabels(['']*(len(input_words)+1), rotation=45, fontsize=5)
    return ax, cax


def draw_multihead_attn_heatmap(
        shape: Tuple,
        input_words: List[str],
        output_words: List[str],
        attentions: torch.Tensor,  # nhead x len_src x len_tgt
        name='attn',
        show_label=True,
        input_label='src',
        output_label='tgt',
        cmap='bone'):
    assert shape[0] * shape[1] == attentions.shape[0], 'nheads do not match plot shape!'
    assert cmap in ['hot', 'bone', 'cool', 'gray', 'spring', 'summer', 'autumn', 'winter'],\
        "param: \'cmap\' should in [hot, bone, cool, gray, spring, summer, autumn, winter]"
    ratio = attentions.shape[2] / attentions.shape[1]
    plt.rcParams['figure.figsize'] = (shape[1]*ratio, shape[0])
    plt.rcParams['font.size'] = 5
    attns = attentions.cpu().numpy()
    # Set up figure with colorbar
    fig = plt.figure()
    for i, attn in enumerate(attns):
        ax, cax = draw_attn_heatmap_subplot(
            fig, shape, i+1, input_words, output_words, attn,
            show_label, input_label, output_label, cmap)
    position = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bot, width, height]
    fig.colorbar(cax, cax=position)
    plt.savefig(name + "_heatmap.pdf", format="pdf")


def draw_attn_bipartite_subplot(
        fig,
        base_shape: int,
        head_idx: int,
        input_words: List[str],
        output_words: List[str],
        attn: torch.Tensor,  # len_src x len_tgt
        name='attn',
        show_label=True,
        cmap=plt.cm.Purples):
    left, right, bottom, top = .1, .9, .05, .95
    mid = (top + bottom)/2.
    layer_sizes = [len(input_words), len(output_words)]
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)
    src_layer_top = v_spacing*(layer_sizes[0]-1)/2. + mid
    tgt_layer_top = v_spacing*(layer_sizes[1]-1)/2. + mid
    src_layer_left = left
    tgt_layer_left = left + h_spacing
    # add nodes and edges
    G = nx.Graph()
    for i in range(layer_sizes[0]):
        G.add_node(input_words[i], pos=(src_layer_left, src_layer_top - i*v_spacing))
        for j in range(layer_sizes[1]):
            G.add_node(output_words[j], pos=(tgt_layer_left, tgt_layer_top - j*v_spacing))
            G.add_edge(input_words[i], output_words[j], weight=attn[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = [edge[-1]['weight'] for edge in G.edges(data=True)]
    ax = fig.add_subplot(base_shape[0], base_shape[1], head_idx)
    nx.draw_networkx_nodes(
        G, pos, node_shape='s', alpha=0)
    edges = nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=0.1, edge_cmap=cmap)
    if show_label:
        nx.draw_networkx_labels(G, pos, font_size=plt.rcParams['font.size'])
    return ax, edges


def draw_multihead_attn_bipartite(
        shape: Tuple,
        input_words: List[str],
        output_words: List[str],
        attentions: torch.Tensor,  # len_src x len_tgt
        name='attn',
        show_label=True):
    ratio = attentions.shape[2] / attentions.shape[1] / 1.3
    plt.rcParams['figure.figsize'] = (shape[1]*ratio, shape[0])
    plt.rcParams['font.size'] = 3
    input_words = [word + '   ' for word in input_words]
    output_words = ['   ' + word for word in output_words]
    attns = attentions.cpu().numpy()
    color_map = plt.cm.Purples
    # draw graph
    fig = plt.figure()
    for i, attn in enumerate(attns):
        ax, edges = draw_attn_bipartite_subplot(
            fig, shape, i+1, input_words, output_words, attn, show_label, color_map)
    position = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bot, width, height]
    edges.cmap = color_map
    fig.colorbar(edges, cax=position)
    plt.savefig(name + "_bipartite.pdf", format="pdf")


if __name__ == '__main__':
    src_len = 10
    tgt_len = 15
    nhead = 16
    attn = torch.randn((nhead, src_len, tgt_len))
    src_label = [str(i) for i in range(src_len)]
    tgt_label = [str(i) for i in range(tgt_len)]
    draw_multihead_attn_heatmap(
        shape=(4, 4), input_words=src_label, output_words=tgt_label,
        attentions=attn, name='attn', show_label=True)
    draw_multihead_attn_bipartite(shape=(4, 4), input_words=src_label, output_words=tgt_label,
        attentions=attn, name='attn', show_label=True)
