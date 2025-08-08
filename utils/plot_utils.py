from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def html_color_for_score(score,min_val=None,max_val=None):
    if min_val is not None:
        norm = (score - min_val) / (max_val - min_val)
        norm = np.clip(norm, 0, 1)  # avoid overflow
    else:
        norm = score
    # Linear green (1) to red (0)
    red = int(255 * (1 - norm))
    green = int(255 * norm)
    color = f'rgb({red},{green},0)'
    return color

def html_colored_tokens_with_colorbar(tokens, scores, num_ticks=6,normalize=False):
    # Prepare colorbar ticks
    if normalize:
        ticks = np.linspace(0, 1, num_ticks)
        min_val =None
        max_val = None
        
    else:
        min_val = min(scores)
        max_val = max(scores)
        ticks = np.linspace(min_val, max_val, num_ticks)
        
    tick_labels = ''.join(
        f'<span style="display:inline-block;position:absolute;left:{i/(num_ticks-1)*100}%;transform:translateX(-50%);font-size:11px;">{tick:.2f}</span>'
        for i, tick in enumerate(ticks)
    )
    # Colorbar HTML
    colorbar_html = f"""
    <div style="position: relative; width: 360px; height: 22px; margin-bottom: 6px; margin-top: 6px;
        background: linear-gradient(to right, rgb(255,0,0), rgb(255,255,0), rgb(0,255,0));
        border-radius: 5px; border:1px solid #888;">
      <div style="position: absolute; width: 100%; top: 22px; left: 0;">{tick_labels}</div>
    </div>
    <div style="width: 360px; text-align:center; margin-bottom:10px; font-size:12px; color:#333;">score</div>
    """
    # Token coloring
    colored = [
        f'<span style="background:{html_color_for_score(s,min_val,max_val)}; color:black; padding:2px 6px; margin:2px; border-radius:4px;">{t}</span>'
        for t, s in zip(tokens, scores)
    ]
    html = colorbar_html + '<div style="margin-top:6px;">' + ' '.join(colored) + '</div>'
    display(HTML(html))

def plot_line(data,xlabel,ylabel,labels=[],title=None):
    if data.ndim == 1:
        data = data.reshape(1,-1)
    x = np.arange(data.shape[1])
    sns.set(style="darkgrid")
    for i,y in enumerate(data):
        if not len(labels):
            sns.lineplot(x=x, y=y, marker="o")
        else:
            sns.lineplot(x=x, y=y, marker="o",label = labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(labels):
        plt.legend()
    plt.show()