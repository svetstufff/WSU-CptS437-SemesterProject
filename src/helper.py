import mpld3

def clear_last_line():
    print ("\033[A                             \033[A")

def save_graph(fig, filename):
    html_str = mpld3.fig_to_html(fig)
    Html_file = open(f'graphs/{filename}.html',"w")
    Html_file.write(html_str)
    Html_file.close()